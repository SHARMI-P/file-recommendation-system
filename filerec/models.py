"""Model wrappers: embedder, reranker and Ollama LLM, each with an offline fallback.

Models are loaded lazily and only once, so the fast path never pays for them.
A global call counter lets the UI show how many model calls each query cost.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from collections import Counter

import numpy as np
import requests

from .config import SETTINGS

log = logging.getLogger(__name__)
_TOKEN = re.compile(r"[a-z0-9]+")


class CallCounter:
    def __init__(self):
        self.c = Counter()

    def add(self, kind: str, n: int = 1):
        self.c[kind] += n

    def snapshot(self) -> dict:
        return dict(self.c)


CALLS = CallCounter()


# ----------------------------------------------------------------------------
# Embeddings
# ----------------------------------------------------------------------------
class HashEmbedder:
    """Offline fallback: hashed word + character-trigram features. No download, works anywhere."""

    name = "hash-embedder (offline fallback)"

    def __init__(self, dim: int):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = _TOKEN.findall(text.lower())
        feats = toks + [t[i:i + 3] for t in toks for i in range(max(1, len(t) - 2))]
        for f in feats:
            h = int(hashlib.md5(f.encode()).hexdigest()[:8], 16)
            v[h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_docs(self, texts):
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text):
        return self._vec(text)


class QwenEmbedder:
    """Qwen3-Embedding via sentence-transformers, truncated to `dim` (Matryoshka)."""

    def __init__(self, model_name: str, dim: int):
        from sentence_transformers import SentenceTransformer
        self.name = model_name
        self.dim = dim
        self.model = SentenceTransformer(model_name, device="cpu", truncate_dim=dim)
        self.lock = threading.Lock()  # background indexing and searches share one model

    def embed_docs(self, texts):
        CALLS.add("embed")
        with self.lock:
            return self.model.encode(list(texts), batch_size=8, normalize_embeddings=True,
                                     convert_to_numpy=True).astype(np.float32)

    def embed_query(self, text):
        CALLS.add("embed")
        with self.lock:
            return self.model.encode([text], prompt_name="query", normalize_embeddings=True,
                                     convert_to_numpy=True)[0].astype(np.float32)


_embedder = None
_emb_lock = threading.Lock()


def get_embedder():
    global _embedder
    with _emb_lock:
        if _embedder is None:
            if SETTINGS.embed_backend == "qwen":
                try:
                    _embedder = QwenEmbedder(SETTINGS.embed_model, SETTINGS.embed_dim)
                except Exception as e:  # noqa: BLE001
                    log.warning("Qwen embedder unavailable (%s); using offline hash embedder", e)
                    _embedder = HashEmbedder(SETTINGS.embed_dim)
            else:
                _embedder = HashEmbedder(SETTINGS.embed_dim)
        return _embedder


# ----------------------------------------------------------------------------
# Image embedder (CLIP): pictures and text in the same vector space
# ----------------------------------------------------------------------------
IMAGE_EXTS = {"png", "jpg", "jpeg"}


class ClipEmbedder:
    """Text-to-image search: 'photo of a beach' lands near pictures of beaches."""

    dim = 512

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.name = model_name
        self.model = SentenceTransformer(model_name, device="cpu")
        self.lock = threading.Lock()

    @staticmethod
    def _load(path):
        from PIL import Image
        im = Image.open(path)
        im.draft("RGB", (448, 448))  # JPEG: decode at reduced size, much faster for camera photos
        im = im.convert("RGB")
        im.thumbnail((448, 448))
        return im

    def embed_image(self, path) -> np.ndarray:
        CALLS.add("image")
        im = self._load(path)
        with self.lock:
            return self.model.encode([im], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        CALLS.add("image")
        with self.lock:
            return self.model.encode([text[:200]], normalize_embeddings=True,
                                     convert_to_numpy=True)[0].astype(np.float32)


class HashImageEmbedder:
    """Offline stand-in used in tests: colour histogram for images, hashed words for text."""

    dim = 512
    name = "hash-image (offline fallback)"

    def embed_image(self, path) -> np.ndarray:
        from PIL import Image
        im = Image.open(path).convert("RGB").resize((32, 32))
        hist = np.array(im.histogram(), dtype=np.float32)[: self.dim]
        v = np.zeros(self.dim, np.float32)
        v[: len(hist)] = hist
        return v / (np.linalg.norm(v) or 1)

    def embed_text(self, text: str) -> np.ndarray:
        return HashEmbedder(self.dim).embed_query(text)


_image_embedder = None
_image_failed = False
_img_lock = threading.Lock()


def get_image_embedder():
    """CLIP model, or None if image search is off or the model cannot load (images stay name-only)."""
    global _image_embedder, _image_failed
    with _img_lock:
        if _image_embedder is None and not _image_failed:
            if SETTINGS.image_backend == "clip":
                try:
                    _image_embedder = ClipEmbedder(SETTINGS.image_model)
                except Exception as e:  # noqa: BLE001
                    log.warning("CLIP image model unavailable (%s); images are searchable by name only", e)
                    _image_failed = True
            elif SETTINGS.image_backend == "hash":
                _image_embedder = HashImageEmbedder()
            else:
                _image_failed = True
        return _image_embedder


# ----------------------------------------------------------------------------
# Reranker
# ----------------------------------------------------------------------------
class OverlapReranker:
    """Offline fallback: query-term coverage with a length penalty. Scores in [0, 1]."""

    name = "overlap-reranker (offline fallback)"

    def score(self, query: str, docs: list[str]) -> list[float]:
        q = set(_TOKEN.findall(query.lower()))
        out = []
        for d in docs:
            dt = Counter(_TOKEN.findall(d.lower()))
            hit = sum(1 for t in q if t in dt) / max(1, len(q))
            out.append(float(hit))
        return out


class QwenReranker:
    """Qwen3-Reranker: a small LM that outputs P(yes | query, document)."""

    INSTRUCTION = "Given a user's search for one of their files, judge whether this file content is what they are looking for"

    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(model_name).eval()
        self.yes = self.tok.convert_tokens_to_ids("yes")
        self.no = self.tok.convert_tokens_to_ids("no")
        prefix = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
                  "and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
                  "<|im_start|>user\n")
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.pre = self.tok.encode(prefix, add_special_tokens=False)
        self.suf = self.tok.encode(suffix, add_special_tokens=False)
        self.max_len = 1024

    def score(self, query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        CALLS.add("rerank")
        pairs = [f"<Instruct>: {self.INSTRUCTION}\n<Query>: {query}\n<Document>: {d}" for d in docs]
        out = []
        for i in range(0, len(pairs), 8):
            batch = pairs[i:i + 8]
            enc = self.tok(batch, padding=False, truncation="longest_first", return_attention_mask=False,
                           max_length=self.max_len - len(self.pre) - len(self.suf))
            enc["input_ids"] = [self.pre + ids + self.suf for ids in enc["input_ids"]]
            enc = self.tok.pad(enc, padding=True, return_tensors="pt", max_length=self.max_len)
            with self.torch.no_grad():
                logits = self.model(**enc).logits[:, -1, :]
            pair = self.torch.stack([logits[:, self.no], logits[:, self.yes]], dim=1)
            out.extend(self.torch.nn.functional.log_softmax(pair, dim=1)[:, 1].exp().tolist())
        return out


_reranker = None
_rr_lock = threading.Lock()


def get_reranker():
    global _reranker
    with _rr_lock:
        if _reranker is None:
            if SETTINGS.rerank_backend == "qwen":
                try:
                    _reranker = QwenReranker(SETTINGS.rerank_model)
                except Exception as e:  # noqa: BLE001
                    log.warning("Qwen reranker unavailable (%s); using overlap fallback", e)
                    _reranker = OverlapReranker()
            else:
                _reranker = OverlapReranker()
        return _reranker


# ----------------------------------------------------------------------------
# LLM via Ollama
# ----------------------------------------------------------------------------
class OllamaLLM:
    def __init__(self, model: str, url: str):
        self.model = model
        self.url = url.rstrip("/")
        self._ok = None

    def available(self) -> bool:
        if not SETTINGS.use_llm:
            return False
        if self._ok is None:
            try:
                tags = requests.get(f"{self.url}/api/tags", timeout=2).json()
                names = {m["name"] for m in tags.get("models", [])}
                self._ok = any(n == self.model or n.startswith(self.model + ":") or n.split(":")[0] == self.model
                               for n in names) or self.model in names
                if not self._ok:
                    log.warning("Ollama is running but model %s is not pulled (run: ollama pull %s)",
                                self.model, self.model)
            except Exception:  # noqa: BLE001
                self._ok = False
        return self._ok

    def json(self, system: str, user: str, schema: dict, max_tokens: int = 300) -> dict | None:
        """Structured output: Ollama constrains generation to the JSON schema."""
        if not self.available():
            return None
        CALLS.add("llm")
        try:
            r = requests.post(f"{self.url}/api/chat", timeout=SETTINGS.llm_timeout, json={
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "format": schema,
                "stream": False,
                "think": False,
                "keep_alive": "60m",  # stay in memory between questions (Ollama's default unloads after 5 min)
                "options": {"temperature": 0, "num_predict": max_tokens},
            })
            r.raise_for_status()
            return json.loads(r.json()["message"]["content"])
        except Exception as e:  # noqa: BLE001
            log.warning("LLM call failed: %s", e)
            return None


_llm = None


def get_llm() -> OllamaLLM:
    global _llm
    if _llm is None:
        _llm = OllamaLLM(SETTINGS.llm_model, SETTINGS.ollama_url)
    return _llm
