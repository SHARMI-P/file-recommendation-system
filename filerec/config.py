"""Central settings. Every value can be overridden with an environment variable."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.getenv(f"FILEREC_{name}", default)


@dataclass
class Settings:
    # Where indexes, the SQLite DB and the vector store live
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", ".filerec_data")).resolve())

    # ---- Models -------------------------------------------------------
    # embed_backend: "qwen" (Qwen3-Embedding via sentence-transformers) or "hash" (offline, no download)
    embed_backend: str = _env("EMBED_BACKEND", "qwen")
    embed_model: str = _env("EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    embed_dim: int = int(_env("EMBED_DIM", "512"))  # Matryoshka truncation: smaller = faster

    # rerank_backend: "qwen" (Qwen3-Reranker) or "overlap" (offline lexical fallback)
    rerank_backend: str = _env("RERANK_BACKEND", "qwen")
    rerank_model: str = _env("RERANK_MODEL", "Qwen/Qwen3-Reranker-0.6B")

    # LLM served by Ollama. On a CPU laptop qwen3:1.7b is the responsive choice; qwen3:4b is stronger.
    llm_model: str = _env("LLM_MODEL", "qwen3:1.7b")
    ollama_url: str = _env("OLLAMA_URL", "http://localhost:11434")
    use_llm: bool = _env("USE_LLM", "1") == "1"
    llm_timeout: float = float(_env("LLM_TIMEOUT", "20"))  # give up and use what we have

    # ---- Indexing -----------------------------------------------------
    chunk_chars: int = 1000
    chunk_overlap: int = 150
    max_file_mb: float = 25.0
    max_chunks_per_file: int = 60
    # Whole-computer mode reads at most this many chunks per file (about 15,000 characters):
    # the start of a document is usually enough to find it, and it keeps CPU indexing practical.
    system_max_chunks: int = int(_env("SYSTEM_MAX_CHUNKS", "15"))
    supported_ext: tuple = (
        ".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".txt", ".md",
        ".py", ".js", ".ts", ".java", ".json", ".html", ".sql", ".yaml", ".yml",
        ".png", ".jpg", ".jpeg",
    )
    use_ocr: bool = _env("USE_OCR", "0") == "1"  # needs Tesseract installed

    # ---- Image search (CLIP: pictures and text in one vector space) -------
    image_backend: str = _env("IMAGE_BACKEND", "clip")      # "clip", or "off"
    image_model: str = _env("IMAGE_MODEL", "clip-ViT-B-32")  # about 600 MB, fast enough on a CPU
    image_min_kb: int = int(_env("IMAGE_MIN_KB", "20"))      # smaller images are icons, skipped

    # ---- Router thresholds (tuned on the demo corpus) ------------------
    fast_name_conf: float = 0.86      # fuzzy filename score needed to stop at the fast path
    hybrid_conf: float = 0.55          # confidence needed to stop at the hybrid path
    complexity_fast: float = 0.30      # complexity below this starts at the fast path
    small_set: int = 5                 # if filters leave this many files or fewer, skip the deep path
    complexity_deep: float = 0.70      # complexity at or above this starts at the deep path

    # ---- Personalization ---------------------------------------------
    half_life_days: float = 10.0
    w_relevance: float = 0.75
    w_frequency: float = 0.06
    w_recency: float = 0.06
    w_temporal: float = 0.03
    w_topic: float = 0.04
    w_filetype: float = 0.02
    w_coaccess: float = 0.04
    personal_query_boost: float = 2.0  # multiplies personal weights for "I worked on..." queries

    cache_ttl_s: int = 300

    @property
    def db_path(self) -> Path:
        return self.data_dir / "filerec.db"

    @property
    def tantivy_dir(self) -> Path:
        return self.data_dir / "tantivy"

    @property
    def qdrant_dir(self) -> Path:
        return self.data_dir / "qdrant"


SETTINGS = Settings()
