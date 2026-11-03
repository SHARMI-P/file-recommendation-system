"""Background ingestion: walk folders, parse, chunk, embed and write to SQLite, Tantivy and Qdrant.

Only new or changed files are re-processed (size + mtime hash), so re-indexing is cheap.
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np

from .config import SETTINGS
from .db import DB
from .models import IMAGE_EXTS, get_embedder, get_image_embedder
from .parsers import chunk, parse
from .store import KeywordIndex, VectorIndex

log = logging.getLogger(__name__)
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".filerec_data", "$RECYCLE.BIN"}

# Windows attributes of OneDrive "online-only" files. Reading such a file makes Windows download it,
# so these are indexed by name and metadata only.
_CLOUD_ONLY = 0x1000 | 0x40000 | 0x400000  # OFFLINE | RECALL_ON_OPEN | RECALL_ON_DATA_ACCESS


def is_cloud_only(st: os.stat_result) -> bool:
    return bool((getattr(st, "st_file_attributes", 0) or 0) & _CLOUD_ONLY)


class Indexer:
    def __init__(self, db: DB, kw: KeywordIndex, vec: VectorIndex):
        self.db, self.kw, self.vec = db, kw, vec
        self.lock = threading.Lock()

    @staticmethod
    def _sig(st: os.stat_result) -> str:
        return hashlib.md5(f"{st.st_size}-{st.st_mtime_ns}".encode()).hexdigest()

    def iter_files(self, root: Path):
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
            for fn in filenames:
                p = Path(dirpath) / fn
                if p.suffix.lower() in SETTINGS.supported_ext and not fn.startswith("~$"):
                    yield p

    def index_file(self, path: Path, commit: bool = True, max_chunks: int | None = None) -> str:
        """Returns 'indexed', 'unchanged' or 'skipped'."""
        path = path.resolve()
        try:
            st = path.stat()
        except FileNotFoundError:
            self.remove_file(path)
            return "skipped"
        if st.st_size > SETTINGS.max_file_mb * 1e6:
            return "skipped"
        sig = self._sig(st)
        existing = self.db.file_by_path(str(path))
        if existing and existing["content_hash"] == sig:
            return "unchanged"

        ext = path.suffix.lower().lstrip(".")
        if ext in IMAGE_EXTS:
            return self._index_image(path, st, sig, commit)
        text, meta = ("", {}) if is_cloud_only(st) else parse(path)
        chunks = chunk(text, SETTINGS.chunk_chars, SETTINGS.chunk_overlap, max_chunks or SETTINGS.max_chunks_per_file)
        # Prefix each chunk with the filename so semantic search also "sees" the name
        name_hint = path.stem.replace("_", " ").replace("-", " ")
        embed_inputs = [f"{name_hint}\n{c}" for c in chunks] or [name_hint]

        with self.lock:
            fid = self.db.upsert_file({
                "path": str(path), "name": path.name, "ext": path.suffix.lower().lstrip("."),
                "folder": path.parent.name, "size": st.st_size,
                "created": getattr(st, "st_birthtime", st.st_ctime), "modified": st.st_mtime,
                "title": meta.get("title"), "author": meta.get("author"),
                "content_hash": sig, "indexed_at": time.time(),
            })
            chunk_ids = self.db.replace_chunks(fid, chunks)
            vecs = get_embedder().embed_docs(embed_inputs)
            if chunks:
                self.vec.add(fid, path.suffix.lower().lstrip("."), st.st_mtime, chunk_ids, vecs)
            else:
                self.vec.delete(fid)
            self.db.set_file_vector(fid, _normalize(vecs.mean(axis=0)))
            self.kw.add(fid, path.name, chunk_ids, chunks)
            if commit:
                self.kw.commit()
        return "indexed"

    def _index_image(self, path: Path, st, sig: str, commit: bool) -> str:
        """A picture: CLIP vector of what it shows (plus OCR text if enabled), searchable by name too."""
        clip = get_image_embedder()
        text, _ = ("", {}) if is_cloud_only(st) else parse(path)  # OCR text when FILEREC_USE_OCR=1
        chunks = chunk(text, SETTINGS.chunk_chars, SETTINGS.chunk_overlap, 5)
        img_vec = None
        if clip is not None and not is_cloud_only(st) and st.st_size >= SETTINGS.image_min_kb * 1024:
            try:
                img_vec = clip.embed_image(path)
            except Exception as e:  # noqa: BLE001  (corrupt or unsupported image)
                log.warning("could not read image %s: %s", path, e)
        with self.lock:
            fid = self.db.upsert_file({
                "path": str(path), "name": path.name, "ext": path.suffix.lower().lstrip("."),
                "folder": path.parent.name, "size": st.st_size,
                "created": getattr(st, "st_birthtime", st.st_ctime), "modified": st.st_mtime,
                "title": None, "author": None, "content_hash": sig, "indexed_at": time.time(),
            })
            chunk_ids = self.db.replace_chunks(fid, chunks)
            self.vec.delete(fid)
            if chunks:  # OCR text is searchable like any document
                name_hint = path.stem.replace("_", " ").replace("-", " ")
                vecs = get_embedder().embed_docs([f"{name_hint}\n{c}" for c in chunks])
                self.vec.add(fid, path.suffix.lower().lstrip("."), st.st_mtime, chunk_ids, vecs)
            if img_vec is not None:
                self.vec.add_image(fid, path.suffix.lower().lstrip("."), st.st_mtime, img_vec)
            self.kw.add(fid, path.name, chunk_ids, chunks)
            if commit:
                self.kw.commit()
        return "indexed"

    def remove_file(self, path: Path):
        row = self.db.file_by_path(str(path))
        if row:
            with self.lock:
                self.kw.delete(row["id"])
                self.kw.commit()
                self.vec.delete(row["id"])
                self.db.delete_file(row["id"])

    def index_folder(self, root: Path, progress=None) -> dict:
        root = Path(root).resolve()
        files = list(self.iter_files(root))
        counts = {"indexed": 0, "unchanged": 0, "skipped": 0, "removed": 0}
        for i, p in enumerate(files):
            counts[self.index_file(p, commit=False)] += 1
            if progress:
                progress(i + 1, len(files), p.name)
        # Drop files that were deleted from disk
        seen = {str(p.resolve()) for p in files}
        for r in self.db.all_files():
            if r["path"].startswith(str(root)) and r["path"] not in seen:
                self.remove_file(Path(r["path"]))
                counts["removed"] += 1
        self.kw.commit()
        return counts


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n else v


class FolderWatcher:
    """Re-indexes files as they change (watchdog). Debounced so a save storm triggers one re-index."""

    def __init__(self, indexer: Indexer, on_change=None):
        self.indexer = indexer
        self.on_change = on_change
        self.observer = None
        self.pending: dict[str, float] = {}
        self.lock = threading.Lock()
        self.roots: set[str] = set()

    def watch(self, root: Path):
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer

        root = str(Path(root).resolve())
        if root in self.roots:
            return
        watcher = self

        class Handler(FileSystemEventHandler):
            def on_any_event(self, event):
                if event.is_directory:
                    return
                for p in {getattr(event, "src_path", None), getattr(event, "dest_path", None)} - {None}:
                    if Path(p).suffix.lower() in SETTINGS.supported_ext:
                        with watcher.lock:
                            watcher.pending[p] = time.time()

        if self.observer is None:
            self.observer = Observer()
            self.observer.daemon = True
            self.observer.start()
            threading.Thread(target=self._drain, daemon=True).start()
        self.observer.schedule(Handler(), root, recursive=True)
        self.roots.add(root)

    def _drain(self):
        while True:
            time.sleep(1.0)
            now = time.time()
            with self.lock:
                ready = [p for p, t in self.pending.items() if now - t > 2.0]
                for p in ready:
                    self.pending.pop(p)
            for p in ready:
                try:
                    status = self.indexer.index_file(Path(p))
                    log.info("watcher: %s %s", status, p)
                    if self.on_change:
                        self.on_change()
                except Exception as e:  # noqa: BLE001
                    log.warning("watcher failed on %s: %s", p, e)
