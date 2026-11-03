"""Search indexes: SQLite FTS5 (BM25 keyword) and Qdrant (dense vectors, embedded local mode)."""
from __future__ import annotations

import logging
import re
import shutil
import sqlite3
import threading

from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (Distance, FieldCondition, Filter, MatchAny, PointIdsList, PointStruct,
                                  Range, VectorParams, FilterSelector, MatchValue)

from .config import SETTINGS

COLLECTION = "chunks"
IMAGES = "images"  # one CLIP vector per picture, point id = file id
_TOKEN = re.compile(r"\w+", re.UNICODE)
log = logging.getLogger(__name__)


class KeywordIndex:
    """BM25 keyword search over file names and chunk text, using SQLite FTS5.

    (Earlier versions used Tantivy. On Windows, Tantivy's commits rename and delete index files,
    which fails with "Access is denied" whenever antivirus or another process has a file open.
    SQLite FTS5 ranks with the same BM25 formula and is reliable on Windows.)

    Rows: one per chunk, or one name-only row for a file whose content is not read yet.
    `kw_map` maps each row to its file so a file's rows can be replaced quickly.
    """

    def __init__(self, path):
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)
        self.path = base.parent / "keyword.db"
        self.lock = threading.RLock()
        self.broken = False
        self.last_error = ""
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
        with self.lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.executescript("""
                CREATE VIRTUAL TABLE IF NOT EXISTS kw USING fts5(
                    name, body, tokenize = 'unicode61 remove_diacritics 2');
                CREATE TABLE IF NOT EXISTS kw_map (
                    rid INTEGER PRIMARY KEY, file_id INTEGER NOT NULL, chunk_id INTEGER NOT NULL);
                CREATE INDEX IF NOT EXISTS idx_kw_map_file ON kw_map(file_id);
            """)
            self.conn.commit()

    def is_empty(self) -> bool:
        with self.lock:
            return self.conn.execute("SELECT 1 FROM kw_map LIMIT 1").fetchone() is None

    # ---- writing --------------------------------------------------------
    @staticmethod
    def _name_tokens(name: str) -> str:
        return re.sub(r"[_\-.]+", " ", name)

    def _delete(self, file_id: int):
        rids = [r[0] for r in self.conn.execute("SELECT rid FROM kw_map WHERE file_id=?", (file_id,))]
        if rids:
            self.conn.executemany("DELETE FROM kw WHERE rowid=?", [(r,) for r in rids])
            self.conn.execute("DELETE FROM kw_map WHERE file_id=?", (file_id,))

    def _insert(self, file_id: int, name: str, chunk_ids, texts):
        name_tokens = self._name_tokens(name)
        rows = list(zip(chunk_ids, texts)) or [(0, "")]
        for cid, text in rows:
            cur = self.conn.execute("INSERT INTO kw_map (file_id, chunk_id) VALUES (?, ?)", (file_id, cid))
            self.conn.execute("INSERT INTO kw (rowid, name, body) VALUES (?, ?, ?)", (cur.lastrowid, name_tokens, text))

    def _safely(self, fn) -> bool:
        try:
            with self.lock:
                fn()
                self.conn.commit()
            return True
        except Exception as e:  # noqa: BLE001
            try:
                self.conn.rollback()
            except Exception:  # noqa: BLE001
                pass
            self.broken, self.last_error = True, f"{type(e).__name__}: {e}"
            log.warning("keyword index write failed: %s", self.last_error)
            return False

    def add(self, file_id: int, name: str, chunk_ids: list[int], texts: list[str]):
        self._safely(lambda: (self._delete(file_id), self._insert(file_id, name, chunk_ids, texts)))

    def add_names(self, items: list[tuple[int, str]]):
        def fn():
            for fid, name in items:
                self._delete(fid)
                self._insert(fid, name, [], [])
        self._safely(fn)

    def delete(self, file_id: int):
        self._safely(lambda: self._delete(file_id))

    def delete_many(self, file_ids):
        def fn():
            for fid in file_ids:
                self._delete(fid)
        self._safely(fn)

    def commit(self):
        """Writes are committed as they happen; kept for compatibility."""
        return None

    # ---- repair ----------------------------------------------------------
    def rebuild(self, db, progress=None) -> int:
        """Recreate the keyword index from the files and chunks stored in the main database."""
        files = db.q("SELECT id, name FROM files")
        n = 0
        with self.lock:
            self.conn.execute("DELETE FROM kw")
            self.conn.execute("DELETE FROM kw_map")
            for i, f in enumerate(files, 1):
                rows = db.q("SELECT id, text FROM chunks WHERE file_id=? ORDER BY idx", (f["id"],))
                self._insert(f["id"], f["name"], [r["id"] for r in rows], [r["text"] for r in rows])
                n += max(1, len(rows))
                if i % 2000 == 0:
                    self.conn.commit()
                    if progress:
                        progress(i, len(files))
            self.conn.commit()
            self.broken, self.last_error = False, ""
        if progress:
            progress(len(files), len(files))
        return n

    # ---- reading -------------------------------------------------------------
    def search(self, text: str, limit: int = 60) -> list[tuple[int, int, float]]:
        """Return (chunk_id, file_id, score); higher score = better match (BM25)."""
        tokens = [t for t in _TOKEN.findall(text.lower()) if len(t) > 1][:16]
        if not tokens:
            return []
        query = " OR ".join('"' + t.replace('"', "") + '"' for t in tokens)
        try:
            with self.lock:
                rows = self.conn.execute(
                    "SELECT m.chunk_id, m.file_id, -bm25(kw) AS score FROM kw JOIN kw_map m ON m.rid = kw.rowid "
                    "WHERE kw MATCH ? ORDER BY bm25(kw) LIMIT ?", (query, limit)).fetchall()
            return [(r[0], r[1], float(r[2])) for r in rows]
        except Exception:  # noqa: BLE001
            return []


class VectorIndex:
    """Qdrant in embedded mode: no server process needed (works on Windows)."""

    def __init__(self, path, dim: int):
        path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(path))
        self.dim = dim
        self.lock = threading.RLock()
        if not self.client.collection_exists(COLLECTION):
            self.client.create_collection(COLLECTION, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
        else:
            size = self.client.get_collection(COLLECTION).config.params.vectors.size
            if size != dim:
                raise RuntimeError(f"Vector index was built with dim={size} but settings say {dim}. "
                                   "Delete the data folder and re-index.")

    def add(self, file_id: int, ext: str, modified: float, chunk_ids: list[int], vecs: np.ndarray):
        with self.lock:
            self._add(file_id, ext, modified, chunk_ids, vecs)

    def _add(self, file_id, ext, modified, chunk_ids, vecs):
        self.delete(file_id)
        if not chunk_ids:
            return
        pts = [PointStruct(id=cid, vector=v.tolist(), payload={"file_id": file_id, "ext": ext, "modified": modified})
               for cid, v in zip(chunk_ids, vecs)]
        self.client.upsert(COLLECTION, pts)

    # ---- pictures --------------------------------------------------------
    def _ensure_images(self, dim: int = 512):
        if not self.client.collection_exists(IMAGES):
            self.client.create_collection(IMAGES, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

    def add_image(self, file_id: int, ext: str, modified: float, vec: np.ndarray):
        with self.lock:
            self._ensure_images(len(vec))
            self.client.upsert(IMAGES, [PointStruct(id=file_id, vector=vec.tolist(),
                                                    payload={"file_id": file_id, "ext": ext, "modified": modified})])

    def image_count(self) -> int:
        with self.lock:
            if not self.client.collection_exists(IMAGES):
                return 0
            return self.client.count(IMAGES).count

    def search_images(self, vec: np.ndarray, limit: int = 40, file_ids=None, exts=None,
                      since: float | None = None) -> list[tuple[int, int, float]]:
        must = []
        if exts:
            must.append(FieldCondition(key="ext", match=MatchAny(any=list(exts))))
        if since:
            must.append(FieldCondition(key="modified", range=Range(gte=since)))
        if file_ids is not None:
            must.append(FieldCondition(key="file_id", match=MatchAny(any=list(file_ids))))
        with self.lock:
            if not self.client.collection_exists(IMAGES):
                return []
            res = self.client.query_points(IMAGES, query=vec.tolist(), limit=limit,
                                           query_filter=Filter(must=must) if must else None).points
        return [(0, p.id, float(p.score)) for p in res]

    def delete(self, file_id: int):
        with self.lock:
            if self.client.collection_exists(IMAGES):
                self.client.delete(IMAGES, points_selector=PointIdsList(points=[file_id]))
            self.client.delete(COLLECTION, points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="file_id", match=MatchValue(value=file_id))])))

    def delete_many(self, chunk_file_ids, picture_ids):
        """Bulk removal: one call per 500 files instead of one per file."""
        with self.lock:
            pics = list(picture_ids)
            if pics and self.client.collection_exists(IMAGES):
                for i in range(0, len(pics), 1000):
                    self.client.delete(IMAGES, points_selector=PointIdsList(points=pics[i:i + 1000]))
            ids = list(chunk_file_ids)
            for i in range(0, len(ids), 500):
                self.client.delete(COLLECTION, points_selector=FilterSelector(
                    filter=Filter(must=[FieldCondition(key="file_id", match=MatchAny(any=ids[i:i + 500]))])))

    def search(self, vec: np.ndarray, limit: int = 60, exts=None, since: float | None = None,
               file_ids=None) -> list[tuple[int, int, float]]:
        must = []
        if exts:
            must.append(FieldCondition(key="ext", match=MatchAny(any=list(exts))))
        if since:
            must.append(FieldCondition(key="modified", range=Range(gte=since)))
        if file_ids is not None:
            must.append(FieldCondition(key="file_id", match=MatchAny(any=list(file_ids))))
        with self.lock:
            res = self.client.query_points(COLLECTION, query=vec.tolist(), limit=limit,
                                           query_filter=Filter(must=must) if must else None).points
        return [(p.id, p.payload["file_id"], float(p.score)) for p in res]

    def close(self):
        self.client.close()


def reset_indexes():
    for p in (SETTINGS.tantivy_dir, SETTINGS.data_dir / "keyword.db", SETTINGS.qdrant_dir, SETTINGS.db_path):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink()
