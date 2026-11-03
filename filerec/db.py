"""SQLite store for file metadata, chunk text, file vectors and the user activity log."""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    ext TEXT NOT NULL,
    folder TEXT,
    size INTEGER,
    created REAL,
    modified REAL,
    title TEXT,
    author TEXT,
    n_chunks INTEGER DEFAULT 0,
    content_hash TEXT,
    indexed_at REAL,
    vector BLOB
);
CREATE INDEX IF NOT EXISTS idx_files_ext ON files(ext);
CREATE INDEX IF NOT EXISTS idx_files_mod ON files(modified);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL,
    idx INTEGER NOT NULL,
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);

CREATE TABLE IF NOT EXISTS activity (
    id INTEGER PRIMARY KEY,
    user_id TEXT NOT NULL,
    file_id INTEGER NOT NULL,
    action TEXT NOT NULL,          -- open | edit | click
    ts REAL NOT NULL,
    session_id TEXT,
    query TEXT
);
CREATE INDEX IF NOT EXISTS idx_act_user ON activity(user_id, ts);

CREATE TABLE IF NOT EXISTS query_log (
    id INTEGER PRIMARY KEY,
    user_id TEXT, query TEXT, path_used TEXT, escalations TEXT,
    latency_ms REAL, model_calls INTEGER, ts REAL
);
"""


_SEP = str.maketrans({"_": " ", "-": " ", ".": " "})


def name_stem(name: str) -> str:
    """'Q3_rates-v4.pptx' -> 'q3 rates v4' (fast: no pathlib or regex, called for every file)."""
    base, _ = os.path.splitext(name.lower())
    return base.translate(_SEP)


class DB:
    """Thread-safe thin wrapper (Streamlit + the folder watcher share one connection)."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self._files_cache = None   # all_files() result; writes mark it stale (see all_files)
        self._files_cache_rows = None
        self._files_cache_t = 0.0
        self._stats_cache = (0.0, None)
        with self.lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    # ---- generic ------------------------------------------------------
    def q(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self.lock:
            return self.conn.execute(sql, args).fetchall()

    def x(self, sql: str, args: tuple = ()) -> int:
        with self.lock:
            cur = self.conn.execute(sql, args)
            self.conn.commit()
            return cur.lastrowid

    # ---- files --------------------------------------------------------
    def file_by_path(self, path: str):
        r = self.q("SELECT * FROM files WHERE path=?", (path,))
        return r[0] if r else None

    def files_by_ids(self, ids) -> dict[int, sqlite3.Row]:
        ids = list(ids)
        if not ids:
            return {}
        ph = ",".join("?" * len(ids))
        return {r["id"]: r for r in self.q(f"SELECT * FROM files WHERE id IN ({ph})", tuple(ids))}

    def all_files(self) -> list[sqlite3.Row]:
        """Every file's metadata. On a whole-computer index this is hundreds of thousands of rows,
        and background indexing writes constantly, so a stale copy up to 60 s old is reused rather
        than reloading on every search."""
        with self.lock:
            fresh_enough = self._files_cache is not None or (
                self._files_cache_rows is not None and time.time() - self._files_cache_t < 60)
            if not fresh_enough:
                self._files_cache_rows = self.q(
                    "SELECT id, path, name, ext, folder, modified, created, size, title FROM files")
                self._files_cache_t = time.time()
                self._name_ids = [r["id"] for r in self._files_cache_rows]
                self._name_stems = [name_stem(r["name"]) for r in self._files_cache_rows]
            self._files_cache = self._files_cache_rows
            return self._files_cache_rows

    def name_index(self) -> tuple[list[int], list[str]]:
        """(file ids, cleaned filename stems) for fuzzy filename matching, built once per reload."""
        with self.lock:
            self.all_files()
            return self._name_ids, self._name_stems

    def refresh_files(self):
        """Force all_files() to reload (after a scan or folder indexing)."""
        with self.lock:
            self._files_cache = None
            self._files_cache_rows = None

    def ids_where(self, exts=None, since: float | None = None, until: float | None = None) -> set[int]:
        sql, args = "SELECT id FROM files WHERE 1=1", []
        if exts:
            sql += f" AND ext IN ({','.join('?' * len(exts))})"
            args += list(exts)
        if since:
            sql += " AND modified >= ?"
            args.append(since)
        if until:
            sql += " AND modified < ?"
            args.append(until)
        return {r["id"] for r in self.q(sql, tuple(args))}

    def delete_files(self, ids):
        ids = list(ids)
        with self.lock:
            for i in range(0, len(ids), 900):
                part = ids[i:i + 900]
                ph = ",".join("?" * len(part))
                self.conn.execute(f"DELETE FROM chunks WHERE file_id IN ({ph})", part)
                self.conn.execute(f"DELETE FROM files WHERE id IN ({ph})", part)
            self.conn.commit()
            self._files_cache = None

    def bulk_upsert_meta(self, metas: list[dict]) -> list[tuple[int, str]]:
        """Insert or update many metadata-only rows in one transaction. Returns (id, name) pairs."""
        cols = ["path", "name", "ext", "folder", "size", "created", "modified", "content_hash", "indexed_at"]
        out = []
        with self.lock:
            cur = self.conn.cursor()
            for m in metas:
                vals = [m.get(c) for c in cols]
                row = cur.execute("SELECT id FROM files WHERE path=?", (m["path"],)).fetchone()
                if row:
                    cur.execute(f"UPDATE files SET {','.join(c + '=?' for c in cols)}, n_chunks=0 WHERE id=?",
                                (*vals, row[0]))
                    cur.execute("DELETE FROM chunks WHERE file_id=?", (row[0],))
                    fid = row[0]
                else:
                    cur.execute(f"INSERT INTO files ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", vals)
                    fid = cur.lastrowid
                out.append((fid, m["name"]))
            self.conn.commit()
            self._files_cache = None
        return out

    def upsert_file(self, meta: dict) -> int:
        with self.lock:
            existing = self.file_by_path(meta["path"])
            cols = ["path", "name", "ext", "folder", "size", "created", "modified",
                    "title", "author", "content_hash", "indexed_at"]
            vals = [meta.get(c) for c in cols]
            self._files_cache = None
            if existing:
                sets = ",".join(f"{c}=?" for c in cols)
                self.conn.execute(f"UPDATE files SET {sets} WHERE id=?", (*vals, existing["id"]))
                self.conn.commit()
                return existing["id"]
            ph = ",".join("?" * len(cols))
            cur = self.conn.execute(f"INSERT INTO files ({','.join(cols)}) VALUES ({ph})", vals)
            self.conn.commit()
            return cur.lastrowid

    def replace_chunks(self, file_id: int, texts: list[str]) -> list[int]:
        with self.lock:
            self.conn.execute("DELETE FROM chunks WHERE file_id=?", (file_id,))
            ids = []
            for i, t in enumerate(texts):
                cur = self.conn.execute("INSERT INTO chunks (file_id, idx, text) VALUES (?,?,?)", (file_id, i, t))
                ids.append(cur.lastrowid)
            self.conn.execute("UPDATE files SET n_chunks=? WHERE id=?", (len(texts), file_id))
            self.conn.commit()
            return ids

    def set_file_vector(self, file_id: int, vec: np.ndarray):
        self.x("UPDATE files SET vector=? WHERE id=?", (vec.astype(np.float32).tobytes(), file_id))

    def file_vectors(self, ids=None) -> dict[int, np.ndarray]:
        if ids is None:
            rows = self.q("SELECT id, vector FROM files WHERE vector IS NOT NULL")
        else:
            ids = list(ids)
            if not ids:
                return {}
            ph = ",".join("?" * len(ids))
            rows = self.q(f"SELECT id, vector FROM files WHERE vector IS NOT NULL AND id IN ({ph})", tuple(ids))
        return {r["id"]: np.frombuffer(r["vector"], dtype=np.float32) for r in rows}

    def delete_file(self, file_id: int):
        with self.lock:
            self.conn.execute("DELETE FROM chunks WHERE file_id=?", (file_id,))
            self.conn.execute("DELETE FROM files WHERE id=?", (file_id,))
            self.conn.commit()
            self._files_cache = None

    def chunks_by_ids(self, ids) -> dict[int, sqlite3.Row]:
        ids = list(ids)
        if not ids:
            return {}
        ph = ",".join("?" * len(ids))
        return {r["id"]: r for r in self.q(f"SELECT * FROM chunks WHERE id IN ({ph})", tuple(ids))}

    def first_chunk(self, file_id: int) -> str:
        r = self.q("SELECT text FROM chunks WHERE file_id=? ORDER BY idx LIMIT 1", (file_id,))
        return r[0]["text"] if r else ""

    # ---- activity -----------------------------------------------------
    def log_activity(self, user_id: str, file_id: int, action: str = "open",
                     ts: float | None = None, session_id: str | None = None, query: str | None = None):
        self.x("INSERT INTO activity (user_id, file_id, action, ts, session_id, query) VALUES (?,?,?,?,?,?)",
               (user_id, file_id, action, ts or time.time(), session_id, query))

    def activity(self, user_id: str) -> list[sqlite3.Row]:
        return self.q("SELECT * FROM activity WHERE user_id=? ORDER BY ts", (user_id,))

    def users(self) -> list[str]:
        return [r["user_id"] for r in self.q("SELECT DISTINCT user_id FROM activity ORDER BY user_id")]

    def log_query(self, **kw):
        self.x("INSERT INTO query_log (user_id, query, path_used, escalations, latency_ms, model_calls, ts) "
               "VALUES (?,?,?,?,?,?,?)",
               (kw.get("user_id"), kw.get("query"), kw.get("path_used"), kw.get("escalations"),
                kw.get("latency_ms"), kw.get("model_calls"), time.time()))

    def stats(self, max_age: float = 10.0) -> dict:
        t, cached = self._stats_cache
        if cached is not None and time.time() - t < max_age:
            return cached
        out = {
            "files": self.q("SELECT COUNT(*) c FROM files")[0]["c"],
            "chunks": self.q("SELECT COUNT(*) c FROM chunks")[0]["c"],
            "meta_only": self.q("SELECT COUNT(*) c FROM files WHERE content_hash LIKE 'meta:%'")[0]["c"],
            "events": self.q("SELECT COUNT(*) c FROM activity")[0]["c"],
            "users": len(self.users()),
        }
        self._stats_cache = (time.time(), out)
        return out
