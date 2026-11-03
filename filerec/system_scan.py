"""Whole-computer indexing in two tiers, plus import of the Windows "Recent files" list.

Tier 1 (minutes): walk every drive and record each document's name, folder, type, size and dates.
          Filename, type and date search then works across the whole computer.
Tier 2 (background, hours on a CPU): read and embed file contents, most useful first:
          recently opened files, then Desktop/Documents/Downloads/OneDrive, then documents
          before code, newest first. Search keeps working while it runs.

System and program folders (Windows, Program Files, AppData, caches, virtual environments...)
are skipped: they hold no user documents and would only slow everything down.
"""
from __future__ import annotations

import hashlib
import logging
import os
import string
import sys
import threading
import time
from pathlib import Path

from .config import SETTINGS
from .models import get_image_embedder

log = logging.getLogger(__name__)

# Folder names never entered (compared lower-case). Names starting with "." or "$" are skipped too.
SKIP_NAMES = {
    "windows", "windows.old", "program files", "program files (x86)", "programdata", "appdata",
    "system volume information", "recovery", "perflogs", "msocache", "config.msi", "intel", "amd",
    "nvidia", "drivers", "onedrivetemp", "$recycle.bin", "$windows.~bt", "$windows.~ws", "$winreagent",
    "node_modules", "__pycache__", "site-packages", "dist-packages", "venv", "env", ".venv",
    "anaconda3", "miniconda3", "go", "library", "cache", "caches", "temp", "tmp", "logs",
    "filerec_data", ".filerec_data", "my_index", "system_index", "models",
}
# Installed apps and games on any drive: their pictures are app graphics, not the user's
APP_NAMES = {"kingsoft", "steam", "steamapps", "steamlibrary", "epic games", "riot games", "xboxgames",
             "windowsapps", "wpsystem", "origin games", "ubisoft game launcher"}
SKIP_NAMES |= APP_NAMES
# A folder holding .dll files is an installed program (Downloads holds .exe installers but not
# .dll files), so it and everything below it is skipped.
PROGRAM_MARKERS = (".dll",)
DOC_EXT = {"pdf", "docx", "pptx", "xlsx", "csv", "txt", "md"}
# What "Scan whole computer" records by default: documents people write and read. Code, data files
# (csv/json) and images are opt-in, because project folders and datasets hold hundreds of
# thousands of them and would bury the documents (and take weeks to read on a CPU).
DEFAULT_SCAN_EXT = (".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".md")
IMAGE_EXT = {"png", "jpg", "jpeg"}


def scan_exts(include_more: bool = False, include_images: bool = False) -> tuple:
    if include_more:
        return SETTINGS.supported_ext
    return DEFAULT_SCAN_EXT + ((".png", ".jpg", ".jpeg") if include_images else ())


def _sig(size: int, mtime_ns: int) -> str:
    return hashlib.md5(f"{size}-{mtime_ns}".encode()).hexdigest()


def drives() -> list[Path]:
    """Every fixed drive on Windows (C:\\, D:\\ ...); the home folder elsewhere."""
    if os.name != "nt":
        return [Path.home()]
    if hasattr(os, "listdrives"):  # Python 3.12+
        found = [Path(d) for d in os.listdrives()]
    else:
        found = [Path(f"{c}:\\") for c in string.ascii_uppercase if os.path.isdir(f"{c}:\\")]
    return [d for d in found if d.is_dir()]


def user_folders() -> list[Path]:
    home = Path.home()
    out = [home / n for n in ("Desktop", "Documents", "Downloads")]
    try:
        out += [p for p in home.iterdir() if p.is_dir() and p.name.lower().startswith("onedrive")]
    except OSError:
        pass
    return [p for p in out if p.is_dir()]


def _norm(p) -> str:
    return os.path.normcase(os.path.abspath(str(p).strip().strip('"')))


def skip_file() -> Path:
    return SETTINGS.data_dir / "skip_folders.txt"


def user_skips() -> list[str]:
    """Folders the user chose to leave out (one per line in skip_folders.txt)."""
    try:
        lines = skip_file().read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return [ln.strip().strip('"') for ln in lines if ln.strip()]


def save_user_skips(folders: list[str]):
    skip_file().parent.mkdir(parents=True, exist_ok=True)
    skip_file().write_text("\n".join(f.strip().strip('"') for f in folders if f.strip()), encoding="utf-8")


def _skip_prefixes() -> list[str]:
    project = Path(__file__).resolve().parents[1]  # this app, its demo workspace and .venv
    paths = [SETTINGS.data_dir, os.getenv("HF_HOME"), os.getenv("OLLAMA_MODELS"), sys.prefix, project]
    return [_norm(p) for p in paths + user_skips() if p]


def under(path: str, prefixes: list[str]) -> bool:
    p = os.path.normcase(path)
    return any(p == pre or p.startswith(pre.rstrip("\\/") + os.sep) for pre in prefixes)


def walk(root: Path, stop: threading.Event | None = None, exts=None):
    """Yield (path, size, mtime, ctime, mtime_ns) for supported files under root.

    Uses os.scandir for speed, never follows symlinks or Windows junctions (which would loop or
    double-count), and ignores folders it is not allowed to read.
    """
    skip_pref = _skip_prefixes()
    stack = [str(root)]
    exts = tuple(exts or SETTINGS.supported_ext)
    while stack:
        if stop is not None and stop.is_set():
            return
        d = stack.pop()
        try:
            it = os.scandir(d)
        except OSError:
            continue
        with it:
            entries = list(it)
        if d != str(root) and any(e.name.lower().endswith(PROGRAM_MARKERS) for e in entries):
            continue  # an installed program
        for e in entries:
            name = e.name
            try:
                if e.is_symlink() or (hasattr(e, "is_junction") and e.is_junction()):
                    continue
                if e.is_dir(follow_symlinks=False):
                    low = name.lower()
                    if low in SKIP_NAMES or name[:1] in ".$":
                        continue
                    if under(e.path, skip_pref):
                        continue
                    stack.append(e.path)
                elif e.is_file(follow_symlinks=False):
                    dot = name.rfind(".")
                    if dot < 0 or name.startswith("~$"):
                        continue
                    if name[dot:].lower() not in exts:
                        continue
                    st = e.stat(follow_symlinks=False)
                    if name[dot + 1:].lower() in IMAGE_EXT and st.st_size < SETTINGS.image_min_kb * 1024:
                        continue  # icons, thumbnails and UI graphics, not pictures people look for
                    if st.st_size == 0 or st.st_size > SETTINGS.max_file_mb * 1e6:
                        continue
                    ctime = getattr(st, "st_birthtime", st.st_ctime)
                    yield e.path, st.st_size, st.st_mtime, ctime, st.st_mtime_ns
            except OSError:
                continue


def scan_metadata(engine, roots: list[Path] | None = None, progress=None,
                  stop: threading.Event | None = None, include_more: bool = False,
                  include_images: bool = False) -> dict:
    """Tier 1: record name/type/size/dates of every document under the roots.

    include_more=False records documents only (DEFAULT_SCAN_EXT); True adds code, csv/json data
    and images. Files of types no longer included are dropped from the index on the next scan.
    """
    db, kw = engine.db, engine.kw
    roots = roots or drives()
    exts = scan_exts(include_more, include_images)
    existing = {r["path"]: (r["id"], r["content_hash"] or "", 0 if (r["content_hash"] or "").startswith("meta:") else 1)
                for r in db.q("SELECT id, path, content_hash, n_chunks FROM files")}
    seen: set[str] = set()
    batch: list[dict] = []
    counts = {"seen": 0, "new": 0, "changed": 0, "removed": 0, "roots": [str(r) for r in roots]}
    t_last = 0.0

    def flush():
        if not batch:
            return
        # Tier 1 only writes SQLite: filename/type/date search reads it directly, and the keyword
        # index gets each file when its content is read in tier 2.
        db.bulk_upsert_meta(batch)
        batch.clear()

    for root in roots:
        for path, size, mtime, ctime, mtime_ns in walk(root, stop, exts):
            counts["seen"] += 1
            seen.add(path)
            sig = _sig(size, mtime_ns)
            old = existing.get(path)
            if old and old[1].endswith(sig):
                continue
            p = Path(path)
            batch.append({"path": path, "name": p.name, "ext": p.suffix.lower().lstrip("."),
                          "folder": p.parent.name, "size": size, "created": ctime, "modified": mtime,
                          "content_hash": "meta:" + sig, "indexed_at": time.time()})
            counts["changed" if old else "new"] += 1
            if len(batch) >= 500:
                flush()
            now = time.time()
            if progress and now - t_last > 0.5:
                progress(counts["seen"], counts["new"] + counts["changed"], str(p.parent))
                t_last = now
        if stop is not None and stop.is_set():
            break
    flush()

    # Forget files that were deleted, or whose type is no longer included (only under scanned roots)
    if not (stop is not None and stop.is_set()):
        root_pref = [os.path.normcase(str(r)) for r in roots]
        gone = [fid for path, (fid, _, n) in existing.items()
                if path not in seen and any(os.path.normcase(path).startswith(rp) for rp in root_pref)]
        counts["removed"] = remove_ids(engine, gone)
    db.refresh_files()
    engine.bump()
    return counts


def remove_ids(engine, ids) -> int:
    """Drop files from every index in bulk (SQLite, keyword index, vectors, pictures)."""
    ids = list(ids)
    if not ids:
        return 0
    db = engine.db
    with_chunks, pictures = [], []
    for i in range(0, len(ids), 900):
        part = ids[i:i + 900]
        for r in db.q(f"SELECT id, ext, n_chunks FROM files WHERE id IN ({','.join('?' * len(part))})", tuple(part)):
            if r["n_chunks"]:
                with_chunks.append(r["id"])
            if r["ext"] in IMAGE_EXT:
                pictures.append(r["id"])
    engine.vec.delete_many(with_chunks, pictures)
    engine.kw.delete_many(ids)
    db.delete_files(ids)
    db.refresh_files()
    engine.bump()
    return len(ids)


def remove_under(engine, folders: list[str]) -> int:
    """Remove every indexed file inside these folders (used by 'Folders to skip')."""
    prefixes = [_norm(f) for f in folders if f.strip()]
    if not prefixes:
        return 0
    rows = engine.db.q("SELECT id, path FROM files")
    return remove_ids(engine, [r["id"] for r in rows if under(r["path"], prefixes)])


def remove_program_files(engine) -> int:
    """Remove files that sit inside installed-program folders (any folder containing .dll files)."""
    rows = engine.db.q("SELECT id, path FROM files")
    verdict: dict[str, bool] = {}

    def is_program(folder: str) -> bool:
        if folder not in verdict:
            try:
                verdict[folder] = any(n.lower().endswith(PROGRAM_MARKERS) for n in os.listdir(folder))
            except OSError:
                verdict[folder] = False
        return verdict[folder]

    bad = []
    for r in rows:
        parts = Path(r["path"]).parents
        # check the file's folder and its ancestors (below the drive root)
        if any(p.name.lower() in APP_NAMES for p in parts) or \
                any(is_program(str(p)) for p in list(parts)[:-1]):
            bad.append(r["id"])
    return remove_ids(engine, bad)


def picture_folders(db, top: int = 10) -> list[tuple[str, int]]:
    """Where indexed pictures come from, grouped two levels above each picture."""
    counts: dict[str, int] = {}
    for r in db.q("SELECT path FROM files WHERE ext IN ('png','jpg','jpeg')"):
        p = Path(r["path"]).parent
        key = str(p.parent if len(p.parts) > 3 else p)
        counts[key] = counts.get(key, 0) + 1
    return sorted(counts.items(), key=lambda kv: -kv[1])[:top]


# ----------------------------------------------------------------------------
# Tier 2: background content indexing
# ----------------------------------------------------------------------------
class ContentWorker:
    def __init__(self, engine):
        self.engine = engine
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.status = {"running": False, "done": 0, "total": 0, "current": "", "failed": 0, "finished": False}
        self.include_more = False
        self.include_images = False
        self.started_at = 0.0

    @property
    def running(self) -> bool:
        return bool(self.thread and self.thread.is_alive())

    def pending(self) -> list:
        db = self.engine.db
        rows = db.q("SELECT id, path, ext, modified FROM files WHERE content_hash LIKE 'meta:%'")
        allowed = {e.lstrip(".") for e in scan_exts(self.include_more, self.include_images)}
        if get_image_embedder() is None and not SETTINGS.use_ocr:
            allowed -= IMAGE_EXT  # nothing to read in a picture without CLIP or OCR
        rows = [r for r in rows if r["ext"] in allowed]
        recent = {r["file_id"] for r in db.q("SELECT DISTINCT file_id FROM activity")}
        homes = [os.path.normcase(str(p)) for p in user_folders()]

        def key(r):
            path = os.path.normcase(r["path"])
            return (r["id"] not in recent, not any(path.startswith(h) for h in homes),
                    r["ext"] not in DOC_EXT, -(r["modified"] or 0))
        return sorted(rows, key=key)

    def start(self):
        if self.running:
            return
        self.stop.clear()
        self.thread = threading.Thread(target=self._run, daemon=True, name="content-indexer")
        self.thread.start()

    def pause(self):
        self.stop.set()

    def _run(self):
        eng = self.engine
        queue = self.pending()
        self.status.update(running=True, done=0, total=len(queue), failed=0, finished=False)
        self.started_at = time.time()
        try:
            for i, r in enumerate(queue, 1):
                if self.stop.is_set():
                    break
                while eng.searching.is_set() and not self.stop.is_set():
                    time.sleep(0.2)  # give the CPU and the model to the user's search
                self.status["current"] = r["path"]
                try:
                    eng.indexer.index_file(Path(r["path"]), commit=True, max_chunks=SETTINGS.system_max_chunks)
                except Exception as e:  # noqa: BLE001
                    self.status["failed"] += 1
                    log.warning("content indexing failed for %s: %s", r["path"], e)
                    # mark it so it is not retried forever
                    eng.db.x("UPDATE files SET content_hash=? WHERE id=?", ("failed", r["id"]))
                self.status["done"] = i
                if i % 10 == 0:
                    eng.kw.commit()
                    eng.bump()
                time.sleep(0.02)  # stay polite to the foreground app
            self.status["finished"] = not self.stop.is_set()
        finally:
            eng.kw.commit()
            eng.bump()
            self.status["running"] = False
            self.status["current"] = ""


# ----------------------------------------------------------------------------
# Windows Recent files -> activity history
# ----------------------------------------------------------------------------
def _u32(b: bytes, i: int) -> int:
    return int.from_bytes(b[i:i + 4], "little")


def lnk_target(path: Path) -> str | None:
    """Read the target path of a Windows .lnk shortcut (MS-SHLLINK), without extra libraries."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) < 0x4C or data[:4] != b"L\x00\x00\x00":
        return None
    flags = _u32(data, 0x14)
    pos = 0x4C
    if flags & 0x01:  # HasLinkTargetIDList
        pos += 2 + int.from_bytes(data[pos:pos + 2], "little")
    if not flags & 0x02:  # HasLinkInfo
        return None
    li = pos
    hdr, li_flags = _u32(data, li + 4), _u32(data, li + 8)
    if not li_flags & 0x01:  # no local path (network share)
        return None

    def cstr(off: int, wide: bool) -> str:
        if wide:
            end = off
            while end + 1 < len(data) and data[end:end + 2] != b"\x00\x00":
                end += 2
            return data[off:end].decode("utf-16-le", errors="ignore")
        end = data.find(b"\x00", off)
        return data[off:end].decode("mbcs" if os.name == "nt" else "latin-1", errors="ignore")

    if hdr >= 0x24 and _u32(data, li + 28):
        base = cstr(li + _u32(data, li + 28), True)
        suffix = cstr(li + _u32(data, li + 32), True) if _u32(data, li + 32) else ""
    else:
        base = cstr(li + _u32(data, li + 16), False)
        suffix = cstr(li + _u32(data, li + 24), False) if _u32(data, li + 24) else ""
    if suffix and not base.endswith("\\"):
        base += "\\"
    return (base + suffix) or None


def recent_dir() -> Path | None:
    appdata = os.getenv("APPDATA")
    if not appdata:
        return None
    d = Path(appdata) / "Microsoft" / "Windows" / "Recent"
    return d if d.is_dir() else None


def import_recent(engine, user_id: str, folder: Path | None = None) -> dict:
    """Turn the Windows Recent list into 'open' events, so personalization starts on day one."""
    folder = folder or recent_dir()
    out = {"shortcuts": 0, "imported": 0, "already": 0, "new_files": 0, "skipped": 0}
    if folder is None:
        out["error"] = "Windows Recent folder not found (only available on Windows)"
        return out
    try:  # pywin32 resolves every shortcut type; the built-in reader covers normal local files
        import win32com.client  # type: ignore
        shell = win32com.client.Dispatch("WScript.Shell")

        def resolve(p):
            return shell.CreateShortCut(str(p)).Targetpath or lnk_target(p)
    except Exception:  # noqa: BLE001
        resolve = lnk_target

    db = engine.db
    for lnk in folder.glob("*.lnk"):
        out["shortcuts"] += 1
        target = resolve(lnk)
        if not target:
            out["skipped"] += 1
            continue
        tp = Path(target)
        if tp.suffix.lower() not in SETTINGS.supported_ext or not tp.is_file():
            out["skipped"] += 1
            continue
        path = os.path.abspath(target)
        row = db.file_by_path(path)
        if row is None:
            st = tp.stat()
            fid = db.bulk_upsert_meta([{
                "path": path, "name": tp.name, "ext": tp.suffix.lower().lstrip("."), "folder": tp.parent.name,
                "size": st.st_size, "created": getattr(st, "st_birthtime", st.st_ctime), "modified": st.st_mtime,
                "content_hash": "meta:" + _sig(st.st_size, st.st_mtime_ns), "indexed_at": time.time()}])[0][0]
            engine.kw.add(fid, tp.name, [], [])
            out["new_files"] += 1
        else:
            fid = row["id"]
        ts = lnk.stat().st_mtime  # Windows updates the shortcut each time the file is opened
        if db.q("SELECT 1 FROM activity WHERE user_id=? AND file_id=? AND ABS(ts-?)<2", (user_id, fid, ts)):
            out["already"] += 1
            continue
        db.log_activity(user_id, fid, "open", ts=ts, session_id="recent")
        out["imported"] += 1
    engine.kw.commit()
    engine.db.refresh_files()
    engine.bump()
    return out
