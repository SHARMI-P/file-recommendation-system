r"""The agentic pipeline as a LangGraph state machine.

    understand -> route -> fast ----(confident)----> rank -> explain -> END
                       \         \--(not confident)--> hybrid
                        \-> hybrid --(confident)-----> rank
                         \         \--(not confident)-> deep
                          \-> deep ------------------> rank

The router chooses the starting path from the query's complexity, then escalates one level at a
time whenever a path's confidence is below its threshold. Every stage records its latency and
model calls in a trace, which the UI shows.
"""
from __future__ import annotations

import atexit
import re
import threading
import time
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from .config import SETTINGS
from .db import DB
from .indexer import FolderWatcher, Indexer
from .models import CALLS
from .personalization import build_profile
from .query_agent import QueryPlan, understand
from .ranking import explain, rank
from .retrieval import allowed_files, deep_path, fast_path, hybrid_path
from .store import KeywordIndex, VectorIndex
from . import system_scan


class State(TypedDict, total=False):
    query: str
    user_id: str
    force_path: str | None
    personalize: bool
    plan: QueryPlan
    profile: Any
    result: Any
    path: str
    escalations: list
    recs: list
    trace: list
    next: str


class Engine:
    """Owns the stores, the indexer, the watcher and the compiled agent graph."""

    def __init__(self):
        SETTINGS.data_dir.mkdir(parents=True, exist_ok=True)
        self.db = DB(SETTINGS.db_path)
        self.kw = KeywordIndex(SETTINGS.tantivy_dir)
        if self.kw.is_empty() and self.db.q("SELECT 1 FROM files LIMIT 1"):
            self.kw.rebuild(self.db)  # first start after switching keyword engines: fill it from SQLite
        self.vec = VectorIndex(SETTINGS.qdrant_dir, SETTINGS.embed_dim)
        self.indexer = Indexer(self.db, self.kw, self.vec)
        self.version = 0
        self.watcher = FolderWatcher(self.indexer, on_change=self.bump)
        self._cache: dict = {}
        self._lock = threading.Lock()
        self.graph = self._build_graph()
        self.content_worker = system_scan.ContentWorker(self)
        self.searching = threading.Event()  # background indexing pauses while this is set
        atexit.register(self.close)

    def close(self):
        try:
            self.content_worker.pause()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.vec.close()
        except Exception:  # noqa: BLE001
            pass

    def bump(self):
        self.version += 1

    # ---- graph nodes ---------------------------------------------------
    def _timed(self, state: State, stage: str, fn, **info):
        before = CALLS.snapshot()
        t0 = time.perf_counter()
        out = fn()
        ms = (time.perf_counter() - t0) * 1000
        after = CALLS.snapshot()
        calls = {k: after.get(k, 0) - before.get(k, 0) for k in after if after.get(k, 0) - before.get(k, 0)}
        state.setdefault("trace", []).append({"stage": stage, "ms": round(ms, 1), "model_calls": calls, **info})
        return out

    def n_understand(self, s: State) -> State:
        plan = self._timed(s, "Query understanding agent", lambda: understand(s["query"]))
        s["trace"][-1]["detail"] = f"parsed by {plan.parsed_by}, complexity {plan.complexity}"
        s["plan"] = plan
        s["profile"] = build_profile(self.db, s["user_id"])
        s["escalations"] = []
        return s

    def n_route(self, s: State) -> State:
        plan = s["plan"]
        # If the filters exclude every file, relax them (time first, then type) instead of returning nothing
        for attr, label in (("since", "time"), ("exts", "file type")):
            allowed = allowed_files(self.db, plan, s["profile"])
            if allowed is None or allowed:
                break
            if getattr(plan, attr):
                what = plan.time_label if attr == "since" else "/".join(plan.exts)
                if attr == "since":
                    plan.since = plan.until = plan.time_label = None
                else:
                    plan.exts = []
                msg = f"no files match the {label} filter ({what}); relaxed it"
                plan.notes.append(msg)
                s["trace"].append({"stage": "Adaptive router agent", "ms": 0.0, "model_calls": {}, "detail": msg,
                                   "escalation": True})
        if s.get("force_path"):
            path, why = s["force_path"], "forced from the UI"
        elif plan.filename or plan.complexity < SETTINGS.complexity_fast or (not plan.keywords and (plan.exts or plan.since)):
            path, why = "fast", "filename or metadata query"
        elif plan.complexity >= SETTINGS.complexity_deep:
            path, why = "deep", "reasoning-heavy query"
        else:
            path, why = "hybrid", "content query of moderate complexity"
        s["path"] = path
        s.setdefault("trace", []).append({"stage": "Adaptive router agent", "ms": 0.0, "model_calls": {},
                                          "detail": f"start at {path} path ({why})"})
        return s

    def n_fast(self, s: State) -> State:
        r = self._timed(s, "Fast path", lambda: fast_path(self.db, s["plan"], s["profile"]))
        s["trace"][-1]["detail"] = f"{r.detail}; confidence {r.confidence:.2f}"
        s["result"], s["path"] = r, "fast"
        is_meta = bool(r.candidates) and "metadata" in next(iter(r.candidates.values())).scores
        if s.get("force_path") == "fast" or r.confidence >= SETTINGS.fast_name_conf or is_meta:
            s["next"] = "rank"
        else:
            s["next"] = "hybrid"
            self._escalate(s, f"fast -> hybrid (confidence {r.confidence:.2f} < {SETTINGS.fast_name_conf})")
        return s

    def n_hybrid(self, s: State) -> State:
        prev = s.get("result")
        r = self._timed(s, "Hybrid path", lambda: hybrid_path(self.db, self.kw, self.vec, s["plan"], s["profile"], prev))
        s["trace"][-1]["detail"] = f"{r.detail}; confidence {r.confidence:.2f}"
        s["result"], s["path"] = r, "hybrid"
        few = r.n_allowed is not None and r.n_allowed <= SETTINGS.small_set
        if s.get("force_path") == "hybrid" or r.confidence >= SETTINGS.hybrid_conf:
            s["next"] = "rank"
        elif few and s.get("force_path") != "deep":
            # Filters left only a handful of files: ranking them needs no reranker or LLM
            s["next"] = "rank"
            s["trace"].append({"stage": "Adaptive router agent", "ms": 0.0, "model_calls": {},
                               "detail": f"only {r.n_allowed} files match the filters: no need for the deep path"})
        else:
            s["next"] = "deep"
            self._escalate(s, f"hybrid -> deep (confidence {r.confidence:.2f} < {SETTINGS.hybrid_conf})")
        return s

    def n_deep(self, s: State) -> State:
        hyb = s.get("result")
        if hyb is None or hyb.path == "fast":
            hyb = self._timed(s, "Hybrid retrieval (for deep)",
                              lambda: hybrid_path(self.db, self.kw, self.vec, s["plan"], s["profile"], hyb))
        r = self._timed(s, "Deep path", lambda: deep_path(self.db, s["plan"], hyb))
        s["trace"][-1]["detail"] = f"{r.detail}; top reranker score {r.confidence:.2f}"
        s["result"], s["path"] = r, "deep"
        return s

    def n_rank(self, s: State) -> State:
        recs = self._timed(s, "Fusion + personalized ranker",
                           lambda: rank(self.db, s["plan"], s["result"], s["profile"], personalize=s.get("personalize", True)))
        s["trace"][-1]["detail"] = f"{len(recs)} results, personalization {'on' if s.get('personalize', True) else 'off'}"
        s["recs"] = recs
        return s

    def n_explain(self, s: State) -> State:
        s["recs"] = self._timed(s, "Explainer agent",
                                lambda: explain(self.db, s["plan"], s["result"], s["recs"], s["profile"]))
        s["trace"][-1]["detail"] = "reasons built from ranking signals"
        return s

    # ---- escalation decisions (made inside nodes; edges only read them) --
    def _escalate(self, s: State, msg: str):
        s["escalations"].append(msg)
        s["trace"].append({"stage": "Adaptive router agent", "ms": 0.0, "model_calls": {},
                           "detail": msg, "escalation": True})

    def e_route(self, s: State) -> str:
        return s["path"]

    def e_next(self, s: State) -> str:
        return s["next"]

    def _build_graph(self):
        g = StateGraph(State)
        for name, fn in [("understand", self.n_understand), ("route", self.n_route), ("fast", self.n_fast),
                         ("hybrid", self.n_hybrid), ("deep", self.n_deep), ("rank", self.n_rank),
                         ("explain", self.n_explain)]:
            g.add_node(name, fn)
        g.set_entry_point("understand")
        g.add_edge("understand", "route")
        g.add_conditional_edges("route", self.e_route, {"fast": "fast", "hybrid": "hybrid", "deep": "deep"})
        g.add_conditional_edges("fast", self.e_next, {"rank": "rank", "hybrid": "hybrid"})
        g.add_conditional_edges("hybrid", self.e_next, {"rank": "rank", "deep": "deep"})
        g.add_edge("deep", "rank")
        g.add_edge("rank", "explain")
        g.add_edge("explain", END)
        return g.compile()

    # ---- public API ----------------------------------------------------
    def search(self, query: str, user_id: str = "default", force_path: str | None = None,
               personalize: bool = True, use_cache: bool = True) -> dict:
        self.searching.set()
        try:
            return self._search(query, user_id, force_path, personalize, use_cache)
        finally:
            self.searching.clear()

    def _search(self, query: str, user_id: str, force_path: str | None, personalize: bool,
                use_cache: bool) -> dict:
        t0 = time.perf_counter()
        key = (_normalize(query), user_id, force_path, personalize, self.version)
        with self._lock:
            hit = self._cache.get(key)
        if use_cache and hit and time.time() - hit[0] < SETTINGS.cache_ttl_s:
            out = dict(hit[1])
            out["cached"] = True
            out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            return out

        before = CALLS.snapshot()
        s: State = self.graph.invoke({"query": query, "user_id": user_id, "force_path": force_path,
                                      "personalize": personalize, "trace": []})
        after = CALLS.snapshot()
        calls = {k: after.get(k, 0) - before.get(k, 0) for k in after if after.get(k, 0) - before.get(k, 0)}
        out = {"query": query, "plan": s["plan"], "path": s["path"], "escalations": s["escalations"],
               "recs": s["recs"], "trace": s["trace"], "model_calls": calls, "cached": False,
               "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "profile": s["profile"]}
        self.db.log_query(user_id=user_id, query=query, path_used=s["path"], escalations=";".join(s["escalations"]),
                          latency_ms=out["latency_ms"], model_calls=sum(calls.values()))
        with self._lock:
            self._cache[key] = (time.time(), out)
        return out

    def open_file(self, user_id: str, file_id: int, query: str | None = None, session_id: str | None = None):
        """Feedback loop: log the open, which changes the user's profile for the next query."""
        self.db.log_activity(user_id, file_id, "open", session_id=session_id, query=query)
        self.bump()

    # ---- whole computer ------------------------------------------------
    def scan_computer(self, progress=None, roots=None, include_more: bool = False,
                      include_images: bool = False) -> dict:
        """Tier 1: names, types and dates of documents (and optionally pictures) on the chosen drives."""
        self.content_worker.include_more = include_more
        self.content_worker.include_images = include_images
        return system_scan.scan_metadata(self, roots, progress, include_more=include_more,
                                         include_images=include_images)

    def start_content_indexing(self, include_more: bool | None = None, include_images: bool | None = None):
        """Tier 2: read and embed file contents in the background, most useful first."""
        if include_more is not None:
            self.content_worker.include_more = include_more
        if include_images is not None:
            self.content_worker.include_images = include_images
        for folder in system_scan.user_folders():
            try:
                self.watcher.watch(folder)
            except Exception:  # noqa: BLE001
                pass
        self.content_worker.start()

    def import_recent(self, user_id: str) -> dict:
        return system_scan.import_recent(self, user_id)

    def index_folder(self, root, progress=None) -> dict:
        out = self.indexer.index_folder(root, progress)
        self.db.refresh_files()
        self.bump()
        return out


def _normalize(q: str) -> str:
    """Cache key: lower-cased content words, order-insensitive, so near-identical phrasings share an entry."""
    words = re.findall(r"[a-z0-9]+", q.lower())
    return " ".join(sorted(set(w for w in words if w not in {"the", "a", "an", "please", "file", "me", "show", "find"})))
