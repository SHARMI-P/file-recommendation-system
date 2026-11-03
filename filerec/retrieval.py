"""The three retrieval paths. Each returns candidate files with per-method scores and a confidence.

  fast   - filename + metadata only. No model calls.
  hybrid - BM25 keyword search (SQLite FTS5) + dense semantic search (Qdrant), fused with RRF.
  deep   - hybrid candidates, re-scored by the Qwen3 reranker, then judged by the LLM.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz, process

from .db import DB, name_stem
from .models import IMAGE_EXTS, get_embedder, get_image_embedder, get_llm, get_reranker
from .personalization import UserProfile
from .query_agent import QueryPlan
from .store import KeywordIndex, VectorIndex

RRF_K = 60
# Passing a long list of allowed file ids into the vector store's filter is very slow (every
# stored vector is checked against the whole list). Above this size we filter by type and date
# inside the vector store instead, fetch more, and drop the rest afterwards.
ID_FILTER_MAX = 1000


def _vector_filter(plan: QueryPlan, allowed: set | None) -> tuple[dict, int]:
    if allowed is None:
        return {"exts": plan.exts or None}, 0
    if len(allowed) <= ID_FILTER_MAX:
        return {"exts": plan.exts or None, "file_ids": list(allowed)}, 0
    return {"exts": plan.exts or None, "since": plan.since}, 120


@dataclass
class Candidate:
    file_id: int
    scores: dict = field(default_factory=dict)   # method -> raw score
    ranks: dict = field(default_factory=dict)    # method -> rank (1-based)
    best_chunk: int | None = None
    matched_terms: list = field(default_factory=list)
    llm_reason: str | None = None
    llm_relevant: bool | None = None


@dataclass
class RetrievalResult:
    path: str
    candidates: dict  # file_id -> Candidate
    confidence: float
    detail: str
    query_vec: np.ndarray | None = None
    n_allowed: int | None = None  # files left after metadata filters (None = no filter)


def allowed_files(db: DB, plan: QueryPlan, profile: UserProfile) -> set[int] | None:
    """Apply metadata filters (type, time). Returns None when there is no filter."""
    if not plan.exts and not plan.since:
        return None
    # SQL does the filtering (indexes on ext and modified): fast even for 500,000 files
    ok = db.ids_where(plan.exts or None, plan.since, plan.until)
    if plan.since and plan.personal:  # also files the user opened in the window, even if not modified
        touched = [f for f, t in profile.last.items()
                   if t >= plan.since and (plan.until is None or t < plan.until)]
        if plan.exts:
            touched = [f for f, r in db.files_by_ids(touched).items() if r["ext"] in plan.exts]
        ok |= set(touched)
    return ok


def _add(cands: dict, method: str, ranked: list[tuple[int, float, int | None]]):
    for rank, (fid, score, chunk) in enumerate(ranked, 1):
        c = cands.setdefault(fid, Candidate(fid))
        c.scores[method] = score
        c.ranks[method] = rank
        if c.best_chunk is None and chunk:
            c.best_chunk = chunk


def _per_file(hits: list[tuple[int, int, float]], allowed: set | None) -> list[tuple[int, float, int]]:
    """Collapse chunk hits to files (max score), keeping the best chunk."""
    best: dict[int, tuple[float, int]] = {}
    for cid, fid, s in hits:
        if allowed is not None and fid not in allowed:
            continue
        if fid not in best or s > best[fid][0]:
            best[fid] = (s, cid)
    return [(f, s, c) for f, (s, c) in sorted(best.items(), key=lambda kv: -kv[1][0])]


# ----------------------------------------------------------------------------
def fast_path(db: DB, plan: QueryPlan, profile: UserProfile) -> RetrievalResult:
    allowed = allowed_files(db, plan, profile)
    target = (plan.filename or " ".join(plan.keywords)).lower()
    target_stem = Path(target).stem if plan.filename else target
    rows = db.all_files() if allowed is None else [r for r in db.all_files() if r["id"] in allowed]
    cands: dict = {}
    if not target_stem:  # metadata-only ("pdfs from last month"): order by the user's recency, then mtime
        rows.sort(key=lambda r: (-(profile.last.get(r["id"], 0)), -(r["modified"] or 0)))
        _add(cands, "metadata", [(r["id"], 0.5, None) for r in rows[:40]])
        return RetrievalResult("fast", cands, 0.9 if rows else 0.0, f"{len(rows)} files match the filters")

    # Fuzzy matching in C over a cached list of cleaned filenames: fast for 500,000 files
    ids, stems = db.name_index()
    if allowed is not None:
        keep = [(i, s) for i, s in zip(ids, stems) if i in allowed]
        ids, stems = [i for i, _ in keep], [s for _, s in keep]
    if not plan.filename:  # "resumes" should match "Resume_2026.docx"
        target_stem = " ".join(w[:-1] if len(w) > 4 and w.endswith("s") and not w.endswith("ss") else w
                               for w in target_stem.split())
    best: dict[int, float] = {}
    for scorer in (fuzz.WRatio, fuzz.token_set_ratio):
        for _, score, i in process.extract(target_stem, stems, scorer=scorer, limit=50, processor=None):
            best[ids[i]] = max(best.get(ids[i], 0.0), score / 100)
    if plan.filename:
        exact = plan.filename.lower()
        for r in rows:
            if r["name"].lower() == exact:
                best[r["id"]] = 1.0
            elif r["id"] in best:
                best[r["id"]] = min(best[r["id"]], 0.97)
    scored = sorted(((fid, s, None) for fid, s in best.items()), key=lambda t: -t[1])
    _add(cands, "filename", scored[:25])
    top = scored[0][1] if scored else 0.0
    second = scored[1][1] if len(scored) > 1 else 0.0
    # Confident only if the name match is strong AND clearly ahead of the runner-up
    conf = top if top - second >= 0.04 or plan.filename else top * 0.8
    return RetrievalResult("fast", cands, round(conf, 3), f"best filename match {top:.2f} among {len(rows)} files")


def hybrid_path(db: DB, kw: KeywordIndex, vec: VectorIndex, plan: QueryPlan, profile: UserProfile,
                prev: RetrievalResult | None = None) -> RetrievalResult:
    allowed = allowed_files(db, plan, profile)
    cands: dict = dict(prev.candidates) if prev else {}

    kw_query = " ".join(plan.keywords) or plan.raw
    bm25 = _per_file(kw.search(kw_query, limit=80), allowed)[:30]
    _add(cands, "bm25", bm25)

    qv = get_embedder().embed_query(plan.semantic_query)
    vf, extra = _vector_filter(plan, allowed)
    dense = _per_file(vec.search(qv, limit=80 + extra, **vf), allowed)[:30]
    _add(cands, "dense", dense)

    # Picture queries ("photo of a beach"): CLIP compares the request with what each image shows
    img = []
    if wants_images(plan) and vec.image_count():
        clip = get_image_embedder()
        if clip is not None:
            img_query = re.sub(r"\b(my|i|me|from|last|this|week|month|year|yesterday|today|recent(ly)?)\b", " ",
                               plan.raw, flags=re.I)
            hits = vec.search_images(clip.embed_text(img_query), limit=40 + extra, **vf)
            img = [(f, s, None) for _, f, s in hits if allowed is None or f in allowed][:30]
            _add(cands, "image", img)

    # Personal queries ("...I worked on last week") also consider the user's recent files
    if plan.personal and profile.n_events:
        pool = [f for f in profile.recent_files(plan.since) if allowed is None or f in allowed]
        fv = db.file_vectors(pool)
        hist = sorted(((f, float(np.dot(qv, fv[f])), None) for f in fv), key=lambda t: -t[1])[:15]
        _add(cands, "history", hist)

    # Which query terms appear in each candidate's best chunk (used by the explainer)
    chunks = db.chunks_by_ids([c.best_chunk for c in cands.values() if c.best_chunk])
    for c in cands.values():
        if c.best_chunk in chunks:
            txt = chunks[c.best_chunk]["text"].lower()
            c.matched_terms = [k for k in plan.keywords if re.search(rf"\b{re.escape(k)}", txt)][:5]

    if not cands and plan.exts and plan.keywords and prev is None:
        # The type filter found nothing about this topic: search all types rather than return nothing
        what = "/".join(plan.exts)
        plan.exts = []
        plan.notes.append(f"nothing about this in {what} files; searched all file types")
        out = hybrid_path(db, kw, vec, plan, profile)
        out.detail = f"no {what} match, searched all types · " + out.detail
        return out

    conf = _hybrid_confidence(bm25, dense)
    detail = f"{len(bm25)} keyword hits, {len(dense)} semantic hits"
    if img:
        detail += f", {len(img)} picture matches"
        if set(plan.exts) <= IMAGE_EXTS:  # only pictures wanted: CLIP has the answer, no LLM needed
            conf = max(conf, 0.9)
    return RetrievalResult("hybrid", cands, conf, detail, qv, len(allowed) if allowed is not None else None)


def wants_images(plan: QueryPlan) -> bool:
    return bool(plan.exts) and bool(set(plan.exts) & IMAGE_EXTS)


def _hybrid_confidence(bm25, dense) -> float:
    if not bm25 and not dense:
        return 0.0
    top_b = [f for f, _, _ in bm25[:3]]
    top_d = [f for f, _, _ in dense[:3]]
    agree = 1.0 if (top_b and top_d and top_b[0] == top_d[0]) else 0.5 if set(top_b) & set(top_d) else 0.0
    d_top = dense[0][1] if dense else 0.0
    d_margin = (dense[0][1] - dense[1][1]) if len(dense) > 1 else d_top
    sim = min(1.0, d_top / 0.6)
    margin = min(1.0, d_margin * 10)
    return round(0.5 * agree + 0.3 * sim + 0.2 * margin, 3)


DEEP_SCHEMA = {
    "type": "object",
    "properties": {"judgements": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "integer"}, "relevant": {"type": "boolean"}, "reason": {"type": "string"}},
        "required": ["id", "relevant", "reason"]}}},
    "required": ["judgements"],
}
DEEP_SYSTEM = ("You are ranking a user's files for their request. For each candidate file decide whether it is "
               "what the user is looking for, and give a reason of at most 12 words grounded in the excerpt. "
               "Return JSON only.")


def deep_path(db: DB, plan: QueryPlan, hybrid: RetrievalResult, top_files: int = 6,
              llm_files: int = 3) -> RetrievalResult:
    cands = hybrid.candidates
    # Pre-rank by RRF over what we have, then spend the expensive models on the best few only
    pre = sorted(cands.values(), key=lambda c: -rrf(c))[:top_files]
    chunks = db.chunks_by_ids([c.best_chunk for c in pre if c.best_chunk])
    files = db.files_by_ids([c.file_id for c in pre])
    docs = []
    for c in pre:
        body = chunks[c.best_chunk]["text"] if c.best_chunk in chunks else db.first_chunk(c.file_id)
        docs.append(f"File: {files[c.file_id]['name']}\n{body[:900]}")  # shorter = faster on a CPU
    scores = get_reranker().score(plan.raw, docs)
    ranked = sorted(zip(pre, scores, docs), key=lambda t: -t[1])
    for rank, (c, s, _) in enumerate(ranked, 1):
        c.scores["rerank"] = float(s)
        c.ranks["rerank"] = rank

    detail = f"reranked {len(pre)} files"
    llm = get_llm()
    if llm.available() and ranked:
        lines = [f"[{i}] {d[:450]}" for i, (_, _, d) in enumerate(ranked[:llm_files])]
        out = llm.json(DEEP_SYSTEM, f"User request: {plan.raw}\n\nCandidates:\n" + "\n\n".join(lines),
                       DEEP_SCHEMA, max_tokens=180)
        if out:
            for j in out.get("judgements", []):
                i = j.get("id")
                if isinstance(i, int) and 0 <= i < min(llm_files, len(ranked)):
                    c = ranked[i][0]
                    c.llm_relevant = bool(j.get("relevant"))
                    c.llm_reason = (j.get("reason") or "").strip()[:160]
            detail += f", LLM judged top {min(llm_files, len(ranked))}"
    conf = float(ranked[0][1]) if ranked else 0.0
    return RetrievalResult("deep", cands, round(conf, 3), detail, hybrid.query_vec)


def rrf(c: Candidate) -> float:
    return sum(1.0 / (RRF_K + r) for m, r in c.ranks.items() if m != "rerank")
