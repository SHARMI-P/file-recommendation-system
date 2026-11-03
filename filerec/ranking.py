"""Fusion + personalized ranker, and the explainer agent.

score = w_rel * relevance + relevance^3 * sum_i (w_i * boost * personal_signal_i)

relevance comes from RRF over all retrieval methods, refined by the reranker and LLM when the deep
path ran. Personal weights are multiplied by `personal_query_boost` for queries like "the deck I
worked on". The explainer builds each reason from the signals that actually produced the score.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import SETTINGS
from .db import DB
from .personalization import UserProfile
from .query_agent import QueryPlan
from .retrieval import Candidate, RetrievalResult, rrf

PERSONAL_KEYS = ["frequency", "recency", "temporal", "topic", "filetype", "coaccess"]


@dataclass
class Recommendation:
    file_id: int
    name: str
    path: str
    ext: str
    modified: float
    score: float
    relevance: float
    signals: dict
    contributions: dict
    reasons: list = field(default_factory=list)
    snippet: str = ""


def weights(plan: QueryPlan, profile: UserProfile) -> dict:
    w = {"relevance": SETTINGS.w_relevance, "frequency": SETTINGS.w_frequency, "recency": SETTINGS.w_recency,
         "temporal": SETTINGS.w_temporal, "topic": SETTINGS.w_topic, "filetype": SETTINGS.w_filetype,
         "coaccess": SETTINGS.w_coaccess}
    if profile.n_events == 0:
        for k in PERSONAL_KEYS:
            w[k] = 0.0
    elif plan.personal:
        for k in PERSONAL_KEYS:
            w[k] *= SETTINGS.personal_query_boost
        if plan.habit_weekday is not None:  # the user described a habit: trust the time pattern most
            w["temporal"] *= 4
    return w


def rank(db: DB, plan: QueryPlan, result: RetrievalResult, profile: UserProfile, k: int = 10,
         personalize: bool = True) -> list[Recommendation]:
    cands: list[Candidate] = list(result.candidates.values())
    if not cands:
        return []
    files = db.files_by_ids([c.file_id for c in cands])
    vecs = db.file_vectors([c.file_id for c in cands]) if personalize else {}
    w = weights(plan, profile) if personalize else {"relevance": 1.0}

    max_rrf = max(rrf(c) for c in cands) or 1
    rel = {}
    for c in cands:
        if result.path == "fast":
            r = c.scores.get("filename", c.scores.get("metadata", 0.5))
        else:
            r = rrf(c) / max_rrf
            if c.scores.get("filename", 0) >= 0.8:  # names are a strong signal people choose deliberately
                r = min(1.0, r + 0.1)
            if "rerank" in c.scores:
                r = 0.4 * r + 0.6 * c.scores["rerank"]
            if c.llm_relevant is True:
                r = min(1.0, r + 0.15)
            elif c.llm_relevant is False:
                r = max(0.0, r - 0.25)
        rel[c.file_id] = r

    sigs = {c.file_id: profile.signals(c.file_id, files[c.file_id]["ext"], vecs.get(c.file_id), plan.habit_weekday)
            for c in cands if c.file_id in files} if personalize else {}
    # Topic similarity is relative: rescale across this candidate set
    if sigs:
        t = [s["topic"] for s in sigs.values()]
        lo, hi = min(t), max(t)
        for s in sigs.values():
            s["topic"] = (s["topic"] - lo) / (hi - lo) if hi > lo else 0.0

    recs = []
    for c in cands:
        f = files.get(c.file_id)
        if f is None:
            continue
        s = sigs.get(c.file_id, {})
        r_ = rel[c.file_id]
        contrib = {"relevance": w["relevance"] * r_}
        # Relevance gate: personal signals re-order relevant files; they cannot lift an
        # irrelevant file that the user merely opens a lot.
        gate = r_ ** 3
        for key in PERSONAL_KEYS:
            contrib[key] = w.get(key, 0.0) * s.get(key, 0.0) * gate
        recs.append(Recommendation(
            file_id=c.file_id, name=f["name"], path=f["path"], ext=f["ext"], modified=f["modified"],
            score=round(sum(contrib.values()), 4), relevance=round(rel[c.file_id], 3), signals=s,
            contributions=contrib))
    recs.sort(key=lambda r: -r.score)
    return recs[:k]


# ----------------------------------------------------------------------------
def _ago(ts: float, now: float) -> str:
    d = (now - ts) / 86400
    if d < 1 / 24:
        return "less than an hour ago"
    if d < 1:
        return f"{int(d * 24)} hours ago"
    if d < 2:
        return "yesterday"
    return f"{int(d)} days ago"


def explain(db: DB, plan: QueryPlan, result: RetrievalResult, recs: list[Recommendation],
            profile: UserProfile) -> list[Recommendation]:
    """Explainer agent: every reason is derived from a signal that contributed to the score."""
    now = profile.now
    chunks = db.chunks_by_ids([result.candidates[r.file_id].best_chunk for r in recs
                               if result.candidates[r.file_id].best_chunk])
    names = {}
    for r in recs:
        c: Candidate = result.candidates[r.file_id]
        reasons = []
        doubt = None
        if c.llm_reason and c.llm_relevant is not False:
            reasons.append(("LLM", c.llm_reason))
        elif c.llm_reason:
            doubt = ("Doubt", "LLM doubts this match: " + c.llm_reason)
        if "filename" in c.scores and c.scores["filename"] >= 0.7:
            reasons.append(("Match", f"Filename matches \"{plan.filename or ' '.join(plan.keywords)}\""))
        if c.matched_terms:
            reasons.append(("Content", "Mentions " + ", ".join(f"\"{t}\"" for t in c.matched_terms)))
        if "image" in c.ranks and c.ranks["image"] <= 5:
            reasons.append(("Meaning", f"The picture looks like your description (rank {c.ranks['image']})"))
        if "dense" in c.ranks and c.ranks["dense"] <= 3:
            reasons.append(("Meaning", f"Semantically close to your request (rank {c.ranks['dense']})"))
        if "rerank" in c.scores and c.scores["rerank"] >= 0.5:
            reasons.append(("Reranker", f"Reranker relevance {c.scores['rerank']:.2f}"))
        if plan.exts and r.ext in plan.exts:
            reasons.append(("Filter", f"File type matches ({r.ext})"))
        if plan.time_label and (r.modified >= (plan.since or 0) or profile.last.get(r.file_id, 0) >= (plan.since or 0)):
            when = plan.time_label if plan.time_label in {"today", "yesterday", "this week", "this month"} \
                else f"in the {plan.time_label}"
            reasons.append(("Filter", f"Active {when}"))

        s = r.signals
        n = profile.count.get(r.file_id, 0)
        if s.get("frequency", 0) >= 0.35 and n:
            e = profile.edits.get(r.file_id, 0)
            extra = f", edited {e} times" if e else ""
            reasons.append(("You", f"You opened it {n} times{extra}"))
        if s.get("recency", 0) >= 0.3:
            reasons.append(("You", f"Last used {_ago(profile.last[r.file_id], now)}"))
        if s.get("temporal", 0) >= 0.5:
            ut = profile.usual_time(r.file_id)
            if ut:
                reasons.append(("Habit", f"You usually open this on {ut}"))
        if s.get("coaccess", 0) >= 0.2:
            partner = profile.coaccess_partners(r.file_id)
            if partner:
                if partner[0] not in names:
                    names.update({k: v["name"] for k, v in db.files_by_ids(partner).items()})
                reasons.append(("Context", f"Often used together with {names.get(partner[0], 'your current files')}"))
        if s.get("topic", 0) >= 0.8 and profile.n_events:
            reasons.append(("Context", "Close to topics you're working on"))
        r.reasons = reasons[:5] or [("Match", "Best available match")]
        if doubt:
            r.reasons.append(doubt)

        if c.best_chunk in chunks:
            r.snippet = chunks[c.best_chunk]["text"][:260].replace("\n", " ")
        else:
            r.snippet = db.first_chunk(r.file_id)[:260].replace("\n", " ") or \
                f"{r.path} · content not indexed yet (found by name, type and date)"
    return recs
