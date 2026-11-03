"""Query understanding agent.

Turns a natural-language request into a structured QueryPlan: intent, file-type and time filters,
keywords, personal signals and a complexity score in [0, 1].

Rules handle the common cases in microseconds. The LLM is consulted only when the query looks
reasoning-heavy, so simple queries never pay for a model call.
"""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field

from .models import get_llm

TYPE_WORDS = {
    "pdf": ["pdf"], "pdfs": ["pdf"],
    "word": ["docx"], "docx": ["docx"], "doc": ["docx"], "document": ["docx", "pdf", "txt", "md"], "documents": ["docx", "pdf", "txt", "md"],
    "report": ["pdf", "docx"],
    "excel": ["xlsx", "csv"], "xlsx": ["xlsx"], "spreadsheet": ["xlsx", "csv"], "sheet": ["xlsx", "csv"],
    "sheets": ["xlsx", "csv"], "csv": ["csv"],
    "ppt": ["pptx"], "pptx": ["pptx"], "deck": ["pptx"], "slides": ["pptx"], "slide": ["pptx"],
    "presentation": ["pptx"], "powerpoint": ["pptx"],
    "code": ["py", "js", "ts", "java", "sql"], "script": ["py", "js", "sql"], "python": ["py"],
    "notebook": ["ipynb"], "sql": ["sql"],
    "image": ["png", "jpg", "jpeg"], "screenshot": ["png", "jpg", "jpeg"], "photo": ["png", "jpg", "jpeg"],
    "images": ["png", "jpg", "jpeg"], "screenshots": ["png", "jpg", "jpeg"], "photos": ["png", "jpg", "jpeg"],
    "picture": ["png", "jpg", "jpeg"], "pictures": ["png", "jpg", "jpeg"], "pic": ["png", "jpg", "jpeg"],
    "pics": ["png", "jpg", "jpeg"], "selfie": ["png", "jpg", "jpeg"], "selfies": ["png", "jpg", "jpeg"],
    "wallpaper": ["png", "jpg", "jpeg"], "png": ["png"], "jpg": ["jpg", "jpeg"], "jpeg": ["jpg", "jpeg"],
    "diagram": ["png", "jpg", "jpeg", "pdf"],
    "notes": ["md", "txt", "docx"], "markdown": ["md"], "txt": ["txt"], "readme": ["md"],
}
# Words that are pure filters and should not be treated as content keywords
FILTER_ONLY = {"pdf", "pdfs", "docx", "doc", "xlsx", "pptx", "ppt", "csv", "txt", "excel", "powerpoint",
               "word", "deck", "slides", "slide", "presentation", "spreadsheet", "sheet", "sheets", "file", "files",
               "document", "documents", "decks", "presentations", "spreadsheets", "docs",
               "image", "images", "photo", "photos", "picture", "pictures", "pic", "pics", "png", "jpg", "jpeg"}

STOP = set("""have has had having got get gets keep kept put saved save wrote write all every
a an the of to in on for with and or from by at is are was were be been it this that these those
i me my mine we our you your he she they them his her its find open show get give where which what whats
look looking search need want please can could would file files doc thing one some any about last this
week weeks month months year today yesterday recent recently ago days day worked working edited opened
sent shared made wrote created used using before after during when did do does
talk talks talking mention mentions mentioning discuss discusses discussing describe describes describing
cover covers explain explains explaining related regarding contains containing""".split())

# In a request for a file, a first-person reference ("I", "my", "we") almost always means
# "a file from my own history", so the user's activity should weigh more.
PERSONAL = re.compile(r"\b(i|i'm|i've|i'd|my|mine|me)\b|\brecent(ly)?\b|\busual(ly)?\b|"
                      r"\balways\b|\bkeep (opening|using)\b", re.I)
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
HABIT_WORDS = {"check", "checks", "keep", "keeps", "opening", "usually", "every", "always", "often", "regularly"}
# Questions that need reasoning over content (deep path). "A document that talks about X" or
# "notes that mention X" are topic searches, which the hybrid path answers in seconds, so they
# are not listed here.
REASONING = re.compile(r"\b(why|how (do|does|did|to|we|is|are|can|should)|reason|decid|decision|compar|versus|vs\.?|"
                       r"difference|summar|best|should)\w*", re.I)
FILENAME = re.compile(r"[\w\-()\[\] ]{1,80}\.(pdf|docx|pptx|xlsx|csv|txt|md|py|js|ts|json|png|jpg|jpeg|sql|yaml|yml)\b",
                      re.I)
QUOTED = re.compile(r"[\"']([^\"']{3,80})[\"']")


@dataclass
class QueryPlan:
    raw: str
    intent: str = "find"
    keywords: list[str] = field(default_factory=list)
    exts: list[str] = field(default_factory=list)
    since: float | None = None
    until: float | None = None
    time_label: str | None = None
    personal: bool = False
    habit_weekday: int | None = None
    filename: str | None = None
    complexity: float = 0.5
    semantic_query: str = ""
    parsed_by: str = "rules"
    notes: list[str] = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        for k in ("since", "until"):
            if d[k]:
                d[k] = time.strftime("%Y-%m-%d %H:%M", time.localtime(d[k]))
        return d


def _time_filter(q: str, now: float) -> tuple[float | None, float | None, str | None]:
    day = 86400
    lt = time.localtime(now)
    midnight = now - (lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec)
    ql = q.lower()
    m = re.search(r"\b(last|past)\s+(\d+)\s+(day|week|month)s?\b", ql)
    if m:
        n, unit = int(m.group(2)), m.group(3)
        span = {"day": 1, "week": 7, "month": 30}[unit] * n
        return now - span * day, None, f"last {n} {unit}s"
    if re.search(r"\btoday\b", ql):
        return midnight, None, "today"
    if re.search(r"\byesterday\b", ql):
        return midnight - day, midnight, "yesterday"
    if re.search(r"\bthis week\b", ql):
        return midnight - lt.tm_wday * day, None, "this week"
    if re.search(r"\b(last|past|previous) week\b", ql):
        return now - 7 * day, None, "last 7 days"
    if re.search(r"\bthis month\b", ql):
        return midnight - (lt.tm_mday - 1) * day, None, "this month"
    if re.search(r"\b(last|past|previous) month\b", ql):
        return now - 30 * day, None, "last 30 days"
    if re.search(r"\b(recent|recently|lately)\b", ql):
        return now - 14 * day, None, "last 14 days"
    return None, None, None


def _complexity(q: str, keywords: list[str], filename: str | None) -> float:
    if filename:
        return 0.1
    score = 0.35
    if REASONING.search(q):
        score += 0.4
    score += min(0.25, 0.04 * max(0, len(keywords) - 3))
    if len(keywords) <= 2 and not REASONING.search(q):
        score -= 0.15
    return round(max(0.05, min(1.0, score)), 2)


LLM_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["find", "open", "compare", "question"]},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "rewritten_query": {"type": "string"},
        "file_types": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["intent", "keywords", "rewritten_query", "file_types"],
}
LLM_SYSTEM = ("You help a file search engine. Given a user's request for one of their files, return JSON with: "
              "intent; the 2-6 most important content keywords (include likely synonyms); a rewritten_query that "
              "describes the content of the wanted file in one plain sentence; and file_types as extensions "
              "(pdf, docx, pptx, xlsx, csv, md, txt, py, png) only if the user implied a type, else [].")


def understand(query: str, now: float | None = None, allow_llm: bool = True) -> QueryPlan:
    now = now or time.time()
    q = query.strip()
    plan = QueryPlan(raw=q)
    ql = q.lower()

    m = FILENAME.search(q) or QUOTED.search(q)
    if m:
        plan.filename = re.sub(r"^(open|find|show|get|launch)\s+", "", m.group(0).strip(" \"'"), flags=re.I)
        plan.notes.append(f"looks like a filename: {plan.filename}")
    if re.match(r"^\s*(open|launch)\b", ql):
        plan.intent = "open"
    if REASONING.search(q):
        plan.intent = "question"

    exts = []
    for w in re.findall(r"[a-z]+", ql):
        exts += TYPE_WORDS.get(w, [])
    plan.exts = sorted(set(exts))
    plan.since, plan.until, plan.time_label = _time_filter(q, now)
    plan.personal = bool(PERSONAL.search(q))
    for i, d in enumerate(WEEKDAYS):
        if re.search(rf"\b{d}s?\b", ql):
            plan.habit_weekday = i
            plan.personal = True
            plan.notes.append(f"habit on {d.title()}s")

    tokens = re.findall(r"[a-z0-9][a-z0-9_\-]*", ql)
    plan.keywords = [t for t in tokens if t not in STOP and t not in FILTER_ONLY and t not in HABIT_WORDS
                     and t.rstrip('s') not in WEEKDAYS and len(t) > 1 and t not in {"i'm", "i've"}]
    plan.complexity = _complexity(q, plan.keywords, plan.filename)
    plan.semantic_query = q

    # Only reasoning-heavy queries justify an LLM call here
    if allow_llm and plan.complexity >= 0.7:
        llm = get_llm()
        out = llm.json(LLM_SYSTEM, q, LLM_SCHEMA, max_tokens=150) if llm.available() else None
        if out:
            plan.parsed_by = "rules + LLM"
            extra = [k.lower() for k in out.get("keywords", []) if k.lower() not in plan.keywords]
            plan.keywords += extra[:6]
            plan.semantic_query = out.get("rewritten_query") or plan.semantic_query
            if out.get("file_types") and not plan.exts:
                plan.exts = [e.lower().lstrip(".") for e in out["file_types"]][:4]
            plan.intent = out.get("intent", plan.intent)
    return plan
