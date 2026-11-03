"""User context engine: turns the activity log into per-file personal signals.

Signals (each in [0, 1]):
  frequency  - exponentially decayed access count (half-life configurable)
  recency    - how recently the user last touched the file
  temporal   - does the user tend to open this file at this weekday / time of day?
  topic      - similarity of the file to the user's current topics of interest
  filetype   - the user's preference for this file type
  coaccess   - how often this file is opened together with what the user is working on right now
"""
from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from .config import SETTINGS
from .db import DB

SESSION_GAP_S = 45 * 60      # a gap longer than this starts a new work session
WORKING_SET_S = 3 * 3600     # files touched in the last 3 hours form the current working set
ACTION_WEIGHT = {"open": 1.0, "edit": 1.5, "click": 0.7}


@dataclass
class UserProfile:
    user_id: str
    now: float
    freq: dict = field(default_factory=dict)
    last: dict = field(default_factory=dict)
    count: Counter = field(default_factory=Counter)
    edits: Counter = field(default_factory=Counter)
    when: dict = field(default_factory=lambda: defaultdict(list))  # fid -> [(weekday, hour)]
    ext_pref: dict = field(default_factory=dict)
    topic_vec: np.ndarray | None = None
    graph: nx.Graph = field(default_factory=nx.Graph)
    working_set: list = field(default_factory=list)
    n_events: int = 0

    # ---- signals ------------------------------------------------------
    def signals(self, fid: int, ext: str, file_vec: np.ndarray | None, weekday: int | None = None) -> dict:
        maxf = max(self.freq.values(), default=0) or 1
        s = {"frequency": self.freq.get(fid, 0) / maxf}
        if fid in self.last:
            days = (self.now - self.last[fid]) / 86400
            s["recency"] = math.exp(-days / 7)
        else:
            s["recency"] = 0.0
        s["temporal"] = self._temporal(fid, weekday)
        s["topic"] = 0.0
        if self.topic_vec is not None and file_vec is not None:
            s["topic"] = max(0.0, float(np.dot(self.topic_vec, file_vec)))
        maxe = max(self.ext_pref.values(), default=0) or 1
        s["filetype"] = self.ext_pref.get(ext, 0) / maxe
        s["coaccess"] = self._coaccess(fid)
        return s

    def _temporal(self, fid: int, weekday: int | None = None) -> float:
        """Habit match. With an explicit weekday from the query ("...I check on Mondays") only the day counts."""
        hist = self.when.get(fid, [])
        if len(hist) < 3:
            return 0.0
        if weekday is not None:
            return sum(1 for d, _ in hist if d == weekday) / len(hist)
        lt = time.localtime(self.now)
        same_day = sum(1 for d, _ in hist if d == lt.tm_wday) / len(hist)
        same_hour = sum(1 for _, h in hist if abs(h - lt.tm_hour) <= 2) / len(hist)
        return 0.5 * same_day + 0.5 * same_hour

    def _coaccess(self, fid: int) -> float:
        if not self.working_set or fid not in self.graph:
            return 0.0
        w = sum(self.graph[fid][o]["weight"] for o in self.working_set if o != fid and self.graph.has_edge(fid, o))
        maxw = max((d["weight"] for _, _, d in self.graph.edges(data=True)), default=1)
        return min(1.0, w / maxw)

    def coaccess_partners(self, fid: int, k: int = 1) -> list[int]:
        if fid not in self.graph:
            return []
        nb = sorted(self.graph[fid].items(), key=lambda kv: -kv[1]["weight"])
        ws = set(self.working_set)
        pref = [n for n, _ in nb if n in ws] + [n for n, _ in nb if n not in ws]
        return pref[:k]

    def usual_time(self, fid: int) -> str | None:
        hist = self.when.get(fid, [])
        if len(hist) < 3:
            return None
        days = Counter(d for d, _ in hist).most_common(1)[0]
        hours = Counter(h for _, h in hist)
        if days[1] / len(hist) >= 0.5:
            name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][days[0]]
            h = hours.most_common(1)[0][0]
            part = "mornings" if h < 12 else "afternoons" if h < 17 else "evenings"
            return f"{name} {part}"
        return None

    def top_files(self, k: int = 20) -> list[int]:
        return [f for f, _ in sorted(self.freq.items(), key=lambda kv: -kv[1])[:k]]

    def recent_files(self, since: float | None = None, k: int = 40) -> list[int]:
        items = [(f, t) for f, t in self.last.items() if since is None or t >= since]
        return [f for f, _ in sorted(items, key=lambda kv: -kv[1])[:k]]


def build_profile(db: DB, user_id: str, now: float | None = None) -> UserProfile:
    now = now or time.time()
    p = UserProfile(user_id=user_id, now=now)
    events = [e for e in db.activity(user_id) if e["ts"] <= now]
    p.n_events = len(events)
    if not events:
        return p

    lam = math.log(2) / (SETTINGS.half_life_days * 86400)
    ext_of = {fid: r["ext"] for fid, r in db.files_by_ids({e["file_id"] for e in events}).items()}
    ext_score = Counter()
    sessions: list[list[int]] = []
    cur, last_ts = [], None
    for e in events:
        fid, ts = e["file_id"], e["ts"]
        if fid not in ext_of:  # file was deleted
            continue
        w = ACTION_WEIGHT.get(e["action"], 1.0) * math.exp(-lam * (now - ts))
        p.freq[fid] = p.freq.get(fid, 0) + w
        p.last[fid] = max(p.last.get(fid, 0), ts)
        p.count[fid] += 1
        if e["action"] == "edit":
            p.edits[fid] += 1
        lt = time.localtime(ts)
        p.when[fid].append((lt.tm_wday, lt.tm_hour))
        ext_score[ext_of[fid]] += w
        if last_ts is not None and ts - last_ts > SESSION_GAP_S:
            sessions.append(cur)
            cur = []
        cur.append(fid)
        last_ts = ts
    sessions.append(cur)

    p.ext_pref = dict(ext_score)
    for s in sessions:
        uniq = list(dict.fromkeys(s))
        for i, a in enumerate(uniq):
            for b in uniq[i + 1:]:
                if p.graph.has_edge(a, b):
                    p.graph[a][b]["weight"] += 1
                else:
                    p.graph.add_edge(a, b, weight=1)

    p.working_set = list(dict.fromkeys(e["file_id"] for e in events if now - e["ts"] <= WORKING_SET_S))

    # Topics of interest: decayed, weighted mean of vectors of recently used files
    recent = {f: w for f, w in p.freq.items() if now - p.last[f] <= 21 * 86400}
    vecs = db.file_vectors(recent.keys())
    if vecs:
        m = sum(vecs[f] * recent[f] for f in vecs) / sum(recent[f] for f in vecs)
        n = np.linalg.norm(m)
        p.topic_vec = m / n if n else None
    return p
