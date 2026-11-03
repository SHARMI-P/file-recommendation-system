"""Streamlit demo UI for the Agentic File Recommendation System.

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from filerec.config import SETTINGS
from filerec.models import get_embedder, get_image_embedder, get_llm, get_reranker
from filerec.personalization import build_profile
from filerec.pipeline import Engine

st.set_page_config(page_title="Agentic File Recommender", page_icon="🗂️", layout="wide")

PATH_COLOR = {"fast": "#1f9d55", "hybrid": "#2b6cb0", "deep": "#b7791f"}
PATH_LABEL = {"fast": "Fast path · filename + metadata", "hybrid": "Hybrid path · BM25 + semantic",
              "deep": "Deep path · rerank + LLM"}
TAG_COLOR = {"Match": "#2b6cb0", "Content": "#2b6cb0", "Meaning": "#2b6cb0", "Reranker": "#b7791f", "LLM": "#b7791f",
             "Filter": "#4a5568", "Doubt": "#9b2c2c", "You": "#2f855a", "Habit": "#2f855a", "Context": "#6b46c1"}
DEMO_EXAMPLES = ["Q3_rates_v4.pptx", "budget file", "the pricing deck I worked on last week",
                 "why did we drop kafka", "pdfs from last month", "slack alerts code", "customer churn analysis"]
# Shown when this index holds the user's own laptop files (no demo files in it)
LAPTOP_EXAMPLES = ["my resume", "pdfs from last week", "the file I worked on yesterday",
                   "presentation I made this month", "screenshots from today", "photo of a group of people",
                   "notes about machine learning"]

st.markdown("""
<style>
.card {border:1px solid rgba(128,128,128,.25); border-radius:10px; padding:12px 14px; margin-bottom:6px}
.fname {font-weight:600; font-size:1.05rem}
.path {color:gray; font-size:.8rem}
.chip {display:inline-block; padding:2px 8px; margin:2px 4px 2px 0; border-radius:12px; font-size:.78rem; color:white}
.badge {display:inline-block; padding:3px 10px; border-radius:6px; color:white; font-weight:600; font-size:.85rem}
.snip {color:gray; font-size:.82rem; margin-top:6px}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Starting engine...")
def get_engine() -> Engine:
    return Engine()


eng = get_engine()
# Demo mode only when the fictional Nimbus Analytics files are in this index; otherwise it is the
# user's own laptop and every demo-specific control and example is hidden.
demo_indexed = bool(eng.db.q("SELECT 1 FROM files WHERE name='Q3_rates_v4.pptx' LIMIT 1"))
EXAMPLES = DEMO_EXAMPLES if demo_indexed else LAPTOP_EXAMPLES
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:8]

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🗂️ File Recommender")
    st.caption("Demo: Nimbus Analytics sample files" if demo_indexed else "💻 Searching the files on this laptop")
    # Users: everyone with history in this index, plus users added in this session.
    # The last selected user and folder are remembered in the data folder across restarts.
    prefs_file = SETTINGS.data_dir / "ui_prefs.json"

    def load_prefs() -> dict:
        try:
            return json.loads(prefs_file.read_text())
        except Exception:  # noqa: BLE001
            return {}

    def save_prefs(**kw):
        p = load_prefs()
        p.update(kw)
        try:
            prefs_file.write_text(json.dumps(p))
        except Exception:  # noqa: BLE001
            pass

    prefs = load_prefs()
    if "known_users" not in st.session_state:
        st.session_state.known_users = set(eng.db.users()) | ({prefs["user"]} if prefs.get("user") else set())
    st.session_state.known_users |= set(eng.db.users())
    if not st.session_state.known_users:
        st.session_state.known_users = {"me"}
    if st.session_state.get("user_select") not in st.session_state.known_users:
        st.session_state.user_select = prefs.get("user") if prefs.get("user") in st.session_state.known_users \
            else sorted(st.session_state.known_users)[0]

    def add_user():
        name = st.session_state.new_user.strip().lower()
        if name:
            st.session_state.known_users.add(name)
            st.session_state.user_select = name
            save_prefs(user=name)
        st.session_state.new_user = ""

    user = st.selectbox("Current user", sorted(st.session_state.known_users), key="user_select",
                        on_change=lambda: save_prefs(user=st.session_state.user_select),
                        help="Personalization uses this user's activity history")
    st.text_input("Add a new user", key="new_user", placeholder="type a name and press Enter", on_change=add_user)
    st.caption(f"Searching as **{user}**")

    st.subheader("Index")
    demo_dir = str(Path("demo_workspace").resolve())
    default_folder = prefs.get("folder") or (demo_dir if demo_indexed else str(Path.home() / "Documents"))
    if not demo_indexed and "demo_workspace" in default_folder:
        default_folder = str(Path.home() / "Documents")  # never suggest the demo on the laptop index
    folder = st.text_input("Folder to index", value=default_folder,
                           help="Index one folder right away (optional). For the whole laptop use 🖥️ Whole computer.")
    if folder != prefs.get("folder"):
        save_prefs(folder=folder)
    c1, c2 = st.columns(2)
    if c1.button("Index folder", width="stretch"):
        if not Path(folder).is_dir():
            st.error("Folder not found")
        else:
            bar = st.progress(0.0, text="Indexing…")
            counts = eng.index_folder(Path(folder), lambda i, n, name: bar.progress(i / n, text=f"{i}/{n} {name[:30]}"))
            bar.empty()
            st.success(f"Indexed {counts['indexed']}, unchanged {counts['unchanged']}, removed {counts['removed']}")
    if c2.button("Watch folder", width="stretch", help="Auto re-index files as they change"):
        eng.watcher.watch(Path(folder))
        st.success("Watching for changes")
    if demo_indexed and st.button("Load demo personas (priya, rahul)", width="stretch",
                 help="Simulates 30 days of file activity for two users. Index the demo workspace first."):
        import importlib.util
        spec = importlib.util.spec_from_file_location("sim", Path(__file__).parent / "scripts" / "simulate_activity.py")
        sim = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sim)
        try:
            n = sim.main(db=eng.db)
            eng.bump()
            st.success(f"Logged {n} activity events")
        except SystemExit as e:
            st.error(str(e))
    with st.expander("🖥️ Whole computer", expanded=False):
        from filerec import system_scan
        all_roots = system_scan.drives()
        st.caption("Drives to search. Windows, Program Files, AppData, installed programs and games (on any "
                   "drive) and other system folders are always skipped.")
        roots = [r for r in all_roots if st.checkbox(str(r), value=True, key=f"drive-{r}")]
        include_images = st.checkbox("Include pictures (find photos and screenshots by what's in them)",
                                     value=False, key="incl-images",
                                     help="PNG and JPG files over 20 KB. Content indexing reads each picture with "
                                          "the CLIP model (about 0.2–0.5 s per picture on a CPU).")
        include_more = st.checkbox("Also include code and CSV/JSON data", value=False,
                                   help="Off = documents only (PDF, Word, PowerPoint, Excel, text, Markdown). "
                                        "Code and dataset folders can hold hundreds of thousands of files.")
        if st.button("1 · Scan whole computer", width="stretch", disabled=not roots,
                     help="Records name, type, size and dates of every document. Takes a few minutes."):
            msg = st.empty()
            try:
                counts = eng.scan_computer(
                    lambda n, ch, d: msg.caption(f"Checked {n:,} files · {ch:,} new or changed · {d[-45:]}"),
                    roots=roots, include_more=include_more, include_images=include_images)
            except RuntimeError as e:
                msg.empty()
                st.error(str(e))
            else:
                msg.empty()
                st.success(f"Checked {counts['seen']:,} files: {counts['new']:,} new, {counts['changed']:,} changed, "
                           f"{counts['removed']:,} removed. Name, type and date search now covers the whole computer.")
        if eng.kw.broken:
            st.warning("Keyword index paused after a Windows write error; search still works by filename "
                       f"and meaning. Error: {eng.kw.last_error[:200]}")
        if st.button("Repair keyword index", width="stretch",
                     help="Rebuilds the keyword (BM25) index from the database in a fresh folder"):
            bar = st.progress(0.0, text="Rebuilding keyword index…")
            try:
                n = eng.kw.rebuild(eng.db, lambda i, t: bar.progress(min(1.0, i / max(t, 1)),
                                                                     text=f"Rebuilding keyword index… {i:,}/{t:,} files"))
                eng.bump()
                bar.empty()
                st.success(f"Keyword index rebuilt ({n:,} entries).")
            except Exception as e:  # noqa: BLE001
                bar.empty()
                st.error(f"Rebuild failed: {e}. Check free space on the drive and the Windows Security "
                         "exclusion for the index folder, then restart the app.")
        st.markdown("**Folders to skip**")
        skip_text = st.text_area("Folders to skip", value="\n".join(system_scan.user_skips()), height=90,
                                 label_visibility="collapsed",
                                 placeholder="One folder per line, e.g.\nD:\\ML\\dataset\nE:\\old_backup",
                                 help="Dataset, backup or game folders you don't want searched. Their files are "
                                      "removed from the index now and skipped by future scans.")
        if st.button("Save and clean up the index", width="stretch",
                     help="Removes files in these folders and in installed-program folders from the index"):
            folders = [ln.strip().strip('"') for ln in skip_text.splitlines() if ln.strip()]
            system_scan.save_user_skips(folders)
            was_running = eng.content_worker.running
            eng.content_worker.pause()
            with st.spinner("Removing skipped and program folders from the index…"):
                n = system_scan.remove_under(eng, folders) + system_scan.remove_program_files(eng)
            st.success(f"Removed {n:,} files from the index." +
                       (" Content indexing was paused; start step 2 again." if was_running else ""))
        if st.button("Where do my pictures come from?", width="stretch",
                     help="Folders with the most indexed pictures. Copy a dataset folder into 'Folders to skip'."):
            pf = system_scan.picture_folders(eng.db)
            if pf:
                st.dataframe(pd.DataFrame(pf, columns=["folder", "pictures"]), hide_index=True, width="stretch")
            else:
                st.caption("No pictures in the index.")
        worker = eng.content_worker
        if worker.running:
            if st.button("Pause content indexing", width="stretch"):
                worker.pause()
        elif st.button("2 · Index file contents (background)", width="stretch",
                       help="Reads and embeds contents, recent files and Desktop/Documents/Downloads first. "
                            "Keeps running while you search."):
            eng.start_content_indexing(include_more=include_more, include_images=include_images)
        if st.button("Import Windows recent files", width="stretch",
                     help="Adds the files Windows lists as recently opened to this user's history"):
            r = eng.import_recent(user)
            if r.get("error"):
                st.warning(r["error"])
            else:
                st.success(f"Imported {r['imported']} recent opens for {user} "
                           f"({r['new_files']} files added, {r['already']} already known).")

        @st.fragment(run_every=3)
        def _content_status():
            stt = eng.content_worker.status
            stats = eng.db.stats()
            if eng.content_worker.running and stt["total"]:
                st.progress(stt["done"] / stt["total"],
                            text=f"Content indexed {stt['done']:,} / {stt['total']:,}")
                elapsed = time.time() - eng.content_worker.started_at
                if stt["done"] >= 5 and elapsed > 0:
                    left = (stt["total"] - stt["done"]) * elapsed / stt["done"]
                    eta = f"{left / 3600:.1f} hours" if left >= 3600 else f"{max(1, round(left / 60))} minutes"
                    st.caption(f"About {eta} left at {elapsed / stt['done']:.1f} s per file · "
                               f"{Path(stt['current']).name[:40]}")
                else:
                    st.caption(f"Estimating time… {Path(stt['current']).name[:40]}")
            elif stats["meta_only"]:
                st.caption(f"{stats['meta_only']:,} files are searchable by name only so far. "
                           "Start step 2 to index their contents.")
        _content_status()

    s = eng.db.stats()
    st.caption(f"{s['files']:,} files ({s['files'] - s['meta_only']:,} with content) · "
               f"{s['events']:,} activity events · {s['users']} users")

    st.subheader("Options")
    force = st.radio("Retrieval path", ["Auto (router decides)", "fast", "hybrid", "deep"], index=0)
    force_path = None if force.startswith("Auto") else force
    personalize = st.toggle("Personalization", value=True)
    really_open = st.toggle("Open files on click", value=False, help="Also launch the file in its app")

    with st.expander("Models"):
        llm = get_llm()
        st.write(f"**Embedder:** {SETTINGS.embed_model if SETTINGS.embed_backend == 'qwen' else 'offline hash'}")
        st.write(f"**Reranker:** {SETTINGS.rerank_model if SETTINGS.rerank_backend == 'qwen' else 'offline overlap'}")
        st.write(f"**Pictures:** {SETTINGS.image_model if SETTINGS.image_backend == 'clip' else SETTINGS.image_backend}"
                 f" · {eng.vec.image_count():,} pictures indexed")
        st.write(f"**LLM:** {SETTINGS.llm_model} — {'✅ connected' if llm.available() else '⚠️ Ollama not reachable (rules only)'}")
        if st.button("Warm up models"):
            with st.spinner("Loading all models (embedder, reranker, pictures, LLM)…"):
                get_embedder().embed_query("warm up")
                get_reranker().score("warm up", ["warm up"])
                if get_image_embedder() is not None:
                    get_image_embedder().embed_text("warm up")
                if llm.available():  # loads the LLM into memory so the first deep question is not slow
                    llm.json("Reply with JSON.", "Say ok.", {"type": "object", "properties": {"ok": {"type": "string"}}},
                             max_tokens=5)
            st.success("Models ready")


def open_file(rec, query):
    eng.open_file(user, rec.file_id, query=query, session_id=st.session_state.session_id)
    st.session_state.last_opened = rec.name
    if really_open:
        try:
            if platform.system() == "Windows":
                os.startfile(rec.path)  # type: ignore[attr-defined]
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", rec.path])
            else:
                subprocess.Popen(["xdg-open", rec.path])
        except Exception as e:  # noqa: BLE001
            st.session_state.open_error = str(e)


def render_results(out, key_prefix: str, compact: bool = False):
    for i, r in enumerate(out["recs"], 1):
        chips = "".join(f'<span class="chip" style="background:{TAG_COLOR.get(t, "#4a5568")}">{txt}</span>'
                        for t, txt in r.reasons)
        folder_rel = Path(r.path).parent.name
        st.markdown(f"""<div class="card"><span class="fname">{i}. {r.name}</span>
            <span class="path"> · {folder_rel} · score {r.score:.3f} (relevance {r.relevance:.2f})</span><br>{chips}
            {'' if compact else f'<div class="snip">{r.snippet}</div>'}</div>""", unsafe_allow_html=True)
        if r.ext in ("png", "jpg", "jpeg") and not compact:
            try:
                st.image(r.path, width=220)
            except Exception:  # noqa: BLE001  (moved, cloud-only or unreadable picture)
                pass
        if not compact:
            cols = st.columns([1, 1, 6])
            cols[0].button("Open", key=f"{key_prefix}-open-{r.file_id}", on_click=open_file, args=(r, out["query"]))
            with cols[1].popover("Why?"):
                df = pd.DataFrame({"signal": list(r.contributions), "contribution": list(r.contributions.values())})
                st.bar_chart(df.set_index("signal"))
                st.json({k: round(v, 3) for k, v in r.signals.items()})


tab_search, tab_compare, tab_profile = st.tabs(["🔎 Search", "⚖️ Compare paths", "👤 User insights"])

# ---------------------------------------------------------------- search
with tab_search:
    st.markdown("#### Ask for a file the way you'd ask a colleague")
    ex_cols = st.columns(len(EXAMPLES))
    for c, ex in zip(ex_cols, EXAMPLES):
        if c.button(ex, width="stretch"):
            st.session_state.query = ex
    query = st.text_input("Query", key="query", placeholder="e.g. " + EXAMPLES[2],
                          label_visibility="collapsed")
    if st.session_state.get("last_opened"):
        st.toast(f"Logged open: {st.session_state.pop('last_opened')} — your profile updated")
    if st.session_state.get("open_error"):
        st.warning(st.session_state.pop("open_error"))

    if query:
        with st.spinner("Thinking…"):
            out = eng.search(query, user, force_path=force_path, personalize=personalize)
        left, right = st.columns([3, 2], gap="large")
        with left:
            path = out["path"]
            esc = f" · escalated: {' → '.join(e.split(' (')[0] for e in out['escalations'])}" if out["escalations"] else ""
            st.markdown(f'<span class="badge" style="background:{PATH_COLOR[path]}">{PATH_LABEL[path]}</span>'
                        f'&nbsp; {out["latency_ms"]:.0f} ms · model calls: {sum(out["model_calls"].values())}'
                        f'{" · ⚡ cached" if out["cached"] else ""}{esc}', unsafe_allow_html=True)
            st.write("")
            if not out["recs"]:
                st.info("No matching files. Try different words or remove a filter.")
            render_results(out, "s")
        with right:
            st.markdown("##### Agent trace")
            for t in out["trace"]:
                calls = ", ".join(f"{k}×{v}" for k, v in t["model_calls"].items()) or "no model calls"
                icon = "⤴️" if t.get("escalation") else "▸"
                st.markdown(f"{icon} **{t['stage']}** — {t['ms']:.0f} ms · {calls}  \n"
                            f"<span style='color:gray;font-size:.85rem'>{t.get('detail', '')}</span>",
                            unsafe_allow_html=True)
            st.markdown("##### Parsed query")
            plan = out["plan"].to_dict()
            st.json({k: v for k, v in plan.items() if v not in (None, [], "")})

# ---------------------------------------------------------------- compare
with tab_compare:
    st.markdown("#### Same query, every path — what does the router save?")
    cq = st.text_input("Query to compare", value="why did we drop kafka" if demo_indexed else
                       "notes about machine learning", key="cq")
    if st.button("Run comparison") and cq:
        rows, outs = [], {}
        for p in ["fast", "hybrid", "deep", None]:
            o = eng.search(cq, user, force_path=p, personalize=personalize, use_cache=False)
            outs[p or "auto"] = o
            rows.append({"path": p or f"auto → {o['path']}", "latency ms": o["latency_ms"],
                         "model calls": sum(o["model_calls"].values()),
                         "top result": o["recs"][0].name if o["recs"] else "—"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Note: the first deep run includes model loading time. Run twice for steady-state latency.")
        st.markdown("#### Personalization on vs off")
        a, b = st.columns(2)
        with a:
            st.markdown("**Without personalization**")
            render_results(eng.search(cq, user, personalize=False, use_cache=False), "off", compact=True)
        with b:
            st.markdown(f"**Personalized for {user}**")
            render_results(eng.search(cq, user, personalize=True, use_cache=False), "on", compact=True)

# ---------------------------------------------------------------- profile
with tab_profile:
    prof = build_profile(eng.db, user)
    if not prof.n_events:
        st.info(f"No activity for {user} yet. Open some files from search results"
                + (", or click 'Load demo personas'." if demo_indexed else
                   ", or click 'Import Windows recent files' in 🖥️ Whole computer."))
    else:
        need = set(prof.count) | set(prof.working_set) | set(prof.graph.nodes)
        files = eng.db.files_by_ids(need)
        m1, m2, m3 = st.columns(3)
        m1.metric("Activity events", prof.n_events)
        m2.metric("Distinct files", len(prof.count))
        m3.metric("Working set now", len(prof.working_set))
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Most relied-on files (decayed frequency)")
            top = [{"file": files[f]["name"], "opens": prof.count[f], "edits": prof.edits.get(f, 0),
                    "usual time": prof.usual_time(f) or "—"} for f in prof.top_files(10) if f in files]
            st.dataframe(pd.DataFrame(top), hide_index=True, width="stretch")
            st.markdown("##### Working on right now")
            st.write(", ".join(files[f]["name"] for f in prof.working_set if f in files) or "—")
        with c2:
            st.markdown("##### Preferred file types")
            st.bar_chart(pd.Series(prof.ext_pref).sort_values(ascending=False))
            st.markdown("##### When this user works (weekday × hour)")
            grid = pd.DataFrame(0, index=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], columns=range(7, 21))
            for hist in prof.when.values():
                for d, h in hist:
                    if h in grid.columns:
                        grid.iloc[d, grid.columns.get_loc(h)] += 1
            st.dataframe(grid.style.background_gradient(cmap="Blues", axis=None), width="stretch")
        st.markdown("##### Files used together (co-access graph)")
        edges = sorted(prof.graph.edges(data=True), key=lambda e: -e[2]["weight"])[:12]
        st.dataframe(pd.DataFrame([{"file A": files[a]["name"], "file B": files[b]["name"], "sessions together": d["weight"]}
                                   for a, b, d in edges if a in files and b in files]),
                     hide_index=True, width="stretch")
