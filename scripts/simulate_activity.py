"""Simulate 30 days of file activity for two personas so personalization has something to learn.

  priya - product manager: pricing, marketing budget, customer research; reviews pricing on Monday
          mornings and fills her timesheet on Friday afternoons.
  rahul - backend engineer: alerting code, the Kafka-to-Postgres migration, infra budget; checks the
          on-call rotation on Monday mornings.

Each persona also has a "current working set" (files opened in the last hour), which drives the
co-access signal. Run this AFTER indexing the demo workspace.

Usage:  python scripts/simulate_activity.py
"""
from __future__ import annotations

import random
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filerec.config import SETTINGS  # noqa: E402
from filerec.db import DB  # noqa: E402

random.seed(42)
DAY = 86400

# routine: (weekdays, hour range, files opened together in one session, probability per matching day)
PERSONAS = {
    "priya": {
        "routines": [
            ([0], (9, 11), ["Q3_rates_v4.pptx", "pricing_experiment_results.xlsx", "leadership_sync_notes.md"], 0.95),
            ([0, 1, 2, 3], (14, 17), ["Marketing_Budget_FY27.xlsx", "launch_plan_slack_alerts.docx"], 0.6),
            ([1, 3], (10, 12), ["final_FINAL_v2.docx", "user_interviews_summary.md", "NPS_survey_responses.csv"], 0.5),
            ([2], (11, 13), ["roadmap_H2_2026.pptx", "feature_requests_backlog.csv"], 0.7),
            ([4], (16, 18), ["timesheet_template.xlsx"], 0.95),
            ([1, 2, 3, 4], (12, 15), ["sales_pipeline_Q3.xlsx", "enterprise_deal_globex.docx"], 0.3),
        ],
        "edits": {"Q3_rates_v4.pptx": 0.6, "Marketing_Budget_FY27.xlsx": 0.3, "launch_plan_slack_alerts.docx": 0.3},
        "working_set": ["client_call_acme_prep.docx", "Q3_rates_v4.pptx"],
    },
    "rahul": {
        "routines": [
            ([0], (9, 10), ["oncall_rotation_oct.csv", "k8s_upgrade_runbook.md"], 0.9),
            ([0, 1, 2, 3, 4], (10, 13), ["alert_evaluator.py", "slack_client.py", "system_design_overview.pdf"], 0.7),
            ([1, 2, 3], (14, 18), ["migrate_events_to_postgres.sql", "notes_0912.md", "incident_postmortem_0904.docx"], 0.5),
            ([3], (15, 17), ["Cloud_Infra_Budget_2026.xlsx", "FY27_company_budget_draft.xlsx"], 0.8),
            ([2, 4], (11, 12), ["API_reference_v3.pdf", "SSO_SAML_design.docx"], 0.4),
            ([4], (16, 17), ["sprint_42_retro.md"], 0.6),
        ],
        "edits": {"alert_evaluator.py": 0.7, "slack_client.py": 0.6, "migrate_events_to_postgres.sql": 0.4},
        "working_set": ["alert_evaluator.py", "slack_client.py"],
    },
}


def main(days: int = 30, db: DB | None = None) -> int:
    db = db or DB(SETTINGS.db_path)
    by_name = {r["name"]: r["id"] for r in db.all_files()}
    if not by_name:
        raise SystemExit("No files indexed yet. Index the demo workspace first (see README).")
    db.x("DELETE FROM activity WHERE user_id IN (?, ?)", tuple(PERSONAS))
    now = time.time()
    lt = time.localtime(now)
    midnight = now - (lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec)

    total = 0
    for user, p in PERSONAS.items():
        for d in range(days, 0, -1):
            day_start = midnight - d * DAY
            wday = time.localtime(day_start + 12 * 3600).tm_wday
            for wdays, (h0, h1), files, prob in p["routines"]:
                if wday not in wdays or random.random() > prob:
                    continue
                ts = day_start + random.uniform(h0, h1) * 3600
                sid = uuid.uuid4().hex[:8]
                for name in files:
                    if name not in by_name:
                        continue
                    action = "edit" if random.random() < p["edits"].get(name, 0) else "open"
                    db.log_activity(user, by_name[name], action, ts=ts, session_id=sid)
                    ts += random.uniform(3, 15) * 60
                    total += 1
        # Current session: what the user is working on right now
        sid = uuid.uuid4().hex[:8]
        for i, name in enumerate(p["working_set"]):
            if name in by_name:
                db.log_activity(user, by_name[name], "open", ts=now - (60 - 20 * i) * 60, session_id=sid)
                total += 1
    print(f"Logged {total} activity events for {', '.join(PERSONAS)}")
    return total


if __name__ == "__main__":
    main()
