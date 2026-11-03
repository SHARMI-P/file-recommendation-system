"""Evaluate retrieval quality and cost on the demo corpus.

Compares the adaptive router against fixed pipelines (always hybrid, always deep) and measures what
personalization adds. Prints Recall@3, MRR, mean latency and model calls per query.

Usage (close the Streamlit app first):  python scripts/evaluate.py
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filerec.pipeline import Engine  # noqa: E402

# (user, query, set of correct file names)
BENCH = [
    ("priya", "Q3_rates_v4.pptx", {"Q3_rates_v4.pptx"}),
    ("priya", "open discount_policy.docx", {"discount_policy.docx"}),
    ("rahul", "k8s upgrade runbook", {"k8s_upgrade_runbook.md"}),
    ("rahul", "oncall rotation", {"oncall_rotation_oct.csv"}),
    ("priya", "the pricing deck I worked on last week", {"Q3_rates_v4.pptx"}),
    ("priya", "customer churn analysis", {"final_FINAL_v2.docx"}),
    ("priya", "budget file", {"Marketing_Budget_FY27.xlsx"}),
    ("rahul", "budget file", {"Cloud_Infra_Budget_2026.xlsx"}),
    ("rahul", "why did we drop kafka", {"notes_0912.md"}),
    ("rahul", "what caused the dashboard slowdown incident", {"incident_postmortem_0904.docx"}),
    ("rahul", "slack alerts code", {"slack_client.py", "alert_evaluator.py"}),
    ("priya", "plan for launching slack alerts", {"launch_plan_slack_alerts.docx"}),
    ("priya", "what do users say about onboarding", {"user_interviews_summary.md", "NPS_survey_responses.csv",
                                                       "final_FINAL_v2.docx"}),
    ("priya", "conversion rates of the price test", {"pricing_experiment_results.xlsx"}),
    ("rahul", "which database did we choose and why", {"ADR-007-database-choice.md"}),
    ("rahul", "sql migration for events table", {"migrate_events_to_postgres.sql"}),
    ("priya", "what does globex need before signing", {"enterprise_deal_globex.docx", "SSO_SAML_design.docx"}),
    ("priya", "prep for the acme call", {"client_call_acme_prep.docx"}),
    ("rahul", "API rate limit", {"API_reference_v3.pdf"}),
    ("priya", "how many leave days do we get", {"leave_policy_2026.pdf"}),
    ("priya", "new joiner onboarding slides", {"untitled_deck.pptx"}),
    ("rahul", "pdfs from last month", {"API_reference_v3.pdf", "Q3_revenue_report.pdf"}),
    # Ambiguous without knowing the user: personalization should resolve these
    ("rahul", "the python file I'm working on", {"alert_evaluator.py", "slack_client.py"}),
    ("priya", "the spreadsheet I check on mondays", {"pricing_experiment_results.xlsx"}),
    ("priya", "the notes I keep opening", {"leadership_sync_notes.md", "user_interviews_summary.md"}),
]

CONFIGS = [
    ("Adaptive router (ours)", None, True),
    ("Always hybrid", "hybrid", True),
    ("Always deep", "deep", True),
    ("Adaptive, no personalization", None, False),
]


def run():
    eng = Engine()
    # Warm the models once so the first query does not carry loading time
    eng.search("warm up the models", "priya", force_path="deep", use_cache=False)
    print(f"{'Configuration':<30}{'Recall@3':>10}{'MRR':>8}{'ms/query':>10}{'model calls':>13}")
    for label, path, pers in CONFIGS:
        rec3, rr, ms, calls = [], [], [], []
        for user, q, gold in BENCH:
            out = eng.search(q, user, force_path=path, personalize=pers, use_cache=False)
            names = [r.name for r in out["recs"]]
            rank = next((i for i, n in enumerate(names, 1) if n in gold), None)
            rec3.append(1.0 if rank and rank <= 3 else 0.0)
            rr.append(1 / rank if rank else 0.0)
            ms.append(out["latency_ms"])
            calls.append(sum(out["model_calls"].values()))
        print(f"{label:<30}{statistics.mean(rec3):>10.2f}{statistics.mean(rr):>8.2f}"
              f"{statistics.mean(ms):>10.0f}{statistics.mean(calls):>13.2f}")
    print("\nPaths chosen by the router:")
    for user, q, _ in BENCH:
        out = eng.search(q, user, use_cache=False)
        print(f"  {out['path']:<7} {q}")


if __name__ == "__main__":
    run()
