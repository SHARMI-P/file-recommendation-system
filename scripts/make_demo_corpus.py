"""Generate a realistic demo workspace for the fictional company "Nimbus Analytics".

Creates ~45 files (PDF, DOCX, PPTX, XLSX, CSV, MD, PY, SQL, PNG) across project folders, with
modification dates spread over the last 60 days. Several files are deliberately badly named
(e.g. Q3_rates_v4.pptx is the pricing deck, notes_0912.md records the Kafka decision) so that
filename search alone fails and semantic search has to do the work.

Usage:  python scripts/make_demo_corpus.py demo_workspace
"""
from __future__ import annotations

import csv
import os
import random
import sys
import time
from pathlib import Path

from docx import Document
from fpdf import FPDF
from openpyxl import Workbook
from pptx import Presentation
from pptx.util import Pt

random.seed(7)
DAY = 86400

# (folder, filename, days_ago, content). Content shape depends on extension:
#   docx/pdf/md/txt: list of paragraphs (first = title); pptx: list of (title, bullets)
#   xlsx/csv: list of rows (first = header); py/sql: source text; png: None
FILES = [
    # ---------------- Product / pricing (Priya) ----------------
    ("Product/Pricing", "Q3_rates_v4.pptx", 4, [
        ("Pricing Proposal for Q3", ["Nimbus Analytics - new subscription rates", "Prepared for leadership review"]),
        ("Current tiers", ["Starter 49 USD per month, up to 5 dashboards", "Team 199 USD per month", "Enterprise custom"]),
        ("Proposed changes", ["Raise Team tier to 229 USD with usage-based add-on", "Introduce annual discount of 18 percent",
                              "Enterprise floor of 2,000 USD per month"]),
        ("Competitor benchmark", ["Competitor A charges 250 USD for comparable seats", "Competitor B bundles alerts for free"]),
        ("Expected impact", ["Revenue uplift of 9 to 12 percent", "Churn risk concentrated in small teams"]),
    ]),
    ("Product/Pricing", "Q2_rates_v2.pptx", 52, [
        ("Pricing Review Q2", ["Previous quarter subscription pricing review"]),
        ("Decision", ["Keep Starter and Team prices unchanged", "Revisit enterprise packaging next quarter"]),
    ]),
    ("Product/Pricing", "pricing_experiment_results.xlsx", 9, [
        ["Variant", "Price USD", "Trials", "Conversions", "Conversion rate"],
        ["Control", 199, 1200, 96, 0.08], ["Variant A", 229, 1180, 89, 0.075], ["Variant B", 249, 1210, 71, 0.059],
    ]),
    ("Product/Pricing", "discount_policy.docx", 30, [
        "Discount Policy for Sales Team",
        "Sales may offer up to 15 percent discount on annual Team plans without approval.",
        "Discounts above 15 percent require approval from the head of sales and finance.",
        "Non-profit and education customers receive a standing 30 percent discount.",
    ]),
    ("Product/Roadmap", "roadmap_H2_2026.pptx", 12, [
        ("H2 2026 Product Roadmap", ["Themes: collaboration, alerting, enterprise readiness"]),
        ("Q3", ["Shared dashboards", "Slack alerts", "SSO with SAML"]),
        ("Q4", ["Anomaly detection", "Audit logs", "Usage-based billing"]),
    ]),
    ("Product/Roadmap", "feature_requests_backlog.csv", 6, [
        ["ID", "Request", "Customer", "Votes", "Status"],
        ["FR-101", "Export dashboards to PDF", "Acme Corp", 42, "Planned"],
        ["FR-117", "Slack alert integration", "Globex", 37, "In progress"],
        ["FR-123", "Dark mode", "Initech", 25, "Backlog"],
        ["FR-130", "Row level security", "Umbrella", 19, "Planned"],
    ]),
    ("Product/Research", "final_FINAL_v2.docx", 15, [
        "Customer Attrition Analysis - Q3",
        "Monthly logo attrition rose from 2.1 percent to 3.4 percent between April and August.",
        "Most cancelling customers were small teams on the Starter plan in their first 90 days.",
        "Exit surveys cite missing onboarding guidance and the lack of Slack alerts.",
        "Recommendation: guided onboarding checklist, proactive outreach in week 2, ship Slack alerts in Q3.",
    ]),
    ("Product/Research", "user_interviews_summary.md", 21, [
        "# User Interview Summary",
        "We interviewed 14 analysts and 6 team leads in August.",
        "Top pain point: building the first dashboard takes too long without templates.",
        "Second: alerts arrive by email only and are easy to miss.",
        "Quote theme: people want to share a dashboard link with their manager in one click.",
    ]),
    ("Product/Research", "NPS_survey_responses.csv", 18, [
        ["Respondent", "Plan", "Score", "Comment"],
        ["r1", "Team", 9, "Love the dashboards"], ["r2", "Starter", 4, "Onboarding was confusing"],
        ["r3", "Enterprise", 8, "Need SSO"], ["r4", "Starter", 3, "Too expensive for what we use"],
    ]),
    ("Marketing", "Marketing_Budget_FY27.xlsx", 3, [
        ["Line item", "Q1", "Q2", "Q3", "Q4", "Total"],
        ["Paid search", 40000, 42000, 45000, 45000, 172000], ["Events", 25000, 10000, 30000, 15000, 80000],
        ["Content and SEO", 12000, 12000, 14000, 14000, 52000], ["Webinars", 5000, 5000, 6000, 6000, 22000],
    ]),
    ("Marketing", "launch_plan_slack_alerts.docx", 7, [
        "Launch Plan - Slack Alerts",
        "Launch date targeted for the second week of October.",
        "Channels: in-app banner, email to Team and Enterprise admins, blog post, LinkedIn.",
        "Success metric: 25 percent of active teams connect Slack within 30 days.",
    ]),
    ("Marketing", "brand_guidelines.pdf", 58, [
        "Nimbus Analytics Brand Guidelines",
        "Primary colour is deep blue, secondary is teal. Use the full logo on light backgrounds.",
        "Tone of voice: clear, confident and friendly. Avoid jargon in customer-facing copy.",
    ]),
    ("Marketing", "webinar_script_dashboards.docx", 26, [
        "Webinar Script - Building Your First Dashboard",
        "Introduce the template gallery, connect a sample data source, add three charts and share.",
    ]),
    ("Sales", "enterprise_deal_globex.docx", 11, [
        "Globex Enterprise Deal Notes",
        "Globex wants 120 seats, SSO and audit logs before signing. Budget approved for Q4.",
        "Proposed price 2,400 USD per month on a two year term.",
    ]),
    ("Sales", "sales_pipeline_Q3.xlsx", 5, [
        ["Account", "Stage", "Value USD", "Owner", "Close month"],
        ["Globex", "Negotiation", 57600, "Anita", "Oct"], ["Acme Corp", "Proposal", 24000, "Vikram", "Nov"],
        ["Initech", "Discovery", 12000, "Anita", "Dec"],
    ]),
    ("Meetings", "leadership_sync_notes.md", 2, [
        "# Leadership Sync",
        "Agreed to take the Q3 pricing proposal to the board with the 229 USD Team tier.",
        "Finance asked for a sensitivity analysis on churn before final approval.",
        "Marketing budget for FY27 approved with a 10 percent cut to events.",
    ]),
    ("Meetings", "weekly_product_review_0929.md", 9, [
        "# Weekly Product Review",
        "Slack alerts are in beta with 12 customers. Two bugs on message formatting.",
        "Shared dashboards design review scheduled for next week.",
    ]),
    ("Meetings", "client_call_acme_prep.docx", 4, [
        "Acme Corp Call Preparation",
        "Acme is evaluating our Team plan against a competitor. Bring the pricing comparison and the PDF export roadmap.",
    ]),
    # ---------------- Engineering (Rahul) ----------------
    ("Engineering/Architecture", "notes_0912.md", 26, [
        "# Notes 12 Sep",
        "Decision: we are dropping Kafka for the event pipeline and moving to Postgres with logical replication plus a job queue.",
        "Reasons: our event volume is around 2,000 events per second which Postgres handles comfortably.",
        "Kafka added operational overhead: three brokers, Zookeeper upgrades, and on-call pages for consumer lag.",
        "Trade-off: we lose replay beyond 7 days, accepted because analytics backfills read from the warehouse.",
    ]),
    ("Engineering/Architecture", "ADR-007-database-choice.md", 40, [
        "# ADR 007: Primary database",
        "Status: accepted. We chose PostgreSQL 16 over MongoDB for the core metadata store.",
        "Reasons: relational integrity for dashboards and permissions, mature tooling, JSONB for flexible fields.",
    ]),
    ("Engineering/Architecture", "system_design_overview.pdf", 33, [
        "Nimbus Platform System Design Overview",
        "Ingestion service receives events over HTTPS and writes them to Postgres.",
        "Query service compiles dashboard definitions into SQL against the ClickHouse warehouse.",
        "Alerting service evaluates rules every minute and sends notifications to email and Slack.",
    ]),
    ("Engineering/Architecture", "IMG_2041.png", 26, None),
    ("Engineering/Architecture", "event_pipeline_diagram.png", 45, None),
    ("Engineering/Infra", "Cloud_Infra_Budget_2026.xlsx", 6, [
        ["Service", "Monthly USD", "Annual USD", "Notes"],
        ["Compute (Kubernetes)", 18000, 216000, "3 node pools"], ["ClickHouse cluster", 9500, 114000, "Warehouse"],
        ["Postgres (managed)", 4200, 50400, "Primary plus replica"], ["Kafka (to be removed)", 3100, 37200, "Decommission in Q4"],
    ]),
    ("Engineering/Infra", "k8s_upgrade_runbook.md", 14, [
        "# Kubernetes Upgrade Runbook",
        "Drain one node pool at a time. Check pod disruption budgets before draining.",
        "Roll back by restoring the previous node pool image if health checks fail.",
    ]),
    ("Engineering/Infra", "incident_postmortem_0904.docx", 34, [
        "Incident Postmortem - 4 September",
        "Dashboards were slow for 47 minutes because Kafka consumer lag delayed event ingestion.",
        "Root cause: a consumer group rebalance loop after a broker restart.",
        "Action items: evaluate replacing Kafka, add lag alerts, document broker restart procedure.",
    ]),
    ("Engineering/Infra", "oncall_rotation_oct.csv", 1, [
        ["Week", "Primary", "Secondary"], ["Oct 5", "Rahul", "Meera"], ["Oct 12", "Meera", "Arjun"],
        ["Oct 19", "Arjun", "Rahul"],
    ]),
    ("Engineering/Code", "alert_evaluator.py", 3, """\"\"\"Evaluate alert rules and dispatch notifications to email and Slack.\"\"\"
import time

def evaluate(rule, metric_value):
    # Fire when the metric crosses the threshold in the configured direction
    if rule["direction"] == "above":
        return metric_value > rule["threshold"]
    return metric_value < rule["threshold"]

def dispatch(rule, channel_client):
    message = f"Alert: {rule['name']} crossed {rule['threshold']}"
    channel_client.send(rule["slack_channel"], message)  # Slack webhook
"""),
    ("Engineering/Code", "slack_client.py", 2, """\"\"\"Thin wrapper around the Slack incoming webhook API with retry and rate limiting.\"\"\"
import time, requests

class SlackClient:
    def __init__(self, webhook_url, max_retries=3):
        self.url = webhook_url
        self.max_retries = max_retries

    def send(self, channel, text):
        for attempt in range(self.max_retries):
            r = requests.post(self.url, json={"channel": channel, "text": text}, timeout=5)
            if r.status_code == 429:  # rate limited
                time.sleep(2 ** attempt)
                continue
            return r.ok
        return False
"""),
    ("Engineering/Code", "migrate_events_to_postgres.sql", 20, """-- Migration: move event storage from Kafka topics to Postgres
CREATE TABLE events (
    id BIGSERIAL PRIMARY KEY,
    tenant_id UUID NOT NULL,
    event_type TEXT NOT NULL,
    payload JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX idx_events_tenant_time ON events (tenant_id, created_at);
-- Logical replication publication for downstream consumers
CREATE PUBLICATION events_pub FOR TABLE events;
"""),
    ("Engineering/Code", "dashboard_query_compiler.py", 28, """\"\"\"Compile a dashboard JSON definition into a ClickHouse SQL query.\"\"\"

def compile_query(widget):
    cols = ", ".join(widget["metrics"])
    sql = f"SELECT {widget['dimension']}, {cols} FROM {widget['table']}"
    if widget.get("filters"):
        sql += " WHERE " + " AND ".join(widget["filters"])
    return sql + f" GROUP BY {widget['dimension']}"
"""),
    ("Engineering", "API_reference_v3.pdf", 24, [
        "Nimbus Public API v3 Reference",
        "Authentication uses bearer tokens. Rate limit is 600 requests per minute per token.",
        "Endpoints: GET /dashboards, POST /dashboards, GET /alerts, POST /alerts, POST /events.",
    ]),
    ("Engineering", "sprint_42_retro.md", 8, [
        "# Sprint 42 Retrospective",
        "Went well: Postgres event pipeline migration finished two days early.",
        "To improve: flaky integration tests in the alerting service.",
    ]),
    ("Engineering", "SSO_SAML_design.docx", 16, [
        "SSO with SAML - Design",
        "Support Okta and Azure AD first. Just-in-time user provisioning with default viewer role.",
        "Enterprise customers including Globex require SSO before contract signature.",
    ]),
    # ---------------- Shared / HR / Finance ----------------
    ("Finance", "FY27_company_budget_draft.xlsx", 10, [
        ["Department", "FY26 USD", "FY27 USD", "Change"],
        ["Engineering", 3200000, 3600000, "12.5%"], ["Marketing", 380000, 326000, "-14%"],
        ["Sales", 900000, 1050000, "16.7%"],
    ]),
    ("Finance", "Q3_revenue_report.pdf", 13, [
        "Q3 Revenue Report",
        "Recurring revenue grew 6 percent quarter on quarter to 1.42 million USD per month.",
        "Net revenue retention 104 percent. Gross churn 3.4 percent, up from 2.1 percent.",
    ]),
    ("Finance", "expense_claims_september.csv", 22, [
        ["Employee", "Category", "Amount USD"], ["Priya", "Travel", 640], ["Rahul", "Conference", 900],
    ]),
    ("HR", "untitled_deck.pptx", 37, [
        ("Welcome to Nimbus", ["New joiner onboarding"]),
        ("Your first week", ["Set up laptop and SSO", "Meet your buddy", "Read the engineering handbook"]),
    ]),
    ("HR", "leave_policy_2026.pdf", 55, [
        "Leave Policy 2026",
        "Employees receive 24 days of paid leave per year plus public holidays.",
    ]),
    ("HR", "team_offsite_agenda.docx", 19, [
        "Team Offsite Agenda - Goa",
        "Day 1: strategy session and pricing workshop. Day 2: roadmap planning and team activities.",
    ]),
    ("Personal", "timesheet_template.xlsx", 0, [
        ["Week", "Project", "Hours"], ["Mon", "Pricing", 8], ["Tue", "Roadmap", 7],
    ]),
    ("Personal", "reading_list.txt", 47, [
        "Reading list: Designing Data-Intensive Applications; Inspired by Marty Cagan; The Mom Test.",
    ]),
    ("Personal", "Screenshot 2026-09-30 at 10.14.22.png", 8, None),
]


def write_docx(path: Path, paras):
    d = Document()
    d.core_properties.title = paras[0]
    d.add_heading(paras[0], level=1)
    for p in paras[1:]:
        d.add_paragraph(p)
    d.save(path)


def write_pdf(path: Path, paras):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_title(paras[0])
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 10, paras[0], new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    for p in paras[1:]:
        pdf.multi_cell(0, 7, p, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
    pdf.output(str(path))


def write_pptx(path: Path, slides):
    prs = Presentation()
    for i, (title, bullets) in enumerate(slides):
        s = prs.slides.add_slide(prs.slide_layouts[0 if i == 0 else 1])
        s.shapes.title.text = title
        body = s.placeholders[1].text_frame
        body.text = bullets[0]
        for b in bullets[1:]:
            para = body.add_paragraph()
            para.text = b
            para.font.size = Pt(20)
    prs.core_properties.title = slides[0][0]
    prs.save(path)


def write_xlsx(path: Path, rows):
    wb = Workbook()
    ws = wb.active
    for r in rows:
        ws.append(r)
    wb.save(path)


def write_csv(path: Path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


def write_png(path: Path):
    try:
        from PIL import Image, ImageDraw
        im = Image.new("RGB", (640, 400), (245, 247, 250))
        dr = ImageDraw.Draw(im)
        for i in range(4):
            dr.rectangle([40 + i * 150, 160, 160 + i * 150, 240], outline=(40, 90, 160), width=3)
        im.save(path)
    except ImportError:  # minimal valid 1x1 PNG
        path.write_bytes(bytes.fromhex("89504e470d0a1a0a0000000d4948445200000001000000010806000000"
                                       "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082"))


def main(out: str):
    root = Path(out).resolve()
    now = time.time()
    for folder, name, days, content in FILES:
        d = root / folder
        d.mkdir(parents=True, exist_ok=True)
        p = d / name
        ext = p.suffix.lower()
        if ext == ".docx":
            write_docx(p, content)
        elif ext == ".pdf":
            write_pdf(p, content)
        elif ext == ".pptx":
            write_pptx(p, content)
        elif ext == ".xlsx":
            write_xlsx(p, content)
        elif ext == ".csv":
            write_csv(p, content)
        elif ext == ".png":
            write_png(p)
        elif ext in {".py", ".sql"}:
            p.write_text(content, encoding="utf-8")
        else:
            p.write_text("\n\n".join(content), encoding="utf-8")
        t = now - days * DAY - random.randint(0, 6 * 3600)
        os.utime(p, (t, t))
    print(f"Created {len(FILES)} files in {root}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "demo_workspace")
