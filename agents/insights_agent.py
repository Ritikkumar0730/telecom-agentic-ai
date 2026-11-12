"""
Generative insights agent.

Pulls aggregated metrics from the database and uses the LLM to generate
actionable business insights — suitable for executive dashboards or daily
business briefings.
"""

from __future__ import annotations
import sqlite3
from typing import Dict, Any

INSIGHTS_PROMPT = """\
You are a senior data scientist at a telecom company. Based on the following
business metrics, generate 3–5 concise, actionable insights for the leadership team.
Focus on churn prevention, fraud risk, revenue growth, and customer satisfaction.

Metrics:
{metrics}

Format each insight as:
INSIGHT: <what is happening>
ACTION: <what to do about it>
"""


class InsightsAgent:
    """
    Generates LLM-powered business insights from the telecom database.

    Parameters
    ----------
    llm  : LLM client with .complete(prompt) method
    conn : seeded telecom SQLite connection
    """

    def __init__(self, llm, conn: sqlite3.Connection):
        self.llm = llm
        self.conn = conn

    def _collect_metrics(self) -> Dict[str, Any]:
        metrics = {}

        # Customer overview
        r = self.conn.execute(
            "SELECT COUNT(*) as total, "
            "SUM(CASE WHEN churn_risk_score > 0.7 THEN 1 ELSE 0 END) as high_risk, "
            "ROUND(AVG(monthly_charge), 2) as avg_arpu, "
            "ROUND(AVG(tenure_months), 1) as avg_tenure "
            "FROM customers"
        ).fetchone()
        metrics["total_customers"] = r["total"]
        metrics["high_churn_risk_customers"] = r["high_risk"]
        metrics["avg_arpu_inr"] = r["avg_arpu"]
        metrics["avg_tenure_months"] = r["avg_tenure"]

        # Plan distribution
        rows = self.conn.execute(
            "SELECT plan_type, COUNT(*) as cnt FROM customers GROUP BY plan_type"
        ).fetchall()
        metrics["plan_distribution"] = {r["plan_type"]: r["cnt"] for r in rows}

        # Fraud overview
        r = self.conn.execute(
            "SELECT COUNT(*) as total, "
            "SUM(flagged) as flagged, "
            "ROUND(SUM(CASE WHEN flagged=1 THEN amount ELSE 0 END), 2) as flagged_value "
            "FROM transactions"
        ).fetchone()
        metrics["total_transactions"] = r["total"]
        metrics["flagged_transactions"] = r["flagged"]
        metrics["flagged_transaction_value_inr"] = r["flagged_value"]

        # Support tickets
        r = self.conn.execute(
            "SELECT COUNT(*) as total, "
            "SUM(CASE WHEN status='open' THEN 1 ELSE 0 END) as open_tickets, "
            "SUM(CASE WHEN category='churn_risk' THEN 1 ELSE 0 END) as churn_tickets "
            "FROM support_tickets"
        ).fetchone()
        metrics["open_support_tickets"] = r["open_tickets"]
        metrics["churn_risk_tickets"] = r["churn_tickets"]

        # Low utilisation (upsell opportunity)
        r = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM customers "
            "WHERE (data_used_gb / data_limit_gb) < 0.4 AND plan_type = 'postpaid'"
        ).fetchone()
        metrics["low_utilisation_postpaid_customers"] = r["cnt"]

        return metrics

    def generate(self) -> Dict[str, Any]:
        metrics = self._collect_metrics()
        metrics_str = "\n".join(f"  {k}: {v}" for k, v in metrics.items())
        prompt = INSIGHTS_PROMPT.format(metrics=metrics_str)
        insights_text = self.llm.complete(prompt, max_tokens=512)
        return {
            "metrics": metrics,
            "insights": insights_text,
        }

    def daily_briefing(self) -> str:
        result = self.generate()
        m = result["metrics"]
        lines = [
            "=" * 60,
            "  TELECOM BUSINESS INTELLIGENCE — DAILY BRIEFING",
            "=" * 60,
            f"  Customers       : {m['total_customers']}",
            f"  High churn risk : {m['high_churn_risk_customers']} "
            f"({100 * m['high_churn_risk_customers'] // max(m['total_customers'], 1)}%)",
            f"  Avg ARPU        : ₹{m['avg_arpu_inr']}",
            f"  Flagged txns    : {m['flagged_transactions']} "
            f"(₹{m['flagged_transaction_value_inr']:,.0f} at risk)",
            f"  Open tickets    : {m['open_support_tickets']}",
            f"  Upsell oppty    : {m['low_utilisation_postpaid_customers']} "
            "low-utilisation postpaid customers",
            "",
            "INSIGHTS",
            "-" * 60,
            result["insights"],
            "=" * 60,
        ]
        return "\n".join(lines)
