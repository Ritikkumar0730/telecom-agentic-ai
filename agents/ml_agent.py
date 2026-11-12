"""
ML agent — natural language interface to churn and fraud models.

Understands queries like:
  "What is the churn risk for customer C002?"
  "Score transaction T004 for fraud."
  "List all high-risk customers."
  "Show me suspicious transactions."
"""

from __future__ import annotations
import re
import sqlite3
from typing import Dict, Any

from models.churn import ChurnPredictor
from models.fraud import FraudDetector


class MLAgent:
    """
    Wraps ChurnPredictor and FraudDetector behind a natural language interface.

    Parameters
    ----------
    llm     : LLM client (used to generate explanations)
    conn    : seeded telecom SQLite connection
    """

    def __init__(self, llm, conn: sqlite3.Connection):
        self.llm = llm
        self.conn = conn
        self._churn = ChurnPredictor()
        self._fraud = FraudDetector()
        self._churn.fit(conn)
        self._fraud.fit(conn)

    # ------------------------------------------------------------------ #

    def query(self, question: str) -> Dict[str, Any]:
        q = question.lower()

        # Churn — single customer
        if "churn" in q or "retention" in q or "cancel" in q or "leave" in q:
            cid = self._extract_id(question, prefix="C")
            if cid:
                result = self._churn.predict_customer(self.conn, cid)
                result["explanation"] = self._explain(question, result)
                return {"type": "churn_single", **result}
            # Churn — all customers
            all_results = self._churn.predict_all(self.conn)
            high = [r for r in all_results if r["risk_band"] == "HIGH"]
            return {
                "type": "churn_all",
                "total_customers": len(all_results),
                "high_risk_count": len(high),
                "high_risk_customers": high,
                "summary": (
                    f"{len(high)} of {len(all_results)} customers are at HIGH churn risk. "
                    "Immediate retention intervention recommended."
                ),
            }

        # Fraud — single transaction
        if "fraud" in q or "suspicious" in q or "anomaly" in q or "flag" in q:
            tid = self._extract_id(question, prefix="T")
            if tid:
                result = self._fraud.score_transaction(self.conn, tid)
                result["explanation"] = self._explain(question, result)
                return {"type": "fraud_single", **result}
            # Fraud — all transactions
            all_results = self._fraud.score_all(self.conn)
            flagged = [r for r in all_results if r["is_suspicious"]]
            return {
                "type": "fraud_all",
                "total_transactions": len(all_results),
                "suspicious_count": len(flagged),
                "suspicious_transactions": flagged,
                "summary": (
                    f"{len(flagged)} of {len(all_results)} transactions flagged as suspicious. "
                    "Immediate review required for HIGH risk items."
                ),
            }

        return {"type": "unknown", "message": "Could not determine ML task from question."}

    def _extract_id(self, text: str, prefix: str) -> str:
        match = re.search(rf"\b({prefix}\d{{3}})\b", text, re.IGNORECASE)
        return match.group(1).upper() if match else ""

    def _explain(self, question: str, result: Dict) -> str:
        summary = str(result)[:300]
        prompt = (
            f"Given these ML model results for a telecom customer analysis:\n{summary}\n\n"
            f"Original question: {question}\n\n"
            "Write a concise 2-sentence business explanation:"
        )
        return self.llm.complete(prompt, max_tokens=150)
