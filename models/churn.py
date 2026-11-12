"""
Customer churn prediction model.

Features
--------
- tenure_months     : how long the customer has been subscribed
- monthly_charge    : monthly bill amount
- data_utilisation  : fraction of data allowance used (0-1)
- support_calls     : number of support interactions
- plan_type_encoded : 0=prepaid, 1=postpaid, 2=enterprise

Model: LogisticRegression (interpretable, fast, production-deployable).
Trained on the seeded demo dataset; replace with real historical data in prod.
"""

from __future__ import annotations
import sqlite3
import numpy as np
from typing import Dict, List, Any

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

PLAN_MAP = {"prepaid": 0, "postpaid": 1, "enterprise": 2}
RISK_BANDS = [(0.7, "HIGH"), (0.4, "MEDIUM"), (0.0, "LOW")]


def _risk_band(prob: float) -> str:
    for threshold, label in RISK_BANDS:
        if prob >= threshold:
            return label
    return "LOW"


class ChurnPredictor:
    """
    Train on customer records from the SQLite DB and score individual customers.

    Usage
    -----
    predictor = ChurnPredictor()
    predictor.fit(conn)
    result = predictor.predict_customer(conn, "C002")
    """

    def __init__(self):
        if not _SKLEARN:
            raise ImportError("scikit-learn is required: pip install scikit-learn")
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, random_state=42)),
        ])
        self._fitted = False
        self.feature_names = [
            "tenure_months", "monthly_charge", "data_utilisation",
            "support_calls", "plan_type_encoded",
        ]

    # ------------------------------------------------------------------ #

    def _extract_features(self, rows) -> np.ndarray:
        X = []
        for r in rows:
            utilisation = r["data_used_gb"] / max(r["data_limit_gb"], 0.01)
            plan_enc = PLAN_MAP.get(r["plan_type"], 1)
            X.append([
                r["tenure_months"],
                r["monthly_charge"],
                utilisation,
                r["support_calls"],
                plan_enc,
            ])
        return np.array(X, dtype=float)

    def fit(self, conn: sqlite3.Connection) -> "ChurnPredictor":
        rows = conn.execute(
            "SELECT * FROM customers"
        ).fetchall()
        if not rows:
            raise ValueError("No customer data found — seed the database first.")
        X = self._extract_features(rows)
        # Use churn_risk_score > 0.5 as pseudo-label for demo
        y = np.array([1 if r["churn_risk_score"] > 0.5 else 0 for r in rows])
        if len(set(y)) < 2:
            y[0] = 1 - y[0]  # ensure both classes present for demo
        self.pipeline.fit(X, y)
        self._fitted = True
        return self

    def predict_proba(self, features: List[float]) -> float:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_proba().")
        X = np.array(features, dtype=float).reshape(1, -1)
        return float(self.pipeline.predict_proba(X)[0, 1])

    def predict_customer(self, conn: sqlite3.Connection, customer_id: str) -> Dict[str, Any]:
        row = conn.execute(
            "SELECT * FROM customers WHERE customer_id = ?", (customer_id,)
        ).fetchone()
        if row is None:
            return {"error": f"Customer {customer_id} not found."}
        X = self._extract_features([row])
        prob = float(self.pipeline.predict_proba(X)[0, 1])
        return {
            "customer_id": customer_id,
            "name": row["name"],
            "churn_probability": round(prob, 4),
            "risk_band": _risk_band(prob),
            "top_risk_factors": self._explain(row),
        }

    def predict_all(self, conn: sqlite3.Connection) -> List[Dict[str, Any]]:
        rows = conn.execute(
            "SELECT * FROM customers ORDER BY churn_risk_score DESC"
        ).fetchall()
        results = []
        for row in rows:
            X = self._extract_features([row])
            prob = float(self.pipeline.predict_proba(X)[0, 1])
            results.append({
                "customer_id": row["customer_id"],
                "name": row["name"],
                "plan_type": row["plan_type"],
                "tenure_months": row["tenure_months"],
                "churn_probability": round(prob, 4),
                "risk_band": _risk_band(prob),
            })
        return results

    def _explain(self, row) -> List[str]:
        reasons = []
        if row["tenure_months"] <= 3:
            reasons.append("Very new customer (tenure ≤ 3 months)")
        if row["support_calls"] >= 4:
            reasons.append(f"High support contact ({row['support_calls']} calls)")
        utilisation = row["data_used_gb"] / max(row["data_limit_gb"], 0.01)
        if utilisation < 0.3:
            reasons.append(f"Low data utilisation ({utilisation:.0%} of allowance used)")
        if row["plan_type"] == "prepaid" and row["monthly_charge"] < 200:
            reasons.append("Low-value prepaid plan — high price sensitivity")
        if not reasons:
            reasons.append("Moderate risk — monitor closely")
        return reasons
