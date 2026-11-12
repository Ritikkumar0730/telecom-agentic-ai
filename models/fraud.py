"""
Transaction fraud / anomaly detection model.

Approach: IsolationForest (unsupervised) — no labelled fraud data needed at
training time, making it practical for cold-start deployment.

Features
--------
- amount          : transaction amount in ₹
- hour_of_day     : 0–23 (late-night transactions are riskier)
- channel_encoded : app=0, online=1, ivr=2, store=3
- is_new_device   : 1 if device_id not seen before for this customer
"""

from __future__ import annotations
import sqlite3
import numpy as np
from typing import Dict, List, Any

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

CHANNEL_MAP = {"app": 0, "online": 1, "ivr": 2, "store": 3}
FRAUD_THRESHOLD = 0.0  # IsolationForest decision function; below = anomaly


class FraudDetector:
    """
    Fit on historical transactions and score new ones.

    Usage
    -----
    detector = FraudDetector()
    detector.fit(conn)
    result = detector.score_transaction(conn, "T004")
    """

    def __init__(self, contamination: float = 0.15):
        if not _SKLEARN:
            raise ImportError("scikit-learn is required: pip install scikit-learn")
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("iso", IsolationForest(
                n_estimators=100,
                contamination=contamination,
                random_state=42,
            )),
        ])
        self._fitted = False
        self._customer_devices: Dict[str, set] = {}

    # ------------------------------------------------------------------ #

    def _build_device_map(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            "SELECT customer_id, device_id FROM transactions WHERE flagged = 0"
        ).fetchall()
        for r in rows:
            self._customer_devices.setdefault(r["customer_id"], set()).add(r["device_id"])

    def _extract_features(self, rows, known_devices: Dict[str, set]) -> np.ndarray:
        X = []
        for r in rows:
            ch = CHANNEL_MAP.get(r["channel"] if isinstance(r, dict) else r["channel"], 0)
            known = known_devices.get(
                r["customer_id"] if isinstance(r, dict) else r["customer_id"], set()
            )
            device = r["device_id"] if isinstance(r, dict) else r["device_id"]
            is_new = 0 if device in known else 1
            X.append([r["amount"], r["hour_of_day"], ch, is_new])
        return np.array(X, dtype=float)

    def fit(self, conn: sqlite3.Connection) -> "FraudDetector":
        self._build_device_map(conn)
        rows = conn.execute("SELECT * FROM transactions").fetchall()
        if not rows:
            raise ValueError("No transaction data found — seed the database first.")
        X = self._extract_features(rows, self._customer_devices)
        self.pipeline.fit(X)
        self._fitted = True
        return self

    def score_transaction(self, conn: sqlite3.Connection, txn_id: str) -> Dict[str, Any]:
        row = conn.execute(
            "SELECT * FROM transactions WHERE txn_id = ?", (txn_id,)
        ).fetchone()
        if row is None:
            return {"error": f"Transaction {txn_id} not found."}
        X = self._extract_features([row], self._customer_devices)
        score = float(self.pipeline.decision_function(X)[0])
        is_fraud = score < FRAUD_THRESHOLD
        return {
            "txn_id": txn_id,
            "customer_id": row["customer_id"],
            "amount": row["amount"],
            "channel": row["channel"],
            "hour_of_day": row["hour_of_day"],
            "anomaly_score": round(score, 4),
            "is_suspicious": is_fraud,
            "risk_level": "HIGH" if score < -0.3 else "MEDIUM" if is_fraud else "LOW",
            "flags": self._explain(row, score),
        }

    def score_all(self, conn: sqlite3.Connection) -> List[Dict[str, Any]]:
        rows = conn.execute(
            "SELECT * FROM transactions ORDER BY amount DESC"
        ).fetchall()
        results = []
        for row in rows:
            X = self._extract_features([row], self._customer_devices)
            score = float(self.pipeline.decision_function(X)[0])
            is_fraud = score < FRAUD_THRESHOLD
            results.append({
                "txn_id": row["txn_id"],
                "customer_id": row["customer_id"],
                "amount": row["amount"],
                "anomaly_score": round(score, 4),
                "is_suspicious": is_fraud,
                "risk_level": "HIGH" if score < -0.3 else "MEDIUM" if is_fraud else "LOW",
            })
        return sorted(results, key=lambda x: x["anomaly_score"])

    def _explain(self, row, score: float) -> List[str]:
        flags = []
        if row["amount"] > 5000:
            flags.append(f"Large transaction amount (₹{row['amount']:,.0f})")
        if row["hour_of_day"] is not None and row["hour_of_day"] < 5:
            flags.append(f"Unusual hour ({row['hour_of_day']:02d}:xx AM)")
        known = self._customer_devices.get(row["customer_id"], set())
        if row["device_id"] not in known:
            flags.append("New/unrecognised device")
        if score < -0.3:
            flags.append("Extreme statistical outlier vs customer baseline")
        if not flags:
            flags.append("Moderate anomaly — further review recommended")
        return flags
