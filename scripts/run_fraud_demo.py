"""
Demo: fraud/anomaly detection on all transactions.

Run with:
    python scripts/run_fraud_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.seed import get_connection
from models.fraud import FraudDetector


def main():
    conn = get_connection()
    detector = FraudDetector()
    detector.fit(conn)

    all_results = detector.score_all(conn)

    print("=" * 72)
    print("  TRANSACTION FRAUD DETECTION REPORT")
    print("=" * 72)
    print(f"{'TxnID':<8} {'CustID':<8} {'Amount':>10} {'Score':>8} {'Risk':<8} {'Suspicious'}")
    print("-" * 72)
    for r in all_results:
        flag = "*** FLAGGED ***" if r["is_suspicious"] else ""
        print(
            f"{r['txn_id']:<8} {r['customer_id']:<8} ₹{r['amount']:>9,.0f} "
            f"{r['anomaly_score']:>8.4f} {r['risk_level']:<8} {flag}"
        )

    flagged = [r for r in all_results if r["is_suspicious"]]
    print(f"\n  Flagged: {len(flagged)}/{len(all_results)} transactions")

    if flagged:
        print("\n  Detailed flags:")
        for txn_id in [r["txn_id"] for r in flagged]:
            detail = detector.score_transaction(conn, txn_id)
            print(f"\n  {txn_id} (Customer {detail['customer_id']}) — ₹{detail['amount']:,.0f}")
            print(f"    Risk level: {detail['risk_level']}  |  Score: {detail['anomaly_score']}")
            for f in detail["flags"]:
                print(f"    • {f}")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
