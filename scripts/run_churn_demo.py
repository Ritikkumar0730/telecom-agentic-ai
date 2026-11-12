"""
Demo: churn prediction across all customers.

Run with:
    python scripts/run_churn_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.seed import get_connection
from models.churn import ChurnPredictor


def main():
    conn = get_connection()
    predictor = ChurnPredictor()
    predictor.fit(conn)

    all_results = predictor.predict_all(conn)

    print("=" * 65)
    print("  CUSTOMER CHURN RISK REPORT")
    print("=" * 65)
    print(f"{'ID':<8} {'Name':<18} {'Plan':<12} {'Tenure':>7} {'P(Churn)':>10} {'Risk':<8}")
    print("-" * 65)
    for r in all_results:
        print(
            f"{r['customer_id']:<8} {r['name']:<18} {r['plan_type']:<12} "
            f"{r['tenure_months']:>7} {r['churn_probability']:>10.2%} {r['risk_band']:<8}"
        )

    high = [r for r in all_results if r["risk_band"] == "HIGH"]
    print(f"\n  HIGH risk customers: {len(high)}/{len(all_results)}")

    if high:
        print("\n  Detailed analysis — top 3 at-risk customers:")
        for r in high[:3]:
            detail = predictor.predict_customer(conn, r["customer_id"])
            print(f"\n  {detail['name']} ({detail['customer_id']})")
            print(f"    Churn probability: {detail['churn_probability']:.2%}")
            for f in detail["top_risk_factors"]:
                print(f"    • {f}")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
