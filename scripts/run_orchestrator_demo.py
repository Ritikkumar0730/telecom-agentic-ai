"""
Demo: multi-agent orchestrator answering diverse business questions.

Run with:
    python scripts/run_orchestrator_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm import get_llm
from data.seed import get_connection
from agents.orchestrator import Orchestrator

QUESTIONS = [
    "What is the churn risk for customer C002?",
    "Show me all high-risk customers likely to cancel.",
    "Is transaction T004 fraudulent?",
    "Show me all suspicious transactions.",
    "What is the average ARPU by plan type?",
    "List the top 10 customers by churn risk score.",
    "What is Airtel's retention policy for at-risk customers?",
    "How should I handle a flagged transaction?",
    "Give me a business insights summary for today.",
]


def main():
    llm = get_llm()
    conn = get_connection()
    orchestrator = Orchestrator(llm, conn)

    print("=" * 70)
    print("  TELECOM AGENTIC AI — ORCHESTRATOR DEMO")
    print(f"  LLM: {llm.model}")
    print("=" * 70)

    for question in QUESTIONS:
        print(f"\nQ: {question}")
        response = orchestrator.run(question)
        print(f"   → Routed to: {response['routed_to']}")
        result = response["result"]

        if response["routed_to"] == "sql_agent":
            print(f"   SQL: {result.get('sql', '')[:80]}...")
            rows = result.get("rows", [])
            if rows:
                for row in rows[:3]:
                    print(f"   {row}")
            print(f"   ({result.get('row_count', 0)} rows returned)")

        elif response["routed_to"] == "ml_agent":
            rtype = result.get("type", "")
            if rtype == "churn_single":
                print(f"   Churn probability : {result.get('churn_probability')}")
                print(f"   Risk band         : {result.get('risk_band')}")
                for f in result.get("top_risk_factors", []):
                    print(f"   • {f}")
            elif rtype == "churn_all":
                print(f"   {result.get('summary')}")
                for c in result.get("high_risk_customers", [])[:3]:
                    print(f"   • {c['name']} ({c['customer_id']}) — {c['churn_probability']:.2%}")
            elif rtype == "fraud_single":
                print(f"   Suspicious        : {result.get('is_suspicious')}")
                print(f"   Risk level        : {result.get('risk_level')}")
                print(f"   Anomaly score     : {result.get('anomaly_score')}")
                for f in result.get("flags", []):
                    print(f"   • {f}")
            elif rtype == "fraud_all":
                print(f"   {result.get('summary')}")
                for t in result.get("suspicious_transactions", [])[:3]:
                    print(f"   • {t['txn_id']} — ₹{t['amount']:,.0f} — {t['risk_level']}")

        elif response["routed_to"] == "rag_agent":
            print(f"   Answer: {result.get('answer', '')[:200]}")
            sources = result.get("sources", [])
            print(f"   Sources: {[s['title'] for s in sources]}")

        elif response["routed_to"] == "insights_agent":
            insights = result.get("insights", "")
            print(insights[:400])

        print()

    print("=" * 70)
    print("  Demo complete. Set OPENAI_API_KEY for real LLM responses.")
    print("=" * 70)


if __name__ == "__main__":
    main()
