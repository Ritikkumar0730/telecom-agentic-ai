"""
LLM abstraction with OpenAI backend and offline MockLLM fallback.

Set OPENAI_API_KEY to use a real model; otherwise MockLLM runs deterministically
so every demo script works without any API credentials.
"""

import os
import re
from typing import List, Dict, Optional


class MockLLM:
    """Deterministic offline LLM for demos and tests."""

    def __init__(self, model: str = "mock-gpt"):
        self.model = model
        self._call_count = 0

    def complete(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        self._call_count += 1
        p = prompt.lower()

        # Agent routing — extract just the question line to avoid matching template keywords
        if "route" in p or "which agent" in p or "classify" in p:
            q = p
            if "question:" in p:
                q = p.split("question:")[-1].split("agent name")[0].strip()
            if any(k in q for k in ("churn", "cancel", "leave", "retention")):
                return "ml_agent"
            if any(k in q for k in ("fraud", "suspicious", "anomaly", "unusual", "transaction")):
                return "ml_agent"
            if any(k in q for k in ("policy", "plan", "offer", "terms", "how does", "what does")):
                return "rag_agent"
            if any(k in q for k in ("how many", "count", "total", "average", "list", "show", "top", "database")):
                return "sql_agent"
            return "insights_agent"

        # SQL generation
        if ("sql" in p or "query" in p) and any(k in p for k in ("select", "count", "table", "show", "list", "how many", "average", "total", "top", "which")):
            if "churn" in p or "risk" in p:
                return "SELECT customer_id, name, churn_risk_score FROM customers ORDER BY churn_risk_score DESC LIMIT 10;"
            if "fraud" in p or "suspicious" in p:
                return "SELECT t.customer_id, t.amount, t.transaction_type, t.created_at FROM transactions t WHERE t.flagged = 1 LIMIT 10;"
            if "revenue" in p or "arpu" in p:
                return "SELECT plan_type, ROUND(AVG(monthly_charge), 2) as avg_arpu, COUNT(*) as customers FROM customers GROUP BY plan_type ORDER BY avg_arpu DESC;"
            if "count" in p:
                return "SELECT COUNT(*) as total FROM customers;"
            return "SELECT customer_id, name, plan_type, monthly_charge, tenure_months FROM customers LIMIT 10;"

        # Insight generation
        if "insight" in p or "summary" in p or "analyse" in p or "analyze" in p or "recommend" in p:
            return (
                "Key findings from the data:\n"
                "1. High-churn risk segment: customers on prepaid plans with tenure < 6 months "
                "and more than 3 support calls show 2.4x higher churn probability.\n"
                "2. Fraud pattern: transactions above ₹5,000 occurring within 2 hours of a "
                "SIM swap event have an 87% fraud hit rate.\n"
                "3. ARPU opportunity: postpaid customers using less than 40% of their data "
                "allowance are strong upsell candidates for bundled OTT plans.\n"
                "Recommended action: trigger a retention offer for the high-churn segment "
                "and flag recent SIM-swap accounts for manual fraud review."
            )

        # RAG / policy questions
        if "policy" in p or "plan" in p or "offer" in p or "terms" in p or "how does" in p:
            return (
                "Based on the retrieved policy documents: Airtel's retention policy offers "
                "customers at high churn risk a one-time discount of up to 20% on their next "
                "three billing cycles, subject to a 6-month lock-in. Eligibility requires "
                "at least 3 months of active subscription and no outstanding dues."
            )

        # Churn explanation
        if "churn" in p or "retention" in p:
            return (
                "The customer shows elevated churn risk due to: (1) declining data usage "
                "over the last 60 days, (2) two unresolved support tickets, and (3) recent "
                "competitor plan comparison activity. Recommend proactive outreach with a "
                "personalised retention offer within the next 7 days."
            )

        # Fraud explanation
        if "fraud" in p or "suspicious" in p or "anomaly" in p:
            return (
                "Transaction flagged as suspicious: amount is 4.2 standard deviations above "
                "this customer's 90-day baseline, occurring at an unusual hour (02:47 AM) "
                "from a new device. Recommend temporary account hold pending customer "
                "verification via registered mobile OTP."
            )

        # Default
        return (
            "Analysis complete. The data indicates actionable patterns across the customer "
            "base. Focus areas: churn prevention in the 0-6 month tenure cohort, fraud "
            "monitoring on high-value transactions, and ARPU growth through targeted "
            "upsell campaigns for underutilising postpaid customers."
        )

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 512, temperature: float = 0.0) -> str:
        combined = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        return self.complete(combined, max_tokens=max_tokens, temperature=temperature)


class OpenAILLM:
    """Thin wrapper around the OpenAI Chat Completions API."""

    def __init__(self, model: str = "gpt-3.5-turbo"):
        from openai import OpenAI
        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model

    def complete(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        return self.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 512, temperature: float = 0.0) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()


def get_llm(model: Optional[str] = None) -> "MockLLM | OpenAILLM":
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAILLM(model=model or "gpt-3.5-turbo")
    return MockLLM(model=model or "mock-gpt")
