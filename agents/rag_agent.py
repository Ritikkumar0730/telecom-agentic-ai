"""
RAG agent — retrieves telecom policy and product knowledge, then answers
questions using retrieved context + LLM generation.

Knowledge base: in-memory policy documents (replace with a real vector store
in production by swapping the retriever implementation).
"""

from __future__ import annotations
import math
from typing import Dict, List, Any

# ---------------------------------------------------------------------------
# Knowledge base: telecom policies and product information
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE = [
    {
        "id": "KB001",
        "title": "Churn Retention Policy",
        "content": (
            "Airtel's retention policy allows account managers to offer high-churn-risk "
            "customers a one-time discount of up to 20% on their next three billing cycles, "
            "subject to a 6-month lock-in. Eligibility: active subscription for at least "
            "3 months with no outstanding dues. Escalation to senior retention team is "
            "triggered when churn probability exceeds 0.80."
        ),
    },
    {
        "id": "KB002",
        "title": "Postpaid Plan Benefits",
        "content": (
            "Airtel postpaid plans range from ₹499 to ₹1,299 per month. All plans include "
            "unlimited local and STD calls, 100 SMS/day, and data allowances from 15 GB to "
            "100 GB. Plans above ₹799 include free OTT subscriptions (Amazon Prime + "
            "Disney+ Hotstar). International roaming packs available from ₹999/week."
        ),
    },
    {
        "id": "KB003",
        "title": "Fraud Investigation SOP",
        "content": (
            "Transactions flagged by the fraud model must be reviewed within 2 hours. "
            "For HIGH risk transactions: (1) place a temporary hold on the account, "
            "(2) send OTP verification to the registered mobile, (3) escalate to the "
            "fraud team if the customer cannot verify within 30 minutes. SIM-swap events "
            "within 24 hours of a large transaction automatically trigger enhanced review."
        ),
    },
    {
        "id": "KB004",
        "title": "Enterprise Plan Terms",
        "content": (
            "Enterprise plans (₹4,999–₹9,999/month) include dedicated account management, "
            "SLA-backed network uptime of 99.9%, priority customer support with 2-hour "
            "resolution SLA, and custom data pooling across up to 50 SIMs. Contracts are "
            "annual with 3-month notice for termination. Volume discounts available for "
            "50+ SIM deployments."
        ),
    },
    {
        "id": "KB005",
        "title": "Data Add-on Policy",
        "content": (
            "Customers who exhaust their monthly data allowance are auto-notified at 80% "
            "and 100% usage. Add-on packs: 1 GB for ₹19, 3 GB for ₹49, 10 GB for ₹149. "
            "Unused add-on data rolls over for one billing cycle. Customers consistently "
            "purchasing add-ons are flagged for upsell to the next plan tier."
        ),
    },
    {
        "id": "KB006",
        "title": "Marketing Segmentation Guidelines",
        "content": (
            "Customer segments for targeted campaigns: (1) High-value postpaid (ARPU > ₹799) "
            "— focus on loyalty rewards and OTT bundles; (2) Low-utilisation postpaid "
            "(data usage < 40%) — downgrade offer or usage coaching; (3) High-data prepaid "
            "(usage > 90%) — upsell to postpaid; (4) At-risk prepaid (tenure < 3 months, "
            "support_calls > 2) — proactive outreach within 48 hours."
        ),
    },
]


def _tfidf_score(query: str, doc: str) -> float:
    """Simple TF-IDF-inspired relevance score (no external library needed)."""
    query_terms = set(query.lower().split())
    doc_lower = doc.lower()
    doc_terms = doc_lower.split()
    if not doc_terms:
        return 0.0
    tf = sum(doc_terms.count(t) for t in query_terms) / len(doc_terms)
    # IDF approximation: penalise very common short words
    idf = sum(1.0 / (1 + math.log1p(doc_lower.count(t))) for t in query_terms)
    return tf * idf


ANSWER_PROMPT = """\
You are a helpful telecom business assistant. Answer the customer/agent question
using ONLY the information in the retrieved policy documents below.

Retrieved documents:
{context}

Question: {question}

Answer concisely and accurately:"""


class RAGAgent:
    """
    Answer policy and product questions using retrieval-augmented generation.

    Parameters
    ----------
    llm     : LLM client with .complete(prompt) method
    top_k   : number of documents to retrieve
    """

    def __init__(self, llm, top_k: int = 2):
        self.llm = llm
        self.top_k = top_k
        self.kb = KNOWLEDGE_BASE

    def _retrieve(self, query: str) -> List[Dict]:
        scored = [
            (doc, _tfidf_score(query, doc["title"] + " " + doc["content"]))
            for doc in self.kb
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[: self.top_k]]

    def query(self, question: str) -> Dict[str, Any]:
        docs = self._retrieve(question)
        context = "\n\n".join(
            f"[{d['id']}] {d['title']}:\n{d['content']}" for d in docs
        )
        prompt = ANSWER_PROMPT.format(context=context, question=question)
        answer = self.llm.complete(prompt, max_tokens=256)
        return {
            "question": question,
            "answer": answer,
            "sources": [{"id": d["id"], "title": d["title"]} for d in docs],
            "retrieved_count": len(docs),
        }
