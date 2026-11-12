"""
Agentic orchestrator — the central brain of the multi-agent system.

Routing strategy
----------------
1. The LLM classifies the incoming question and returns an agent name.
2. The orchestrator dispatches to the correct specialist agent.
3. Results are returned with routing metadata for transparency.

Agents
------
sql_agent      : business analytics via natural language → SQL
ml_agent       : churn prediction and fraud detection
rag_agent      : policy and product knowledge (RAG)
insights_agent : LLM-generated business insights from live metrics
"""

from __future__ import annotations
import sqlite3
from typing import Dict, Any

from agents.sql_agent import BusinessSQLAgent
from agents.ml_agent import MLAgent
from agents.rag_agent import RAGAgent
from agents.insights_agent import InsightsAgent

ROUTE_PROMPT = """\
You are a routing agent for a telecom business intelligence system.
Classify the following question and return ONLY one of these agent names:

  sql_agent      — business analytics, counts, averages, rankings, data queries
  ml_agent       — churn risk, fraud detection, anomaly scoring, ML predictions
  rag_agent      — policy questions, plan details, terms, offers, procedures
  insights_agent — business insights, summaries, recommendations, briefings

Question: {question}

Agent name (one word only):"""


class Orchestrator:
    """
    Multi-agent orchestrator for telecom business intelligence.

    Parameters
    ----------
    llm  : LLM client with .complete(prompt) method
    conn : seeded telecom SQLite connection
    """

    def __init__(self, llm, conn: sqlite3.Connection):
        self.llm = llm
        self.conn = conn
        self._agents = {
            "sql_agent":      BusinessSQLAgent(llm, conn),
            "ml_agent":       MLAgent(llm, conn),
            "rag_agent":      RAGAgent(llm),
            "insights_agent": InsightsAgent(llm, conn),
        }

    def _route(self, question: str) -> str:
        prompt = ROUTE_PROMPT.format(question=question)
        raw = self.llm.complete(prompt, max_tokens=10).strip().lower()
        # Normalise — accept partial matches
        for name in self._agents:
            if name.split("_")[0] in raw:
                return name
        return "insights_agent"  # safe default

    def run(self, question: str) -> Dict[str, Any]:
        """
        Route a question to the best agent and return its response.

        Returns
        -------
        dict with keys: question, routed_to, result
        """
        agent_name = self._route(question)
        agent = self._agents[agent_name]

        if agent_name == "insights_agent":
            result = agent.generate()
        else:
            result = agent.query(question)

        return {
            "question": question,
            "routed_to": agent_name,
            "result": result,
        }

    def run_batch(self, questions: list) -> list:
        return [self.run(q) for q in questions]
