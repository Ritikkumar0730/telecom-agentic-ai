"""
Unit tests for telecom-agentic-ai.

Run with:
    pytest tests/ -v
    pytest tests/ -v -m "not integration"   # offline only
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm import MockLLM
from data.seed import get_connection
from models.churn import ChurnPredictor
from models.fraud import FraudDetector
from agents.sql_agent import BusinessSQLAgent
from agents.ml_agent import MLAgent
from agents.rag_agent import RAGAgent
from agents.insights_agent import InsightsAgent
from agents.orchestrator import Orchestrator


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture(scope="module")
def conn():
    return get_connection()

@pytest.fixture(scope="module")
def llm():
    return MockLLM()

@pytest.fixture(scope="module")
def churn_model(conn):
    m = ChurnPredictor()
    m.fit(conn)
    return m

@pytest.fixture(scope="module")
def fraud_model(conn):
    m = FraudDetector()
    m.fit(conn)
    return m

@pytest.fixture(scope="module")
def orchestrator(llm, conn):
    return Orchestrator(llm, conn)


# ------------------------------------------------------------------ #
# Data seed tests
# ------------------------------------------------------------------ #

def test_seed_customers(conn):
    rows = conn.execute("SELECT COUNT(*) as n FROM customers").fetchone()
    assert rows["n"] == 15

def test_seed_transactions(conn):
    rows = conn.execute("SELECT COUNT(*) as n FROM transactions").fetchone()
    assert rows["n"] == 15

def test_seed_fraud_flags(conn):
    rows = conn.execute("SELECT COUNT(*) as n FROM transactions WHERE flagged = 1").fetchone()
    assert rows["n"] == 3

def test_seed_support_tickets(conn):
    rows = conn.execute("SELECT COUNT(*) as n FROM support_tickets").fetchone()
    assert rows["n"] == 10


# ------------------------------------------------------------------ #
# Churn model tests
# ------------------------------------------------------------------ #

def test_churn_model_fits(churn_model):
    assert churn_model._fitted

def test_churn_predict_customer(churn_model, conn):
    result = churn_model.predict_customer(conn, "C002")
    assert "churn_probability" in result
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["risk_band"] in ("LOW", "MEDIUM", "HIGH")
    assert isinstance(result["top_risk_factors"], list)

def test_churn_predict_all(churn_model, conn):
    results = churn_model.predict_all(conn)
    assert len(results) == 15
    for r in results:
        assert "churn_probability" in r
        assert r["risk_band"] in ("LOW", "MEDIUM", "HIGH")

def test_churn_high_risk_customers_exist(churn_model, conn):
    results = churn_model.predict_all(conn)
    high = [r for r in results if r["risk_band"] == "HIGH"]
    assert len(high) >= 1

def test_churn_unknown_customer(churn_model, conn):
    result = churn_model.predict_customer(conn, "C999")
    assert "error" in result

def test_churn_probability_range(churn_model, conn):
    for cid in ["C001", "C002", "C003", "C006", "C008"]:
        r = churn_model.predict_customer(conn, cid)
        assert 0.0 <= r["churn_probability"] <= 1.0


# ------------------------------------------------------------------ #
# Fraud model tests
# ------------------------------------------------------------------ #

def test_fraud_model_fits(fraud_model):
    assert fraud_model._fitted

def test_fraud_score_transaction(fraud_model, conn):
    result = fraud_model.score_transaction(conn, "T004")
    assert "is_suspicious" in result
    assert result["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    assert isinstance(result["flags"], list)

def test_fraud_flagged_transactions_detected(fraud_model, conn):
    all_results = fraud_model.score_all(conn)
    flagged = [r for r in all_results if r["is_suspicious"]]
    assert len(flagged) >= 1

def test_fraud_score_range(fraud_model, conn):
    result = fraud_model.score_transaction(conn, "T001")
    assert isinstance(result["anomaly_score"], float)

def test_fraud_unknown_transaction(fraud_model, conn):
    result = fraud_model.score_transaction(conn, "T999")
    assert "error" in result


# ------------------------------------------------------------------ #
# SQL agent tests
# ------------------------------------------------------------------ #

def test_sql_agent_basic_query(llm, conn):
    agent = BusinessSQLAgent(llm, conn)
    result = agent.query("How many customers are there?")
    assert "sql" in result
    assert "rows" in result
    assert result["error"] is None or result["attempts"] > 1

def test_sql_agent_returns_rows(llm, conn):
    agent = BusinessSQLAgent(llm, conn)
    result = agent.query("Show top customers by churn risk")
    assert isinstance(result["rows"], list)

def test_sql_agent_schema_cached(llm, conn):
    agent = BusinessSQLAgent(llm, conn)
    _ = agent._get_schema()
    assert agent._schema_cache is not None
    schema2 = agent._get_schema()
    assert schema2 == agent._schema_cache


# ------------------------------------------------------------------ #
# RAG agent tests
# ------------------------------------------------------------------ #

def test_rag_agent_returns_answer(llm):
    agent = RAGAgent(llm)
    result = agent.query("What is the churn retention policy?")
    assert "answer" in result
    assert len(result["answer"]) > 10

def test_rag_agent_retrieves_sources(llm):
    agent = RAGAgent(llm)
    result = agent.query("How do I handle a fraudulent transaction?")
    assert len(result["sources"]) >= 1
    assert all("id" in s and "title" in s for s in result["sources"])

def test_rag_agent_top_k(llm):
    agent = RAGAgent(llm, top_k=3)
    result = agent.query("Tell me about plans")
    assert result["retrieved_count"] == 3


# ------------------------------------------------------------------ #
# Insights agent tests
# ------------------------------------------------------------------ #

def test_insights_agent_metrics(llm, conn):
    agent = InsightsAgent(llm, conn)
    result = agent.generate()
    m = result["metrics"]
    assert m["total_customers"] == 15
    assert m["total_transactions"] == 15
    assert m["flagged_transactions"] == 3
    assert "insights" in result

def test_insights_daily_briefing(llm, conn):
    agent = InsightsAgent(llm, conn)
    briefing = agent.daily_briefing()
    assert "TELECOM" in briefing
    assert "INSIGHTS" in briefing


# ------------------------------------------------------------------ #
# Orchestrator tests
# ------------------------------------------------------------------ #

def test_orchestrator_routes_churn(orchestrator):
    result = orchestrator.run("What is the churn risk for customer C002?")
    assert result["routed_to"] == "ml_agent"
    assert result["result"]["type"] in ("churn_single", "churn_all")

def test_orchestrator_routes_fraud(orchestrator):
    result = orchestrator.run("Is transaction T004 suspicious?")
    assert result["routed_to"] == "ml_agent"

def test_orchestrator_routes_policy(orchestrator):
    result = orchestrator.run("What does the policy say about plan terms and offers?")
    assert result["routed_to"] == "rag_agent"
    assert "answer" in result["result"]

def test_orchestrator_routes_sql(orchestrator):
    result = orchestrator.run("Show me the total count of customers in the database.")
    assert result["routed_to"] == "sql_agent"

def test_orchestrator_has_question_in_result(orchestrator):
    q = "Show me suspicious transactions"
    result = orchestrator.run(q)
    assert result["question"] == q

def test_orchestrator_batch(orchestrator):
    questions = [
        "What is churn risk for C003?",
        "Show fraudulent transactions.",
    ]
    results = orchestrator.run_batch(questions)
    assert len(results) == 2
    assert all("routed_to" in r for r in results)
