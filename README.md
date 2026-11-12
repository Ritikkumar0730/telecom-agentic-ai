# Telecom Agentic AI

A production-grade **multi-agent business intelligence system** built for telecom operators. Four specialist AI agents — SQL analytics, ML predictions, RAG policy retrieval, and business insights — are orchestrated by an LLM-based router that automatically routes any natural language question to the right agent.

Built to demonstrate the full AI/ML stack required for a **Data Scientist role at Airtel**: ML modelling, Generative AI, LLMs, RAG, agentic orchestration, churn prediction, fraud detection, and business analytics.

---

## Architecture

```
User Question
      │
      ▼
 Orchestrator  ← LLM routing (MockLLM or OpenAI)
      │
      ├─► SQL Agent      ─ NL → SQL with self-correction
      ├─► ML Agent       ─ churn risk · fraud scoring
      ├─► RAG Agent      ─ TF-IDF retrieval over policy KB
      └─► Insights Agent ─ LLM-generated executive briefings
```

| Agent | Capability | ML / AI technique |
|---|---|---|
| `sql_agent` | Natural language → SQL queries | LLM + self-correction loop |
| `ml_agent` | Churn & fraud predictions | Logistic Regression, IsolationForest |
| `rag_agent` | Policy & plan Q&A | TF-IDF retrieval + LLM generation |
| `insights_agent` | Business briefings | LLM over live metrics |

---

## Features

- **Agentic orchestration** — LLM decides which specialist handles each query
- **Customer churn prediction** — Logistic Regression pipeline with risk bands (LOW / MEDIUM / HIGH) and human-readable explanations
- **Transaction fraud detection** — Unsupervised IsolationForest; no labelled fraud data needed at training time
- **Text-to-SQL with self-correction** — Automatically retries on SQL errors, correcting the query up to 3 times
- **RAG knowledge base** — 6 policy documents; TF-IDF similarity retrieval with cited sources
- **Generative insights** — LLM summarises live DB metrics into executive-level briefings
- **Offline-first** — `MockLLM` runs deterministically without any API key; swap in `OpenAILLM` by setting `OPENAI_API_KEY`
- **29 unit tests** — 100% pass rate covering all agents, models, and routing

---

## Project Structure

```
telecom-agentic-ai/
├── llm.py                    # LLM abstraction (MockLLM + OpenAI backend)
├── requirements.txt
├── data/
│   └── seed.py               # Demo telecom SQLite DB (customers, transactions, calls, tickets)
├── models/
│   ├── churn.py              # ChurnPredictor (LogisticRegression pipeline)
│   └── fraud.py              # FraudDetector (IsolationForest pipeline)
├── agents/
│   ├── orchestrator.py       # Central router — classifies and dispatches questions
│   ├── sql_agent.py          # BusinessSQLAgent — NL → SQL with schema introspection
│   ├── ml_agent.py           # MLAgent — churn & fraud via natural language
│   ├── rag_agent.py          # RAGAgent — TF-IDF retrieval over policy documents
│   └── insights_agent.py     # InsightsAgent — generative business insights
├── scripts/
│   ├── run_orchestrator_demo.py
│   ├── run_churn_demo.py
│   ├── run_fraud_demo.py
│   └── run_insights_demo.py
└── tests/
    └── test_telecom_ai.py    # 29 unit tests (100% pass)
```

---

## Quick Start

```bash
git clone https://github.com/Ritikkumar0730/telecom-agentic-ai
cd telecom-agentic-ai
pip install -r requirements.txt
```

### Run the orchestrator demo

```bash
python scripts/run_orchestrator_demo.py
```

Sample output:
```
Q: What is the churn risk for customer C002?
→ Routed to: ml_agent
   Risk: HIGH (0.82) | Factors: declining data usage, 2 open tickets

Q: What does the policy say about plan terms and offers?
→ Routed to: rag_agent
   Source: KB003 — Retention Policy Guidelines

Q: Show me the total count of customers in the database.
→ Routed to: sql_agent
   SQL: SELECT COUNT(*) as total FROM customers;
   Result: 15 customers
```

### Run individual demos

```bash
python scripts/run_churn_demo.py      # churn risk for all 15 customers
python scripts/run_fraud_demo.py      # fraud scoring for all 15 transactions
python scripts/run_insights_demo.py   # daily business briefing
```

### Run tests

```bash
pytest tests/ -v
```

---

## Using a Real LLM

Set your OpenAI API key and the system automatically upgrades from MockLLM to GPT:

```bash
export OPENAI_API_KEY=sk-...
python scripts/run_orchestrator_demo.py
```

---

## ML Models

### Churn Predictor (`models/churn.py`)

| Feature | Description |
|---|---|
| `tenure_months` | How long the customer has been active |
| `monthly_charge` | Current plan charge (₹) |
| `data_utilisation` | `data_used_gb / data_limit_gb` |
| `support_calls` | Support tickets raised |
| `plan_type_encoded` | prepaid=0, postpaid=1, enterprise=2 |

Pipeline: `StandardScaler → LogisticRegression`

Risk bands: LOW (<0.4) · MEDIUM (0.4–0.7) · HIGH (>0.7)

### Fraud Detector (`models/fraud.py`)

| Feature | Description |
|---|---|
| `amount` | Transaction amount (₹) |
| `hour_of_day` | 0–23 (late-night is riskier) |
| `channel_encoded` | app=0, online=1, ivr=2, store=3 |
| `is_new_device` | 1 if device not seen before for this customer |

Pipeline: `StandardScaler → IsolationForest(contamination=0.15)`  
No labelled fraud data needed — anomaly-based, cold-start ready.

---

## Demo Database

Seeded SQLite in-memory database with realistic telecom data:

| Table | Rows | Description |
|---|---|---|
| `customers` | 15 | plan type, monthly charge, data usage, tenure, support calls, churn risk score |
| `transactions` | 15 | amount, channel, hour, device, 3 pre-flagged fraud rows |
| `calls` | 8 | call records with duration and type |
| `support_tickets` | 10 | issue type, resolution status, priority |

---

## Skills Demonstrated

`Python` · `scikit-learn` · `SQL / SQLite` · `LLMs` · `RAG` · `Agentic AI` · `Multi-agent orchestration` · `Churn prediction` · `Fraud detection` · `NL-to-SQL` · `TF-IDF retrieval` · `Generative AI` · `pytest`
