"""
Seed a demo telecom SQLite database with realistic synthetic data.

Tables
------
customers     — subscriber profiles, plan details, churn risk score
transactions  — payment and top-up events, fraud flag
calls         — CDR-style call records
support_tickets — customer service interactions
"""

import sqlite3
import random
from typing import Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id     TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    age             INTEGER,
    city            TEXT,
    plan_type       TEXT NOT NULL,       -- prepaid / postpaid / enterprise
    monthly_charge  REAL NOT NULL,
    data_limit_gb   REAL NOT NULL,
    data_used_gb    REAL NOT NULL,
    tenure_months   INTEGER NOT NULL,
    support_calls   INTEGER DEFAULT 0,
    churn_risk_score REAL DEFAULT 0.0,  -- 0-1, higher = more likely to churn
    is_churned      INTEGER DEFAULT 0,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transactions (
    txn_id          TEXT PRIMARY KEY,
    customer_id     TEXT REFERENCES customers(customer_id),
    amount          REAL NOT NULL,
    transaction_type TEXT NOT NULL,     -- recharge / bill_pay / roaming / addon
    channel         TEXT NOT NULL,      -- app / ivr / store / online
    device_id       TEXT,
    hour_of_day     INTEGER,
    flagged         INTEGER DEFAULT 0,  -- 1 = suspicious
    fraud_score     REAL DEFAULT 0.0,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS calls (
    call_id         TEXT PRIMARY KEY,
    customer_id     TEXT REFERENCES customers(customer_id),
    duration_sec    INTEGER,
    call_type       TEXT,               -- local / isd / roaming
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS support_tickets (
    ticket_id       TEXT PRIMARY KEY,
    customer_id     TEXT REFERENCES customers(customer_id),
    category        TEXT NOT NULL,      -- billing / network / device / churn_risk
    status          TEXT NOT NULL,      -- open / resolved / escalated
    created_at      TEXT NOT NULL
);
"""

CUSTOMERS = [
    ("C001", "Amit Sharma",    28, "Bangalore", "postpaid",   799.0, 40.0, 38.2, 24, 1, 0.12, 0, "2024-11-01"),
    ("C002", "Priya Nair",     34, "Mumbai",    "postpaid",   999.0, 75.0, 12.1, 3,  4, 0.78, 0, "2026-08-15"),
    ("C003", "Rahul Verma",    22, "Delhi",     "prepaid",    199.0, 2.0,  1.9,  2,  3, 0.81, 0, "2026-09-01"),
    ("C004", "Sunita Patel",   45, "Ahmedabad", "enterprise", 4999.0,500.0,320.0,60, 0, 0.05, 0, "2021-01-10"),
    ("C005", "Karan Mehta",    29, "Pune",      "postpaid",   599.0, 20.0, 19.8, 18, 2, 0.21, 0, "2025-05-20"),
    ("C006", "Deepa Reddy",    38, "Hyderabad", "prepaid",    99.0,  1.0,  0.1,  1,  5, 0.91, 0, "2026-10-05"),
    ("C007", "Vikram Singh",   52, "Chennai",   "postpaid",   1299.0,100.0,88.0, 48, 0, 0.08, 0, "2022-07-14"),
    ("C008", "Anjali Kumar",   26, "Kolkata",   "prepaid",    299.0, 3.0,  0.4,  4,  6, 0.87, 0, "2026-07-22"),
    ("C009", "Suresh Iyer",    41, "Bangalore", "enterprise", 9999.0,1000.0,750.0,36,1, 0.10, 0, "2023-03-01"),
    ("C010", "Meena Joshi",    31, "Jaipur",    "postpaid",   699.0, 30.0, 5.2,  6,  3, 0.62, 0, "2026-05-10"),
    ("C011", "Arjun Das",      24, "Bangalore", "prepaid",    149.0, 1.5,  1.4,  1,  2, 0.74, 0, "2026-10-20"),
    ("C012", "Ritu Agarwal",   35, "Delhi",     "postpaid",   899.0, 50.0, 48.9, 30, 0, 0.11, 0, "2024-01-05"),
    ("C013", "Mohit Saxena",   27, "Noida",     "postpaid",   499.0, 15.0, 14.8, 8,  4, 0.55, 0, "2026-03-18"),
    ("C014", "Kavitha Rao",    49, "Chennai",   "enterprise", 7499.0,750.0,600.0,72, 0, 0.04, 0, "2020-06-01"),
    ("C015", "Nitin Khanna",   33, "Gurgaon",   "postpaid",   799.0, 40.0, 7.3,  5,  5, 0.72, 0, "2026-06-12"),
]

TRANSACTIONS = [
    ("T001", "C001", 799.0,  "bill_pay",  "app",    "D001", 10, 0, 0.02, "2026-11-01"),
    ("T002", "C002", 999.0,  "bill_pay",  "online", "D002", 9,  0, 0.05, "2026-11-01"),
    ("T003", "C003", 199.0,  "recharge",  "store",  "D003", 14, 0, 0.03, "2026-11-02"),
    ("T004", "C006", 8500.0, "recharge",  "app",    "D099", 2,  1, 0.94, "2026-11-03"),  # fraud
    ("T005", "C008", 12000.0,"roaming",   "ivr",    "D088", 3,  1, 0.91, "2026-11-03"),  # fraud
    ("T006", "C005", 599.0,  "bill_pay",  "app",    "D005", 11, 0, 0.01, "2026-11-01"),
    ("T007", "C007", 1299.0, "bill_pay",  "online", "D007", 10, 0, 0.02, "2026-11-01"),
    ("T008", "C010", 699.0,  "bill_pay",  "app",    "D010", 15, 0, 0.04, "2026-11-01"),
    ("T009", "C011", 149.0,  "recharge",  "store",  "D011", 8,  0, 0.06, "2026-11-04"),
    ("T010", "C013", 15000.0,"addon",     "app",    "D077", 2,  1, 0.89, "2026-11-05"),  # fraud
    ("T011", "C012", 899.0,  "bill_pay",  "online", "D012", 12, 0, 0.01, "2026-11-01"),
    ("T012", "C015", 799.0,  "bill_pay",  "app",    "D015", 9,  0, 0.03, "2026-11-01"),
    ("T013", "C004", 4999.0, "bill_pay",  "online", "D004", 10, 0, 0.01, "2026-11-01"),
    ("T014", "C009", 9999.0, "bill_pay",  "online", "D009", 11, 0, 0.02, "2026-11-01"),
    ("T015", "C014", 7499.0, "bill_pay",  "online", "D014", 10, 0, 0.01, "2026-11-01"),
]

CALLS = [
    ("CL001", "C001", 342,  "local",   "2026-11-01"),
    ("CL002", "C002", 120,  "local",   "2026-11-02"),
    ("CL003", "C003", 45,   "local",   "2026-11-02"),
    ("CL004", "C005", 780,  "isd",     "2026-11-03"),
    ("CL005", "C007", 1200, "roaming", "2026-11-01"),
    ("CL006", "C010", 90,   "local",   "2026-11-04"),
    ("CL007", "C012", 450,  "local",   "2026-11-02"),
    ("CL008", "C013", 60,   "local",   "2026-11-05"),
]

TICKETS = [
    ("TK001", "C002", "churn_risk", "open",     "2026-11-01"),
    ("TK002", "C002", "billing",    "escalated","2026-10-20"),
    ("TK003", "C003", "network",    "open",     "2026-11-02"),
    ("TK004", "C006", "churn_risk", "open",     "2026-10-30"),
    ("TK005", "C008", "billing",    "open",     "2026-11-01"),
    ("TK006", "C010", "network",    "resolved", "2026-10-15"),
    ("TK007", "C013", "billing",    "open",     "2026-11-03"),
    ("TK008", "C015", "churn_risk", "escalated","2026-11-02"),
    ("TK009", "C001", "device",     "resolved", "2026-10-28"),
    ("TK010", "C005", "network",    "resolved", "2026-11-01"),
]


def seed(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.executemany(
        "INSERT OR IGNORE INTO customers VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        CUSTOMERS,
    )
    conn.executemany(
        "INSERT OR IGNORE INTO transactions VALUES (?,?,?,?,?,?,?,?,?,?)",
        TRANSACTIONS,
    )
    conn.executemany(
        "INSERT OR IGNORE INTO calls VALUES (?,?,?,?,?)",
        CALLS,
    )
    conn.executemany(
        "INSERT OR IGNORE INTO support_tickets VALUES (?,?,?,?,?)",
        TICKETS,
    )
    conn.commit()


def get_connection(db_path: str = ":memory:") -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    seed(conn)
    return conn
