"""
Business analytics SQL agent.

Converts natural language business questions into SQL queries against the
telecom SQLite database, executes them, and returns structured results.
Self-correction loop retries on SQL errors (up to max_retries times).
"""

from __future__ import annotations
import sqlite3
import re
from typing import Dict, List, Optional, Any, Tuple

SCHEMA_PROMPT = """\
You are an expert SQL analyst for a telecom company. Given the database schema below,
write a valid SQLite query that answers the business question.

Schema:
{schema}

Question: {question}

Rules:
- Write ONLY the SQL query, nothing else.
- Use only table and column names from the schema.
- End with a semicolon.

SQL:"""

CORRECTION_PROMPT = """\
This SQL query failed. Fix it.

Question: {question}
Schema: {schema}
Broken SQL: {sql}
Error: {error}

Corrected SQL:"""


def _extract_sql(text: str) -> str:
    text = re.sub(r"```(?:sql)?\s*", "", text).strip().rstrip("`").strip()
    match = re.search(
        r"(SELECT|INSERT|UPDATE|DELETE|WITH).*?;", text, re.DOTALL | re.IGNORECASE
    )
    return match.group(0).strip() if match else text.strip()


class BusinessSQLAgent:
    """
    Answer business analytics questions using natural language → SQL → results.

    Parameters
    ----------
    llm         : LLM client with .complete(prompt) method
    conn        : persistent SQLite connection (seeded telecom DB)
    max_retries : self-correction iterations on SQL error
    """

    def __init__(self, llm, conn: sqlite3.Connection, max_retries: int = 3):
        self.llm = llm
        self.conn = conn
        self.max_retries = max_retries
        self._schema_cache: Optional[str] = None

    def _get_schema(self) -> str:
        if self._schema_cache:
            return self._schema_cache
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [r[0] for r in cursor.fetchall()]
        parts = []
        for table in tables:
            info = self.conn.execute(f"PRAGMA table_info({table})").fetchall()
            cols = ", ".join(f"{r[1]} {r[2]}" for r in info)
            try:
                rows = self.conn.execute(f"SELECT * FROM {table} LIMIT 2").fetchall()
                sample = "; ".join(str(dict(r)) for r in rows)
                parts.append(f"Table: {table} ({cols})\nSample: {sample}")
            except Exception:
                parts.append(f"Table: {table} ({cols})")
        self._schema_cache = "\n\n".join(parts)
        return self._schema_cache

    def _execute(self, sql: str) -> Tuple[List[Dict], Optional[str]]:
        try:
            cur = self.conn.execute(sql)
            cols = [d[0] for d in cur.description] if cur.description else []
            return [dict(zip(cols, row)) for row in cur.fetchall()], None
        except Exception as e:
            return [], str(e)

    def query(self, question: str) -> Dict[str, Any]:
        schema = self._get_schema()
        prompt = SCHEMA_PROMPT.format(schema=schema, question=question)
        sql = _extract_sql(self.llm.complete(prompt, max_tokens=256))
        rows, error = self._execute(sql)
        attempts = 1

        while error and attempts <= self.max_retries:
            fix_prompt = CORRECTION_PROMPT.format(
                question=question, schema=schema, sql=sql, error=error
            )
            sql = _extract_sql(self.llm.complete(fix_prompt, max_tokens=256))
            rows, error = self._execute(sql)
            attempts += 1

        return {
            "question": question,
            "sql": sql,
            "rows": rows,
            "row_count": len(rows),
            "error": error,
            "attempts": attempts,
        }
