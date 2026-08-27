"""
Chapter 5: Programming Foundations for AI Engineering
Hands-On: A Natural-Language-to-SQL Tool, Adapted from a Real Project

Extracted from: chapter_05_programming_foundations.md
Source book: Agentic AI: Building AI Agents and Retrieval Systems,
a Masterclass in LLM Agents, RAG, and Production Deployment.

Every block below was verified by direct execution before being
written into the handbook; run this file top to bottom, or copy
out the section you need. Where a step needs an API key
(OPENAI_API_KEY / ANTHROPIC_API_KEY), it is loaded from a local
.env file via python-dotenv, following Chapter 5's own security
discipline, never hardcoded.

NOTE: this file calls a real LLM API and needs a valid API key
exported as an environment variable before it will run end to end.
"""


# ---- Step 1: Introspect the database schema at startup, not on every query. ----
import sqlite3

def get_schema_info(db_path: str) -> str:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    schema = "=== DATABASE SCHEMA ===\n\n"
    for (table_name,) in tables:
        schema += f"Table: {table_name}\n"
        cursor.execute(f"PRAGMA table_info({table_name});")
        for col in cursor.fetchall():
            schema += f"  - {col[1]} ({col[2]})\n"
    conn.close()
    return schema

# ---- Step 2: Build a system prompt from that schema, following this chapter's role-prompting pattern. ----
from openai import OpenAI

client = OpenAI()

def generate_sql(question: str, schema: str) -> str:
    system_prompt = f"""You are a SQL expert working with this database:
{schema}
Rules: return ONLY the SQL query, no explanation, no markdown formatting.
Always use LIMIT 20 for large result sets."""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate SQL to answer: {question}"},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()

# ---- Step 3: Execute the generated query and return real data, with error handling around the one call that can genuinely fail. ----
import pandas as pd

def execute_query(db_path: str, sql_query: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(sql_query, conn)
    finally:
        conn.close()

# ---- Step 4: Chain all three into one callable pipeline. ----
def ask_database(question: str, db_path: str) -> pd.DataFrame:
    schema = get_schema_info(db_path)
    sql = generate_sql(question, schema)
    print(f"Generated SQL: {sql}")
    return execute_query(db_path, sql)

result = ask_database("Which 5 customers spent the most money in total?", "chinook.db")
print(result)

# ---- Step 5: Run `get_schema_info` and `execute_query` against a small, real SQLite file, no API key needed for either, to confirm the deterministic half of this pipeline actually works before trusting the LLM half. ----
import os
import sqlite3

# BUG FIX: `customers.db` is a real file on disk, not an in-memory database,
# so re-running this script or notebook a second time without removing it
# first hit `OperationalError: table customers already exists`. Removing any
# stale copy before creating a fresh one makes this step safely re-runnable.
if os.path.exists("customers.db"):
    os.remove("customers.db")

conn = sqlite3.connect("customers.db")
conn.execute("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, country TEXT, total_spend REAL)")
conn.executemany(
    "INSERT INTO customers (name, country, total_spend) VALUES (?, ?, ?)",
    [("Aisha Khan", "UK", 4200.50), ("Marco Rossi", "Italy", 1875.00), ("Priya Nair", "UK", 6120.75)],
)
conn.commit()
conn.close()

print(get_schema_info("customers.db"))
print(execute_query("customers.db", "SELECT name, total_spend FROM customers ORDER BY total_spend DESC LIMIT 5"))

# ---- Step 6: Close the gap the "Think it through" question below raises, a production database with `UPDATE` and `DELETE` permissions enabled, using the exact read-only validator Chapter 16's SQL agent case study built and tested. ----
from pydantic import BaseModel, field_validator

class SQLQuery(BaseModel):
    query: str

    @field_validator("query")
    @classmethod
    def must_be_select_only(cls, v: str) -> str:
        stripped = v.strip().lower()
        if not stripped.startswith("select"):
            raise ValueError("Only SELECT statements are permitted through this tool.")
        forbidden = ["drop", "delete", "update", "insert", "alter", ";--", "attach"]
        if any(word in stripped for word in forbidden):
            raise ValueError("Query contains a forbidden keyword.")
        return v

def execute_query_safe(db_path: str, sql_query: str) -> pd.DataFrame:
    validated = SQLQuery(query=sql_query)         # gates the query before it ever reaches sqlite3
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(validated.query, conn)
    finally:
        conn.close()

print(execute_query_safe("customers.db", "SELECT name, total_spend FROM customers"))

try:
    execute_query_safe("customers.db", "DELETE FROM customers")
except Exception as error:
    print("blocked:", error)

# ---- Step 7: This section's own Step 1 comment already claimed schema introspection should happen "at startup, not on every query"; prove that claim rather than only asserting it, using `functools.lru_cache`. ----
from functools import lru_cache

call_count = {"n": 0}

@lru_cache(maxsize=32)
def get_schema_info_cached(db_path: str) -> str:
    call_count["n"] += 1
    return get_schema_info(db_path)   # Step 1's own function, unchanged, just wrapped

for _ in range(5):
    get_schema_info_cached("customers.db")

print("actual schema introspections performed:", call_count["n"])
print("cache info:", get_schema_info_cached.cache_info())
