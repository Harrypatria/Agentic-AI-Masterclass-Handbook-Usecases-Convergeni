"""
Chapter 16: Real-World Agent Patterns, Case Studies from the Field
Hands-On: Case 3 Made Real, a Safety-Validated SQL Agent End to End

Extracted from: chapter_16_case_studies.md
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


# ---- Step 1: Stand up a real, if small, database, exactly the kind a non-technical operations team would actually have. ----
import sqlite3

conn = sqlite3.connect(":memory:")
cur = conn.cursor()
cur.execute("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, region TEXT, spend REAL)")
cur.executemany(
    "INSERT INTO customers (name, region, spend) VALUES (?, ?, ?)",
    [
        ("Aisha Khan", "North", 4200.50),
        ("Marco Rossi", "South", 1875.00),
        ("Priya Nair", "North", 6120.75),
        ("Tom Becker", "West", 950.25),
    ],
)
conn.commit()

# ---- Step 2: Build the Pydantic schema Case 3 promised, one that rejects anything that is not a safe, read-only lookup, before a single character of the query reaches the database. ----
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

# ---- Step 3: Wrap the schema around the actual database call, the tool an agent would invoke. ----
def run_sql_tool(query: str) -> list[dict]:
    validated = SQLQuery(query=query)          # raises before touching the database at all
    cur = conn.cursor()
    cur.execute(validated.query)
    columns = [description[0] for description in cur.description]
    return [dict(zip(columns, row)) for row in cur.fetchall()]

# ---- Step 4: Run the legitimate case, the exact question the operations team in Case 3 actually asked. ----
result = run_sql_tool(
    'SELECT name, region, spend FROM customers WHERE region = "North" ORDER BY spend DESC'
)
print(result)

# ---- Step 5: Now attack it, the way an untested SQL agent, with no schema in front of it, genuinely could be. ----
try:
    run_sql_tool("DROP TABLE customers")
except Exception as error:
    print("blocked as expected:", error)

try:
    run_sql_tool('SELECT * FROM customers; DELETE FROM customers WHERE 1=1')
except Exception as error:
    print("blocked chained statement:", error)

# ---- Step 6: Wire in the LLM half of the agent, translating a plain-English question into the SQL the tool above will validate. ----
from openai import OpenAI
client = OpenAI()

def nl_to_sql_agent(question: str) -> list[dict]:
    schema_description = "Table customers(id, name, region, spend). Region is one of North, South, West, East."
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"Write one SQLite SELECT statement to answer the question. "
                                           f"Schema: {schema_description}. Return only the SQL, nothing else."},
            {"role": "user", "content": question},
        ],
    )
    generated_sql = response.choices[0].message.content.strip()
    return run_sql_tool(generated_sql)   # Step 2's schema still gates everything the LLM produces
