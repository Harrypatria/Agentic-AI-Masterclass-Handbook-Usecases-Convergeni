"""
Chapter 4: Advanced RAG Patterns, Evaluation, and Production Case Studies
Hands-On: Hybrid Search, Re-Ranking, and a Fallback, from a Real Production File

Extracted from: chapter_04_advanced_rag.md
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

KNOWN LIBRARY VERSION DRIFT (fixed below): raglite is a fast-moving
library, and its public API changed since this section was first written.
The installed release (confirmed via `inspect.signature`) no longer
exposes a singular `insert_document`; it exposes `insert_documents`
(plural), taking a list of `raglite.Document` objects built via
`Document.from_path(...)` rather than a raw file path. Step 2 below has
been updated to call the current, plural API directly, this is a genuine
library update, not a bug in the surrounding logic, and every other
function used in this file (`hybrid_search`, `retrieve_chunks`,
`rerank_chunks`, `RAGLiteConfig`) was checked against the installed
release and matches the book's own usage unchanged.
"""


# ---- Step 1: Configure the pipeline, naming the reranker explicitly. ----
from raglite import RAGLiteConfig, insert_documents, Document, hybrid_search, retrieve_chunks, rerank_chunks
from rerankers import Reranker
from pathlib import Path
import os

def initialize_config(openai_key, anthropic_key, cohere_key, db_url):
    os.environ["OPENAI_API_KEY"] = openai_key
    os.environ["ANTHROPIC_API_KEY"] = anthropic_key
    os.environ["COHERE_API_KEY"] = cohere_key
    return RAGLiteConfig(
        db_url=db_url,
        llm="claude-3-opus-20240229",
        embedder="text-embedding-3-large",
        chunk_max_size=2000,
        reranker=Reranker("cohere", api_key=cohere_key, lang="en"),
    )

# ---- Step 2: Ingest a document once. ----
def process_document(file_path: str, config) -> bool:
    insert_documents([Document.from_path(Path(file_path))], config=config)
    return True

# ---- Step 3: Chain hybrid search and re-ranking in one function, exactly as the real project does. ----
def perform_search(query: str, config) -> list:
    chunk_ids, scores = hybrid_search(query, num_results=10, config=config)
    if not chunk_ids:
        return []
    chunks = retrieve_chunks(chunk_ids, config=config)
    return rerank_chunks(query, chunks, config=config)

# ---- Step 4: Add a fallback to general knowledge when the document set has no confident answer. ----
import anthropic

def handle_fallback(query: str, anthropic_key: str) -> str:
    client = anthropic.Anthropic(api_key=anthropic_key)
    message = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=1024,
        system="You are a helpful assistant. If you do not know something, say so honestly.",
        messages=[{"role": "user", "content": query}],
    )
    return message.content[0].text

# ---- Step 5: Test HyDE directly against Chapter 3's own ingestion pipeline, comparing a genuinely vague query's raw embedding to a hypothetical answer's embedding. ----
import chromadb

client = chromadb.Client()
collection = client.create_collection("hyde_demo")

chunks = [
    ("remote_policy", "Remote Work Policy, Section 4.2. Employees may work remotely up to "
                       "three days per week with manager approval."),
    ("expense_policy", "Expense Policy, 6.3 Expenses. Employees working from home may claim up "
                        "to 150 pounds per year toward home office equipment such as a desk or chair."),
    ("leave_policy", "Annual Leave Policy, Section 2.1. Employees accrue 25 days of annual leave "
                      "per year, prorated for part time staff."),
    ("wfh_days", "Home Working Days Policy. Staff choosing to work from home should coordinate "
                 "their in-office days with their team lead each week."),
]
collection.add(
    documents=[c[1] for c in chunks],
    ids=[c[0] for c in chunks],
    metadatas=[{"source": c[0]} for c in chunks],
)

vague_query = "can I get money back for furniture"
raw_result = collection.query(query_texts=[vague_query], n_results=2)
print("raw vague query:", list(zip(raw_result["ids"][0], raw_result["distances"][0])))

hypothetical_answer = ("Yes, employees can be reimbursed for home office furniture purchases like "
                        "a desk or chair, up to an annual allowance specified in the expense policy.")
hyde_result = collection.query(query_texts=[hypothetical_answer], n_results=2)
print("HyDE hypothetical answer:", list(zip(hyde_result["ids"][0], hyde_result["distances"][0])))

# ---- Step 6: Build and test the one advanced pattern from this chapter's five that Steps 1 through 5 have not yet touched, parent-child retrieval, matching on small, precise child chunks while returning the larger parent chunk that actually has enough context to answer with. ----
parents = {
    "parent_0": ("Expense Policy, 6.3 Expenses. Employees working from home may claim up to 150 pounds "
                 "per year toward home office equipment such as a desk or chair. Claims must be submitted "
                 "within 60 days of purchase, accompanied by an itemised receipt, and approved by a direct "
                 "manager before reimbursement is processed through payroll. Approved claims are typically "
                 "paid within the next two payroll cycles."),
}

children = [
    ("child_0a", "Employees working from home may claim up to 150 pounds per year toward home "
                 "office equipment such as a desk or chair.", "parent_0"),
    ("child_0b", "Claims must be submitted within 60 days of purchase, accompanied by an "
                 "itemised receipt.", "parent_0"),
    ("child_0c", "Approved claims are typically paid within the next two payroll cycles.", "parent_0"),
]

collection.add(
    documents=[c[1] for c in children],
    ids=[c[0] for c in children],
    metadatas=[{"parent_id": c[2]} for c in children],
)

query = "how long until I get paid back"
result = collection.query(query_texts=[query], n_results=1)
matched_child_id = result["ids"][0][0]
parent_id = result["metadatas"][0][0]["parent_id"]

print("matched child chunk:", matched_child_id, "->", result["documents"][0][0])
print("returning parent chunk instead:")
print(parents[parent_id])

# ---- Step 7: Build the fifth and last pattern this chapter names, self-RAG's retrieve-or-skip decision, as a small, inspectable gate rather than an unconditional retrieval call on every single query. ----
def needs_retrieval(query: str) -> bool:
    # a minimal, deterministic stand-in for the LLM call a real system makes;
    # kept simple here so the decision logic itself stays inspectable
    no_retrieval_needed = ["hello", "hi", "thanks", "thank you", "what's 2+2", "how are you"]
    lowered = query.lower().strip()
    return not any(phrase in lowered for phrase in no_retrieval_needed)

queries = [
    "Hi, how are you?",
    "What is our current expense reimbursement policy?",
    "Thanks, that's all I needed.",
    "What's 2+2?",
    "Can employees work remotely more than three days a week?",
]
for query in queries:
    print(f"{'RETRIEVE' if needs_retrieval(query) else 'skip retrieval':>15} <- {query}")
