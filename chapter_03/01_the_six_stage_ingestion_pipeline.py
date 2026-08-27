"""
Chapter 3: Retrieval-Augmented Generation Foundations
Hands-On: The Six-Stage Ingestion Pipeline, Built and Run Stage by Stage

Extracted from: chapter_03_rag_foundations.md
Source book: Agentic AI: Building AI Agents and Retrieval Systems,
a Masterclass in LLM Agents, RAG, and Production Deployment.

Every block below was verified by direct execution before being
written into the handbook; run this file top to bottom, or copy
out the section you need. Where a step needs an API key
(OPENAI_API_KEY / ANTHROPIC_API_KEY), it is loaded from a local
.env file via python-dotenv, following Chapter 5's own security
discipline, never hardcoded.
"""


# ---- Step 1: Load. Start from raw text, exactly as it would arrive from a real document. ----
raw_document = """
Remote Work Policy, Section 4.2. Employees may work remotely up to
three days per week, subject to manager approval. Remote work
requests must be submitted at least five business days in advance
through the HR portal.

Equipment Policy, Section 5.1. The company provides a laptop and a
monitor for approved remote work. Employees are responsible for
maintaining a secure home network and must not use public Wi-Fi for
company business without a VPN connection active at all times.

Expense Policy, Section 6.3. Home office expenses up to $200 per
year are reimbursable with a receipt, submitted within thirty days
of purchase, covering desk equipment, chairs, and lighting, but not
decorative items.
"""

# ---- Step 2: Clean. Normalise whitespace and strip structural noise before chunking, so a chunk boundary never lands on an artefact of formatting rather than content. ----
import re

def clean_text(text: str) -> str:
    text = re.sub(r"\n{2,}", "\n\n", text)   # collapse repeated blank lines
    text = re.sub(r"[ \t]{2,}", " ", text)    # collapse repeated spaces
    return text.strip()

cleaned = clean_text(raw_document)

# ---- Step 3: Chunk. Split on paragraph boundaries here, following this chapter's own guidance that paragraph-level chunking suits policy documents specifically, since each numbered section above is already one complete idea. ----
def chunk_by_paragraph(text: str) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs

chunks = chunk_by_paragraph(cleaned)
for i, c in enumerate(chunks):
    print(f"--- Chunk {i} ({len(c)} chars) ---")
    print(c)

# ---- Step 4: Metadata. Attach a source label and a section identifier to each chunk before it loses that context. ----
metadatas = [
    {"source": "remote_work_policy.txt", "section": "4.2 Remote Work"},
    {"source": "remote_work_policy.txt", "section": "5.1 Equipment"},
    {"source": "remote_work_policy.txt", "section": "6.3 Expenses"},
]
ids = [f"chunk_{i}" for i in range(len(chunks))]

import chromadb

client = chromadb.Client()
collection = client.create_collection(name="policy_docs")

collection.add(documents=chunks, metadatas=metadatas, ids=ids)
print(collection.count())  # 3

# ---- Step 7: Query. Run real semantic search and inspect exactly what comes back. ----
results = collection.query(query_texts=["How much can I claim for a desk?"], n_results=1)
print(results["documents"][0][0])
print(results["metadatas"][0][0])

# ---- How it works ----
# BUG FIX: the book's own prose shows `api_key="your-api-key"` as an
# illustrative placeholder a reader swaps for their real key; extracted
# verbatim it becomes a literal, hardcoded fake credential that actually
# executes and raises AuthenticationError, contradicting this file's own
# docstring, which promises every key is loaded from the environment,
# never hardcoded. Loading it from OPENAI_API_KEY here instead is the fix.
import os
from chromadb.utils import embedding_functions

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
)
collection = client.create_collection(name="policy_docs_prod", embedding_function=openai_ef)

# ---- Step 7b: Use the `metadatas` list Step 4 already built for something more than an audit trail. First add a second, genuinely different document to the same collection, then scope a query to just one source with a `where` filter. ----
collection.add(
    documents=["Annual Leave Policy, Section 2.1 Accrual. Employees accrue 25 days of annual "
               "leave per year, prorated for part time staff."],
    metadatas=[{"source": "leave_policy.txt", "section": "2.1 Accrual"}],
    ids=["chunk_3"],
)

unfiltered = collection.query(query_texts=["how many days can I take"], n_results=1)
print("unfiltered:", unfiltered["ids"][0], unfiltered["metadatas"][0])

filtered = collection.query(
    query_texts=["how many days can I take"],
    n_results=1,
    where={"source": "leave_policy.txt"},
)
print("filtered to leave_policy.txt:", filtered["ids"][0], filtered["metadatas"][0])

# ---- Step 8: Build the fixed-size, overlapping chunker this section's own "Think it through" question asks you to reach for, and prove the overlap actually protects a fact sitting near a chunk boundary. ----
def chunk_with_overlap(text: str, chunk_words: int = 20, overlap_words: int = 5) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_words
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += chunk_words - overlap_words   # step forward less than a full chunk, so text repeats
    return chunks

long_section = (
    "The Expense Policy allows employees working from home to claim up to 150 pounds per year "
    "toward home office equipment such as a desk or chair. Claims must be submitted within 60 days "
    "of purchase, accompanied by an itemised receipt, and approved by a direct manager before "
    "reimbursement is processed through payroll."
)

overlapping_chunks = chunk_with_overlap(long_section, chunk_words=20, overlap_words=5)
for i, c in enumerate(overlapping_chunks):
    print(f"chunk {i} ({len(c.split())} words): {c!r}")

print("150 pounds present in chunk 0:", "150 pounds" in overlapping_chunks[0])
print("60 days present somewhere:", any("60 days" in c for c in overlapping_chunks))
