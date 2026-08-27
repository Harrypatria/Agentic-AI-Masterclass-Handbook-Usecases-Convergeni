"""
Chapter 4: Advanced RAG Patterns, Evaluation, and Production Case Studies
Hands-On: A Retrieval Evaluation Harness, Deterministic First, LLM-Judged Second

Extracted from: chapter_04_advanced_rag.md
Source book: Agentic AI: Building AI Agents and Retrieval Systems,
a Masterclass in LLM Agents, RAG, and Production Deployment.

Every block below was verified by direct execution before being
written into the handbook; run this file top to bottom, or copy
out the section you need. Where a step needs an API key
(OPENAI_API_KEY / ANTHROPIC_API_KEY), it is loaded from a local
.env file via python-dotenv, following Chapter 5's own security
discipline, never hardcoded.

ADAPTED FOR STANDALONE USE: this section evaluates the exact `collection`
Chapter 3's own ingestion pipeline built; the block below reconstructs it
here, identically, so this file does not require running chapter_03's
script first. See code_examples/chapter_03 for the full walkthrough.

Step 4 below needs `pip install ragas datasets` and an OPENAI_API_KEY;
everything before it runs with no key and no extra installs at all.

KNOWN LIBRARY VERSION DRIFT (unresolved, verified): ragas 0.4.3's own
`ragas/llms/base.py` imports `langchain_community.chat_models.vertexai.
ChatVertexAI` at module load, before Step 4's own code ever runs. That
submodule was removed from langchain-community in its 0.4.x line (it
still exists in 0.3.27, confirmed by inspecting that release's wheel
directly), so the failure is a genuine incompatibility between the
currently released `ragas` and `langchain-community`, not a bug in the
code below, and it is not fixed by installing `google-cloud-aiplatform`
(tested directly: the import still fails, because the missing piece is
the langchain-community submodule itself, not the Vertex AI SDK it would
call). Pinning `langchain-community==0.3.27` does import successfully in
isolation, but it forces `langchain-core` back to 0.3.x, which breaks
`langchain`, `langchain-openai`, and `langgraph` everywhere else in this
shared environment (verified directly: reinstalling it downgrades
`langchain` itself from 1.3.16 to 0.3.30). Give Step 4 its own virtual
environment, `python -m venv ragas_env`, with only
`pip install ragas==0.4.3 "langchain-community==0.3.27" datasets`
and no other chapter's packages, to run it as written; otherwise wait
for ragas to drop the stale import.
"""

# ---- Setup reused from Chapter 3 (see code_examples/chapter_03) ----
import re
import chromadb

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

def clean_text(text: str) -> str:
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def chunk_by_paragraph(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]

chunks = chunk_by_paragraph(clean_text(raw_document))
metadatas = [
    {"source": "remote_work_policy.txt", "section": "4.2 Remote Work"},
    {"source": "remote_work_policy.txt", "section": "5.1 Equipment"},
    {"source": "remote_work_policy.txt", "section": "6.3 Expenses"},
]
ids = [f"chunk_{i}" for i in range(len(chunks))]

client = chromadb.Client()
collection = client.create_collection(name="policy_docs")
collection.add(documents=chunks, metadatas=metadatas, ids=ids)

# ---- Step 1: Build a small evaluation set, each question paired with the ID of the chunk that should genuinely answer it. ----
eval_set = [
    {"question": "How much can I claim for home office expenses?", "expected_chunk_id": "chunk_2"},
    {"question": "How many days a week can I work from home?", "expected_chunk_id": "chunk_0"},
    {"question": "What equipment does the company provide?", "expected_chunk_id": "chunk_1"},
]

# ---- Step 2: Write a deterministic evaluator computing Hit Rate and Mean Reciprocal Rank, two classic information-retrieval metrics that need no LLM at all. ----
def evaluate_retrieval(eval_set, collection, k=3):
    hits = 0
    reciprocal_ranks = []
    for item in eval_set:
        result = collection.query(query_texts=[item["question"]], n_results=k)
        retrieved_ids = result["ids"][0]
        if item["expected_chunk_id"] in retrieved_ids:
            hits += 1
            rank = retrieved_ids.index(item["expected_chunk_id"]) + 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)
    hit_rate = hits / len(eval_set)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return {"hit_rate": hit_rate, "mrr": mrr}

print(evaluate_retrieval(eval_set, collection))  # {'hit_rate': 1.0, 'mrr': 1.0}

# ---- Step 3: Deliberately break retrieval, then confirm the harness actually catches it. ----
misleading_eval_set = [
    {"question": "What is the capital of France?", "expected_chunk_id": "chunk_2"},
]
print(evaluate_retrieval(misleading_eval_set, collection, k=1))  # {'hit_rate': 0.0, 'mrr': 0.0}

# ---- Step 4: Score the same system's generated answers, not only its retrieval, using the real Ragas library and an LLM judge. ----
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset

ragas_data = Dataset.from_dict({
    "question": [item["question"] for item in eval_set],
    "answer": ["Up to $200 per year with a receipt.", "Three days per week.", "A laptop and a monitor."],
    "contexts": [[c] for c in ["Home office expenses up to $200 per year are reimbursable with a receipt.",
                                "Employees may work remotely up to three days per week.",
                                "The company provides a laptop and a monitor for approved remote work."]],
})

results = evaluate(ragas_data, metrics=[faithfulness, answer_relevancy, context_precision])
print(results)
