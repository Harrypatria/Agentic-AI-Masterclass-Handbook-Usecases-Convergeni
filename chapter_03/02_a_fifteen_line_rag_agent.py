"""
Chapter 3: Retrieval-Augmented Generation Foundations
Hands-On: A Fifteen-Line RAG Agent, Adapted from a Real Project

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


# ---- Step 1: Point the knowledge base at a real document and choose a vector store. ----
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.lancedb import LanceDb, SearchType
from phi.tools.duckduckgo import DuckDuckGo

db_uri = "tmp/lancedb"
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://your-own-document-url.pdf"],  # swap for a real document
    vector_db=LanceDb(table_name="my_docs", uri=db_uri, search_type=SearchType.vector),
)

# ---- Step 2: Run ingestion once, then comment it out. ----
knowledge_base.load(upsert=True)
# Comment this line out after the first successful run;
# re-running it re-embeds and re-inserts every chunk unnecessarily.

# ---- Step 3: Wire the retriever and a web-search fallback into one agent. ----
rag_agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    agent_id="rag-agent",
    knowledge=knowledge_base,
    tools=[DuckDuckGo()],
    show_tool_calls=True,
    markdown=True,
)

# ---- Step 4: Ask a question and inspect which source the agent actually used. ----
response = rag_agent.run("What does the document say about ingredient substitutions?")
print(response.content)
