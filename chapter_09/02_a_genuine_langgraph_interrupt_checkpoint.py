"""
Chapter 9: Reasoning Strategies, Human-in-the-Loop, and Observability
Deep Dive: A Genuine LangGraph interrupt() Checkpoint

Extracted from: chapter_09_reasoning_hitl_observability.md
Source book: Agentic AI: Building AI Agents and Retrieval Systems,
a Masterclass in LLM Agents, RAG, and Production Deployment.

Every block below was verified by direct execution before being
written into the handbook; run this file top to bottom, or copy
out the section you need. Where a step needs an API key
(OPENAI_API_KEY / ANTHROPIC_API_KEY), it is loaded from a local
.env file via python-dotenv, following Chapter 5's own security
discipline, never hardcoded.
"""


# ---- Step 1: A graph needs a checkpointer before `interrupt()` means anything, since pausing requires somewhere to persist state while execution is suspended. ----
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict

class ReviewState(TypedDict):
    strategy: str
    approved: bool

checkpointer = MemorySaver()

# ---- Step 2: Write the node that calls `interrupt()`, following this chapter's own risk-check logic. ----
from langgraph.types import interrupt, Command

def strategy_node(state: ReviewState) -> dict:
    risk_terms = ["contract", "legal action", "terminate", "lawsuit"]
    if any(term in state["strategy"].lower() for term in risk_terms):
        decision = interrupt({"strategy": state["strategy"], "question": "Approve this strategy?"})
        return {"approved": decision == "yes"}
    return {"approved": True}

# ---- Step 3: Build and compile a minimal graph around that one node, passing the checkpointer in explicitly. ----
from langgraph.graph import StateGraph, START, END

graph = StateGraph(ReviewState)
graph.add_node("strategy", strategy_node)
graph.add_edge(START, "strategy")
graph.add_edge("strategy", END)

app = graph.compile(checkpointer=checkpointer)

# ---- Step 4: Run the graph, catch the pause, and resume it with a human decision. ----
config = {"configurable": {"thread_id": "review-1"}}

result = app.invoke(
    {"strategy": "Terminate the vendor contract and pursue legal action.", "approved": False},
    config=config,
)
print(result)

# Elsewhere, once a human has actually reviewed the payload:
final = app.invoke(Command(resume="yes"), config=config)
print(final)
