"""
Chapter 8: Advanced Agent Architectures, Multi-Agent Orchestration
Deep Dive: The Same Supervisor Team, Rebuilt on LangGraph's StateGraph

Extracted from: chapter_08_multi_agent.md
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


# ---- Step 1: Define the shared state every node will read and write. ----
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class TeamState(TypedDict):
    messages: Annotated[list, add_messages]
    next: str

# ---- Step 2: Define each specialist as a plain function that reads the state and returns an update to it. ----
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

def researcher_node(state: TeamState) -> dict:
    result = llm.invoke([("system", "Find relevant legal precedents."), *state["messages"]])
    return {"messages": [result]}

def analyst_node(state: TeamState) -> dict:
    result = llm.invoke([("system", "Review contract terms and risks."), *state["messages"]])
    return {"messages": [result]}

def strategist_node(state: TeamState) -> dict:
    result = llm.invoke([("system", "Recommend a legal strategy."), *state["messages"]])
    return {"messages": [result]}

# ---- Step 3: Define the supervisor as a routing function, returning the name of whichever node should run next. ----
def supervisor_node(state: TeamState) -> dict:
    routing_prompt = (
        "Given the conversation so far, which specialist should act next: "
        "researcher, analyst, strategist, or FINISH if the task is complete?"
    )
    decision = llm.invoke([("system", routing_prompt), *state["messages"]])
    return {"next": decision.content.strip()}

def route(state: TeamState) -> str:
    return state["next"] if state["next"] != "FINISH" else "__end__"

# ---- Step 4: Assemble the graph, wiring the supervisor's routing function to a conditional edge. ----
from langgraph.graph import StateGraph, START, END

graph = StateGraph(TeamState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("researcher", researcher_node)
graph.add_node("analyst", analyst_node)
graph.add_node("strategist", strategist_node)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", route, {
    "researcher": "researcher", "analyst": "analyst",
    "strategist": "strategist", "__end__": END,
})
for specialist in ["researcher", "analyst", "strategist"]:
    graph.add_edge(specialist, "supervisor")

team = graph.compile()

# ---- Step 5: Invoke the compiled graph exactly as you would call a single function. ----
result = team.invoke({"messages": [("user", "Review this vendor contract for termination risk.")], "next": ""})
print(result["messages"][-1].content)

# ---- Step 8: Build that swarm alternative directly, no `supervisor` node deciding each step, each specialist handing off straight to the next. ----
def researcher(state: dict) -> dict:
    state["findings"] = f"Background research on {state['topic']} complete."
    state["next"] = "analyst"
    return state

def analyst(state: dict) -> dict:
    state["analysis"] = f"Analysis of findings: {state['findings']} suggests moderate risk."
    state["next"] = "strategist"
    return state

def strategist(state: dict) -> dict:
    state["strategy"] = f"Recommended strategy based on: {state['analysis']}"
    state["next"] = None
    return state

AGENTS = {"researcher": researcher, "analyst": analyst, "strategist": strategist}

def run_swarm(topic: str, start: str = "researcher") -> dict:
    state = {"topic": topic, "next": start}
    steps_taken = []
    while state["next"] is not None:
        current = state["next"]
        steps_taken.append(current)
        state = AGENTS[current](state)
    state["steps_taken"] = steps_taken
    return state

result = run_swarm("vendor contract termination risk")
print("steps taken:", result["steps_taken"])
print("final strategy:", result["strategy"])

# ---- Step 6: Prove the "Think it through" question below is not hypothetical, by running the exact failure it describes against a standalone copy of `route`. ----
def route(state: dict) -> str:
    next_step = state.get("next", "")
    valid = {"researcher", "analyst", "strategist"}
    return next_step if next_step in valid else "__end__"

print(route({"next": "researcher"}))                                   # the happy path
print(route({"next": "Researcher"}))                                    # a capitalised LLM reply
print(route({"next": "the researcher should look into this"}))         # free text, not a literal

# ---- Step 7: Fix it with Chapter 9's structured-output discipline, an enum a model's output either satisfies or is rejected against, rather than a bare string a routing dictionary merely hopes matches. ----
from enum import Enum
from pydantic import BaseModel

class NextAgent(str, Enum):
    researcher = "researcher"
    analyst = "analyst"
    strategist = "strategist"
    end = "__end__"

class SupervisorDecision(BaseModel):
    next: NextAgent
    reason: str

good = SupervisorDecision(next="researcher", reason="Contract needs background research first.")
print(good.next.value)

try:
    SupervisorDecision(next="Researcher", reason="bad casing")
except Exception as error:
    print("rejected as expected:", type(error).__name__)
