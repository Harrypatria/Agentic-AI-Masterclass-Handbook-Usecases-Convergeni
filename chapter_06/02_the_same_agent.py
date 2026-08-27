"""
Chapter 6: Building Your First AI Agent
Deep Dive: The Same Agent, Rebuilt on LangChain's AgentExecutor

Extracted from: chapter_06_first_agent.md
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

KNOWN LIBRARY VERSION DRIFT (fixed below): current langchain releases
(1.x) removed `create_react_agent` from `langchain.agents` entirely and
replaced the whole `AgentExecutor` pattern this section originally used.
Confirmed directly: `from langchain.agents import create_react_agent`
raises `ImportError`; `langgraph.prebuilt.create_react_agent` still
exists but is itself deprecated (`LangGraphDeprecatedSinceV10`, to be
removed in v2) in favour of the current, stable replacement used below,
`langchain.agents.create_agent`, which takes a plain `system_prompt`
string instead of a `ChatPromptTemplate` and returns a compiled
LangGraph graph, not an `AgentExecutor`, called with
`.invoke({"messages": [...]})` and read back from `result["messages"][-1]`
rather than `result["output"]`. Steps 2 through 4 below have been
rewritten to this current API; the tool definition in Step 1 and the
observability logic in Steps 5 and 6 are untouched, since neither
depends on the agent-construction API that changed.
"""


# ---- Step 1: Define a tool with LangChain's `@tool` decorator, following Chapter 9's Pydantic-backed schema discipline. ----
from langchain.tools import tool
from pydantic import BaseModel

class StockPriceInput(BaseModel):
    ticker: str

@tool(args_schema=StockPriceInput)
def get_stock_price(ticker: str) -> str:
    """Look up the current price for a stock ticker."""
    import yfinance as yf
    price = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
    return f"{ticker}: ${price:.2f}"

# ---- Step 2: Write the system prompt the agent will reason inside. `create_agent` takes this as a plain string, not a `ChatPromptTemplate`; the "{agent_scratchpad}" placeholder the old API needed is handled internally by the graph now. ----
system_prompt = "You are a financial analyst. Use tools to answer accurately."

# ---- Step 3: Wire the model, the tools, and the prompt into an agent with `langchain.agents.create_agent`, the current stable replacement for the old `create_react_agent` + `AgentExecutor` pair. ----
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [get_stock_price]

agent = create_agent(llm, tools, system_prompt=system_prompt)

# ---- Step 4: Run it and read the final answer back from the last message in the returned state, `create_agent`'s graph returns the whole conversation, not a single `output` key. ----
result = agent.invoke({"messages": [("user", "What is the current price of AAPL?")]})
print(result["messages"][-1].content)

# ---- Step 5: Reproduce the silent max-iterations cutoff the "Think it through" question below asks about, in plain Python, so the failure is something you have actually seen happen rather than only imagined. ----
def run_bounded_loop(goal: str, max_iterations: int = 6) -> dict:
    iterations = 0
    scratchpad = []
    finished = False

    while iterations < max_iterations and not finished:
        iterations += 1
        scratchpad.append(f'step {iterations}: still gathering information for "{goal}"')
        finished = False   # a poorly scoped goal, standing in for one that never actually resolves

    return {
        "output": scratchpad[-1] if scratchpad else None,
        "iterations_used": iterations,
        "hit_cap": iterations >= max_iterations and not finished,
    }

print(run_bounded_loop("compare every S&P 500 stock pairwise", max_iterations=6))

# ---- Step 6: Add the observability signal Chapter 9 would call for, so a silent cutoff gets flagged rather than quietly returned as if it were complete. ----
def run_bounded_loop_with_flag(goal: str, max_iterations: int = 6) -> dict:
    result = run_bounded_loop(goal, max_iterations)
    if result["hit_cap"]:
        result["review_flag"] = "INCOMPLETE: max_iterations reached before agent reported AgentFinish"
    return result

print(run_bounded_loop_with_flag("compare every S&P 500 stock pairwise"))
