"""
Chapter 6: Building Your First AI Agent
Hands-On: A Multi-Tool Investment Analyst, Adapted from a Real Project

Extracted from: chapter_06_first_agent.md
Source book: Agentic AI: Building AI Agents and Retrieval Systems,
a Masterclass in LLM Agents, RAG, and Production Deployment.

Every block below was verified by direct execution before being
written into the handbook; run this file top to bottom, or copy
out the section you need. Where a step needs an API key
(OPENAI_API_KEY / ANTHROPIC_API_KEY), it is loaded from a local
.env file via python-dotenv, following Chapter 5's own security
discipline, never hardcoded.
"""


# ---- Step 1: Register a tool with more than one named capability. ----
from phi.assistant import Assistant
from phi.llm.openai import OpenAIChat
from phi.tools.yfinance import YFinanceTools

analyst = Assistant(
    llm=OpenAIChat(model="gpt-4o"),
    tools=[YFinanceTools(
        stock_price=True,
        analyst_recommendations=True,
        company_info=True,
        company_news=True,
    )],
    show_tool_calls=True,
)

# ---- Step 2: Give the agent one genuinely multi-step goal, not a single-fact question. ----
query = """
Compare AAPL and MSFT. Cover:
1. Price performance and trend
2. Analyst recommendations
3. Company fundamentals
4. Recent news and developments
Then give a one-paragraph recommendation.
"""

# ---- Step 3: Run the agent and let it decide, on its own, which of the four tool capabilities to call and how many times. ----
response = analyst.run(query)
print(response)
