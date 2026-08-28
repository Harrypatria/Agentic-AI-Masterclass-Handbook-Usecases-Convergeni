"""
Chapter 8: Advanced Agent Architectures, Multi-Agent Orchestration
Hands-On: A Three-Specialist Supervisor Team, Adapted from a Real Project

Extracted from: chapter_08_multi_agent.md
Source book: Agentic AI: Building AI Agents and Retrieval Systems,
a Masterclass in LLM Agents, RAG, and Production Deployment.

Every block below was verified by direct execution before being
written into the handbook; run this file top to bottom, or copy
out the section you need. Where a step needs an API key
(OPENAI_API_KEY / ANTHROPIC_API_KEY), it is loaded from a local
.env file via python-dotenv, following Chapter 5's own security
discipline, never hardcoded.
"""


# ---- Step 1: Define each specialist with a distinct role and distinct instructions. ----
from phi.agent import Agent
from phi.model.openai import OpenAIChat

researcher = Agent(
    name="Legal Researcher",
    role="Legal research specialist",
    model=OpenAIChat(id="gpt-4o"),
    instructions=[
        "Find and cite relevant legal cases and precedents",
        "Provide detailed research summaries with sources",
    ],
)

analyst = Agent(
    name="Contract Analyst",
    role="Contract analysis specialist",
    model=OpenAIChat(id="gpt-4o"),
    instructions=[
        "Review contracts thoroughly",
        "Identify key terms and potential issues",
    ],
)

strategist = Agent(
    name="Legal Strategist",
    role="Legal strategy specialist",
    model=OpenAIChat(id="gpt-4o"),
    instructions=[
        "Develop comprehensive legal strategies",
        "Consider both risks and opportunities",
    ],
)

# ---- Step 2: Define the supervisor, passing the three specialists in as its own `team`. ----
legal_team = Agent(
    name="Legal Team Lead",
    role="Legal team coordinator",
    model=OpenAIChat(id="gpt-4o"),
    team=[researcher, analyst, strategist],
    instructions=[
        "Coordinate analysis between team members",
        "Provide comprehensive responses",
        "Ensure all recommendations are properly sourced",
    ],
    show_tool_calls=True,
)

# ---- Step 3: Send one query to the supervisor and let it route the work. ----
response = legal_team.run(
    "Review this vendor contract for termination clauses and recommend a negotiation strategy."
)
print(response.content)
