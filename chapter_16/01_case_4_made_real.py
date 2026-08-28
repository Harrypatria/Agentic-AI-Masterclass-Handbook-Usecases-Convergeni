"""
Chapter 16: Real-World Agent Patterns, Case Studies from the Field
Hands-On: Case 4 Made Real, Routing Temperature by Task Shape

Extracted from: chapter_16_case_studies.md
Source book: Agentic AI: Building AI Agents and Retrieval Systems,
a Masterclass in LLM Agents, RAG, and Production Deployment.

Every block below was verified by direct execution before being
written into the handbook; run this file top to bottom, or copy
out the section you need. Where a step needs an API key
(OPENAI_API_KEY / ANTHROPIC_API_KEY), it is loaded from a local
.env file via python-dotenv, following Chapter 5's own security
discipline, never hardcoded.
"""


# ---- == Hands-On: Case 4 Made Real, Routing Temperature by Task Shape == ----
def recommended_temperature(task_description: str) -> dict:
    creative_signals = ["script", "casting", "tagline", "brainstorm", "creative", "story idea"]
    factual_signals = ["classify", "extract", "compliance", "calculate", "lookup", "verify"]
    lowered = task_description.lower()
    is_creative = any(s in lowered for s in creative_signals)
    is_factual = any(s in lowered for s in factual_signals)
    if is_factual and not is_creative:
        return {"temperature": 0.0, "reason": "factual task, one correct answer expected"}
    if is_creative and not is_factual:
        return {"temperature": 0.8, "reason": "creative task, variety is the point"}
    return {"temperature": 0.3, "reason": "mixed or unclear signal, default to a moderate value"}

tasks = [
    "Classify this contract clause as compliant or non-compliant",
    "Generate a script outline and casting suggestions for a new film concept",
    "Extract the vendor name and invoice amount from this PDF",
    "Brainstorm five taglines for a summer marketing campaign",
]
for task in tasks:
    print(recommended_temperature(task), "<-", task[:45])
