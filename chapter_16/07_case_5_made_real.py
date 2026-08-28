"""
Chapter 16: Real-World Agent Patterns, Case Studies from the Field
Hands-On: Case 5 Made Real, a Triage Router That Automates the Majority and Escalates the Rest

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


# ---- Step 1: Define the escalation signals and the simple, automatable categories a real support inbox sees every day. ----
import re

ESCALATION_SIGNALS = [
    "lawsuit", "legal action", "cancel my account", "furious", "unacceptable",
    "refund immediately", "speak to a manager", "never again",
]

def sentiment_flag(text: str) -> bool:
    lowered = text.lower()
    return any(signal in lowered for signal in ESCALATION_SIGNALS)

def classify_support_ticket(ticket_text: str) -> dict:
    simple_patterns = {
        "order_status": re.compile(r"where is my order|order status|track my order", re.I),
        "return_policy": re.compile(r"return policy|how do i return|refund policy", re.I),
    }
    for category, pattern in simple_patterns.items():
        if pattern.search(ticket_text):
            return {"category": category, "route": "automated", "escalate": False}

    if sentiment_flag(ticket_text):
        return {"category": "complaint", "route": "human", "escalate": True}

    return {"category": "general", "route": "automated", "escalate": False}

# ---- Step 2: Run it against four realistic tickets, three that should resolve automatically and one that genuinely should not. ----
tickets = [
    "Where is my order? It was supposed to arrive yesterday.",
    "What is your return policy for electronics?",
    "This is unacceptable, I want a refund immediately or I will cancel my account.",
    "Can you tell me your opening hours on weekends?",
]
for ticket in tickets:
    print(classify_support_ticket(ticket), "<-", ticket[:50])
