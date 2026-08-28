"""
Chapter 9: Reasoning Strategies, Human-in-the-Loop, and Observability
Hands-On: Adding a Human Checkpoint to a Real Sequential Pipeline

Extracted from: chapter_09_reasoning_hitl_observability.md
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


# ---- Step 1: Reproduce the real project's chained-call pattern, condensed to two stages. ----
from openai import OpenAI

client = OpenAI()

def call_openai(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content

def context_analysis(company: str, objective: str) -> str:
    return call_openai(f"Analyse the context for a meeting with {company} whose objective is: {objective}")

def strategy_development(company: str, context: str) -> str:
    return call_openai(f"Based on this context about {company}:\n{context}\nDraft a meeting strategy and agenda.")

# ---- Step 2: Add a risk check that flags a specific condition, following this chapter's four named high-stakes categories. ----
def needs_human_review(strategy_text: str) -> bool:
    """A minimal stand-in for a real risk classifier.
    Chapter 13 covers the production version of this check."""
    risk_terms = ["contract", "legal action", "terminate", "lawsuit"]
    return any(term in strategy_text.lower() for term in risk_terms)

# ---- Step 3: Wire the checkpoint into the pipeline as a genuine pause, not a log line. ----
def run_pipeline(company: str, objective: str):
    context = context_analysis(company, objective)
    strategy = strategy_development(company, context)

    if needs_human_review(strategy):
        print("--- PAUSED FOR HUMAN REVIEW ---")
        print(strategy)
        decision = input("Approve this strategy? (yes/no): ")
        if decision.strip().lower() != "yes":
            return "Strategy rejected by reviewer; pipeline halted for re-planning."

    return f"Approved strategy for {company}:\n{strategy}"

print(run_pipeline("Acme Corp", "renegotiate the vendor contract terms"))

# ---- Step 4: Answer this section's own first "Think it through" question with a real, constructed false negative, not a hypothetical one. ----
risky_but_missed = (
    "Recommend Acme stop all payments immediately and involve outside counsel "
    "to pursue damages given the vendor breach."
)
print("flagged:", needs_human_review(risky_but_missed))

safe_and_correctly_skipped = "Recommend scheduling a quarterly check-in call to review deliverables."
print("flagged:", needs_human_review(safe_and_correctly_skipped))

# ---- Step 5: Replace the keyword list with the structured-output pattern Chapter 14 used for the health copilot's own risk classification, closing the gap Step 4 just demonstrated. ----
from pydantic import BaseModel

class RiskAssessment(BaseModel):
    requires_human_review: bool
    reasoning: str

def needs_human_review_llm(strategy_text: str) -> RiskAssessment:
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a risk reviewer. Judge whether this strategy text "
                                           "involves legal, financial, or relationship risk serious enough "
                                           "to need human sign-off before acting on it, regardless of the "
                                           "specific words used."},
            {"role": "user", "content": strategy_text},
        ],
        response_format=RiskAssessment,
    )
    return response.choices[0].message.parsed
