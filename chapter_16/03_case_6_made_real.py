"""
Chapter 16: Real-World Agent Patterns, Case Studies from the Field
Hands-On: Case 6 Made Real, Answering the Property Question Directly

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


# ---- == Hands-On: Case 6 Made Real, Answering the Property Question Directly == ----
import pandas as pd

df = pd.DataFrame({
    "neighbourhood": ["Riverside", "Riverside", "Old Town", "Old Town", "Hillcrest", "Hillcrest"],
    "property_type": ["Flat", "House", "Flat", "House", "Flat", "House"],
    "price": [285000, 410000, 245000, 360000, 320000, 495000],
})

def average_price_by(df: pd.DataFrame, group_col: str) -> dict:
    return df.groupby(group_col)["price"].mean().round(0).to_dict()

def answer_data_question(df: pd.DataFrame, question: str) -> dict:
    lowered = question.lower()
    if "neighbourhood" in lowered and "highest" in lowered:
        averages = average_price_by(df, "neighbourhood")
        top = max(averages, key=averages.get)
        return {"answer": top, "value": averages[top], "all_averages": averages}
    if "property type" in lowered or "property_type" in lowered:
        return {"all_averages": average_price_by(df, "property_type")}
    return {"error": "question not recognised"}

print(answer_data_question(df, "Which neighbourhood has the highest average price?"))
print(answer_data_question(df, "Now break that down by property type"))
