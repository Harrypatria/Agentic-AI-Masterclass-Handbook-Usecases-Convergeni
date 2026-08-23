"""
Chapter 2: Prompt Engineering, Twenty Techniques from Zero-Shot to Optimisation
Hands-On: Measuring the Ladder's First Three Rungs on One Real Question

Extracted from: chapter_02_prompt_engineering.md
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


# ---- Step 1: Write one reusable call function. ----
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ask(prompt: str, temperature: float = 0) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content

# ---- Step 2: Pose the same reasoning question three ways, climbing the ladder one rung at a time. ----
question = (
    "A shop had 23 apples. They sold 14 and then received a delivery "
    "of 2 crates, 8 apples per crate. How many apples do they have now?"
)

zero_shot = ask(question)

few_shot = ask(
    "Q: A farm had 10 hens. It sold 3 and bought 5 more. How many hens now?\n"
    "A: 10 minus 3 is 7, plus 5 is 12. Answer: 12.\n\n"
    f"Q: {question}\nA:"
)

chain_of_thought = ask(f"{question}\nLet's think step by step.")

# ---- Step 3: Print all three outputs side by side and inspect them. ----
for label, output in [("Zero-shot", zero_shot), ("Few-shot", few_shot),
                        ("Chain-of-thought", chain_of_thought)]:
    print(f"--- {label} ---\n{output}\n")

# ---- Step 4: Build the fourth rung, self-consistency, the answer to this section's own "Think it through" question, and test its vote-tallying logic against a real, deliberately imperfect sample. ----
from collections import Counter
import re

def extract_final_number(text: str) -> str:
    numbers = re.findall(r"-?\d+", text)
    return numbers[-1] if numbers else None

def self_consistency_vote(responses: list) -> tuple:
    answers = [extract_final_number(r) for r in responses]
    counts = Counter(a for a in answers if a is not None)
    if not counts:
        return None, {}
    winner, _ = counts.most_common(1)[0]
    return winner, dict(counts)

five_samples = [ask(f"{question}\nLet's think step by step.", temperature=0.7) for _ in range(5)]
winner, tally = self_consistency_vote(five_samples)
print("winner:", winner)
print("tally:", tally)
