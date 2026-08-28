"""
Chapter 7: No-Code Agent Automation with n8n
Hands-On: Reading a Real n8n Workflow's Exported JSON

Extracted from: chapter_07_nocode_n8n.md
Source book: Agentic AI: Building AI Agents and Retrieval Systems,
a Masterclass in LLM Agents, RAG, and Production Deployment.

Every block below was verified by direct execution before being
written into the handbook; run this file top to bottom, or copy
out the section you need. Where a step needs an API key
(OPENAI_API_KEY / ANTHROPIC_API_KEY), it is loaded from a local
.env file via python-dotenv, following Chapter 5's own security
discipline, never hardcoded.

ADAPTED FOR STANDALONE USE: Step 6's simulated reply includes a real emoji
character, which raises a UnicodeEncodeError on a default Windows console
(cp1252). The two lines below reconfigure stdout to UTF-8 first, so this
file prints correctly regardless of the terminal's own default encoding.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ---- Step 6: Read the JSON, then prove you actually understood it, by simulating the same execution path in plain Python. ----
def run_workflow(incoming_message: str, respond_fn) -> dict:
    context = {"message": {"text": incoming_message}}
    context["output"] = respond_fn(context["message"]["text"])   # the "AI Agent" step
    return context                                                # what the "Telegram" node reads

def fake_llm_agent(user_text: str) -> str:
    return f'Sure! Here is your answer to "{user_text}" \U0001F642'

result = run_workflow("What is the weather like today?", fake_llm_agent)
print(result)

# ---- Step 7: Add the router node the "Think it through" question below asks for, and prove the branch actually works against two real messages. ----
extended_connections = {
    "Telegram Trigger": {"main": [[{"node": "Router"}]]},
    "Router": {
        "main": [
            [{"node": "AI Agent (support)"}],   # output 0: condition True
            [{"node": "AI Agent (general)"}],   # output 1: condition False
        ]
    },
}

def router_condition(message_text: str) -> bool:
    keywords = ["refund", "broken", "cancel", "complaint"]
    return any(word in message_text.lower() for word in keywords)

def run_extended_workflow(message_text: str) -> str:
    branch = router_condition(message_text)
    return extended_connections["Router"]["main"][0 if branch else 1][0]["node"]

print(run_extended_workflow("My order arrived broken, I want a refund"))
print(run_extended_workflow("What time do you close on Fridays?"))
