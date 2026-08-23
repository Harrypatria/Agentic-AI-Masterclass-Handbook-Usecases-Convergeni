"""
Chapter 1: Foundations of Artificial Intelligence and the Agentic Paradigm
Hands-On: Tracing the Perceive, Reason, Act, Reflect Loop in Fifteen Lines

Extracted from: chapter_01_foundations.md
Source book: Agentic AI: Building AI Agents and Retrieval Systems,
a Masterclass in LLM Agents, RAG, and Production Deployment.

Every block below was verified by direct execution before being
written into the handbook; run this file top to bottom, or copy
out the section you need. Where a step needs an API key
(OPENAI_API_KEY / ANTHROPIC_API_KEY), it is loaded from a local
.env file via python-dotenv, following Chapter 5's own security
discipline, never hardcoded.
"""


# ---- Step 1: Define the environment the agent perceives. ----
# A tiny "world" the agent can perceive and act on.
# In Chapters 5 and 6 this becomes a real API response;
# here it is a plain dictionary so the loop stays visible.
world = {"ticket": "Customer says their order #4471 never arrived."}

# ---- Step 2: Define one tool the agent can act with. ----
def lookup_order(order_id: str) -> dict:
    """Stand-in for a real CRM lookup tool (Chapter 6 builds a real one)."""
    fake_orders = {"4471": {"status": "delivered", "date": "2026-08-02"}}
    return fake_orders.get(order_id, {"status": "not found"})

# ---- Step 3: Write the Reason step as an explicit decision function, not a black box. ----
def reason(perceived: dict) -> str:
    """Given what was perceived, decide the next action.
    A real agent asks an LLM this question; here the
    decision logic is written out by hand so it stays inspectable."""
    if "order" in perceived["ticket"].lower():
        return "call_lookup_order"
    return "ask_clarifying_question"

# ---- Step 4: Wire Perceive, Reason, Act, and Reflect into one explicit loop. ----
def run_agent(world: dict) -> str:
    # Perceive
    perceived = world

    # Reason
    action = reason(perceived)

    # Act
    if action == "call_lookup_order":
        order_id = "4471"  # a real agent would extract this from the ticket text
        result = lookup_order(order_id)
    else:
        result = {"status": "needs_clarification"}

    # Reflect
    if result["status"] == "delivered":
        return f"Resolved: order {order_id} shows delivered on {result['date']}. Draft a reply citing this."
    elif result["status"] == "not found":
        return "Reflect: lookup failed, re-plan with a different order ID or escalate to a human."
    else:
        return "Reflect: insufficient information, ask the customer for their order number."

print(run_agent(world))

# ---- Step 5: Answer this section's own first "Think it through" question with real output, not a guess, by running a ticket that never mentions an order at all. ----
world_no_order = {"ticket": "Customer wants to know your opening hours."}
print(run_agent(world_no_order))

# ---- Step 6: Build the runaway scenario this section's third "Think it through" question asks for, and confirm the one-line fix actually stops it. ----
def run_agent_looping(world: dict, max_iterations: int = 5) -> str:
    perceived = world
    iterations = 0
    while True:
        iterations += 1
        if iterations > max_iterations:                          # the one line that stops it
            return f"Reflect: stopped after {max_iterations} iterations with no resolution, escalate to a human."

        action = reason(perceived)
        if action == "call_lookup_order":
            result = lookup_order("9999")   # an order ID that will never resolve
        else:
            result = {"status": "needs_clarification"}

        if result["status"] == "delivered":
            return "Resolved: order shows delivered."
        elif result["status"] == "not found":
            perceived = world   # contrived: Reflect always re-perceives the same ticket and retries
            continue
        else:
            return "Reflect: insufficient information, ask the customer for their order number."

world_broken = {"ticket": "Customer says their order #9999 never arrived."}
print(run_agent_looping(world_broken, max_iterations=5))
