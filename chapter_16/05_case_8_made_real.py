"""
Chapter 16: Real-World Agent Patterns, Case Studies from the Field
Hands-On: Case 8 Made Real, Scoring the Pilot That Never Scaled

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


# ---- == Hands-On: Case 8 Made Real, Scoring the Pilot That Never Scaled == ----
def score_readiness(checklist: dict) -> dict:
    required = ["env_discipline", "containerised", "cicd_pipeline", "evaluation_metrics", "named_kpi"]
    missing = [item for item in required if not checklist.get(item, False)]
    return {
        "ready_for_production": len(missing) == 0,
        "missing": missing,
        "score": f"{len(required) - len(missing)}/{len(required)}",
    }

case_8_prototype = {
    "env_discipline": False, "containerised": False, "cicd_pipeline": False,
    "evaluation_metrics": False, "named_kpi": False,
}
print(score_readiness(case_8_prototype))

hardened_version = {
    "env_discipline": True, "containerised": True, "cicd_pipeline": True,
    "evaluation_metrics": True, "named_kpi": True,
}
print(score_readiness(hardened_version))
