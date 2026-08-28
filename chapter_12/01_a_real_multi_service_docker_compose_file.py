"""
Chapter 12: Deploying Agentic AI to Production
Hands-On: A Real Multi-Service Docker Compose File, Read Line by Line

Extracted from: chapter_12_deployment.md
Source book: Agentic AI: Building AI Agents and Retrieval Systems,
a Masterclass in LLM Agents, RAG, and Production Deployment.

Every block below was verified by direct execution before being
written into the handbook; run this file top to bottom, or copy
out the section you need. Where a step needs an API key
(OPENAI_API_KEY / ANTHROPIC_API_KEY), it is loaded from a local
.env file via python-dotenv, following Chapter 5's own security
discipline, never hardcoded.
"""


# ---- Step 4: Answer this section's own first "Think it through" question with real, tested code, rewriting both plain-text passwords to the `${VARIABLE_NAME}` pattern and confirming the substitution actually works. ----
import os
import re

os.environ["POSTGRES_PASSWORD"] = "a-real-secret-not-in-git"
os.environ["N8N_BASIC_AUTH_PASSWORD"] = "another-real-secret"

compose_template = '''
POSTGRES_PASSWORD: "${POSTGRES_PASSWORD}"
N8N_BASIC_AUTH_PASSWORD: "${N8N_BASIC_AUTH_PASSWORD}"
'''

def substitute_env_vars(template: str) -> str:
    def replace(match):
        var_name = match.group(1)
        return os.environ.get(var_name, f"MISSING:{var_name}")
    return re.sub(r"\$\{([A-Z_]+)\}", replace, template)

print(substitute_env_vars(compose_template))

def substitute_env_vars(template: str) -> str:
    def replace(match):
        var_name = match.group(1)
        return os.environ.get(var_name, f"MISSING:{var_name}")
    return re.sub(r"\$\{([A-Z0-9_]+)\}", replace, template)   # 0-9 added
