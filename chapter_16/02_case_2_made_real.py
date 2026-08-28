"""
Chapter 16: Real-World Agent Patterns, Case Studies from the Field
Hands-On: Case 2 Made Real, Checking Whether a Citation Actually Grounds Its Claim

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


# ---- == Hands-On: Case 2 Made Real, Checking Whether a Citation Actually Grounds Its Claim == ----
def verify_citation_grounded(claim: str, cited_source: str, min_shared_words: int = 3) -> dict:
    claim_words = set(w.strip(".,%$").lower() for w in claim.split())
    source_words = set(w.strip(".,%$").lower() for w in cited_source.split())
    shared = claim_words & source_words
    return {"grounded": len(shared) >= min_shared_words, "shared_terms": sorted(shared)}

source_doc = ("Q3 earnings report: revenue grew 12% year-over-year to $4.2 billion, "
              "driven primarily by cloud services growth of 18%.")

grounded_claim = "Revenue grew 12% year-over-year to $4.2 billion in Q3."
fabricated_claim = "The company announced a major new product line launching in Q4."

print(verify_citation_grounded(grounded_claim, source_doc))
print(verify_citation_grounded(fabricated_claim, source_doc))
