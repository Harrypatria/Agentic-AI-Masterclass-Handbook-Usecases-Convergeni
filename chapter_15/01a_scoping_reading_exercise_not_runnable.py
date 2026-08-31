"""
Chapter 15: The Capstone Project, From Scoping to Demo Day
Hands-On: Scoping a Real Repository as if It Were a Capstone Submission
(Step 1 of 3: the reading exercise)

Extracted from: chapter_15_capstone.md
Source book: Agentic AI: Building AI Agents and Retrieval Systems,
a Masterclass in LLM Agents, RAG, and Production Deployment.

THIS FILE IS FOR READING, NOT RUNNING. It is a real excerpt from the
masterclass's own agen_layanan_dukungan_pelanggan project, quoted here
as evidence to run the chapter's own scoping template and architecture
checklist against, exactly as this hands-on section's own "Technical
requirements" line states: "No installation needed for this section; it
is a structured reading exercise against a real, already-written
codebase." `Memory` is that project's own import, not defined here, and
running this file directly will raise a NameError, which is expected.
"""

# ---- Step 1: Read the project's own memory configuration, the clearest architectural evidence available. ----
config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333,
        }
    }
}
memory = Memory.from_config(config)  # noqa: F821 -- intentionally undefined, see module docstring
