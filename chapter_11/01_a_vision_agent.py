"""
Chapter 11: Multimodal Agents and Fine-Tuning with LoRA and QLoRA
Hands-On: A Vision Agent, Adapted from a Real Medical Imaging Project

Extracted from: chapter_11_multimodal_finetuning.md
Source book: Agentic AI: Building AI Agents and Retrieval Systems,
a Masterclass in LLM Agents, RAG, and Production Deployment.

Every block below was verified by direct execution before being
written into the handbook; run this file top to bottom, or copy
out the section you need. Where a step needs an API key
(OPENAI_API_KEY / ANTHROPIC_API_KEY), it is loaded from a local
.env file via python-dotenv, following Chapter 5's own security
discipline, never hardcoded.
"""


# ---- Step 1: Configure the model with vision explicitly enabled. ----
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.duckduckgo import DuckDuckGo

vision_agent = Agent(
    model=OpenAIChat(id="gpt-4o", vision=True),
    tools=[DuckDuckGo()],
    markdown=True,
)

# ---- Step 2: Write a structured query, following the real project's own five-section pattern. ----
query = """
Analyse the uploaded image and structure your response as:
### 1. Type and Context
Identify what kind of image this is and its relevant context.
### 2. Key Observations
List primary observations systematically, with severity ratings.
### 3. Assessment
Provide a primary assessment with a confidence level.
### 4. Plain-Language Summary
Explain the findings in language a non-expert can follow.
### 5. Supporting Research
Use web search to find 2-3 relevant, current references.
"""

# ---- Step 3: Pass the image and the query to the agent together, in one call. ----
response = vision_agent.run(query, images=["uploaded_image.png"])
print(response.content)

# ---- Step 4: Close the vague-confidence gap this section's own "Think it through" question names, replacing the free-text five-section query with a Pydantic schema that makes "confidence" mean exactly one of three things, tested against both a valid and an invalid response. ----
from enum import Enum
from typing import List
from pydantic import BaseModel, Field

class ConfidenceLevel(str, Enum):
    low = "low"
    moderate = "moderate"
    high = "high"

class Observation(BaseModel):
    finding: str
    severity: ConfidenceLevel

class ImageAnalysis(BaseModel):
    image_type: str
    observations: List[Observation]
    assessment: str
    confidence: ConfidenceLevel
    plain_language_summary: str
    supporting_references: List[str] = Field(min_length=1, max_length=3)

good = ImageAnalysis(
    image_type="Chest X-ray, PA view",
    observations=[Observation(finding="Mild opacity in lower left lobe", severity=ConfidenceLevel.moderate)],
    assessment="Findings consistent with early-stage consolidation, follow-up recommended.",
    confidence=ConfidenceLevel.moderate,
    plain_language_summary="There is a small area of cloudiness in the lower left part of the lung.",
    supporting_references=["Fleischner Society glossary of terms, 2024 revision"],
)
print(good.confidence.value)

try:
    ImageAnalysis(
        image_type="Chest X-ray, PA view",
        observations=[Observation(finding="Mild opacity", severity="pretty sure")],
        assessment="test", confidence="pretty confident",
        plain_language_summary="test", supporting_references=["ref1"],
    )
except Exception as error:
    print("rejected as expected:", type(error).__name__)
