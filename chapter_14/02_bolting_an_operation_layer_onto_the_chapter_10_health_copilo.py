"""
Chapter 14: The ITDO Framework, From Prediction to Operational Action
Hands-On: Bolting an Operation Layer onto the Chapter 10 Health Copilot

Extracted from: chapter_14_itdo_framework.md
Source book: Agentic AI: Building AI Agents and Retrieval Systems,
a Masterclass in LLM Agents, RAG, and Production Deployment.

Every block below was verified by direct execution before being
written into the handbook; run this file top to bottom, or copy
out the section you need.

ADAPTED FOR STANDALONE USE: this section reuses Chapter 10's own
`predict_diabetes_proba`, `flag_abnormal_features`, and `explain_diagnosis`;
the block below reconstructs all three here so this file does not require
running chapter_10's script first. `explain_diagnosis` calls a real GPT-4o
model when OPENAI_API_KEY is set (loaded via python-dotenv, following
Chapter 5's own security discipline, never hardcoded), and otherwise falls
back to a clearly-labelled deterministic stand-in, so the full pipeline
still runs end to end with no key at all.
"""

# ---- Setup reused from Chapter 10 (see code_examples/chapter_10) ----
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from pydantic import BaseModel

df = pd.read_csv("dataset/diabetes.csv")
cols_to_check = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols_to_check] = df[cols_to_check].replace(0, np.nan)
X = df.drop(columns=["Outcome"])
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)),
])
rf_pipeline.fit(X_train, y_train)

def predict_diabetes_proba(patient_data: list) -> float:
    return rf_pipeline.predict_proba([patient_data])[0][1]

NORMAL_RANGES = {
    "Glucose": (70, 99), "BloodPressure": (60, 80),
    "BMI": (18.5, 24.9), "Age": (0, 120),
}

def flag_abnormal_features(patient_data: list, feature_names: list) -> list:
    flags = []
    for name, value in zip(feature_names, patient_data):
        if name in NORMAL_RANGES:
            low, high = NORMAL_RANGES[name]
            if value < low or value > high:
                flags.append(f"{name} = {value} (normal range: {low}-{high})")
    return flags

FEATURE_NAMES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                  "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

class ClinicalExplanation(BaseModel):
    risk_level: str
    key_factors: list[str]
    recommendation: str
    disclaimer: str

def explain_diagnosis(probability: float, flags: list) -> ClinicalExplanation:
    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI
        client = OpenAI()
        system_prompt = (
            "You are a clinical communication assistant, not a diagnosing physician. "
            "Reason step by step using ONLY the flagged factors provided, never invent "
            "a factor not listed. Think first, then produce your final answer as the "
            "requested structured fields."
        )
        user_prompt = (
            f"Model-predicted diabetes probability: {probability:.0%}.\n"
            f"Flagged out-of-range factors: {flags if flags else 'none flagged'}.\n"
            f"Step 1, reason about what these specific flags suggest. "
            f"Step 2, state a risk_level (low, moderate, high). "
            f"Step 3, list key_factors drawn only from the flags above. "
            f"Step 4, give one plain-language recommendation. "
            f"Step 5, include a disclaimer that this is not a medical diagnosis."
        )
        response = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}],
            response_format=ClinicalExplanation,
        )
        return response.choices[0].message.parsed
    # no API key set: a deterministic stand-in so this file still completes end to end
    return ClinicalExplanation(
        risk_level="high" if probability >= 0.7 else "moderate",
        key_factors=flags,
        recommendation="[set OPENAI_API_KEY for a real, model-generated recommendation]",
        disclaimer="This is not a medical diagnosis.",
    )


# ---- Step 1: Add the Trigger layer, a threshold check the Chapter 10 pipeline never had. ----
def check_trigger(probability: float, threshold: float = 0.70) -> bool:
    """Insight alone is not action. This is the Rules Engine layer:
    convert a passive probability into an active signal."""
    return probability >= threshold

# ---- Step 2: Add the Operation layer, a tracked task with a named owner and a deadline, not just a printed explanation. ----
from datetime import datetime, timedelta

def create_operation_task(patient_id: str, explanation: dict) -> dict:
    """The layer a dashboard alone never provides: a real, tracked action."""
    return {
        "patient_id": patient_id,
        "task": "Schedule a follow-up consultation and share this explanation",
        "owner": "duty_clinician",
        "deadline": (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d"),
        "context": explanation,
        "status": "open",
    }

# ---- Step 3: Chain all four ITDO layers into one pipeline, reusing Chapter 10's own `predict_diabetes_proba`, `flag_abnormal_features`, and `explain_diagnosis` unchanged, exactly as this chapter's own Technical Requirements line promised. ----
def run_itdo_pipeline(patient_id: str, patient_data: list) -> dict:
    probability = predict_diabetes_proba(patient_data)             # Insight
    if not check_trigger(probability):                              # Trigger
        return {"status": "below threshold, no action taken", "probability": probability}

    flags = flag_abnormal_features(patient_data, FEATURE_NAMES)     # grounding for Decision
    explanation = explain_diagnosis(probability, flags)             # Decision
    task = create_operation_task(patient_id, explanation.model_dump())  # Operation
    return task

print(run_itdo_pipeline("P-1042", [2, 148, 72, 35, 155, 33.6, 0.627, 50]))
print(run_itdo_pipeline("P-3311", [7, 194, 68, 28, 200, 35.9, 0.745, 41]))

# ---- Step 4: Prove the domain-agnostic claim from earlier in this chapter directly, retargeting the exact same four-function shape at the banking churn example this chapter's own worked example named, not a hypothetical. ----
def predict_churn_proba(customer_features: dict) -> float:
    # stand-in for a trained XGBoost classifier's own predict_proba
    products_held = customer_features.get("products_held", 1)
    return 0.82 if products_held == 1 else 0.18

def check_trigger_churn(probability: float, threshold: float = 0.60) -> bool:
    return probability >= threshold

def create_operation_task_churn(customer_id: str, driver: str) -> dict:
    return {
        "customer_id": customer_id,
        "task": f"Trigger loyalty offer via CRM, top driver: {driver}",
        "owner": "retention_crm_bot",
        "deadline": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
        "status": "open",
    }

def run_churn_itdo_pipeline(customer_id: str, customer_features: dict) -> dict:
    probability = predict_churn_proba(customer_features)               # Insight
    if not check_trigger_churn(probability):                            # Trigger
        return {"status": "below threshold, no action taken", "probability": probability}
    driver = "only one product held" if customer_features.get("products_held", 1) == 1 else "other"
    return create_operation_task_churn(customer_id, driver)              # Decision + Operation

print(run_churn_itdo_pipeline("C-8821", {"products_held": 1}))
print(run_churn_itdo_pipeline("C-4410", {"products_held": 3}))
