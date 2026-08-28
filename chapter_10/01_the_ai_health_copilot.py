"""
Chapter 10: Machine Learning Integration for Agentic Systems
Hands-On: The AI Health Copilot, End to End Through Every CRISP-DM Phase

Extracted from: chapter_10_ml_integration.md
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


# ---- Step 1: Load the data and look at its basic shape before doing anything else. ----
import pandas as pd

df = pd.read_csv("dataset/diabetes.csv")
print(df.shape)
print(df.head())
print(df["Outcome"].value_counts())

# ---- Step 2: Audit for a specific, well-documented data quality problem before trusting any statistic. ----
cols_to_check = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols_to_check:
    zero_count = (df[col] == 0).sum()
    print(f"{col}: {zero_count} zero values ({zero_count / len(df):.1%})")

# ---- Step 3: Convert the invalid zeros into genuine, explicit missing values. ----
import numpy as np

df[cols_to_check] = df[cols_to_check].replace(0, np.nan)
print(df.isnull().sum())

# ---- Step 4: Split into training and test sets before any further preparation, not after. ----
from sklearn.model_selection import train_test_split

X = df.drop(columns=["Outcome"])
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- Step 5: Impute and scale inside one `Pipeline`, following this chapter's own best-practice rule from earlier. ----
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

preprocessing = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# ---- Step 6: Train two candidate models inside the same pipeline shape, so they are genuinely comparable. ----
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

rf_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)),
])
rf_pipeline.fit(X_train, y_train)

logreg_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000)),
])
logreg_pipeline.fit(X_train, y_train)

# ---- Step 7: Score both models on the held-out test set, never seen during training. ----
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

for name, pipeline in [("Random Forest", rf_pipeline), ("Logistic Regression", logreg_pipeline)]:
    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]
    print(f"--- {name} ---")
    print("Accuracy:", round(accuracy_score(y_test, pred), 4))
    print("AUC:", round(roc_auc_score(y_test, proba), 4))
    print(confusion_matrix(y_test, pred))

# ---- Step 8: Check the result is not a lucky split with five-fold cross-validation. ----
from sklearn.model_selection import cross_val_score

scores = cross_val_score(rf_pipeline, X, y, cv=5, scoring="roc_auc")
print("AUC across 5 folds:", [round(s, 3) for s in scores])
print("Mean AUC:", round(scores.mean(), 4), "Std:", round(scores.std(), 4))

# ---- Step 8b: Before accepting `max_depth=5, n_estimators=200` as final, search a small grid of alternatives properly, using the same 5-fold discipline Step 8 just validated, rather than picking hyperparameters by hand and hoping. ----
from sklearn.model_selection import GridSearchCV

param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [3, 5, 7],
}

grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
grid_search.fit(X_train, y_train)

print("best params:", grid_search.best_params_)
print("best CV AUC:", round(grid_search.best_score_, 4))

# ---- Step 9: Save the fitted pipeline exactly as the real project's own `.sav` files were produced. ----
import pickle

with open("saved_models/diabetes_model.sav", "wb") as f:
    pickle.dump(rf_pipeline, f)

# ---- Step 10: Expose a genuine probability, not a bare label, so downstream logic has something to reason about. ----
def predict_diabetes_proba(patient_data: list) -> float:
    return rf_pipeline.predict_proba([patient_data])[0][1]

# ---- Step 11: Deterministically flag which specific features are outside a normal clinical range, in code, before the LLM ever sees the case. ----
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

# ---- Step 12: Feed only the deterministic flags, not the raw feature vector, into a structured, role-based, chain-of-thought prompt. ----
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()

class ClinicalExplanation(BaseModel):
    risk_level: str
    key_factors: list[str]
    recommendation: str
    disclaimer: str

def explain_diagnosis(probability: float, flags: list) -> ClinicalExplanation:
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

# ---- Step 13: Chain every phase into one callable pipeline, wrapped exactly as Chapter 6's tool-registry pattern requires. ----
FEATURE_NAMES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                  "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

def run_health_copilot(patient_data: list) -> dict:
    probability = predict_diabetes_proba(patient_data)
    flags = flag_abnormal_features(patient_data, FEATURE_NAMES)
    explanation = explain_diagnosis(probability, flags)
    return {"probability": probability, "explanation": explanation.model_dump()}

patient = [2, 148, 72, 35, 155, 33.6, 0.627, 50]  # a real, positive-outcome row from the dataset
print(run_health_copilot(patient))
