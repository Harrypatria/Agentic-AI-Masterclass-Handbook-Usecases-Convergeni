"""
Chapter 14: The ITDO Framework, From Prediction to Operational Action
Hands-On: Real SHAP Values, and the Log-Odds Trap

Extracted from: chapter_14_itdo_framework.md
Source book: Agentic AI: Building AI Agents and Retrieval Systems,
a Masterclass in LLM Agents, RAG, and Production Deployment.

Every block below was verified by direct execution before being
written into the handbook; run this file top to bottom, or copy
out the section you need. Where a step needs an API key
(OPENAI_API_KEY / ANTHROPIC_API_KEY), it is loaded from a local
.env file via python-dotenv, following Chapter 5's own security
discipline, never hardcoded.

ADAPTED FOR STANDALONE USE: this section's own text builds on
X_train/X_test/y_train/y_test from Chapter 10's Phase 3 pipeline; the
block below reconstructs them here, identically, so this file does not
require running chapter_10's script first. See code_examples/chapter_10
for the full, worked CRISP-DM walkthrough these lines are drawn from.
"""

# ---- Setup reused from Chapter 10 (see code_examples/chapter_10) ----
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset/diabetes.csv")
cols_to_check = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols_to_check] = df[cols_to_check].replace(0, np.nan)
df[cols_to_check] = df[cols_to_check].fillna(df[cols_to_check].median())
X = df.drop(columns=["Outcome"])
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- Step 1: Train the Gradient Boosting model this chapter has been discussing, on the exact cleaned data Chapter 10 built. ----
from sklearn.ensemble import GradientBoostingClassifier

gb_model = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train)

# ---- Step 2: Build the explainer and compute SHAP values for the entire held-out test set in one call. ----
import shap

explainer = shap.TreeExplainer(gb_model)
shap_values = explainer.shap_values(X_test)
print(shap_values.shape)  # (154, 8): one row per patient, one column per feature

# ---- Step 3: Explain one specific patient, the same kind of individual case the Decision layer above reads from. ----
patient_row = X_test.iloc[[0]]
patient_shap = explainer.shap_values(patient_row)[0]

for feature_name, contribution in zip(X_test.columns, patient_shap):
    print(f"{feature_name}: {contribution:+.4f}")

# ---- Step 4: The log-odds trap. Try to reconstruct the model's own probability the naive way, then the correct way. ----
import numpy as np

base_value = float(np.ravel(explainer.expected_value)[0])
naive_sum = base_value + patient_shap.sum()
actual_proba = gb_model.predict_proba(patient_row)[0][1]

print("base value:", round(base_value, 4))
print("base + sum(shap), treated as a probability:", round(naive_sum, 4))
print("gb_model.predict_proba (the real answer):", round(actual_proba, 4))

def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))

reconstructed_proba = sigmoid(base_value + patient_shap.sum())
print("sigmoid(base + sum(shap)):", round(reconstructed_proba, 4))

# ---- Step 5: Rank features by global importance across the whole test set, not just one patient, producing Table 14.1. ----
mean_abs_shap = np.abs(shap_values).mean(axis=0)
ranking = sorted(zip(X_test.columns, mean_abs_shap), key=lambda pair: pair[1], reverse=True)
for feature_name, importance in ranking:
    print(f"{feature_name}: {importance:.4f}")
