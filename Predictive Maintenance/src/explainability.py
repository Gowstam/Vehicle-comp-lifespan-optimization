"""
explainability.py — SHAP + LLM Explainability for Predictive Maintenance
Updated for openai>=1.0.0 (new client API)
Generates feature-level SHAP importance and business-friendly LLM narratives.
"""

import shap
import json
import joblib
import numpy as np
import pandas as pd
import os
from openai import OpenAI

# ---------------- CONFIGURATION ----------------
OPENAI_API_KEY="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"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------- SHAP COMPUTATION ----------------
def compute_shap_values(model, X_sample):
    """Compute SHAP values safely for one or more rows of features."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    # Flatten all arrays and align lengths
    shap_vals = np.array(shap_values.values).flatten()
    feat_vals = np.array(X_sample.iloc[0].values).flatten()
    feat_names = np.array(X_sample.columns)

    n = min(len(feat_names), len(feat_vals), len(shap_vals))
    shap_df = pd.DataFrame({
        "feature": feat_names[:n],
        "feature_value": feat_vals[:n],
        "shap_value": shap_vals[:n]
    }).sort_values(by="shap_value", ascending=False)

    return shap_df


# ---------------- LLM NARRATIVE GENERATION ----------------
def generate_llm_explanation(component, shap_df, top_k=5):
    """Use GPT-4-Turbo to translate SHAP features into an engineer-friendly narrative."""
    top_features = shap_df.head(top_k).to_dict(orient="records")
    prompt = f"""
    You are a predictive maintenance expert.
    Explain why the {component} may fail soon based on the top contributing features below.
    Provide a concise, insightful paragraph suitable for maintenance engineers.

    Features (importance order):
    {json.dumps(top_features, indent=2)}

    Return a single paragraph explanation (no bullet points).
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
        )
        explanation = response.choices[0].message.content.strip()
    except Exception as e:
        explanation = f"[Error generating explanation: {e}]"

    return explanation


# ---------------- WRAPPER FOR ONE COMPONENT ----------------
def explain_prediction(component, model_path, feature_list, row_df, save_dir=None):
    """
    Compute SHAP values + LLM narrative for a given component.
    Optionally saves SHAP CSV for visualization in Streamlit or Power BI.
    """
    model = joblib.load(model_path)
    X = row_df[feature_list].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    shap_df = compute_shap_values(model, X)
    narrative = generate_llm_explanation(component, shap_df)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        shap_path = os.path.join(save_dir, f"shap_{component}.csv")
        shap_df.to_csv(shap_path, index=False)

    return shap_df, narrative
