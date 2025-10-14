"""
app.py — Streamlit Dashboard for Predictive Maintenance
Loads trained models and predicts failure & RUL
for selected vehicle/date, plus integrates a maintenance LLM chatbot.
"""

import streamlit as st
import pandas as pd
import joblib
import json
import os
import numpy as np
from llm_helper import chat_with_maintenance_ai

# ---------------- Configuration ----------------
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="centered")

BASE_DIR = "/Users/gowthamsivaraman/Desktop/Vehicle Maintenance Project/Predictive Maintenance"
MODEL_DIR = os.path.join(BASE_DIR, "models")
COMPONENTS = ["engine", "battery", "brake"]
HORIZON = 7

# ---------------- Load Models ----------------
@st.cache_resource
def load_models():
    clf_models, reg_models, features_map = {}, {}, {}
    for comp in COMPONENTS:
        clf_path = os.path.join(MODEL_DIR, f"{comp}_rf_binary_model.joblib")
        reg_path = os.path.join(MODEL_DIR, f"{comp}_xgb_rul_model.joblib")
        feats_path = os.path.join(MODEL_DIR, f"{comp}_features.json")

        if os.path.exists(clf_path):
            clf_models[comp] = joblib.load(clf_path)
        if os.path.exists(reg_path):
            reg_models[comp] = joblib.load(reg_path)
        if os.path.exists(feats_path):
            with open(feats_path) as f:
                features_map[comp] = json.load(f)
    return clf_models, reg_models, features_map


clf_models, reg_models, features_map = load_models()
if not clf_models:
    st.error(" No trained models found in `/models`. Please run main_pipeline.py first.")
    st.stop()

# ---------------- File Upload ----------------
st.title("🚗 Predictive Maintenance Dashboard")
st.markdown("""
Upload a **feature-engineered CSV file** to view  
predicted failures and Remaining Useful Life (RUL) for each component.
""")

uploaded = st.file_uploader("Upload telemetry CSV file", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.success(f"✅ File uploaded successfully! Shape: {df.shape}")

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Detect vehicle_id column
    vehicle_col = None
    for c in ["vehicle_id", "VehicleID", "veh_id"]:
        if c in df.columns:
            vehicle_col = c
            break

    if not vehicle_col:
        st.error(" No vehicle_id column found in dataset.")
        st.stop()

    # --- Use first record (read-only mode) ---
    row = df.iloc[[0]].copy()
    selected_vehicle = str(row[vehicle_col].values[0])
    selected_date = str(row["timestamp"].values[0])

    # ---------------- Vehicle and Date ----------------
    st.subheader("Vehicle and Date Information")
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Selected Vehicle", selected_vehicle, disabled=True)
    with col2:
        st.text_input("Selected Date", selected_date, disabled=True)

    # ---------------- Perform Predictions ----------------
    preds_fail, preds_rul, expected_fail = {}, {}, {}
    for comp in COMPONENTS:
        feats = [f for f in features_map.get(comp, []) if f in row.columns]
        if not feats:
            continue
        X = row[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        if comp in clf_models:
            preds_fail[comp] = int(clf_models[comp].predict(X)[0])

        if comp in reg_models:
            rul_val = float(reg_models[comp].predict(X)[0])
            preds_rul[comp] = round(rul_val, 2)
            timestamp = pd.to_datetime(row["timestamp"].values[0]).normalize()
            days_to_add = int(np.round(rul_val)) if rul_val > 0 else 0
            expected_fail[comp] = (
                str((timestamp + pd.Timedelta(days=days_to_add)).date())
                if rul_val > 0 else "N/A"
            )

    # ---------------- Display Results ----------------
    st.subheader("Prediction Results")
    results_df = pd.DataFrame({
        "Component": list(preds_fail.keys()),
        "Pred_Fail_in_Next_7D": [preds_fail[c] for c in preds_fail],
        "Pred_RUL_Days": [preds_rul.get(c, np.nan) for c in preds_fail],
        "Expected_Failure_Date": [expected_fail.get(c, "N/A") for c in preds_fail]
    })
    results_df["Pred_RUL_Days"] = results_df["Pred_RUL_Days"].round(0).astype(int)
    st.dataframe(results_df, use_container_width=True)

    # ---------------- Download Results ----------------
    st.download_button(
        "Download Results as JSON",
        data=json.dumps(results_df.to_dict(orient="records"), indent=2).encode("utf-8"),
        file_name="prediction_results.json",
        mime="application/json"
    )

    # ---------------- Maintenance AI Chat ----------------
    st.markdown("---")
    st.subheader(" Ask the Maintenance AI")

    user_question = st.text_input("Ask about component behavior or anomalies:")
    if user_question:
        # Example: choose 'battery' context (can extend to detect automatically)
        component = "battery"
        feats = [f for f in features_map.get(component, []) if f in row.columns]

        context = {
            "component": component,
            "predictions": {
                "Pred_Fail_in_Next_7D": preds_fail.get(component),
                "Pred_RUL_Days": preds_rul.get(component),
                "Expected_Failure_Date": expected_fail.get(component)
            },
            "features": {k: float(row[k]) for k in feats}
        }

        with st.spinner("Analyzing telemetry patterns..."):
            response = chat_with_maintenance_ai(context, user_question)
        st.markdown(f"**AI Response:** {response}")

else:
    st.info("⬆️ Upload a telemetry CSV file to begin prediction.")

st.markdown("---")
st.caption("Built using Streamlit, Machine Learning, and OpenAI GPT-4o-mini.")
