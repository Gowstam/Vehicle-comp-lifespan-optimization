"""
main_pipeline.py — Predictive Maintenance Pipeline with Explainability
Trains RandomForest classifiers and XGBoost regressors,
predicts failure risk and RUL, and generates SHAP + LLM explanations.
"""

import os
import json
import joblib
import warnings
import pandas as pd
import numpy as np
from src.anomaly_detection import detect_anomalies_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    mean_absolute_error, mean_squared_error
)
from xgboost import XGBRegressor

# ---------------- Local Imports ----------------
from src.data_loader import load_data
from src.data_cleaning import inspect_data, clean_zero_values
from src.feature_engineering import (
    add_features,
    rebuild_failure_date_column,
    regenerate_daily_timestamps,
    build_component_labels,
)
from src.feature_selection import select_all_components
from src.explainability import explain_prediction

# ---------------- CONFIGURATION ----------------
COMPONENTS = ["engine", "battery", "brake"]
HORIZON = 7
RUL_CAP = 60
RANDOM_STATE = 42

BASE_DIR = "/Users/gowthamsivaraman/Desktop/Vehicle Maintenance Project/Predictive Maintenance"
OUT_DIR = os.path.join(BASE_DIR, "output_simple")
ART_DIR = os.path.join(OUT_DIR, "artifacts")
MODEL_DIR = os.path.join(BASE_DIR, "models")

for d in [OUT_DIR, ART_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

# ----------- LOAD DATA ----------------
df = load_data("data/synthetic_telemetry_data.csv")
print(" Step 1 completed: Data loaded.")
inspect_data(df)
df = clean_zero_values(df)

# ----------- TIMESTAMP & FAILURE DATE REBUILD -----------------
df = regenerate_daily_timestamps(df)
df = rebuild_failure_date_column(df)
print(" Step 2 completed: Timestamps and failure dates rebuilt.")


# ------------ BUILD FAILED COLUMNS ------------------
for comp in COMPONENTS:
    df[f"{comp}_failed"] = (
        df["failure_type"]
        .astype(str)
        .str.lower()
        .str.contains(comp)
        .astype(int)
    )
print(" Added failed binary columns.")

# ------------- LABEL CREATION --------------------
for comp in COMPONENTS:
    df = build_component_labels(df, comp, horizon_days=HORIZON)
print(" Step 3 completed: Added fail_in_next_7d and RUL columns.")

# Save labeled dataset
labelled_path = os.path.join(OUT_DIR, "telemetry_labeled_7d.csv")
df.to_csv(labelled_path, index=False)
print(f"💾 Labeled dataset saved to: {labelled_path}")

# -------------- FEATURE ENGINEERING ----------------------
df = add_features(df)
print(" Step 4 completed: Feature engineering done.")

# Save dataset with features
features_path = os.path.join(OUT_DIR, "telemetry_labeled_with_features_7d.csv")
df.to_csv(features_path, index=False)
print(f"💾 Feature-engineered dataset saved to: {features_path}")

# ------------------ ANOMALY DETECTION ------------------
# Transformer-based Anomaly Detection 

from src.anomaly_detection import detect_anomalies_transformer
import matplotlib.pyplot as plt
import os

print("\n=== Selecting Sensor Features for Anomaly Detection ===")

sensor_features = [
    "engine_temp_c", "engine_rpm", "oil_pressure_psi", "coolant_temp_c",
    "fuel_level_percent", "fuel_consumption_lph", "engine_load_percent",
    "throttle_pos_percent", "air_flow_rate_gps", "exhaust_gas_temp_c",
    "vibration_level", "engine_hours",
    "brake_fluid_level_psi", "brake_pad_wear_mm", "brake_temp_c",
    "brake_pedal_pos_percent",
    "wheel_speed_fl_kph", "wheel_speed_fr_kph",
    "wheel_speed_rl_kph", "wheel_speed_rr_kph",
    "vehicle_speed_kph",
    "battery_voltage_v", "battery_current_a", "battery_temp_c",
    "alternator_output_v", "battery_charge_percent", "battery_health_percent",
    "ambient_temp_c", "humidity_percent",
    "odometer_reading"
]

sensor_features = [c for c in sensor_features if c in df.columns]
print(f"{len(sensor_features)} sensor features selected for anomaly detection.")

# Run Transformer anomaly detection
print("\n=== Detecting Anomalies using Transformer Autoencoder ===")
df = detect_anomalies_transformer(df, feature_cols=sensor_features, threshold_factor=2.0, epochs=20)
print(" Transformer anomaly detection complete. Columns added: ['anomaly_score', 'is_anomaly']")

# Plot anomaly results (per vehicle)
print("\n=== Plotting Anomaly Detection Results ===")
ANOM_DIR = os.path.join(OUT_DIR, "anomaly_figures")
os.makedirs(ANOM_DIR, exist_ok=True)

for veh in df["vehicle_id"].unique():
    g = df[df["vehicle_id"] == veh].dropna(subset=["timestamp", "anomaly_score"])
    g = g.sort_values("timestamp")

    plt.figure(figsize=(10, 5))
    plt.plot(g["timestamp"], g["anomaly_score"], label="Anomaly Score", color="blue", linewidth=1.5)

    anomalies = g[g["is_anomaly"] == 1]
    if not anomalies.empty:
        plt.scatter(anomalies["timestamp"], anomalies["anomaly_score"], color="red", s=40, label="Anomaly", zorder=3)

    plt.title(f"🚗 Vehicle {veh} — Transformer Anomaly Detection")
    plt.xlabel("Timestamp")
    plt.ylabel("Reconstruction Error (Anomaly Score)")
    plt.legend()
    plt.grid(alpha=0.5, linestyle="--")
    plt.tight_layout()

    plot_path = os.path.join(ANOM_DIR, f"{veh}_anomaly_plot.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()

print(f"\n All anomaly plots saved to: {ANOM_DIR}")


# -------------- FEATURE SELECTION ---------------------
print("\n=== STEP 5: Correlation & Dendrogram-based Feature Selection ===")
engine_feats, battery_feats, brake_feats = select_all_components(
    df, out_dir=ART_DIR, top_n=25, corr_threshold=0.9
)
print(" Step 5 completed: Feature selection done for all components.")

print("\n=== 🧩 FINAL SELECTED FEATURES PER COMPONENT ===")
def summarize_features(name, feats):
    print(f"\n{name.upper()} ({len(feats)} features):")
    print(", ".join(feats) if feats else "⚠️ None selected")

summarize_features("Engine", engine_feats)
summarize_features("Battery", battery_feats)
summarize_features("Brake", brake_feats)

# Save feature summary
feature_summary_path = os.path.join(ART_DIR, "selected_features_summary.json")
with open(feature_summary_path, "w") as f:
    json.dump({
        "engine_features": engine_feats,
        "battery_features": battery_feats,
        "brake_features": brake_feats
    }, f, indent=2)
print(f"\n Feature selection summary saved to: {feature_summary_path}")

# ------------------- REMOVE POST-FAILURE DATA -------------------------
for comp in COMPONENTS:
    fail_flag = f"{comp}_failed"
    if fail_flag in df.columns:
        df = df[df[fail_flag] == 0]
print(" Step 6 completed: Post-failure data removed.")


# ------------------- PREP FEATURE MAP ------------------------
COMPONENT_MAP = {
    "engine": [f for f in engine_feats if f in df.columns],
    "battery": [f for f in battery_feats if f in df.columns],
    "brake": [f for f in brake_feats if f in df.columns],
}

# ----------------- TRAINING TEST DATA SPLIT -------------------
def chronological_split(df_model, feature_cols, label_col, train_frac: float = 0.8):
    """Chronological split per vehicle."""
    train_parts, test_parts = [], []
    for _, g in df_model.groupby("vehicle_id"):
        g = g.sort_values("timestamp")
        cutoff_idx = int(len(g) * train_frac)
        train_parts.append(g.iloc[:cutoff_idx])
        test_parts.append(g.iloc[cutoff_idx:])
    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    cutoff_ts = pd.to_datetime(train_df["timestamp"], errors="coerce").max()
    print(f"Chronological split done (train up to {cutoff_ts.date()})")
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    X_train, X_test = train_df[feature_cols], test_df[feature_cols]
    Y_train, Y_test = train_df[label_col], test_df[label_col]
    return cutoff_ts, X_train, X_test, Y_train, Y_test

# ----------------- CLASSIFICATION (Binary - Fail in 7 Days) -------------------
print("\n=== STEP 8: Training RandomForest Binary Classifiers ===")

for comp, feats in COMPONENT_MAP.items():
    feats = [f for f in feats if f not in ["anomaly_score", "is_anomaly"]]
    label_col = f"{comp}_fail_in_next_{HORIZON}d"
    if label_col not in df.columns or not feats:
        print(f" Skipping {comp.upper()} — missing label or no features.")
        continue

    print(f"\nTraining {comp.upper()} model ({len(feats)} features)...")
    df_sub = df.dropna(subset=[label_col]).copy()
    df_sub[label_col] = df_sub[label_col].astype(int)

    _, X_train, X_test, Y_train, Y_test = chronological_split(df_sub, feats, [label_col])

    model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_estimators=400,
        max_depth=None,
        n_jobs=-1
    )
    model.fit(X_train, Y_train.values.ravel())

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    roc = roc_auc_score(Y_test, y_prob)
    pr = average_precision_score(Y_test, y_prob)
    cm = confusion_matrix(Y_test, y_pred, labels=[0, 1]).tolist()

    print(f"{comp.upper()} → ROC-AUC: {roc:.3f}, PR-AUC: {pr:.3f}")
    print("Confusion Matrix:", cm)

    joblib.dump(model, os.path.join(MODEL_DIR, f"{comp}_rf_binary_model.joblib"))
    with open(os.path.join(MODEL_DIR, f"{comp}_features.json"), "w") as f:
        json.dump(feats, f, indent=2)

print(" All binary classifiers trained and saved.")

# ---------------- RUL REGRESSION (XGBoost) ----------------------------
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("\n=== STEP 9: Training LightGBM RUL Regressors ===")

for comp in COMPONENTS:
    label_col = f"{comp}_RUL_days"
    feats = COMPONENT_MAP.get(comp, [])
    
    if label_col not in df.columns or not feats:
        print(f" Skipping {comp.upper()} — missing RUL column or no features.")
        continue

    print(f"\nTraining {comp.upper()} RUL model ({len(feats)} features)...")

    # --- Prepare data ---
    df_reg = df.copy()
    df_reg[label_col] = pd.to_numeric(df_reg[label_col], errors="coerce")
    df_reg = df_reg.replace([np.inf, -np.inf], np.nan).dropna(subset=[label_col])
    df_reg[label_col] = df_reg[label_col].clip(upper=RUL_CAP)

    # --- Chronological Split ---
    _, X_train, X_test, Y_train, Y_test = chronological_split(df_reg, feats, [label_col])

    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test  = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
    Y_train = Y_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    Y_test  = Y_test.replace([np.inf, -np.inf], np.nan).fillna(0)

    # --- Train LightGBM Regressor ---
    lgb_reg = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=10,
        reg_lambda=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        silent=True
    )

    lgb_reg.fit(
        X_train, 
        Y_train.values.ravel(),
        eval_set=[(X_test, Y_test.values.ravel())],
        eval_metric="rmse"
    )

    # --- Evaluate Performance ---
    Y_pred = lgb_reg.predict(X_test)
    mae = mean_absolute_error(Y_test, Y_pred)
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))

    print(f"{comp.upper()} → MAE: {mae:.3f}, RMSE: {rmse:.3f}")

    # --- Save Model (same filename pattern for compatibility) ---
    joblib.dump(lgb_reg, os.path.join(MODEL_DIR, f"{comp}_xgb_rul_model.joblib"))

print(" RUL models trained and saved.")


# ---------------- SINGLE-ROW PREDICTION + LLM EXPLAINABILITY ---------------
def predict_single_day(date_str: str, vehicle_id: str = "VEH0000"):
    df_sorted = df.sort_values(["vehicle_id", "timestamp"]).reset_index(drop=True)
    df_sorted["timestamp"] = pd.to_datetime(df_sorted["timestamp"])
    vehicle_id = vehicle_id.strip().upper()

    # ---------------- Load models and features ----------------
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

    # ---------------- Select target row ----------------
    target_date = pd.to_datetime(date_str)
    row = df_sorted[
        (df_sorted["vehicle_id"] == vehicle_id)
        & (df_sorted["timestamp"].dt.normalize() == target_date.normalize())
    ]
    if row.empty:
        raise ValueError(f"No record found for vehicle {vehicle_id} on {date_str}")

    pos_idx = row.index[0]

    # ---------------- Perform predictions ----------------
    cls_preds, rul_preds = {}, {}
    for comp in COMPONENTS:
        feats = features_map.get(comp, [])
        if not feats:
            continue

        # Ensure all features exist
        for f in feats:
            if f not in row.columns:
                row[f] = 0.0

        X = row[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Classification: fail_in_next_7d
        if comp in clf_models:
            cls_preds[comp] = int(clf_models[comp].predict(X)[0])

        # Regression: Remaining Useful Life
        if comp in reg_models:
            raw_pred = float(reg_models[comp].predict(X)[0])
            rul_int = max(0, int(round(raw_pred)))  # round to nearest day
            rul_preds[comp] = rul_int

    # ---------------- Expected failure dates ----------------
    expected_failures = {
        comp: str((target_date + pd.Timedelta(days=int(rul))).date()) if rul > 0 else None
        for comp, rul in rul_preds.items()
    }

    # ---------------- Ground truth labels ----------------
    true_labels = {}
    for comp in COMPONENTS:
        fail_col = f"{comp}_fail_in_next_{HORIZON}d"
        rul_col = f"{comp}_RUL_days"

        fail_val = row[fail_col].iloc[0] if fail_col in row else np.nan
        rul_val = row[rul_col].iloc[0] if rul_col in row else np.nan

        true_labels[fail_col] = None if pd.isna(fail_val) else int(fail_val)
        true_labels[rul_col] = None if pd.isna(rul_val) else float(rul_val)

    # ---------------- Final result ----------------
    results = {
        "vehicle_id": vehicle_id,
        "row_index_for_vehicle": int(pos_idx),
        "timestamp": str(row["timestamp"].values[0]),
        "pred_fail_in_next_7d": cls_preds,
        "pred_rul_days": rul_preds,
        "expected_failure_date": expected_failures,
        "true_labels": true_labels,
    }

    return results, row, features_map


print("\n=== LLM-BASED EXPLAINABILITY NARRATIVES ===")
try:
    results, row, features_map = predict_single_day("2024-01-05", vehicle_id="VEH0000")
    print(json.dumps(results, indent=2))

    explanations = {}
    for comp in COMPONENTS:
        model_path = os.path.join(MODEL_DIR, f"{comp}_rf_binary_model.joblib")
        feats = features_map.get(comp, [])
        if os.path.exists(model_path) and feats:
            shap_df, narrative = explain_prediction(comp, model_path, feats, row)
            explanations[comp] = narrative
            print(f"\n[{comp.upper()}]\n{narrative}\n")

    explain_path = os.path.join(OUT_DIR, "llm_explainability_summary.json")
    with open(explain_path, "w") as f:
        json.dump(explanations, f, indent=2)
    print(f" LLM explanations saved to: {explain_path}")

except Exception as e:
    print(" Explainability step failed:", e)
