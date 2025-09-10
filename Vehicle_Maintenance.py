

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Optional
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder  

# Path setup: where to read input and write outputs
CSV_PATH = "/Users/gowthamsivaraman/Desktop/Vehicle Maintenance Project/synthetic_telemetry_data.csv"
OUT_DIR = "/Users/gowthamsivaraman/Desktop/Vehicle Maintenance Project/output"
FIG_DIR = os.path.join(OUT_DIR, "figures")
ART_DIR = os.path.join(OUT_DIR, "artifacts")

for d in [OUT_DIR, FIG_DIR, ART_DIR]:
    os.makedirs(d, exist_ok=True)

# Load & SYNTHESIZE a clean daily timestamp per vehicle
df = pd.read_csv(CSV_PATH)


if "timestamp" in df.columns:
    df = df.drop(columns=["timestamp"])
df = df.sort_values(["vehicle_id"]).copy()
df["day_idx"] = df.groupby("vehicle_id").cumcount()
df["timestamp"] = pd.Timestamp("2024-01-01") + pd.to_timedelta(df["day_idx"], unit="D")
df = df.drop(columns=["day_idx"])

for c in ["harsh_brakes", "harsh_accels"]:
    if c in df.columns:
        df[c] = df[c].fillna(0).astype(int)

#Create labels from failure_type using REAL calendar days
#        * if failure_type == "No failure" → no failure on that row
#        * else → failure on that row
from typing import Optional

def create_labels(
    input_df: pd.DataFrame,
    horizon_days: int = 6,            
    include_today: bool = True,       
    failure_type_filter: Optional[str] = None 
) -> pd.DataFrame:
    df_tmp = input_df.copy()

    ft = df_tmp.get("failure_type", pd.Series(["No failure"] * len(df_tmp), index=df_tmp.index))
    ft = ft.fillna("No failure").astype(str).str.strip()
    if failure_type_filter is None:
        df_tmp["failed"] = (ft.str.casefold() != "no failure".casefold()).astype(int)
    else:
        df_tmp["failed"] = (ft.str.casefold() == failure_type_filter.casefold()).astype(int)
    if "timestamp" not in df_tmp.columns:
        raise ValueError("timestamp column is required to compute time-based labels.")
    df_tmp["timestamp"] = pd.to_datetime(df_tmp["timestamp"], errors="coerce")
    df_tmp["date_norm"] = df_tmp["timestamp"].dt.normalize()

    def per_vehicle(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("timestamp").copy()

        
        fail_dates = (
            g.loc[g["failed"] == 1, "date_norm"]
             .dropna()
             .drop_duplicates()
             .values
        )

        cur_dates = g["date_norm"].values
        rul_days_full = np.full(len(g), np.nan, dtype=float)

        if fail_dates.size > 0:
            idxs = np.searchsorted(fail_dates, cur_dates, side="left")
            has_next = idxs < len(fail_dates)

            next_fail_dates = np.empty(len(g), dtype="datetime64[ns]")
            next_fail_dates[:] = np.datetime64("NaT")
            next_fail_dates[has_next] = fail_dates[idxs[has_next]]

            day_gap = (next_fail_dates - cur_dates) / np.timedelta64(1, "D")
            rul_days_full = day_gap.astype(float)  

        if include_today:
            mask_keep = (rul_days_full >= 0) & (rul_days_full <= horizon_days)
        else:
            mask_keep = (rul_days_full > 0) & (rul_days_full <= horizon_days)

        rul_windowed = np.where(mask_keep, rul_days_full, np.nan)

        g["RUL_days"] = rul_windowed
        g["fail_in_next_7d"] = mask_keep.astype(int) 

        g = g.drop(columns=["date_norm"])
        return g

    return df_tmp.groupby("vehicle_id", group_keys=False).apply(per_vehicle)

df_labeled = create_labels(df, horizon_days=7, include_today=True, failure_type_filter=None)

#  Feature engineering
#    - Rolling means & diffs for time-series signals
#    - Issue/behavior rollups
#    - Vehicle age, odometer deltas
def add_features(input_df: pd.DataFrame) -> pd.DataFrame:
    df_tmp = input_df.copy()
    roll_candidates = [
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
    roll_cols = [c for c in roll_candidates if c in df_tmp.columns]

    flag_cols = [c for c in [
        "engine_failure_imminent", "brake_issue_imminent", "battery_issue_imminent",
        "abs_fault_indicator" 
    ] if c in df_tmp.columns]

    def per_vehicle(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("timestamp").copy()

        for c in roll_cols:
            g[f"{c}_roll3_mean"] = g[c].rolling(window=3, min_periods=1).mean()
            g[f"{c}_roll7_mean"] = g[c].rolling(window=7, min_periods=1).mean()
            g[f"{c}_diff1"] = g[c].diff(1)

        for c in flag_cols:
            g[f"{c}_roll7_sum"] = g[c].rolling(7, min_periods=1).sum()
            g[f"{c}_roll14_sum"] = g[c].rolling(14, min_periods=1).sum()

        if "model_year" in g.columns:
            g["vehicle_age"] = g["timestamp"].dt.year - g["model_year"]

        if "odometer_reading" in g.columns:
            g["odometer_delta"] = g["odometer_reading"].diff(1)

        return g

    return df_tmp.groupby("vehicle_id", group_keys=False).apply(per_vehicle)

df_feat = add_features(df_labeled)

df_feat.to_csv(os.path.join(ART_DIR, "engineered_features_full.csv"), index=False)


le = LabelEncoder()
if "brand" in df_feat.columns:
    df_feat["brand_enc"] = le.fit_transform(df_feat["brand"].astype(str))

# Build train/test split by time (chronological split)
exclude_cols = {
    "timestamp", "vehicle_id",
    "failed", "RUL_days", "fail_in_next_7d",
    "failure_type", "brand",
    "failure_date", "date_norm", "next_fail_date"
}

candidate_cols = [c for c in df_feat.columns if c not in exclude_cols]

feature_cols = (
    df_feat[candidate_cols]
    .select_dtypes(include=[np.number])
    .columns
    .tolist()
)

df_model = df_feat.dropna(subset=feature_cols).copy()

all_ts = np.sort(df_model["timestamp"].unique())
cut_idx = int(0.8 * len(all_ts)) if len(all_ts) else 0
cutoff_ts = all_ts[cut_idx] if len(all_ts) else pd.Timestamp("2099-01-01")

train_df = df_model[df_model["timestamp"] <= cutoff_ts]
test_df  = df_model[df_model["timestamp"] >  cutoff_ts]

X_train, y_train_cls = train_df[feature_cols], train_df["fail_in_next_7d"].astype(int)
X_test,  y_test_cls  = test_df[feature_cols],  test_df["fail_in_next_7d"].astype(int)

# Classification model: Predict failure in next 7 days (inclusive 0..7)

clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    n_jobs=-1,
    random_state=7,
    class_weight="balanced_subsample"
)
clf.fit(X_train, y_train_cls)
y_pred_cls = clf.predict(X_test)

cls_report = classification_report(y_test_cls, y_pred_cls, output_dict=True, zero_division=0)
cls_cm = confusion_matrix(y_test_cls, y_pred_cls)

with open(os.path.join(ART_DIR, "classification_report.json"), "w") as f:
    json.dump(cls_report, f, indent=2)
np.savetxt(os.path.join(ART_DIR, "classification_confusion_matrix.csv"), cls_cm, fmt="%d", delimiter=",")

print("Confusion Matrix:")
print("TN   FP")
print("FN   TP")
print(cls_cm)

importances_cls = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)[:25]*100
plt.figure()
importances_cls.iloc[::-1].plot(kind="barh")
plt.title("Top 25 Feature Importances — Classification")
plt.xlabel("Importance (%)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "classification_feature_importances.png"), dpi=150)
plt.close()

# Regression model: Predict RUL (capped at 60 days)

df_model["RUL_days_capped"] = df_model["RUL_days"].clip(upper=60)

train_df_r = df_model[df_model["timestamp"] <= cutoff_ts]
test_df_r  = df_model[df_model["timestamp"] >  cutoff_ts]

X_train_r = train_df_r[feature_cols]
y_train_r = train_df_r["RUL_days_capped"].fillna(60)

X_test_r  = test_df_r[feature_cols]
y_test_r  = test_df_r["RUL_days_capped"].fillna(60)


if len(X_train_r) == 0 or len(X_test_r) == 0:
    print("[WARN] Regression split has empty train or test set. Skipping RUL model training.")
else:
    reg = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=7
    )
    reg.fit(X_train_r, y_train_r)
    y_pred_r = reg.predict(X_test_r)

    reg_mae = float(mean_absolute_error(y_test_r, y_pred_r))
    reg_mse = float(mean_squared_error(y_test_r, y_pred_r))
    reg_rmse = float(np.sqrt(reg_mse))

    print("RUL Regression Metrics:")
    print(f"  MAE : {reg_mae:.4f}")
    print(f"  RMSE: {reg_rmse:.4f}")

    with open(os.path.join(ART_DIR, "regression_metrics.json"), "w") as f:
        json.dump({"MAE": reg_mae, "RMSE": reg_rmse}, f, indent=2)

    importances_reg = (
        pd.Series(reg.feature_importances_, index=feature_cols)
          .sort_values(ascending=False)[:25] * 100.0
    )

    plt.figure()
    importances_reg.iloc[::-1].plot(kind="barh")
    plt.title("Top 25 Feature Importances — Regression (RUL)")
    plt.xlabel("Importance (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "regression_feature_importances.png"), dpi=150)
    plt.close()

# “Real-time” scoring demo: last 30 rows of one vehicle’s series
if not df_model.empty and "reg" in locals():
    sample_vid = df_model["vehicle_id"].value_counts().index[0]  
    vdf = (
        df_model[df_model["vehicle_id"] == sample_vid]
        .sort_values("timestamp")
        .dropna(subset=feature_cols)
        .copy()
    )
    tail = vdf.tail(30).copy() 

    rows = []
    for _, row in tail.iterrows():
        
        x_df = row[feature_cols].to_frame().T       
        p_fail = float(clf.predict_proba(x_df)[0, 1])  
        est_rul = float(reg.predict(x_df)[0])         

        rows.append({
            "timestamp": row["timestamp"],
            "vehicle_id": sample_vid,
            "p_fail_next_7d": p_fail,
            "est_RUL_days": est_rul
        })

    scores_df = pd.DataFrame(rows)
    scores_df.to_csv(os.path.join(ART_DIR, f"realtime_scores_{sample_vid}.csv"), index=False)

    # Plots
    plt.figure()
    plt.plot(scores_df["timestamp"], scores_df["p_fail_next_7d"])
    plt.title(f"Failure Risk (next 7d) — {sample_vid}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"realtime_risk_{sample_vid}.png"), dpi=150)
    plt.close()

    plt.figure()
    plt.plot(scores_df["timestamp"], scores_df["est_RUL_days"])
    plt.title(f"Estimated RUL (days) — {sample_vid}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"realtime_rul_{sample_vid}.png"), dpi=150)
    plt.close()

# Artifact index: JSON pointer to everything produced
index = {
    "source_dataset": os.path.abspath(CSV_PATH),
    "cutoff_timestamp": str(cutoff_ts),
    "artifacts": {
        "classification_report": os.path.join(ART_DIR, "classification_report.json"),
        "classification_confusion_matrix": os.path.join(ART_DIR, "classification_confusion_matrix.csv"),
        "regression_metrics": os.path.join(ART_DIR, "regression_metrics.json"),
        "engineered_features_full": os.path.join(ART_DIR, "engineered_features_full.csv"),
        "realtime_scores_csv": os.path.join(ART_DIR, f"realtime_scores_{sample_vid}.csv") if 'scores_df' in locals() else None
    },
    "figures": {
        "classification_feature_importances": os.path.join(FIG_DIR, "classification_feature_importances.png"),
        "regression_feature_importances": os.path.join(FIG_DIR, "regression_feature_importances.png"),
        "realtime_risk": os.path.join(FIG_DIR, f"realtime_risk_{sample_vid}.png") if 'scores_df' in locals() else None,
        "realtime_rul": os.path.join(FIG_DIR, f"realtime_rul_{sample_vid}.png") if 'scores_df' in locals() else None
    }
}
with open(os.path.join(OUT_DIR, "ARTIFACTS_INDEX.json"), "w") as f:
    json.dump(index, f, indent=2)

print("Done. Artifacts saved under:", os.path.abspath(OUT_DIR))
print("Classification & regression feature importances plotted.")
print("Sample per-vehicle risk/RUL time series generated.")

