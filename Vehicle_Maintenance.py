"""
Predictive Maintenance — 7-Day 
- Classification: engine/battery/brake fail in next 7d (3 models) — RandomForest
- Regression: engine/battery/brake RUL days (3 models) — XGBoost
"""

import os
import re
import json
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import (classification_report,confusion_matrix,roc_auc_score,average_precision_score,mean_absolute_error,mean_squared_error)

CSV_PATH= os.environ.get("CSV_PATH","/Users/gowthamsivaraman/Desktop/Vehicle Maintenance Project/synthetic_telemetry_data.csv")
OUT_DIR = os.environ.get("OUT_DIR","/Users/gowthamsivaraman/Desktop/Vehicle Maintenance Project/output_simple",)
FIG_DIR = os.path.join(OUT_DIR, "figures")
ART_DIR = os.path.join(OUT_DIR, "artifacts")
for d in [OUT_DIR, FIG_DIR, ART_DIR]:
    os.makedirs(d, exist_ok=True)

COMPONENTS = ["engine", "battery", "brake"]
HORIZON = 7
RUL_CAP = 7
RANDOM_STATE = 42

# Configuring Timestamp and Failure Date 
def regenerate_daily_timestamps(df):
    d = df.copy()
    if "timestamp" in d.columns:
        d = d.drop(columns=["timestamp"])
    d["day_idx"] = d.groupby("vehicle_id").cumcount()
    d["timestamp"] = pd.Timestamp("2024-01-01") + pd.to_timedelta(d["day_idx"], unit="D")
    d.drop(columns=["day_idx"], inplace=True)
    return d

def rebuild_failure_date_column(d):
    df = d.copy()
    ft = (df.get("failure_type").astype(str).str.lower().str.strip().str.replace(r"\s+", " ", regex=True))

    is_engine = ft.str.contains(r"\bengine_failed\b", regex=True)
    is_brake  = ft.str.contains(r"\bbrake_failed\b", regex=True)
    is_batt   = ft.str.contains(r"\bbattery_failed\b", regex=True)
    is_no     = ft.str.fullmatch(r"no failure", na=False)

    df["failure_date"] = "NA"
    if "timestamp" in df.columns:
        df.loc[(is_engine | is_brake | is_batt), "failure_date"] = pd.to_datetime(
            df.loc[(is_engine | is_brake | is_batt), "timestamp"]).dt.strftime("%Y-%m-%d")
    df.loc[is_no, "failure_date"] = "NA"
    return df

# Preprocessing

def daily_timestamps(df):
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce") #timestamp column is converted to pandas datetime64 type.
    return d

def build_component_labels(input_df, component, horizon_days, include_today,):
    """
    Adds target features: <component>_RUL_days, <component>_fail_in_next_{horizon_days}d
    """
    df_tmp = daily_timestamps(input_df)

    # If any of these columns exist, convert them to int (0/1).
    for c in ["abs_fault_indicator", "engine_failure_imminent", "brake_issue_imminent", "battery_issue_imminent"]:
        if c in df_tmp.columns:
            df_tmp[c] = pd.to_numeric(df_tmp[c], errors="coerce").fillna(0).astype(int)

    token = f"{component}_failed"
    df_tmp[f"{component}_failed"] = (
        df_tmp.get("failure_type")
              .astype(str).str.lower().str.strip().str.contains(fr"\b{re.escape(token)}\b", regex=True, na=False).astype(int))

    #If a failure_date column exists, convert it to datetime (normalized to midnight)
    
    fail_date_col = pd.to_datetime(df_tmp["failure_date"], errors="coerce").dt.normalize()
    event_date = pd.to_datetime(df_tmp["timestamp"], errors="coerce").dt.normalize()
    df_tmp["_comp_event_date"] = pd.to_datetime(event_date) # actual failure date to use later when calculating (RUL).
    df_tmp["date_norm"] = pd.to_datetime(df_tmp["timestamp"]).dt.normalize()

    def per_vehicle(g):
        fail_dates = (g.loc[g[f"{component}_failed"] == 1, "_comp_event_date"].sort_values().values)
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
        mask_keep = ((rul_days_full >= 0) & (rul_days_full <= horizon_days)) if include_today \
                    else ((rul_days_full > 0)  & (rul_days_full <= horizon_days))
        g[f"{component}_RUL_days"] = rul_days_full
        g[f"{component}_fail_in_next_{horizon_days}d"] = mask_keep.astype(int)
        return g

    out = df_tmp.groupby("vehicle_id", group_keys=False).apply(per_vehicle)
    return out.drop(columns=[
    c for c in ["date_norm", "_comp_event_date"] if c in out.columns]) #Removes columns not needed for training.

# Columns never to use as features
EXCLUDE_ALWAYS = {"timestamp", "vehicle_id", "failure_type", "failure_date","brand", "date_norm", "next_fail_date"}

def select_features(df_feat,extra_keep):
    exclude_cols = set(EXCLUDE_ALWAYS)
    leakage_cols = []
    for c in df_feat.columns:
        s = str(c)
        if s.endswith("_RUL_days"):
            leakage_cols.append(s)
        if "_fail_in_next_" in s and s.endswith("d"):
            leakage_cols.append(s)
        if s in ("engine_failed", "battery_failed", "brake_failed"):
            leakage_cols.append(s)
    if extra_keep: 
        leakage_cols.extend(extra_keep)
    exclude_cols.update(leakage_cols)

    candidate_cols = [c for c in df_feat.columns if c not in exclude_cols]
    feature_cols = df_feat[candidate_cols].select_dtypes(include=[np.number]).columns.tolist()
    return sorted(set(feature_cols))

def chronological_split(df_model, feature_cols, label_col, train_frac: float = 0.8):
    
    train_parts, test_parts = [], []
    for vid, g in df_model.groupby("vehicle_id"):
        g = g.sort_values("timestamp")
        cutoff_idx = int(len(g) * train_frac)
        train_parts.append(g.iloc[:cutoff_idx])
        test_parts.append(g.iloc[cutoff_idx:])
    train_df = pd.concat(train_parts, ignore_index=True)
    test_df  = pd.concat(test_parts,  ignore_index=True)
    cutoff_ts = pd.to_datetime(train_df["timestamp"], errors="coerce").max()

    return (cutoff_ts,train_df[feature_cols],test_df[feature_cols],train_df[label_col],test_df[label_col])

# ------------------ Optimized Feature Engineering ----------------------

def add_features(input_df):
    df = input_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Flag columns that has 0/1
    FLAG_COLS = ["engine_failure_imminent", "brake_issue_imminent", "battery_issue_imminent", "abs_fault_indicator",
                 "harsh_brakes", "harsh_accels"]
    for c in [c for c in FLAG_COLS if c in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    BASE = ["engine_temp_c","engine_rpm","oil_pressure_psi","coolant_temp_c", "fuel_level_percent","fuel_consumption_lph","engine_load_percent", "throttle_pos_percent","air_flow_rate_gps","exhaust_gas_temp_c", "vibration_level","engine_hours", "brake_fluid_level_psi","brake_pad_wear_mm","brake_temp_c","brake_pedal_pos_percent", "wheel_speed_fl_kph","wheel_speed_fr_kph","wheel_speed_rl_kph","wheel_speed_rr_kph","vehicle_speed_kph", "battery_voltage_v","battery_current_a","battery_temp_c", "alternator_output_v", "battery_charge_percent","battery_health_percent", "ambient_temp_c","humidity_percent","odometer_reading"]
    cols = [c for c in BASE if c in df.columns]

    def rolling_slope(s, window):
        #Slope per day from a quick linear fit over each rolling window.
        x = np.arange(window, dtype=float)
        x_mean = x.mean()
        denom = ((x - x_mean) ** 2).sum()

        def _fit(y):
            if np.isnan(y).any() or denom == 0:
                if np.isnan(y).all():
                    return np.nan
            # slope = cov(x,y)/var(x)
            y = y.astype(float)
            y_mean = y.mean()
            num = ((x - x_mean) * (y - y_mean)).sum()
            return num / denom if denom != 0 else np.nan

        return s.rolling(window, min_periods=window).apply(_fit, raw=True)

    def days_since(cond):
        """Days since condition was True (per row)."""
        curr_idx = np.arange(len(cond))
        last = np.where(cond.values, curr_idx, np.nan)
        last = pd.Series(last).ffill().values
        out = np.where(np.isnan(last), np.nan, curr_idx - last)
        return pd.Series(out, index=cond.index)

    def per_vehicle(g):
        g = g.sort_values("timestamp").copy()
        blocks = []

        if cols:
            r = g[cols]
            # Rolling stats
            blocks.append(r.rolling(7,  min_periods=1).mean().add_suffix("_roll7_mean"))
            blocks.append(r.rolling(14, min_periods=1).mean().add_suffix("_roll14_mean"))
            blocks.append(r.rolling(7,  min_periods=2).std().add_suffix("_roll7_std"))
            blocks.append(r.rolling(14, min_periods=2).std().add_suffix("_roll14_std"))

            # Deltas
            d1  = r.diff(1).add_suffix("_diff1")
            d7  = r.diff(7).add_suffix("_diff7")
            pc1 = r.pct_change(1).add_suffix("_pct1")
            pc7 = r.pct_change(7).add_suffix("_pct7")
            blocks += [d1, d7, pc1, pc7]

            # slopes
            for w in (7, 14):
                slopes = r.apply(lambda s: rolling_slope(s, w))
                blocks.append(slopes.add_suffix(f"_slope{w}"))

            # cumulative negative deltas
            neg_d1 = r.diff(1).clip(upper=0).cumsum().add_suffix("_cum_negdiff1")
            blocks.append(neg_d1)

            # Days since last drop/spike 
            drop = r.diff(1) < 0
            spike = r.diff(1) > 0
            ds_drop  = drop.apply(days_since).add_suffix("_days_since_drop")
            ds_spike = spike.apply(days_since).add_suffix("_days_since_spike")
            blocks += [ds_drop, ds_spike]

        # Fault flags cumulative counts
        fcols = [c for c in FLAG_COLS if c in g.columns]
        if fcols:
            f = g[fcols]
            blocks += [
                f.cumsum().add_suffix("_cum"),    # Creates cumulative counts and rolling sums over 7 and 14 days.
                f.rolling(7,  min_periods=1).sum().add_suffix("_sum7"),
                f.rolling(14, min_periods=1).sum().add_suffix("_sum14"),
            ]

        extra = []
        if "odometer_reading" in g.columns:
            extra.append((g["odometer_reading"].diff(1).rename("odometer_delta")).to_frame())

        if blocks or extra:
            g = pd.concat([g] + blocks + extra, axis=1)
        return g

    return df.groupby("vehicle_id", group_keys=False).apply(per_vehicle)


# Load Data 

df_raw = pd.read_csv(CSV_PATH)
df_raw = regenerate_daily_timestamps(df_raw)
df_raw = rebuild_failure_date_column(df_raw)


# Build Labels & Features

# Build labels for 7d horizon only (also creates RUL columns)
df_labeled_all = df_raw.copy()
for comp in COMPONENTS:
    df_labeled_all = build_component_labels(df_labeled_all, comp, horizon_days=HORIZON, include_today=True)

labeled_csv = os.path.join(OUT_DIR, "telemetry_labeled_7d.csv") # Save labeled CSV
df_labeled_all.to_csv(labeled_csv, index=False)

df_feat = add_features(df_labeled_all)
df_feat = df_feat.replace([np.inf, -np.inf], np.nan)

features_csv = os.path.join(OUT_DIR, "telemetry_labeled_with_features_7d.csv") # Save features CSV
df_feat.to_csv(features_csv, index=False)

# Classification (3 models, 7d) 
for component in COMPONENTS:
    label_col = f"{component}_fail_in_next_{HORIZON}d"
    df_tmp = df_feat

    df_local = df_tmp.dropna(subset=[label_col]).copy()
    df_local[label_col] = df_local[label_col].astype(int)

    feature_cols = select_features(df_local, extra_keep=[label_col])
    df_local[feature_cols] = df_local[feature_cols].replace([np.inf, -np.inf], np.nan).clip(-1e6, 1e6)

    df_model = df_local.dropna(subset=feature_cols).copy()
    
    cutoff_ts, X_train, X_test, y_train, y_test = chronological_split(df_model, feature_cols, label_col)

    ros = RandomOverSampler(sampling_strategy='auto', random_state=RANDOM_STATE)
    X_train_os, y_train_os = ros.fit_resample(X_train, y_train)

    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train_os, y_train_os)
    
    # positive-class probability (if single-class training)
    proba = clf.predict_proba(X_test)
    classes = list(clf.classes_)
    pos_idx = classes.index(1) if 1 in classes else 1

    y_prob = proba[:, pos_idx]
    y_pred = (y_prob >= 0.5).astype(int)

    roc = roc_auc_score(y_test, y_prob) 
    pr  = average_precision_score(y_test, y_prob)
    metrics = {"roc_auc": float(roc),
               "pr_auc":  float(pr)}

    cls_rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()

    print(f"{component.upper()} | {HORIZON}d Failure Prediction")
    print("ROC-AUC:", metrics["roc_auc"], " PR-AUC:", metrics["pr_auc"])
    print("Confusion Matrix:", cm)

    importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    top20 = importances.head(20) * 100.0
    plt.figure(); top20.iloc[::-1].plot(kind="barh")
    plt.title(f"{component.upper()} — Top 20 Feature Importances ({HORIZON}d)")
    plt.xlabel("Importance (%)"); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{component}_clf_featimp_{HORIZON}d.png"), dpi=150)
    plt.close()

    with open(os.path.join(ART_DIR, f"{component}_cls_report_{HORIZON}d.json"), "w") as f:
        json.dump(cls_rep, f, indent=2)
    with open(os.path.join(ART_DIR, f"{component}_metrics_{HORIZON}d.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(ART_DIR, f"{component}_cm_{HORIZON}d.json"), "w") as f:
        json.dump(cm, f, indent=2)

# Regression (XGBoost) 

for component in COMPONENTS:
    label_col = f"{component}_RUL_days"
    df_local = df_feat.copy()

    # Keep rows that either fail within 7 days OR have no future failure (NaN)
    mask_keep = df_local[label_col].isna() | (df_local[label_col] <= RUL_CAP)
    df_local = df_local.loc[mask_keep].copy()
    df_local[label_col] = df_local[label_col].clip(upper=RUL_CAP).fillna(RUL_CAP)

    feature_cols = select_features(df_local, extra_keep=[label_col])
    df_model = df_local.dropna(subset=feature_cols).copy()

    cutoff_ts, X_train, X_test, y_train, y_test = chronological_split(df_model, feature_cols, label_col)

    reg = XGBRegressor(n_estimators=800,learning_rate=0.05,max_depth=8,subsample=0.9,colsample_bytree=0.8, reg_lambda=1.0,
    reg_alpha=0.0, random_state=RANDOM_STATE, n_jobs=-1, eval_metric="rmse")

    reg.fit(X_train, y_train,eval_set=[(X_test, y_test)],verbose=False)

    y_pred = reg.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    print(f"=== {component.upper()} — RUL Regression (XGBoost) ===")
    print("MAE:", round(mae, 3), " RMSE:", round(rmse, 3))

    importances = pd.Series(reg.feature_importances_, index=feature_cols).sort_values(ascending=False)
    top20 = importances.head(20) * 100.0
    plt.figure(); top20.iloc[::-1].plot(kind="barh")
    plt.title(f"{component.upper()} — Top 20 Feature Importances (RUL, XGBoost)")
    plt.xlabel("Importance (%)"); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{component}_rul_featimp_xgb.png"), dpi=150)
    plt.close()


# Artifact Index 

index = {
    "source_dataset": os.path.abspath(CSV_PATH),
    "components": COMPONENTS,
    "horizon": HORIZON,
    "labeled_csv": os.path.join(OUT_DIR, "telemetry_labeled_7d.csv"),
    "features_csv": os.path.join(OUT_DIR, "telemetry_labeled_with_features_7d.csv"),
}
with open(os.path.join(OUT_DIR, "ARTIFACTS_INDEX.json"), "w") as f:
    json.dump(index, f, indent=2)

print("\nDone. Trained 3 classification models (7d) and 3 XGBoost regression models.")
print("Artifacts under:", os.path.abspath(OUT_DIR))
print("Saved labeled data:", index['labeled_csv'])
print("Saved labeled + features:", index['features_csv'])
