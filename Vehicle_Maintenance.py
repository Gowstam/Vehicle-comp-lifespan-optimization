"""
Predictive Maintenance — 7-Day
- classification model (multi-label): predicts fail_in_next_7d for engine, battery, brake
- regression model (multi-output): predicts RUL_days for engine, battery, brake (XGBoost)
- Keeps your preprocessing + feature engineering
"""

import os
import re
import json
from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, average_precision_score, mean_absolute_error, mean_squared_error)
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputClassifier
from imblearn.over_sampling import RandomOverSampler
import joblib

# Configuration 
CSV_PATH = os.environ.get("CSV_PATH","/Users/gowthamsivaraman/Desktop/Vehicle Maintenance Project/synthetic_telemetry_data.csv")
OUT_DIR = os.environ.get("OUT_DIR", "/Users/gowthamsivaraman/Desktop/Vehicle Maintenance Project/output_simple")
FIG_DIR = os.path.join(OUT_DIR, "figures")
ART_DIR = os.path.join(OUT_DIR, "artifacts")
for d in [OUT_DIR, FIG_DIR, ART_DIR]:
    os.makedirs(d, exist_ok=True)

COMPONENTS = ["engine", "battery", "brake"]
HORIZON = 7
# For training target capping of RUL:
RUL_CAP = 60
RANDOM_STATE = 42

# Timestamp & Failure Date 
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
        df.loc[(is_engine | is_brake | is_batt), "failure_date"] = pd.to_datetime(df.loc[(is_engine | is_brake | is_batt), "timestamp"]).dt.strftime("%Y-%m-%d")
    df.loc[is_no, "failure_date"] = "NA"
    return df

# Preprocessing 

def daily_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    return d

def build_component_labels(input_df,component, horizon_days,include_today):
    """
    Adds: <component>_RUL_days, <component>_fail_in_next_{horizon_days}d
    """
    df_tmp = daily_timestamps(input_df)

    token = f"{component}_failed"
    df_tmp[f"{component}_failed"] = (df_tmp.get("failure_type").astype(str).str.lower().str.strip()
                                     .str.contains(fr"\b{re.escape(token)}\b", regex=True, na=False).astype(int))

    # event date = current row timestamp for failure rows
    event_date = pd.to_datetime(df_tmp["timestamp"], errors="coerce").dt.normalize()
    df_tmp["_comp_event_date"] = pd.to_datetime(event_date)
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

        if include_today:
            mask_keep = (rul_days_full >= 0) & (rul_days_full <= horizon_days)
        else:
            mask_keep = (rul_days_full > 0) & (rul_days_full <= horizon_days)

        g[f"{component}_RUL_days"] = rul_days_full
        g[f"{component}_fail_in_next_{horizon_days}d"] = mask_keep.astype(int)
        return g

    out = df_tmp.groupby("vehicle_id", group_keys=False).apply(per_vehicle)
    return out.drop(columns=[c for c in ["date_norm", "_comp_event_date"] if c in out.columns])

# Columns never to use as features
EXCLUDE_ALWAYS = {"timestamp", "vehicle_id", "failure_type", "failure_date","brand", "date_norm", "next_fail_date"}

def select_features(df_feat, extra_remove):
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
    if extra_remove:
        leakage_cols.extend(extra_remove)

    exclude_cols.update(leakage_cols)

    candidate_cols = [c for c in df_feat.columns if c not in exclude_cols]
    feature_cols = df_feat[candidate_cols].select_dtypes(include=[np.number]).columns.tolist()
    return sorted(set(feature_cols))

def chronological_split(df_model, feature_cols, label_cols, train_frac: float = 0.8):
    train_parts, test_parts = [], []
    for _, g in df_model.groupby("vehicle_id"):
        g = g.sort_values("timestamp")
        cutoff_idx = int(len(g) * train_frac)
        train_parts.append(g.iloc[:cutoff_idx])
        test_parts.append(g.iloc[cutoff_idx:])
    train_df = pd.concat(train_parts, ignore_index=True)
    test_df  = pd.concat(test_parts,  ignore_index=True)
    cutoff_ts = pd.to_datetime(train_df["timestamp"], errors="coerce").max()
    X_train = train_df[feature_cols]
    X_test  = test_df[feature_cols]
    Y_train = train_df[label_cols]
    Y_test  = test_df[label_cols]
    return cutoff_ts, X_train, X_test, Y_train, Y_test

# Feature Engineering

def add_features(input_df):
    df = input_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    FLAG_COLS = ["engine_failure_imminent", "brake_issue_imminent", "battery_issue_imminent","abs_fault_indicator", "harsh_brakes", "harsh_accels"]
    for c in [c for c in FLAG_COLS if c in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    BASE = ["engine_temp_c","engine_rpm","oil_pressure_psi","coolant_temp_c","fuel_level_percent","fuel_consumption_lph","engine_load_percent",
        "throttle_pos_percent","air_flow_rate_gps","exhaust_gas_temp_c","vibration_level","engine_hours",
        "brake_fluid_level_psi","brake_pad_wear_mm","brake_temp_c","brake_pedal_pos_percent","wheel_speed_fl_kph","wheel_speed_fr_kph","wheel_speed_rl_kph","wheel_speed_rr_kph",
        "vehicle_speed_kph","battery_voltage_v","battery_current_a","battery_temp_c","alternator_output_v","battery_charge_percent","battery_health_percent","ambient_temp_c","humidity_percent","odometer_reading"]
    cols = [c for c in BASE if c in df.columns]

    def rolling_slope(s, window):
        x = np.arange(window, dtype=float)
        x_mean = x.mean()
        denom = ((x - x_mean) ** 2).sum()

        def _fit(y):
            if np.isnan(y).any() or denom == 0:
                if np.isnan(y).all():
                    return np.nan
            y = y.astype(float)
            y_mean = y.mean()
            num = ((x - x_mean) * (y - y_mean)).sum()
            return num / denom if denom != 0 else np.nan

        return s.rolling(window, min_periods=window).apply(_fit, raw=True)

    def days_since(cond):
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
        fcols = [c for c in ["engine_failure_imminent","brake_issue_imminent","battery_issue_imminent","abs_fault_indicator","harsh_brakes","harsh_accels"] if c in g.columns]
        if fcols:
            f = g[fcols]
            blocks += [
                f.cumsum().add_suffix("_cum"),
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

# Features Build 
df_labeled_all = df_raw.copy()
for comp in COMPONENTS:
    df_labeled_all = build_component_labels(df_labeled_all, comp, horizon_days=HORIZON, include_today=True)

labeled_csv = os.path.join(OUT_DIR, "telemetry_labeled_7d.csv")
df_labeled_all.to_csv(labeled_csv, index=False)

df_feat = add_features(df_labeled_all)
df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
print(df_feat.shape)


# Build Targets
# Classification labels (3 columns)
CLS_LABEL_COLS = [f"{c}_fail_in_next_{HORIZON}d" for c in COMPONENTS]
# Regression labels (3 columns)
RUL_LABEL_COLS = [f"{c}_RUL_days" for c in COMPONENTS]
for col in RUL_LABEL_COLS:
    if col in df_feat.columns:
        df_feat[col] = df_feat[col].fillna(0)
features_csv = os.path.join(OUT_DIR, "telemetry_labeled_with_features_7d.csv")
df_feat.to_csv(features_csv, index=False)

# Feature Selection 
all_labels = CLS_LABEL_COLS + RUL_LABEL_COLS
feature_cols = select_features(df_feat, extra_remove=all_labels)

df_feat[feature_cols] = (df_feat[feature_cols].replace([np.inf, -np.inf], np.nan).clip(-1e6, 1e6))

# Multi-label classifier 
df_cls = df_feat.dropna(subset=CLS_LABEL_COLS).copy()
for col in CLS_LABEL_COLS:
    df_cls[col] = df_cls[col].astype(int)

cutoff_ts, Xc_train, Xc_test, Yc_train, Yc_test = chronological_split(df_cls, feature_cols, CLS_LABEL_COLS)

eng = Yc_train.iloc[:, 0].astype(int).values  # engine_fail_in_next_7d
bat = Yc_train.iloc[:, 1].astype(int).values  # battery_fail_in_next_7d
brk = Yc_train.iloc[:, 2].astype(int).values  # brake_fail_in_next_7d
y_key = (eng << 2) | (bat << 1) | brk          # values 0,1..7 (000,001,010,..)

# Oversample the target
ros = RandomOverSampler(random_state=RANDOM_STATE)
Xc_train_os, y_key_os = ros.fit_resample(Xc_train, y_key)

# Unpack back into three binary labels after oversampling
eng_os = ((y_key_os >> 2) & 1).astype(int)
bat_os = ((y_key_os >> 1) & 1).astype(int)
brk_os = (y_key_os & 1).astype(int)
Yc_train_os = pd.DataFrame(np.column_stack([eng_os, bat_os, brk_os]), columns=CLS_LABEL_COLS)

# Fit a multi-label classifier (3 binary outputs)
base_rf = RandomForestClassifier(n_estimators=400, max_depth=None, class_weight="balanced_subsample",random_state=RANDOM_STATE,
    n_jobs=-1)
clf_multi = MultiOutputClassifier(base_rf, n_jobs=None)
clf_multi.fit(Xc_train_os, Yc_train_os)

Yc_prob_list = clf_multi.predict_proba(Xc_test)  # list of 3 arrays 
Yc_pred = clf_multi.predict(Xc_test)

print("\n===  Multi-label Classifier — 7d Failures (engine/battery/brake) ===")
per_comp_metrics: Dict[str, Dict[str, float]] = {}
for i, comp in enumerate(COMPONENTS):
    y_true = Yc_test.iloc[:, i].values
    y_pred = Yc_pred[:, i]
    y_prob = Yc_prob_list[i][:, 1]  # positive class prob

    # Metrics
    roc = roc_auc_score(y_true, y_prob)
    pr  = average_precision_score(y_true, y_prob)
    cm  = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    print(f"- {comp.upper()}")
    print("  ROC-AUC:", round(float(roc), 3), " PR-AUC:", round(float(pr), 3))
    print("  Confusion Matrix [ [TN, FP], [FN, TP] ]:", cm)
    per_comp_metrics[comp] = {
        "roc_auc": float(roc),
        "pr_auc":  float(pr),
        "cm": cm,
    }

# save model and feature list 
joblib.dump(clf_multi, os.path.join(ART_DIR, f"multi_label_rf_{HORIZON}d.joblib"))
with open(os.path.join(ART_DIR, f"multi_label_features_{HORIZON}d.json"), "w") as f:
    json.dump(feature_cols, f, indent=2)

# Regression
df_reg = df_feat.copy()
# Keep rows that either fail within 7 days OR have no future failure (NaN) for each component
mask_any = pd.Series(True, index=df_reg.index)
for col in RUL_LABEL_COLS:
    mask = df_reg[col].isna() | (df_reg[col] <= RUL_CAP)
    mask_any &= mask

df_reg = df_reg.loc[mask_any].copy()
for col in RUL_LABEL_COLS:
    df_reg[col] = df_reg[col].clip(upper=RUL_CAP).fillna(RUL_CAP)

cutoff_ts_r, Xr_train, Xr_test, Yr_train, Yr_test = chronological_split(df_reg, feature_cols, RUL_LABEL_COLS)

# Multi-output regressor (XGBoost)
xgb = XGBRegressor(n_estimators=800,learning_rate=0.05,max_depth=8, subsample=0.9,colsample_bytree=0.8,reg_lambda=1.0,reg_alpha=0.0,random_state=RANDOM_STATE,n_jobs=-1,
    eval_metric="rmse")
reg_multi = MultiOutputRegressor(xgb, n_jobs=None)
reg_multi.fit(Xr_train, Yr_train)

Yr_pred = pd.DataFrame(reg_multi.predict(Xr_test), columns=RUL_LABEL_COLS, index=Yr_test.index)

print("\n=== Multi-output Regressor — RUL (engine/battery/brake) ===")
for i, comp in enumerate(COMPONENTS):
    label = f"{comp}_RUL_days"
    mae  = mean_absolute_error(Yr_test[label], Yr_pred[label])
    rmse = np.sqrt(mean_squared_error(Yr_test[label], Yr_pred[label]))
    print(f"- {comp.upper()}  MAE: {mae:.3f}  RMSE: {rmse:.3f}")

# Save regressor and feature list
joblib.dump(reg_multi, os.path.join(ART_DIR, f"multi_output_xgb_reg_{HORIZON}d_cap{RUL_CAP}.joblib"))
with open(os.path.join(ART_DIR, f"multi_output_reg_features_{HORIZON}d_cap{RUL_CAP}.json"), "w") as f:
    json.dump(feature_cols, f, indent=2)



# Single-Row Prediction
FEATURES_CSV_PATH = os.path.join(OUT_DIR, "telemetry_labeled_with_features_7d.csv")

def predict_all_from_dataset_position(vehicle_id, pos_idx):
 
    clf_path  = os.path.join(ART_DIR, f"multi_label_rf_{HORIZON}d.joblib")
    reg_path  = os.path.join(ART_DIR, f"multi_output_xgb_reg_{HORIZON}d_cap{RUL_CAP}.joblib")
    feats_path = os.path.join(ART_DIR, f"multi_label_features_{HORIZON}d.json")  # same features for both

    clf_multi = joblib.load(clf_path)
    reg_multi = joblib.load(reg_path)
    feature_cols = json.load(open(feats_path))

    df_all = pd.read_csv(FEATURES_CSV_PATH)
    if "timestamp" in df_all.columns:
        df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], errors="coerce")

    vdf = df_all[df_all["vehicle_id"] == vehicle_id].sort_values("timestamp").reset_index(drop=True)
    row = vdf.iloc[[pos_idx]].copy()
    X = row[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Classification: list of 3 prob arrays
    prob_list = clf_multi.predict_proba(X)
    preds = clf_multi.predict(X)[0]

    cls_probs = {comp: float(prob_list[i][0, 1]) for i, comp in enumerate(COMPONENTS)}
    cls_preds = {comp: int(preds[i]) for i, comp in enumerate(COMPONENTS)}

    # Regression: 3 preds
    rul_preds = reg_multi.predict(X)[0]
    rul = {comp: float(rul_preds[i]) for i, comp in enumerate(COMPONENTS)}

    # Ground truth for comparison (if present)
    value = {
        f"{comp}_fail_in_next_{HORIZON}d": (None if pd.isna(row.get(f"{comp}_fail_in_next_{HORIZON}d", pd.Series([np.nan])).values[0])
                                            else int(row[f"{comp}_fail_in_next_{HORIZON}d"].values[0]))
        for comp in COMPONENTS
    }
    for comp in COMPONENTS:
        k = f"{comp}_RUL_days"
        val = row.get(k, pd.Series([np.nan])).values[0]
        value[k] = None if pd.isna(val) else float(val)

    return {
        "vehicle_id": vehicle_id,
        "row_index_for_vehicle": pos_idx,
        "timestamp": str(row["timestamp"].values[0]) if "timestamp" in row.columns else None,
        "pred_fail_in_next_7d": cls_preds,
        "pred_rul_days": rul,
        "true_labels": value,
    }

result = predict_all_from_dataset_position(vehicle_id="VEH0000", pos_idx=8)
print(json.dumps(result, indent=2))

# Artifact Index 
index = {
    "source_dataset": os.path.abspath(CSV_PATH),
    "components": COMPONENTS,
    "horizon": HORIZON,
    "labeled_csv": os.path.join(OUT_DIR, "telemetry_labeled_7d.csv"),
    "features_csv": os.path.join(OUT_DIR, "telemetry_labeled_with_features_7d.csv"),
    "multi_label_model": os.path.join(ART_DIR, f"multi_label_rf_{HORIZON}d.joblib"),
    "multi_output_reg_model": os.path.join(ART_DIR, f"multi_output_xgb_reg_{HORIZON}d_cap{RUL_CAP}.joblib"),
}
with open(os.path.join(OUT_DIR, "ARTIFACTS_INDEX.json"), "w") as f:
    json.dump(index, f, indent=2)

print("\nDone. Trained ONE multi-label classifier and ONE multi-output regressor.")
print("Artifacts under:", os.path.abspath(OUT_DIR))
print("Saved labeled data:", index['labeled_csv'])
print("Saved labeled + features:", index['features_csv'])
print("Classifier:", index['multi_label_model'])
print("Regressor:", index['multi_output_reg_model'])
