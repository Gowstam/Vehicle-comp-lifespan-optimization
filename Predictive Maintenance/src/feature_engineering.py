"""
feature_eng.py
--------------
Feature engineering and labeling utilities for Predictive Maintenance.
Includes:
• regenerate_daily_timestamps()  — assign daily timestamps per vehicle
• rebuild_failure_date_column()  — extract failure dates from text
• daily_timestamps()  — normalize and parse timestamps
• build_component_labels()  — compute RUL and 7-day failure flags per component
• add_features()  — rolling, delta, slope, and cumulative telemetry features
"""

import numpy as np
import pandas as pd
import re

# Timestamp Utilities
def regenerate_daily_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rebuilds sequential daily timestamps starting from 2024-01-01
    for each vehicle, ensuring consistent temporal spacing.
    """
    d = df.copy()
    if "timestamp" in d.columns:
        d = d.drop(columns=["timestamp"])
    d["day_idx"] = d.groupby("vehicle_id").cumcount()
    d["timestamp"] = pd.Timestamp("2024-01-01") + pd.to_timedelta(d["day_idx"], unit="D")
    d.drop(columns=["day_idx"], inplace=True)
    return d


def rebuild_failure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives a `failure_date` column from the `failure_type` string.
    Example values:
        - "engine_failed"
        - "battery_failed"
        - "brake_failed"
        - "no failure"
    """
    d = df.copy()
    ft = (
        d.get("failure_type")
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    is_engine = ft.str.contains(r"\bengine_failed\b", regex=True)
    is_brake = ft.str.contains(r"\bbrake_failed\b", regex=True)
    is_batt = ft.str.contains(r"\bbattery_failed\b", regex=True)
    is_no = ft.str.fullmatch(r"no failure", na=False)

    d["failure_date"] = "NA"
    if "timestamp" in d.columns:
        mask_fail = is_engine | is_brake | is_batt
        d.loc[mask_fail, "failure_date"] = pd.to_datetime(
            d.loc[mask_fail, "timestamp"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
    d.loc[is_no, "failure_date"] = "NA"
    return d


def daily_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures timestamp column is parsed correctly and normalized to daily granularity."""
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    return d

# Label Creation — RUL and Failure Labels
import pandas as pd
import numpy as np
import re

import pandas as pd
import numpy as np

def build_component_labels(
    df: pd.DataFrame,
    component: str,
    horizon_days: int = 7
) -> pd.DataFrame:
    """
    Builds predictive-maintenance labels for a component:
      • <component>_fail_in_next_<horizon_days>d — binary label (0/1)
      • <component>_RUL_days                    — remaining useful life in days

    Uses existing <component>_failed events to look forward in time per vehicle.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    fail_flag = f"{component}_failed"
    label_col = f"{component}_fail_in_next_{horizon_days}d"
    rul_col = f"{component}_RUL_days"

    # if failure flag not found → skip
    if fail_flag not in df.columns:
        print(f"⚠️ Missing '{fail_flag}', skipping {component.upper()} label creation.")
        df[label_col] = np.nan
        df[rul_col] = np.nan
        return df

    df[label_col] = 0
    df[rul_col] = np.nan

    for vid, g in df.groupby("vehicle_id"):
        g = g.sort_values("timestamp").copy()
        fail_dates = g.loc[g[fail_flag] == 1, "timestamp"].values

        if len(fail_dates) == 0:
            df.loc[g.index, [label_col, rul_col]] = [0, np.nan]
            continue

        cur = g["timestamp"].values
        next_fail = np.full(len(cur), np.datetime64("NaT"), dtype="datetime64[ns]")

        # compute next failure timestamp
        for i, t in enumerate(cur):
            next_after = fail_dates[fail_dates > t]
            if len(next_after) > 0:
                next_fail[i] = next_after[0]

        # compute RUL in days
        rul_days = (next_fail - cur) / np.timedelta64(1, "D")

        # mark failures within next N days
        mask = (rul_days >= 0) & (rul_days <= horizon_days)
        df.loc[g.index, label_col] = mask.astype(int)
        df.loc[g.index, rul_col] = np.where(np.isfinite(rul_days), rul_days, np.nan)

    print(f"Created labels for {component.upper()}: '{label_col}', '{rul_col}'")
    return df

# Feature Engineering — Telemetry
def add_features(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds rolling, delta, slope, and cumulative features for telemetry sensors.
    Generates trends across engine, brake, and battery metrics.
    """
    df = input_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # fault indicator columns
    FLAG_COLS = [
        "engine_failure_imminent", "brake_issue_imminent", "battery_issue_imminent",
        "abs_fault_indicator", "harsh_brakes", "harsh_accels"
    ]
    for c in [c for c in FLAG_COLS if c in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # numeric telemetry base columns
    BASE = [
        "engine_temp_c","engine_rpm","oil_pressure_psi","coolant_temp_c",
        "fuel_level_percent","fuel_consumption_lph","engine_load_percent",
        "throttle_pos_percent","air_flow_rate_gps","exhaust_gas_temp_c",
        "vibration_level","engine_hours","brake_fluid_level_psi",
        "brake_pad_wear_mm","brake_temp_c","brake_pedal_pos_percent",
        "wheel_speed_fl_kph","wheel_speed_fr_kph","wheel_speed_rl_kph",
        "wheel_speed_rr_kph","vehicle_speed_kph","battery_voltage_v",
        "battery_current_a","battery_temp_c","alternator_output_v",
        "battery_charge_percent","battery_health_percent",
        "ambient_temp_c","humidity_percent","odometer_reading"
    ]
    cols = [c for c in BASE if c in df.columns]

    # helper functions
    def rolling_slope(s, window):
        x = np.arange(window, dtype=float)
        x_mean = x.mean()
        denom = ((x - x_mean) ** 2).sum()

        def _fit(y):
            if np.isnan(y).any() or denom == 0:
                if np.isnan(y).all():
                    return np.nan
            y_mean = y.mean()
            num = ((x - x_mean) * (y - y_mean)).sum()
            return num / denom if denom != 0 else np.nan

        return s.rolling(window, min_periods=window).apply(_fit, raw=True)

    def days_since(cond):
        curr_idx = np.arange(len(cond))
        last = np.where(cond.values, curr_idx, np.nan)
        last = pd.Series(last).ffill().values
        return pd.Series(np.where(np.isnan(last), np.nan, curr_idx - last), index=cond.index)

    # per-vehicle feature computation
    def per_vehicle(g):
        g = g.sort_values("timestamp").copy()
        blocks = []

        if cols:
            r = g[cols]

            # rolling statistics
            blocks += [
                r.rolling(7, min_periods=1).mean().add_suffix("_roll7_mean"),
                r.rolling(14, min_periods=1).mean().add_suffix("_roll14_mean"),
                r.rolling(7, min_periods=2).std().add_suffix("_roll7_std"),
                r.rolling(14, min_periods=2).std().add_suffix("_roll14_std"),
            ]

            # deltas and pct changes
            blocks += [
                r.diff(1).add_suffix("_diff1"),
                r.diff(7).add_suffix("_diff7"),
                r.pct_change(1).add_suffix("_pct1"),
                r.pct_change(7).add_suffix("_pct7"),
            ]

            # slopes (trends)
            for w in (7, 14):
                slopes = r.apply(lambda s: rolling_slope(s, w))
                blocks.append(slopes.add_suffix(f"_slope{w}"))

            # cumulative negative changes
            blocks.append(r.diff(1).clip(upper=0).cumsum().add_suffix("_cum_negdiff1"))

            # days since drop/spike
            drop = r.diff(1) < 0
            spike = r.diff(1) > 0
            blocks += [
                drop.apply(days_since).add_suffix("_days_since_drop"),
                spike.apply(days_since).add_suffix("_days_since_spike"),
            ]

        # cumulative fault flag behavior
        fcols = [c for c in FLAG_COLS if c in g.columns]
        if fcols:
            f = g[fcols]
            blocks += [
                f.cumsum().add_suffix("_cum"),
                f.rolling(7,  min_periods=1).sum().add_suffix("_sum7"),
                f.rolling(14, min_periods=1).sum().add_suffix("_sum14"),
            ]

        # odometer delta
        if "odometer_reading" in g.columns:
            blocks.append(g["odometer_reading"].diff(1).rename("odometer_delta").to_frame())

        return pd.concat([g] + blocks, axis=1)

    return df.groupby("vehicle_id", group_keys=False).apply(per_vehicle)
