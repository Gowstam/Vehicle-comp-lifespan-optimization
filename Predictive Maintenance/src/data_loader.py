"""
data_loader.py
Loads and normalizes telemetry data.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# === Configuration ===
CSV_PATH = os.environ.get(
    "CSV_PATH",
    "/Users/gowthamsivaraman/Desktop/Vehicle Maintenance Project/Predictive Maintenance/data/synthetic_telemetry_data.csv"
)
OUT_DIR = os.environ.get(
    "OUT_DIR",
    "/Users/gowthamsivaraman/Desktop/Vehicle Maintenance Project/Predictive Maintenance/"
)
FIG_DIR = os.path.join(OUT_DIR, "figures")
ART_DIR = os.path.join(OUT_DIR, "artifacts")

for d in [OUT_DIR, FIG_DIR, ART_DIR]:
    os.makedirs(d, exist_ok=True)


# === Data Loading ===
def load_data(path: str = CSV_PATH):
    """Loads the telemetry CSV and sorts by vehicle and timestamp."""
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values(["vehicle_id", "timestamp"])
    return df


# === Normalization ===
def normalize_data(df: pd.DataFrame):
    """
    Applies z-score normalization (mean=0, std=1)
    to all numeric columns except identifiers and target labels.
    """
    exclude_cols = ["vehicle_id", "timestamp", "failure_type"]
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols
    ]
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df, scaler
