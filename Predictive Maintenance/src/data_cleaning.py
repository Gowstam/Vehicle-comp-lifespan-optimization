"""
data_cleaning.py
Performs dataset inspection and cleaning for predictive maintenance telemetry data.
Includes:
- Step 1A: Data inspection (shape, missing, duplicates, zeros)
- Step 1B: Zero-value cleaning and verification
"""

import pandas as pd

def inspect_data(df: pd.DataFrame):
    print("Step 1A: Inspecting dataset quality...")

    # Dataset shape
    rows, cols = df.shape
    print(f"Total rows: {rows:,} | Total columns: {cols:,}")

    # Missing values
    missing_counts = df.isna().sum()
    total_missing = missing_counts.sum()
    if total_missing > 0:
        print(f"Found {total_missing:,} missing values across {(missing_counts > 0).sum()} columns.")
        print("Columns with missing values (top 10):")
        print(missing_counts[missing_counts > 0].sort_values(ascending=False).head(10))
    else:
        print("No missing values detected.")

    # Duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        print(f"Found {duplicate_count:,} duplicate rows.")
    else:
        print("No duplicate rows found.")

    # Zero values (excluding binary/imminent columns)
    binary_cols = [
        "engine_failure_imminent", "brake_issue_imminent", "battery_issue_imminent",
        "abs_fault_indicator", "harsh_brakes", "harsh_accels"
    ]
    num_cols = df.select_dtypes(include=["number"]).columns
    zero_check_cols = [c for c in num_cols if c not in binary_cols]

    zero_counts = (df[zero_check_cols] == 0).sum()
    total_zero = zero_counts.sum()
    if total_zero > 0:
        print(f"Found {total_zero:,} zero values across {(zero_counts > 0).sum()} numeric (non-binary) columns.")
        print(zero_counts[zero_counts > 0].sort_values(ascending=False).head(10))
    else:
        print("No zero values found in numeric (non-binary) columns.")

    print("Step 1A completed: Data inspection summary printed.")


def clean_zero_values(df: pd.DataFrame):

    # Columns where zeros are likely invalid
    cols_replace_zeros = [
        "throttle_pos_percent", "fuel_consumption_lph",
        "engine_load_percent", "air_flow_rate_gps"
    ]

    for col in cols_replace_zeros:
        if col in df.columns:
            non_zero_median = df.loc[df[col] > 0, col].median()
            df[col] = df[col].replace(0, non_zero_median)

    print("Skipped replacing zeros in 'vehicle_speed_kph' (valid when stationary).")

    # Recheck zero values after cleaning
    binary_cols = [
        "engine_failure_imminent", "brake_issue_imminent", "battery_issue_imminent",
        "abs_fault_indicator", "harsh_brakes", "harsh_accels"
    ]
    num_cols = df.select_dtypes(include=["number"]).columns
    zero_check_cols = [c for c in num_cols if c not in binary_cols]
    zero_counts_after = (df[zero_check_cols] == 0).sum()
    total_zero_after = zero_counts_after.sum()

    print("\nRe-checking zero values after cleaning...")
    if total_zero_after > 0:
        print(f"{total_zero_after:,} zero values remain across {(zero_counts_after > 0).sum()} numeric (non-binary) columns.")
        print(zero_counts_after[zero_counts_after > 0].sort_values(ascending=False).head(10))
    else:
        print("No zero values remain in numeric (non-binary) columns after cleaning.")

    print("Data Cleaning done.")
    return df
