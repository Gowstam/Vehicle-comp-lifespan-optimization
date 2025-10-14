"""
feature_selection.py
--------------------
Feature correlation & redundancy filtering utilities for Predictive Maintenance.

Functions:
• select_component_features() — correlation-based selection for a single component
• select_all_components() — wrapper that runs selection for all components (engine, battery, brake)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

# remove all known label/target columns to prevent leakage
def remove_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    label_patterns = [
        "_fail_in_next_", "_RUL_", "_failed", "failure_type", "flag",
        "target", "indicator", "_label"
    ]
    cols_to_drop = [c for c in df.columns if any(pat in c for pat in label_patterns)]
    return df.drop(columns=cols_to_drop, errors="ignore")

#  Single-Component Feature Selection
def select_component_features(
    df: pd.DataFrame,
    component: str = "engine",
    out_dir: str = None,
    top_n: int = 25,
    corr_threshold: float = 0.9,
    plot: bool = True
) -> list:
    """
    Identify top N features correlated with a component’s fail_in_next_7d target,
    then remove redundant highly correlated ones (label-safe).
    """

    target = f"{component}_fail_in_next_7d"
    print(f"\n=== 🔍 Feature correlation analysis for {target.upper()} ===")

    if target not in df.columns:
        print(f" Target column '{target}' not found, skipping.")
        return []

    # Remove label columns first
    df_nolabel = remove_label_columns(df)

    # select numeric columns
    sub_df = df_nolabel.select_dtypes(include=[np.number]).copy()

    # Append target back (for correlation)
    sub_df[target] = pd.to_numeric(df[target], errors="coerce")

    # Compute correlation matrix safely
    corr_matrix = sub_df.corr(numeric_only=True)
    corr_target = corr_matrix[target].dropna().abs().sort_values(ascending=False)

    if corr_target.empty:
        print(f" No valid correlation data for {target}. Returning [].")
        return []

    # Top-N correlated features
    top_features = corr_target.head(top_n).index.tolist()
    print(f"Top {len(top_features)} correlated features for {component.upper()}:")
    print(corr_target.head(top_n).round(3).to_string())

    # Redundancy filtering
    corr_top = sub_df[top_features].corr().abs()
    upper = corr_top.where(np.triu(np.ones(corr_top.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]
    selected_feats = [f for f in top_features if f not in to_drop and f != target]

    print(f" Selected {len(selected_feats)} unique features after redundancy filtering.")

    # Dendrogram Plot
    if plot and len(selected_feats) > 1:
        try:
            linkage_matrix = linkage(corr_top, method="ward")
            plt.figure(figsize=(10, 6))
            dendrogram(linkage_matrix, labels=corr_top.columns, leaf_rotation=90)
            plt.title(f"{component.upper()} Feature Dendrogram (Top {top_n})")
            plt.tight_layout()

            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
                dend_path = os.path.join(out_dir, f"{component}_dendrogram.png")
                plt.savefig(dend_path, dpi=200)
                print(f"📊 Dendrogram saved to: {dend_path}")
            plt.close()
        except Exception as e:
            print(f" Could not generate dendrogram for {component.upper()}: {e}")

    # Save JSON Feature List
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        json_path = os.path.join(out_dir, f"{component}_selected_features.json")
        with open(json_path, "w") as f:
            json.dump(selected_feats, f, indent=2)
        print(f"Saved selected feature list → {json_path}")

    return selected_feats

# All Components Wrapper
def select_all_components(
    df: pd.DataFrame,
    out_dir: str = None,
    top_n: int = 25,
    corr_threshold: float = 0.9
):
    """Run feature selection for ENGINE, BATTERY, and BRAKE in one call."""
    engine_feats = select_component_features(df, "engine", out_dir, top_n, corr_threshold)
    battery_feats = select_component_features(df, "battery", out_dir, top_n, corr_threshold)
    brake_feats = select_component_features(df, "brake", out_dir, top_n, corr_threshold)

    print("\n=== 🧩 Final Selected Feature Summary ===")
    print(f"ENGINE  → {len(engine_feats)} features")
    print(f"BATTERY → {len(battery_feats)} features")
    print(f"BRAKE   → {len(brake_feats)} features")

    shared_engine_battery = set(engine_feats) & set(battery_feats)
    shared_engine_brake   = set(engine_feats) & set(brake_feats)
    shared_battery_brake  = set(battery_feats) & set(brake_feats)

    print(f"\n=== 🔄 Feature Overlap Summary ===")
    print(f"Engine ∩ Battery → {len(shared_engine_battery)}")
    print(f"Engine ∩ Brake   → {len(shared_engine_brake)}")
    print(f"Battery ∩ Brake  → {len(shared_battery_brake)}")

    return engine_feats, battery_feats, brake_feats

#  CLI Entry Point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run feature correlation analysis.")
    parser.add_argument("--csv", required=True, help="Path to telemetry CSV file with features.")
    parser.add_argument("--out", default="./feature_analysis", help="Output directory for results.")
    parser.add_argument("--top", type=int, default=25, help="Top N features to analyze.")
    parser.add_argument("--corr", type=float, default=0.9, help="Correlation threshold.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    engine_feats, battery_feats, brake_feats = select_all_components(
        df, args.out, args.top, args.corr
    )

    print("\n=== Summary ===")
    print(f"Engine:  {len(engine_feats)} features")
    print(f"Battery: {len(battery_feats)} features")
    print(f"Brake:   {len(brake_feats)} features")
