# Vehicle-comp-lifespan-optimization
This project focuses on analyzing vehicle telemetry data and predicting component failures using Python-based data analysis and machine learning. The goal is to improve vehicle reliability, efficiency, and component lifespan by identifying patterns in synthetic telemetry data and forecasting potential failures.

Folder Structure:
Vehicle-Comp-Lifespan-Optimization/
│
├── predictive_maintenance_7d.py              # Main training script
├── synthetic_telemetry_data.csv              # Input dataset (synthetic or user-provided)
│
├── output_simple/
│   ├── telemetry_labeled_7d.csv              # Data with failure labels
│   ├── telemetry_labeled_with_features_7d.csv# Data after feature engineering
│   ├── artifacts/
│   │   ├── multi_label_rf_7d.joblib          # RandomForest classifier
│   │   ├── multi_output_xgb_reg_7d_cap60.joblib  # XGBoost regressor
│   │   ├── feature JSON files                # Stored feature columns
│   ├── ARTIFACTS_INDEX.json                  # Index of generated artifacts
│   └── figures/                              # Optional visualizations


1. Data Preprocessing
Normalizes timestamps to daily intervals.
Identifies component failures and assigns dates.
Builds labels:
<component>_fail_in_next_7d (binary)
<component>_RUL_days (numeric Remaining Useful Life)

2. Feature Engineering
Rolling averages & standard deviations (7/14-day windows).
Differences, percentage changes, and linear slopes.
Fault indicators (ABS, harsh brakes, etc.) with cumulative counts.
“Days since last drop/spike” to detect degradation patterns.

3. Model Training
RandomForestClassifier (multi-label) — Predicts failures in next 7 days.
XGBoostRegressor (multi-output) — Predicts RUL for each component.
Uses chronological split per vehicle to avoid time leakage.

How to Run:
1. Install Dependencies:
pip install numpy pandas matplotlib scikit-learn xgboost imbalanced-learn joblib

2. In predictive_maintenance_7d.py, update:
CSV_PATH = "/path/to/synthetic_telemetry_data.csv"
OUT_DIR = "/path/to/output"

3. Run the Script:
python predictive_maintenance_7d.py

This will:
Generate labeled datasets
Create engineered features
Train classification & regression models
Save all outputs in the output_simple/ directory
