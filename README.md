Vehicle Component Lifespan Optimization
This project develops a Predictive Maintenance and Explainable AI System for vehicles using Python and machine learning. It analyzes telemetry data to forecast component failures and estimate Remaining Useful Life (RUL) — helping prevent costly breakdowns and optimize maintenance schedules.

🚘 Overview
The system processes vehicle telemetry logs (engine, brake, battery, etc.) and predicts:
Whether each component is likely to fail within the next 7 days and the estimated Remaining Useful Life (RUL) in days.
Key contributing factors influencing each prediction through explainability dashboards.
The results can be visualized via an interactive web dashboard (app.py) that allows users to upload telemetry data and explore predictions with detailed reasoning.

⚙️ Core Pipeline
1. Data Preprocessing
Converts raw telemetry into uniform daily time intervals.
Detects and timestamps component failures.
Builds target labels:
*_fail_in_next_7d → Binary classification target.
*_RUL_days → Regression target for RUL estimation.

2. Feature Engineering
Creates predictive signals from sensor data:
Rolling statistics (7-day, 14-day averages & std deviations).
Rate of change and linear slope trends.
Fault indicator counters and cumulative degradation metrics.
"Days since last anomaly" signals for slow degradation tracking.

3. Model Training
Two sets of models are trained per component:
RandomForestClassifier (multi-label) → Predicts upcoming failures (binary).
XGBoostRegressor (multi-output) → Predicts remaining useful life in days.
Data is split chronologically per vehicle to prevent time leakage.

Explainability & Maintenance AI
Feature Importance and SHAP Explainability
For every prediction, the system computes:
Feature importance plots for top contributing factors.
SHAP summary to show how each telemetry metric (RPM, temperature, brake pressure, etc.) influenced the decision.
These insights are available both in console logs and dashboard visualizations.

Maintenance AI Assistant
An intelligent assistant module called “Ask Maintenance AI” provides natural-language insights:
Explains “Why did the model predict a brake failure?”
Suggests recommended actions (e.g., “Check brake fluid level”, “Inspect alternator output”).
Provides data-driven preventive maintenance suggestions based on predicted risk levels.
This feature enhances interpretability and supports decision-making for engineers or fleet managers.

🖥️ Dashboard Integration (app.py)
A lightweight Streamlit dashboard is integrated for interactive predictions.
Features: Upload a CSV file or single telemetry row.
Instantly get: Predicted failure risk (Yes/No) for each component and estimated Remaining Useful Life (RUL in days) with Maintenance AI suggestions.

How to Run
1. Install Dependencies
    pip install numpy pandas matplotlib scikit-learn xgboost imbalanced-learn joblib shap flask streamlit
2. Update Configuration
    In main_pipeline.py:  
    CSV_PATH = "/path/to/synthetic_telemetry_data.csv"
    OUT_DIR = "/path/to/output"
3. Train Models
    python main_pipeline.py
    This will:
    Preprocess data & label failures
    Perform feature engineering
    Train RandomForest and XGBoost models
    Save trained models and features to /output/artifacts/
4. Launch Dashboard
    Run python app.py


📊 Evaluation Metrics
Task	Model	Metrics
Failure Prediction:	RandomForestClassifier	ROC-AUC, PR-AUC, Confusion Matrix
RUL Estimation:	XGBoostRegressor	MAE, RMSE
Explainability	SHAP	Local & Global Feature Attribution

📈 Sample Outputs
telemetry_labeled_with_features_7d.csv — Labeled dataset after feature engineering.
multi_label_rf_7d.joblib — Failure prediction model.
multi_output_xgb_reg_7d_cap60.joblib — RUL prediction model.
shap_explanations/ — SHAP feature contribution plots.
ARTIFACTS_INDEX.json — Model metadata and feature index.

🧩 Technologies Used
Python, NumPy, Pandas, Scikit-learn, XGBoost, Matplotlib, SHAP, Streamlit, Joblib, Imbalanced-learn, OpenAI, LLM.


Example Use Cases
Predict EV component failures and battery degradation.
Schedule proactive maintenance in logistics fleets.
Integrate with IoT telemetry dashboards.
Use “Ask Maintenance AI” for live fault diagnosis.
