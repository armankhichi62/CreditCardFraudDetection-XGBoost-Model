
Streamlit deployment:💳 Credit Card Fraud Risk Detection System

A machine learning–based Credit Card Fraud Risk Detection application built using XGBoost and deployed with Streamlit.
The system predicts fraud risk probability and classifies transactions into risk levels instead of making rigid fraud/not-fraud decisions.

🚀 Live Demo (Streamlit App)

👉 Streamlit App:
https://creditcardfrauddetectxgboost.streamlit.app/
📌 Project Overview

Credit card fraud detection is a highly imbalanced classification problem, where fraudulent transactions are rare compared to legitimate ones.
Instead of forcing binary predictions, this project focuses on risk-based decision making, which is closer to how real banking systems operate.

Key Highlights:

Uses XGBoost, which performs well on tabular and imbalanced datasets

Outputs fraud probability

Converts probability into Low / Medium / High Risk

Interactive Streamlit web application

Professional banking-style UI

🧠 Machine Learning Approach

Dataset: IEEE-CIS Fraud Detection Dataset

Model: XGBoost Classifier

Handling Imbalance:

No oversampling (to preserve real-world probabilities)

Risk interpretation via thresholding

Preprocessing:

StandardScaler for numeric features

LabelEncoder for categorical features

Output: Probability-based fraud risk score

🚦 Risk Classification Logic

Fraud probabilities are naturally low due to class imbalance.
Hence, predictions are interpreted as risk levels:

Fraud Probability	Risk Level

This approach is industry-correct and commonly used in banking systems.

🖥️ Application Features

📊 Fraud probability prediction

🚦 Risk-level classification

🎚️ Adjustable risk threshold slider

📈 Feature importance visualization

🎨 Banking-style UI with background image & overlay

⚡ Fast and interactive Streamlit interface
< 0.015	🟢 Low Risk
0.015 – 0.03	🟡 Medium Risk
≥ 0.03	🔴 High Risk
