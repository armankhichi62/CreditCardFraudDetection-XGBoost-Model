import streamlit as st
import numpy as np
import pickle
from xgboost import XGBClassifier

st.set_page_config(
    page_title="Fraud Detection - XGBoost",
    layout="centered"
)

# -------------------------------
# LOAD MODEL + PREPROCESSORS
# -------------------------------
@st.cache_resource
def load_artifacts():
     with open("xgb_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
         
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)

    with open("features.pkl", "rb") as f:
        features = pickle.load(f)

    return xgb_model, scaler, encoders, features


xgb_model, scaler, encoders, features = load_artifacts()

numeric_features = features["numeric"]
categorical_features = features["categorical"]

# -------------------------------
# UI
# -------------------------------
st.title("💳 Credit Card Fraud Detection (XGBoost)")
st.write("Enter transaction details to predict fraud")

with st.form("input_form"):

    st.subheader("Numeric Features")
    num_inputs = {}
    cols = st.columns(2)
    for i, col in enumerate(numeric_features):
        with cols[i % 2]:
            num_inputs[col] = st.number_input(col, value=0.0)

    st.subheader("Categorical Features")
    cat_inputs = {}
    for col in categorical_features:
        cat_inputs[col] = st.text_input(col)

    submitted = st.form_submit_button("Predict")

# -------------------------------
# PREDICTION
# -------------------------------
if submitted:

    threshold = st.slider("Fraud Threshold", 0.0, 1.0, 0.50, 0.01)

    try:
        # Numeric
        X_num = np.array([[num_inputs[c] for c in numeric_features]])
        X_num_scaled = scaler.transform(X_num)

        # Categorical
        cat_encoded = []
        for col in categorical_features:
            le = encoders[col]
            val = str(cat_inputs[col])

            if val in le.classes_:
                enc = int(le.transform([val])[0])
            else:
                enc = int(le.transform([le.classes_[0]])[0])

            cat_encoded.append(enc)

        # Final input
        final_input = np.hstack([X_num_scaled, np.array([cat_encoded])])

        # XGBoost prediction (NO DMatrix)
        prob = float(xgb_model.predict_proba(final_input)[0][1])
        pred = 1 if prob > threshold else 0

        # Output
        st.success("Prediction Complete")
        st.write(f"**Fraud Probability:** `{prob:.4f}`")
        st.write(f"**Threshold Used:** `{threshold}`")
        st.write(f"**Prediction:** {'⚠️ FRAUD' if pred else '✔️ LEGIT'}")

    except Exception as e:
        st.error(f"Error: {e}")

