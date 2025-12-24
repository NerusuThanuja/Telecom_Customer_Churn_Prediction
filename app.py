import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ===============================
# App Config
# ===============================
st.set_page_config(
    page_title="Telecom Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Telecom Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn using key business features.")

# ===============================
# Load Model & Features
# ===============================
MODEL_PATH = "churn_model.pkl"
FEATURES_PATH = "features.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
    st.error("❌ Model files not found. Please retrain and push churn_model.pkl & features.pkl to GitHub.")
    st.stop()

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)

# ===============================
# Input Section (ONLY 5 FEATURES)
# ===============================
st.subheader("Enter Customer Details")

tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=100,
    value=12
)

monthly_charge = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=500.0,
    value=70.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=20000.0,
    value=1500.0
)

contract_type = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

payment_method = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
)

# ===============================
# Encoding (MUST match training)
# ===============================
contract_map = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}

payment_map = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer": 2,
    "Credit card": 3
}

input_data = pd.DataFrame([[
    tenure,
    monthly_charge,
    total_charges,
    contract_map[contract_type],
    payment_map[payment_method]
]], columns=feature_names)

# ===============================
# Prediction
# ===============================
if st.button("🔍 Predict Churn"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"❌ Customer is likely to churn (Probability: {probability:.2%})")
    else:
        st.success(f"✅ Customer is likely to stay (Probability: {(1 - probability):.2%})")

# ===============================
# Footer
# ===============================
st.markdown("---")
st.caption("Built with ❤️ using Machine Learning & Streamlit")