import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(page_title="Telecom Customer Churn", layout="centered")

# Load model and features
model = joblib.load("churn_model.pkl")
features = joblib.load("features.pkl")

# Title
st.title("📊 Telecom Customer Churn Prediction")
st.markdown("Predict whether a telecom customer is likely to **churn or stay**.")

st.divider()

# User inputs
tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)

contract = st.selectbox(
    "Contract Type",
    options=["Month-to-month", "One year", "Two year"]
)

internet_service = st.selectbox(
    "Internet Service",
    options=["DSL", "Fiber optic", "No"]
)

senior_citizen = st.selectbox(
    "Senior Citizen",
    options=["No", "Yes"]
)

# Encode inputs
contract_map = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}

internet_map = {
    "DSL": 0,
    "Fiber optic": 1,
    "No": 2
}

senior_map = {
    "No": 0,
    "Yes": 1
}

# Predict
if st.button("🔍 Predict Churn"):
    input_df = pd.DataFrame([[
        tenure,
        monthly_charges,
        contract_map[contract],
        internet_map[internet_service],
        senior_map[senior_citizen]
    ]], columns=features)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to CHURN\n\nChurn Probability: **{probability:.2%}**")
    else:
        st.success(f"✅ Customer is likely to STAY\n\nChurn Probability: **{probability:.2%}**")