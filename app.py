import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("churn_model.pkl")

st.set_page_config(
    page_title="Telecom Customer Churn Predictor",
    layout="centered"
)

st.title("📡 Telecom Customer Churn Prediction")
st.markdown(
    "Predict whether a telecom customer is likely to **churn** using key business indicators."
)

# Inputs
tenure = st.slider("Customer Tenure (months)", 0, 72, 12)
monthly_charges = st.slider("Monthly Charges", 20.0, 150.0, 70.0)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

payment_method = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["Yes", "No"]
)

# Encoding maps (must match training)
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

internet_map = {
    "DSL": 0,
    "Fiber optic": 1,
    "No": 2
}

tech_map = {
    "No": 0,
    "Yes": 1
}

# Prepare input
input_data = np.array([[
    tenure,
    monthly_charges,
    contract_map[contract],
    payment_map[payment_method],
    internet_map[internet_service],
    tech_map[tech_support]
]])

# Prediction
if st.button("Predict Churn"):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("❌ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is likely to STAY")