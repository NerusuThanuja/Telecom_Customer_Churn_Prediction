import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Telecom Customer Churn Prediction")

@st.cache_resource
def train_model():
    df = pd.read_csv("telco.csv")

    # Rename columns safely
    df.columns = df.columns.str.strip()

    # Map target column
    churn_col = None
    for col in df.columns:
        if col.lower() in ["churn", "churn label", "customer status"]:
            churn_col = col
            break

    if churn_col is None:
        st.error("Churn column not found in dataset")
        st.stop()

    df[churn_col] = df[churn_col].astype(str).str.lower().map(
        {"yes": 1, "no": 0, "churned": 1, "active": 0}
    )

    # Select ONLY 5 professional features
    selected_features = [
        "tenure",
        "monthlycharges",
        "contract",
        "internetservice",
        "seniorcitizen"
    ]

    # Normalize column names
    df.columns = df.columns.str.lower()

    df = df[selected_features + [churn_col]].dropna()

    # Encode categorical features
    df["contract"] = df["contract"].map({
        "month-to-month": 0,
        "one year": 1,
        "two year": 2
    })

    df["internetservice"] = df["internetservice"].map({
        "dsl": 0,
        "fiber optic": 1,
        "no": 2
    })

    X = df[selected_features]
    y = df[churn_col]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=6,
        random_state=42
    )
    model.fit(X_train, y_train)

    return model

model = train_model()

# ---------------- UI ---------------- #

st.title("📊 Telecom Customer Churn Prediction")
st.write("Enter customer details below:")

tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
monthly_charges = st.number_input("Monthly Charges", value=75.0)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)
contract_val = {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract]

internet = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)
internet_val = {"DSL": 0, "Fiber optic": 1, "No": 2}[internet]

senior = st.selectbox(
    "Senior Citizen",
    ["No", "Yes"]
)
senior_val = 1 if senior == "Yes" else 0

if st.button("Predict"):
    input_data = [[
        tenure,
        monthly_charges,
        contract_val,
        internet_val,
        senior_val
    ]]

    # STEP 3 GOES HERE 👇
    input_df = pd.DataFrame(
        input_data,
        columns=[
            "Tenure",
            "Monthly Charges",
            "Contract",
            "Internet Service",
            "Senior Citizen"
        ]
    )

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is likely to STAY")

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("❌ Customer is likely to churn")
    else:
        st.success("✅ Customer is NOT likely to churn")