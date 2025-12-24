import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Telecom Customer Churn Prediction")

@st.cache_resource
def train_and_get_model():
    df = pd.read_csv("telco.csv")

    # 🔍 Automatically detect churn column
    possible_targets = [
        "churn", "churn label", "customer status",
        "churn_category", "exited"
    ]

    target_column = None
    for col in df.columns:
        if col.strip().lower() in possible_targets:
            target_column = col
            break

    if target_column is None:
        st.error(f"❌ No churn column found. Available columns: {list(df.columns)}")
        st.stop()

    # Convert target to binary
    df[target_column] = df[target_column].astype(str).str.lower().map(
        {"yes": 1, "no": 0, "churned": 1, "active": 0}
    )

    df = df.dropna(subset=[target_column])

    # Encode categorical features
    for col in df.select_dtypes(include="object").columns:
        if col != target_column:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        random_state=42
    )
    model.fit(X_train, y_train)

    return model, X.columns

# ---------------- UI ---------------- #

st.title("📊 Telecom Customer Churn Prediction")

model, feature_names = train_and_get_model()

st.subheader("Enter Customer Details")

user_input = []
for feature in feature_names:
    value = st.number_input(feature, value=0.0)
    user_input.append(value)

if st.button("Predict Churn"):
    prediction = model.predict([user_input])[0]
    if prediction == 1:
        st.error("❌ Customer is likely to churn")
    else:
        st.success("✅ Customer is NOT likely to churn")