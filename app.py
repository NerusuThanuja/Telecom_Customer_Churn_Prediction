import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Telecom Customer Churn Prediction")

@st.cache_resource
def train_and_get_model():
    df = pd.read_csv("telco.csv")

    # 🔍 Find churn column safely
    churn_col = None
    for col in df.columns:
        if col.lower() == "churn":
            churn_col = col
            break

    if churn_col is None:
        st.error("❌ No 'Churn' column found in dataset")
        st.stop()

    # Encode target
    df[churn_col] = df[churn_col].map({"Yes": 1, "No": 0})

    # Encode categorical features
    for col in df.select_dtypes(include="object").columns:
        if col != churn_col:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    X = df.drop(churn_col, axis=1)
    y = df[churn_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model, X.columns

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
        st.success("✅ Customer is not likely to churn")