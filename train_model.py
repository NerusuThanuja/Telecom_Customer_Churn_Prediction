import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("telco.csv")

# Select only 5 professional features
selected_features = [
    "tenure",
    "MonthlyCharges",
    "Contract",
    "InternetService",
    "SeniorCitizen"
]

target = "Churn"

# Keep only required columns
df = df[selected_features + [target]].dropna()

# Encode categorical features
df["Contract"] = df["Contract"].map({
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
})

df["InternetService"] = df["InternetService"].map({
    "DSL": 0,
    "Fiber optic": 1,
    "No": 2
})

df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

# Split data
X = df[selected_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

# Save model and feature list
joblib.dump(model, "churn_model.pkl")
joblib.dump(selected_features, "features.pkl")

print("✅ Model trained and saved successfully")