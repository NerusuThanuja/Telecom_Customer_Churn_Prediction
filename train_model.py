import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from preprocess import preprocess_data

# Load dataset
df = pd.read_csv("telco.csv")

# Preprocess data
X, y = preprocess_data(df)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=6,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "churn_model.pkl")

print("✅ Model trained and saved successfully")