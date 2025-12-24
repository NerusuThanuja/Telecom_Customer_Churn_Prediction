import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ============================
# PATHS
# ============================
DATASET_PATH = "telco.csv"
MODEL_PATH = "churn_model.pkl"
FEATURES_PATH = "features.pkl"

# ============================
# LOAD DATA
# ============================
df = pd.read_csv(DATASET_PATH)

# ============================
# NORMALIZE COLUMN NAMES
# ============================
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("\nColumns after normalization:")
print(df.columns.tolist())

# ============================
# TARGET COLUMN (FINAL)
# ============================
if "churn_label" in df.columns:
    TARGET = "churn_label"
elif "churn" in df.columns:
    TARGET = "churn"
else:
    raise ValueError("❌ No churn column found")

# ============================
# DROP ALL NON-INPUT CHURN COLUMNS
# ============================
DROP_COLS = [
    "churn_score",
    "cltv",
    "churn_category",
    "churn_reason"
]

for col in DROP_COLS:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

# ============================
# SELECT ONLY 5 PROFESSIONAL FEATURES
# ============================
FEATURES = [
    "tenure_in_months",
    "monthly_charge",
    "total_charges",
    "contract",
    "payment_method"
]

missing = [f for f in FEATURES if f not in df.columns]
if missing:
    raise ValueError(f"❌ Missing required feature columns: {missing}")

# ============================
# KEEP REQUIRED DATA ONLY
# ============================
df = df[FEATURES + [TARGET]].dropna()

# ============================
# ENCODE CATEGORICAL FEATURES
# ============================
encoders = {}

for col in FEATURES:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

# ============================
# ENCODE TARGET
# ============================
if df[TARGET].dtype == "object":
    df[TARGET] = df[TARGET].str.lower().map({"yes": 1, "no": 0})

# ============================
# TRAIN / TEST SPLIT
# ============================
X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================
# TRAIN MODEL
# ============================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ============================
# EVALUATE
# ============================
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"\n✅ Model Accuracy: {accuracy:.2f}")

# ============================
# SAVE MODEL & FEATURES
# ============================
joblib.dump(model, MODEL_PATH)
joblib.dump(FEATURES, FEATURES_PATH)

print("\n🎉 TRAINING COMPLETE")
print("Saved files:")
print("-", MODEL_PATH)
print("-", FEATURES_PATH)