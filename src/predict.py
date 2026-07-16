import joblib
import pandas as pd

# ----------------------------
# Load Trained Model
# ----------------------------
model = joblib.load("models/random_forest_model.pkl")

print("Model Loaded Successfully.\n")

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("data/processed/final_dataset.csv")

# Remove URL column
X = df.drop(columns=["URL", "label"])

# Actual labels
y = df["label"]

# ----------------------------
# Predict First 10 URLs
# ----------------------------
predictions = model.predict(X.head(10))

print("Predictions:")
print(predictions)

print("\nActual Labels:")
print(y.head(10).values)