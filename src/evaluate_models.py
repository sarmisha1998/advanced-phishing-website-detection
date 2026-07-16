import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ---------------------------------
# Load Saved Model
# ---------------------------------
model = joblib.load("models/random_forest_model.pkl")

print("Model Loaded Successfully.\n")

# ---------------------------------
# Load Dataset
# ---------------------------------
df = pd.read_csv("data/processed/final_dataset.csv")

X = df.drop(columns=["URL", "label"])
y = df["label"]

# ---------------------------------
# Train/Test Split
# ---------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ---------------------------------
# Predictions
# ---------------------------------
y_pred = model.predict(X_test)

# ---------------------------------
# Accuracy
# ---------------------------------
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:")
print(accuracy)

# ---------------------------------
# Classification Report
# ---------------------------------
print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

# ---------------------------------
# Confusion Matrix
# ---------------------------------
print("\nConfusion Matrix:\n")

print(confusion_matrix(y_test, y_pred))