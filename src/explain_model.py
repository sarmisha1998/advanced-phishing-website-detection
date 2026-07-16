import os
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# -------------------------
# Create output folder
# -------------------------
os.makedirs("outputs/reports/figures", exist_ok=True)

# -------------------------
# Load Model
# -------------------------
model = joblib.load("models/random_forest_model.pkl")

print("Model Loaded Successfully.")

# -------------------------
# Load Dataset
# -------------------------
df = pd.read_csv("data/processed/final_dataset.csv")

X = df.drop(columns=["URL", "label"])
y = df["label"]

# -------------------------
# Train/Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# -------------------------
# Sample Data
# -------------------------
X_sample = X_test.iloc[:200]

# -------------------------
# SHAP Explainer
# -------------------------
explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_sample)

# -------------------------
# SHAP Compatibility
# -------------------------
if isinstance(shap_values, list):
    shap_values = shap_values[1]

elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    shap_values = shap_values[:, :, 1]

# -------------------------
# SHAP Summary Plot
# -------------------------
plt.figure(figsize=(10,6))

shap.summary_plot(
    shap_values,
    X_sample,
    show=False
)

plt.tight_layout()

plt.savefig(
    "outputs/reports/figures/shap_summary.png",
    dpi=300
)

plt.close()

print("Saved shap_summary.png")

# -------------------------
# SHAP Bar Plot
# -------------------------
plt.figure(figsize=(10,6))

shap.summary_plot(
    shap_values,
    X_sample,
    plot_type="bar",
    show=False
)

plt.tight_layout()

plt.savefig(
    "outputs/reports/figures/shap_bar.png",
    dpi=300
)

plt.close()

print("Saved shap_bar.png")

print("SHAP Images Generated Successfully.")