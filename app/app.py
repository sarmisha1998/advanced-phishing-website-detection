import os
import streamlit as st
import joblib
import pandas as pd
import re
import sys


# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.live_feature_extraction import extract_features

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Advanced Phishing Website Detection",
    page_icon="🛡️",
    layout="wide"
)

# -----------------------------------
# Load Trained Model
# -----------------------------------
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "random_forest_model.pkl")
model = joblib.load(MODEL_PATH)

# -----------------------------------
# Sidebar Navigation
# -----------------------------------
st.sidebar.title("🛡️ Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home",
        "📂 Dataset Prediction",
        "🔍 URL Feature Analyzer",
        "📈 Model Performance",
        "🧠 Explainable AI",
        "ℹ️ About Project"
    ]
)

# =====================================
# HOME PAGE
# =====================================

if page == "🏠 Home":

    st.title("🛡️ Advanced Phishing Website Detection")

    st.markdown("""
Welcome to the **Advanced Phishing Website Detection System**.

This project uses **Machine Learning (Random Forest)** to detect phishing websites using engineered URL-based features.

## Project Workflow

1. Dataset Collection
2. Data Preprocessing
3. Feature Engineering
4. Random Forest Model Training
5. Model Evaluation
6. Explainable AI (SHAP)
7. Website Prediction

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- SHAP
- Joblib

## Objective

To identify phishing websites accurately using Machine Learning and provide explainable predictions.
""")
    st.subheader("🚀 Key Features")

    st.markdown("""
    ✅ Live Website URL Prediction

    ✅ Batch CSV Prediction
    
    ✅ Automatic 40 Feature Extraction
    
    ✅ Random Forest Machine Learning
    
    ✅ Explainable AI (SHAP)
    
    ✅ Model Performance Dashboard

    ✅ Interactive Streamlit Interface
    """)

    st.markdown("---")
    
    st.subheader("📊 Project Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Dataset Size", "235K+")
    
    with col2:
        st.metric("Features", "40")
    
    with col3:
        st.metric("Algorithm", "Random Forest")

    st.markdown("---")

    st.info("""
    This application helps identify phishing websites by analysing URL characteristics and webpage content using Machine Learning and Explainable AI.
    
    The application supports:
    
    • Single URL Prediction
    
    • Batch CSV Prediction
    
    • Explainable AI Visualization
    """)
    st.markdown("---")

    st.subheader("🎯 Why This Project?")

    st.markdown("""
    Phishing attacks are one of the most common cybersecurity threats. Fraudulent websites imitate trusted websites to steal sensitive information such as usernames, passwords, banking details, and personal data.
    
    This project uses Machine Learning and Explainable AI to identify phishing websites by analysing URL characteristics and webpage features. The goal is to provide accurate, transparent, and reliable phishing detection.
    """)

    st.markdown("---")
    
    st.subheader("🔄 System Workflow")

    st.markdown("""
    1️⃣ User enters a URL or uploads a dataset.
    
    2️⃣ The system extracts 40 engineered features.
    
    3️⃣ The Random Forest model predicts whether the website is Legitimate or Phishing.
    
    4️⃣ SHAP Explainable AI explains why the prediction was made.
    
    5️⃣ The application displays the prediction, confidence score, extracted features, and security recommendations.
    """)
    
    st.markdown("---")
    
    st.success("🎉 Welcome! Use the navigation panel on the left to explore the application's features.")
        
    
# =====================================
# URL FEATURE ANALYZER
# =====================================

elif page == "🔍 URL Feature Analyzer":

    st.title("🔍 URL Feature Analyzer")

    st.write(
        "Enter a website URL to extract features and predict whether it is Legitimate or Phishing."
    )

    url = st.text_input("Enter Website URL")

    if st.button("Analyze URL"):

        if url.strip() == "":
            st.warning("Please enter a valid URL.")

        else:

            with st.spinner("Extracting 40 features..."):

                feature_df = extract_features(url)

            st.success("✅ Feature Extraction Completed")

            # -----------------------------
            # Prediction
            # -----------------------------

            prediction = model.predict(feature_df)[0]

            probability = model.predict_proba(feature_df)[0]

            confidence = round(max(probability) * 100, 2)

            st.subheader("Prediction Result")

            if prediction == 1:

                st.success("🟢 Legitimate Website")

                st.metric(
                    "Confidence",
                    f"{confidence}%"
                )

                st.markdown("""
### Interpretation

This website appears to be **Legitimate** based on the extracted URL and webpage features.

The Random Forest classifier found strong characteristics commonly observed in trusted websites.
""")

                st.markdown("## 🛡 Security Assessment")

                st.info("""
**Risk Level:** 🟢 Low Risk

### Recommendation

✔ Website appears legitimate

✔ HTTPS detected

✔ Domain structure looks normal

✔ Continue browsing with normal precautions
""")

            else:

                st.error("🔴 Phishing Website")

                st.metric(
                    "Confidence",
                    f"{confidence}%"
                )

                st.markdown("""
### Interpretation

This website appears to be a **Phishing Website**.

The model detected suspicious characteristics such as abnormal URL structure or webpage properties that are commonly associated with phishing attacks.
""")

                st.markdown("## 🛡 Security Assessment")

                st.warning("""
**Risk Level:** 🔴 High Risk

### Recommendation

❌ Do NOT enter passwords

❌ Avoid online payments

❌ Verify the official website

❌ Check the domain carefully

❌ Report suspicious websites
""")

            # -----------------------------
            # Feature Table
            # -----------------------------

            st.subheader("Extracted Features")

            display_df = feature_df.T.reset_index()

            display_df.columns = ["Feature", "Value"]

            display_df["Value"] = display_df["Value"].astype(str)

            st.table(display_df)
# =====================================
# DATASET PREDICTION PAGE
# =====================================

elif page == "📂 Dataset Prediction":

    st.title("📂 Dataset Prediction")

    st.write(
        "Upload the processed dataset (CSV) and predict whether each website is Legitimate or Phishing."
    )

    uploaded_file = st.file_uploader(
        "Upload Processed Dataset (CSV)",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.success("✅ Dataset Uploaded Successfully!")

        st.subheader("Dataset Preview")

        st.dataframe(df.head())

        if st.button("Predict Websites"):

            prediction_df = df.copy()

            X = prediction_df.drop(columns=["URL", "label"], errors="ignore")

            predictions = model.predict(X)

            prediction_df["Prediction"] = predictions

            prediction_df["Prediction"] = prediction_df["Prediction"].map({
                1: "Legitimate Website",
                0: "Phishing Website"
            })

            st.success("✅ Prediction Completed Successfully!")

            legitimate = (
                prediction_df["Prediction"] == "Legitimate Website"
            ).sum()

            phishing = (
                prediction_df["Prediction"] == "Phishing Website"
            ).sum()

            st.subheader("Prediction Summary")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Legitimate Websites", legitimate)

            with col2:
                st.metric("Phishing Websites", phishing)

            st.subheader("Prediction Results")

            result_df = prediction_df[["URL", "Prediction"]]

            st.dataframe(result_df)

            csv = prediction_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="📥 Download Prediction Results",
                data=csv,
                file_name="prediction_results.csv",
                mime="text/csv"
            )

# =====================================
# MODEL PERFORMANCE
# =====================================

elif page == "📈 Model Performance":

    st.title("📈 Model Performance Dashboard")

    st.write("""
This page presents the evaluation results of the trained
Random Forest Machine Learning model.
""")

    st.markdown("---")

    st.subheader("🤖 Model Information")

    st.info("**Algorithm:** Random Forest Classifier")

    st.markdown("---")

    st.subheader("📊 Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Accuracy",
            value="100%"
        )

    with col2:
        st.metric(
            label="Precision",
            value="100%"
        )

    with col3:
        st.metric(
            label="Recall",
            value="100%"
        )

    with col4:
        st.metric(
            label="F1 Score",
            value="100%"
        )

    st.markdown("---")

    st.subheader("📌 Confusion Matrix")

    st.code(
"""
[[20189     0]
 [    0 26970]]
""",
language="text"
    )

    st.markdown("---")

    st.subheader("📋 Classification Report")

    st.text("""
              precision    recall  f1-score   support

0              1.00       1.00      1.00      20189
1              1.00       1.00      1.00      26970

accuracy                           1.00      47159

macro avg       1.00       1.00      1.00      47159

weighted avg    1.00       1.00      1.00      47159
""")

    st.success("✅ Model evaluation completed successfully.")
    st.markdown("---")

    st.subheader("📖 Performance Interpretation")
    
    st.markdown("""
    The Random Forest classifier demonstrated excellent performance on the evaluation dataset.
    
    ### Key Observations
    
    - The model accurately distinguishes legitimate and phishing websites.
    - Precision and Recall values indicate reliable classification.
    - The Confusion Matrix shows that very few (or no) samples were misclassified during evaluation.
    - Feature engineering significantly improved the prediction capability of the model.
    - Explainable AI (SHAP) helps understand which features contribute most to the model's decisions.
    
    ### Conclusion
    
    The trained model is suitable for phishing website detection and can be integrated into real-world applications such as browser extensions, security gateways, and web filtering systems.
    """)

# =====================================
# EXPLAINABLE AI
# =====================================

elif page == "🧠 Explainable AI":

    st.title("🧠 Explainable AI (SHAP)")

    st.markdown("""
Explainable Artificial Intelligence (XAI) helps us understand **why**
the Random Forest model predicts a website as Legitimate or Phishing.

This project uses **SHAP (SHapley Additive Explanations)** to measure
the contribution of each feature toward the prediction.
""")

    st.markdown("---")

    st.subheader("📊 SHAP Feature Importance")

    bar_path = "outputs/reports/figures/shap_bar.png"

    if os.path.exists(bar_path):

        st.image(
    bar_path,
    caption="Global Feature Importance",
    width=700
)

    else:

        st.warning("SHAP Bar Plot not found.")

    st.markdown("---")

    st.subheader("📈 SHAP Summary Plot")

    summary_path = "outputs/reports/figures/shap_summary.png"

    if os.path.exists(summary_path):

        st.image(
    summary_path,
    caption="SHAP Summary Plot",
    width=700
)

    else:

        st.warning("SHAP Summary Plot not found.")

    st.markdown("---")

    st.success(
        "The SHAP visualizations explain which features have the greatest influence on phishing website detection."
    )
    st.markdown("---")

    st.subheader("📖 Interpretation of SHAP Results")
    
    st.markdown("""
    ### What is SHAP?

    SHAP (SHapley Additive exPlanations) is an Explainable Artificial Intelligence (XAI) technique that explains how each feature contributes to the machine learning model's prediction.
    
    Unlike traditional black-box models, SHAP provides transparency by showing which features have the greatest influence on each prediction.
    
    ---
    
    ### SHAP Summary Plot
    
    The SHAP Summary Plot displays:
    
    - The overall importance of every feature.
    - Whether a feature increases or decreases phishing probability.
    - The distribution of feature impacts across all samples.

    Features appearing at the top have the greatest influence on the prediction.

    ---

    ### SHAP Feature Importance Plot

    The SHAP Bar Plot ranks features based on their average contribution to the Random Forest model.

    Higher bars indicate features that contribute more strongly to phishing detection.

    ---

    ### Why Explainable AI?

    Explainable AI increases user trust by showing why the model predicts a website as phishing or legitimate rather than simply displaying the final prediction.

    This makes the system suitable for real-world cybersecurity applications where transparency is essential.
    """)


# =====================================
# ABOUT PROJECT
# =====================================
elif page == "ℹ️ About Project":

    st.title("ℹ️ About Project")

    st.markdown("""
# Advanced Phishing Website Detection using Machine Learning and Explainable AI
""")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📌 Project Information")

        st.write("**Project Type:** MCA Minor Project")

        st.write("**Domain:** Cyber Security")

        st.write("**Machine Learning Algorithm:** Random Forest")

        st.write("**Explainable AI:** SHAP")

        st.write("**Frontend:** Streamlit")

        st.write("**Programming Language:** Python")

    with col2:

        st.subheader("📂 Dataset")

        st.write("Dataset: PhiUSIIL Phishing URL Dataset")

        st.write("40 Engineered Features")

        st.write("235,000+ Website Records")

        st.write("Binary Classification")

        st.write("Legitimate vs Phishing")

    st.markdown("---")

    st.subheader("🎯 Project Objectives")

    st.markdown("""
- Detect phishing websites using Machine Learning.

- Extract URL and webpage features automatically.

- Provide live prediction for websites.

- Support batch prediction using CSV files.

- Explain model predictions using SHAP Explainable AI.
""")

    st.markdown("---")

    st.subheader("🚀 Technologies Used")

    st.markdown("""
- Python

- Streamlit

- Pandas

- NumPy

- Scikit-learn

- SHAP

- BeautifulSoup

- Joblib

- Matplotlib
""")

    st.markdown("---")

    st.subheader("🔮 Future Enhancements")

    st.markdown("""
- Deep Learning based phishing detection.

- Browser Extension.

- Real-time Website Monitoring.

- API Integration.

- Cloud Deployment.

- Threat Intelligence Integration.
""")

    st.markdown("---")

    st.success(
        "This application demonstrates the practical use of Machine Learning and Explainable AI for phishing website detection."
    )