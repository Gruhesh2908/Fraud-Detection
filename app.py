import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

# =========================================================
# Page Config
# =========================================================
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# =========================================================
# Load Models & Artifacts  (✅ define FIRST)
# =========================================================
@st.cache_resource
def load_all_models():
    models = {
        "Logistic Regression": joblib.load("models/logistic_regression.pkl"),
        "Random Forest": joblib.load("models/random_forest.pkl"),
    }

    # ✅ Load XGBoost safely (fix version mismatch error)
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("models/xgboost.json")
    models["XGBoost"] = xgb_model

    scaler = joblib.load("models/lr_scaler.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")

    return models, scaler, feature_columns

# ✅ IMPORTANT: Call loader BEFORE using models anywhere
models, scaler, feature_columns = load_all_models()

# =========================================================
# Sidebar (✅ now models exists)
# =========================================================
st.sidebar.header("⚙ Model Settings")

selected_model_name = st.sidebar.selectbox(
    "Choose Model",
    list(models.keys())
)

threshold = st.sidebar.slider(
    "Fraud Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

# =========================================================
# Header
# =========================================================
st.markdown(
    """
    <h1 style='text-align:center;'>💳 Credit Card Fraud Detection</h1>
    <p style='text-align:center; font-size:18px;'>
    Compare Logistic Regression, Random Forest & XGBoost models
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# =========================================================
# Prediction Helper (✅ works for all models)
# =========================================================
def predict_with_model(input_df, model_name):
    model = models[model_name]

    if model_name == "Logistic Regression":
        input_scaled = scaler.transform(input_df)
        proba = model.predict_proba(input_scaled)[:, 1][0]
    else:
        proba = model.predict_proba(input_df)[:, 1][0]

    pred = 1 if proba >= threshold else 0
    return pred, proba

# =========================================================
# Tabs
# =========================================================
tab1, tab2, tab3 = st.tabs([
    "🔍 Single Transaction",
    "📂 Batch CSV Prediction",
    "📘 About Models"
])

# =========================================================
# 🔍 SINGLE TRANSACTION
# =========================================================
with tab1:
    st.subheader("Enter Transaction Details")

    c1, c2, c3 = st.columns(3)

    with c1:
        step = st.number_input("Step", min_value=0)
        amount = st.number_input("Amount", min_value=0.0)
        oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0)

    with c2:
        newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0)
        oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0)

    with c3:
        newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0)
        isFlaggedFraud = st.selectbox("Is Flagged Fraud?", [0, 1])

    txn_type = st.selectbox(
        "Transaction Type",
        ["CASH_OUT", "TRANSFER", "PAYMENT", "DEBIT"]
    )

    orig_balance_diff = oldbalanceOrg - newbalanceOrig
    dest_balance_diff = newbalanceDest - oldbalanceDest

    if st.button("🔍 Predict", use_container_width=True):

        input_data = {
            "step": step,
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest,
            "isFlaggedFraud": isFlaggedFraud,
            "orig_balance_diff": orig_balance_diff,
            "dest_balance_diff": dest_balance_diff
        }

        # One-hot encoding for type columns
        for col in feature_columns:
            if col.startswith("type_"):
                input_data[col] = 1 if col == f"type_{txn_type}" else 0

        input_df = pd.DataFrame([input_data])

        # Align columns exactly
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[feature_columns]

        pred, prob = predict_with_model(input_df, selected_model_name)

        st.markdown("---")
        st.subheader("Prediction Result")
        st.write(f"**Model Used:** {selected_model_name}")
        st.write(f"**Threshold:** {threshold}")

        if pred == 1:
            st.error(f"🚨 Fraud Detected\n\nProbability: {prob:.2%}")
        else:
            st.success(f"✅ Legitimate Transaction\n\nConfidence: {(1-prob):.2%}")

# =========================================================
# 📂 BATCH CSV
# =========================================================
with tab2:
    st.subheader("Upload CSV File")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        raw_df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(raw_df.head())

        df = raw_df.copy()

        # Feature engineering
        df["orig_balance_diff"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
        df["dest_balance_diff"] = df["newbalanceDest"] - df["oldbalanceDest"]

        # One-hot encoding
        df = pd.get_dummies(df, columns=["type"], drop_first=True)

        # Align columns
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0

        df = df[feature_columns]

        # Predict probabilities
        if selected_model_name == "Logistic Regression":
            df_scaled = scaler.transform(df)
            proba = models[selected_model_name].predict_proba(df_scaled)[:, 1]
        else:
            proba = models[selected_model_name].predict_proba(df)[:, 1]

        results_df = raw_df.copy()
        results_df["Fraud_Probability"] = proba
        results_df["Prediction"] = (proba >= threshold).astype(int)

        st.subheader("Prediction Results")
        st.dataframe(results_df.head())

        st.download_button(
            "⬇ Download Results CSV",
            results_df.to_csv(index=False),
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )

# =========================================================
# 📘 ABOUT
# =========================================================
with tab3:
    st.markdown(
        """
        ### Models Available
        - **Logistic Regression (Class Weighted)**: baseline, interpretable  
        - **Random Forest (Class Weighted)**: best balance, high precision  
        - **XGBoost (Weighted)**: high recall, strong performance after tuning  

        ### Notes
        - Threshold slider helps control false positives vs recall
        - Use Batch Prediction for testing large files
        """
    )

    st.info("This app is for educational & demo purposes.")

# =========================================================
# Footer
# =========================================================
st.markdown(
    """
    <hr>
    <p style='text-align:center; font-size:14px;'>
    Fraud Detection System | Machine Learning Project
    </p>
    """,
    unsafe_allow_html=True
)
