import pandas as pd
import streamlit as st
import joblib
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌱",
    layout="centered"
)

DATA_FILE = Path(__file__).parent / "crop-yield.csv"
MODEL_FILE = Path(__file__).parent / "crop_yield_model.pkl"

st.title("🌱 Crop Yield Prediction")
st.write("Enter the farm conditions below to predict the expected crop yield.")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "Dataset `crop-yield.csv` not found. "
        "Please add it to the repository root and redeploy."
    )
    st.stop()

TARGET = "Crop_Yield_ton_per_hectare"

if TARGET not in df.columns:
    st.error(f"Target column `{TARGET}` not found in the dataset.")
    st.stop()

features = [col for col in df.columns if col != TARGET]

numeric_features = df[features].select_dtypes(include="number").columns.tolist()
categorical_features = [col for col in features if col not in numeric_features]

try:
    model = load_model(MODEL_FILE)
except FileNotFoundError:
    st.error(
        "Trained model `crop_yield_model.pkl` not found. "
        "Please add it to the repository root and redeploy."
    )
    st.stop()

with st.form("prediction_form"):
    inputs = {}

    st.subheader("Soil, nutrients and weather")
    for col in numeric_features:
        median_value = pd.to_numeric(df[col], errors="coerce").median()
        if pd.isna(median_value):
            median_value = 0.0
        inputs[col] = st.number_input(
            col.replace("_", " ").title(),
            value=float(median_value),
            format="%.2f"
        )

    st.subheader("Crop and farm management")
    for col in categorical_features:
        options = df[col].dropna().unique().tolist()
        if not options:
            st.error(f"No available values for {col}")
            st.stop()
        inputs[col] = st.selectbox(
            col.replace("_", " ").title(),
            options=options
        )

    submitted = st.form_submit_button("Predict Crop Yield")

if submitted:
    input_df = pd.DataFrame([inputs])[features]
    try:
        prediction = float(model.predict(input_df)[0])
        st.success(f"Predicted yield: **{prediction:.2f} tonnes per hectare**")
        st.divider()
        st.subheader("Model comparison")
        st.write(
            "This app uses an XGBoost pipeline trained on the crop-yield dataset, "
            "selected after comparing Linear Regression, Ridge, Random Forest, "
            "Extra Trees, Gradient Boosting, and XGBoost."
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")
