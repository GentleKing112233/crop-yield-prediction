import pandas as pd
import streamlit as st
import joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌱",
    layout="centered"
)

DATA_FILE = Path(__file__).parent / "crop-yield.csv"
TARGET = "Crop_Yield_ton_per_hectare"

st.title("🌱 Crop Yield Prediction")
st.write("Enter the farm conditions below to predict the expected crop yield.")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)

try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "Dataset `crop-yield.csv` not found. "
        "Please add it to the repository root and redeploy."
    )
    st.stop()

if TARGET not in df.columns:
    st.error(f"Target column `{TARGET}` not found in the dataset.")
    st.stop()

df = df.dropna(subset=[TARGET]).copy()

features = [col for col in df.columns if col != TARGET]
X = df.drop(columns=[TARGET])
y = df[TARGET]

numeric_features = X.select_dtypes(include="number").columns.tolist()
categorical_features = [col for col in features if col not in numeric_features]

@st.cache_resource
def train_model(_df, _target, _features, _numeric, _categorical):
    X = _df.drop(columns=[_target])
    y = _df[_target]

    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_transformer, _numeric),
        ("cat", categorical_transformer, _categorical)
    ])

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X, y)
    return model

try:
    model = train_model(df, TARGET, features, numeric_features, categorical_features)
except Exception as e:
    st.error(f"Model training failed: {e}")
    st.stop()

st.success("Model trained successfully!")

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
        st.subheader("About this model")
        st.write(
            "This app trains an XGBoost pipeline on the crop-yield dataset at startup. "
            "It compares favorably against Linear Regression, Ridge, Random Forest, "
            "Extra Trees, and Gradient Boosting baselines."
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")
