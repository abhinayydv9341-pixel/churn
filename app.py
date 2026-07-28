"""Streamlit app: score a single customer against the trained model."""
import joblib
import pandas as pd
import streamlit as st

from src.preprocess import CATEGORICAL, encode

st.title("Customer Churn Predictor")

try:
    bundle = joblib.load("models/best.pkl")
except FileNotFoundError:
    st.error("No model found. Run `python -m src.train` first.")
    st.stop()

model, columns = bundle["model"], bundle["columns"]

tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly = st.slider("Monthly charges", 15.0, 130.0, 70.0)
contract = st.selectbox("Contract", ["month-to-month", "one-year", "two-year"])
internet = st.selectbox("Internet", ["fiber", "dsl", "none"])
support = st.checkbox("Tech support", value=False)

row = pd.DataFrame(
    [{
        "tenure": tenure,
        "monthly_charges": monthly,
        "contract": contract,
        "internet": internet,
        "tech_support": int(support),
    }]
)
X = encode(row).reindex(columns=columns, fill_value=0)
prob = float(model.predict_proba(X)[0, 1])

st.metric("Churn probability", f"{prob:.0%}")
st.progress(prob)
st.write("**High risk**" if prob > 0.5 else "**Likely to stay**")
