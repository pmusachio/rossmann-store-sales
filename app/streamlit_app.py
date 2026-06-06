"""Streamlit app for Rossmann sales forecasts."""

from __future__ import annotations

import json

import pandas as pd
import requests
import streamlit as st


st.set_page_config(page_title="Rossmann Store Sales", layout="wide")
st.title("Rossmann Store Sales")

api_url = st.text_input("API URL", "http://127.0.0.1:8000/rossmann/predict")
uploaded = st.file_uploader("CSV with store/date records", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    df = pd.read_csv("data/sample/store_1_scoring.csv")

st.dataframe(df.head(30), use_container_width=True, hide_index=True)

if st.button("Forecast", type="primary"):
    response = requests.post(api_url, json={"records": df.to_dict(orient="records")}, timeout=60)
    response.raise_for_status()
    result = pd.DataFrame(response.json())
    st.metric("Total forecast", f"{result['prediction'].sum():,.2f}")
    if "date" in result.columns:
        result["date"] = pd.to_datetime(result["date"])
        st.line_chart(result.set_index("date")["prediction"])
    st.dataframe(result, use_container_width=True, hide_index=True)

with st.expander("Payload example"):
    st.code(json.dumps({"records": df.head(2).to_dict(orient="records")}, indent=2, default=str), language="json")
