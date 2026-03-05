import datetime

import pandas as pd
import requests
import streamlit as st

# Page config
st.set_page_config(page_title="Rossmann Sales Predictor", page_icon="📈", layout="wide")

st.title("🛍️ Rossmann Store Sales Predictor")
st.markdown(
    "Use this dashboard to get sales predictions from our machine learning model served via FastAPI."
)

st.sidebar.header("Store Properties")

# Collect inputs
store_id = st.sidebar.number_input("Store ID", min_value=1, value=1115)
day_of_week = st.sidebar.slider("Day of Week (1-7)", 1, 7, datetime.date.today().isoweekday())
date = st.sidebar.date_input("Date", datetime.date.today())
customers = st.sidebar.number_input("Expected Customers", value=500, step=50)
is_open = st.sidebar.selectbox("Is Open?", [1, 0])
promo = st.sidebar.selectbox("Running Promo?", [1, 0])
state_holiday = st.sidebar.selectbox("State Holiday", ["0", "a", "b", "c"])
school_holiday = st.sidebar.selectbox("School Holiday", [1, 0])
store_type = st.sidebar.selectbox("Store Type", ["a", "b", "c", "d"])
assortment = st.sidebar.selectbox("Assortment", ["a", "b", "c"])

# Advanced features
with st.sidebar.expander("Advanced Competitor Features"):
    comp_dist = st.number_input("Competition Distance", value=100.0)
    comp_month = st.number_input("Competition Open Since (Month)", value=1.0)
    comp_year = st.number_input("Competition Open Since (Year)", value=2010.0)
    promo2 = st.selectbox("Promo2 Running?", [1, 0], index=0)
    promo2_week = st.number_input("Promo2 Since Week", value=1.0)
    promo2_year = st.number_input("Promo2 Since Year", value=2015.0)
    promo_interval = st.text_input("Promo Interval", "Jan,Apr,Jul,Oct")

if st.button("Predict Sales 🚀"):
    data = [
        {
            "Store": store_id,
            "DayOfWeek": day_of_week,
            "Date": date.strftime("%Y-%m-%d"),
            "Customers": customers,
            "Open": is_open,
            "Promo": promo,
            "StateHoliday": state_holiday,
            "SchoolHoliday": school_holiday,
            "StoreType": store_type,
            "Assortment": assortment,
            "CompetitionDistance": comp_dist,
            "CompetitionOpenSinceMonth": comp_month,
            "CompetitionOpenSinceYear": comp_year,
            "Promo2": promo2,
            "Promo2SinceWeek": promo2_week,
            "Promo2SinceYear": promo2_year,
            "PromoInterval": promo_interval,
        }
    ]

    API_URL = "http://127.0.0.1:8000/predict"

    try:
        with st.spinner("Connecting to the Model API..."):
            response = requests.post(API_URL, json=data)

        if response.status_code == 200:
            result = response.json()
            if "error" in result:
                st.error(result["error"])
            else:
                prediction = result[0].get("Prediction", 0)
                st.success(f"### 🎉 Expected Sales: 💶 € {prediction:,.2f}")

                # Show dataframe interpretation
                st.write("---")
                st.markdown("**Processed Input Data Context:**")
                st.dataframe(pd.DataFrame(result))
        else:
            st.error(f"Error {response.status_code}: Could not fetch predictions.")

    except requests.exceptions.ConnectionError:
        st.error("🚨 Could not connect to the API. Is the FastAPI server running on port 8000?")
