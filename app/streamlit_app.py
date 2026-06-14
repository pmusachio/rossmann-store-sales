"""Interactive store-sales forecasting dashboard.

Predicts a Rossmann store's daily sales from its characteristics, the calendar and
the promotion flag, and shows the promotion uplift and the daily sales pattern.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.predict import Predictor  # noqa: E402

D = config.DRACULA
st.set_page_config(page_title="Store Sales Forecast", layout="wide")
st.markdown(
    f"""<style>
    .stApp {{ background-color: {D['background']}; color: {D['foreground']}; }}
    section[data-testid="stSidebar"] {{ background-color: {D['current_line']}; }}
    h1, h2, h3 {{ color: {D['purple']}; }}
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_predictor() -> Predictor:
    return Predictor()


def style_axes(ax):
    ax.set_facecolor(D["background"])
    for s in ax.spines.values():
        s.set_color(D["current_line"])
    ax.tick_params(colors=D["foreground"])
    ax.xaxis.label.set_color(D["foreground"])
    ax.yaxis.label.set_color(D["foreground"])
    ax.grid(True, axis="y", color=D["current_line"], linestyle="--", alpha=0.4)


def weekly_chart(predictor, base):
    days = list(range(1, 8))
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    sales = [predictor.predict_one({**base, "DayOfWeek": d}) for d in days]
    fig, ax = plt.subplots(figsize=(6, 3.2), facecolor=D["background"])
    ax.bar(names, sales, color=D["purple"], edgecolor=D["current_line"])
    ax.set_ylabel("Predicted sales")
    style_axes(ax)
    fig.tight_layout()
    return fig


def main():
    try:
        predictor = load_predictor()
    except FileNotFoundError:
        st.error("Model artifact not found. Run the pipeline before launching the app.")
        return

    st.title("Rossmann Store Sales — Daily Forecast")
    st.markdown(
        "Predicts a store's daily sales from its profile, the calendar and the promotion flag, "
        "to support staffing and inventory planning."
    )

    with st.sidebar:
        st.header("Store and day")
        store_type = st.selectbox("Store type", ["a", "b", "c", "d"], index=2)
        assortment = st.selectbox("Assortment", ["a", "b", "c"], index=0)
        comp_distance = st.number_input("Competition distance (m)", 20.0, 80000.0, 1270.0, 100.0)
        day_of_week = st.selectbox("Day of week", list(range(1, 8)), index=3,
                                   format_func=lambda d: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d-1])
        date = st.date_input("Date", value=pd.to_datetime("2015-08-01"))
        promo = st.selectbox("Promotion running", [1, 0], format_func=lambda v: "Yes" if v else "No")
        school_holiday = st.selectbox("School holiday", [0, 1], format_func=lambda v: "Yes" if v else "No")
        run = st.button("Forecast sales", type="primary")

    base = {"Store": 1, "DayOfWeek": day_of_week, "Date": str(date), "Open": 1, "Promo": promo,
            "StateHoliday": "0", "SchoolHoliday": school_holiday, "StoreType": store_type,
            "Assortment": assortment, "CompetitionDistance": comp_distance,
            "CompetitionOpenSinceMonth": 9, "CompetitionOpenSinceYear": 2008,
            "Promo2": 0, "Promo2SinceWeek": None, "Promo2SinceYear": None, "PromoInterval": ""}

    if run:
        sales = predictor.predict_one(base)
        no_promo = predictor.predict_one({**base, "Promo": 0})
        with_promo = predictor.predict_one({**base, "Promo": 1})
        uplift = (with_promo - no_promo) / no_promo * 100 if no_promo else 0
        st.subheader("Forecast")
        c = st.columns(3)
        c[0].metric("Predicted daily sales", f"${sales:,.0f}")
        c[1].metric("Promotion uplift", f"{uplift:+.0f}%")
        c[2].metric("Median store-day", f"${predictor.median_sales:,.0f}")
        left, right = st.columns(2)
        with left:
            st.pyplot(weekly_chart(predictor, base))
        with right:
            st.markdown("**Most influential features (model-wide)**")
            imp = pd.DataFrame(predictor.top_features(6)).rename(
                columns={"feature": "Feature", "importance": "Permutation importance (RMSE)"})
            st.dataframe(imp, hide_index=True, width="stretch")


if __name__ == "__main__":
    main()
