"""Feature engineering for Rossmann sales forecasting.

Follows the Scikit-Learn transformer API so every step plugs into a Pipeline,
as described in Aurélien Géron – *Hands-On Machine Learning* Ch. 2.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .data import normalize_columns


# ──────────────────────────────────────────────────────────────────────────────
# Lookup tables
# ──────────────────────────────────────────────────────────────────────────────

MONTH_MAP = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}

ASSORTMENT_MAP = {"a": "basic", "b": "extra", "c": "extended"}
HOLIDAY_MAP = {
    "a": "public_holiday", "b": "easter_holiday",
    "c": "christmas", "0": "regular_day", 0: "regular_day",
}

# Features consumed by the model after transformation
FEATURE_COLUMNS = [
    "store",
    "day_of_week",
    "promo",
    "school_holiday",
    "state_holiday",
    "store_type",
    "assortment",
    "competition_distance",
    "competition_open_since_month",
    "competition_open_since_year",
    "promo2",
    "promo2_since_week",
    "promo2_since_year",
    "is_promo",
    "year",
    "month",
    "day",
    "week_of_year",
    "competition_time_month",
    "promo_time_week",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "day_sin",
    "day_cos",
    "week_of_year_sin",
    "week_of_year_cos",
]

CATEGORICAL_FEATURES = ["state_holiday", "store_type", "assortment"]
NUMERIC_FEATURES = [col for col in FEATURE_COLUMNS if col not in CATEGORICAL_FEATURES]


# ──────────────────────────────────────────────────────────────────────────────
# Custom Scikit-Learn Transformer
# ──────────────────────────────────────────────────────────────────────────────

class RossmannFeatureTransformer(BaseEstimator, TransformerMixin):
    """Transform raw Rossmann data into model-ready features.

    Implements the Scikit-Learn transformer contract (fit / transform) so it
    can be composed inside a ``sklearn.pipeline.Pipeline``.  The ``fit`` step
    is stateless for this transformer — it only needs to be called because the
    Pipeline API requires it.

    Parameters
    ----------
    training : bool
        When True, rows with ``open == 0`` *and* ``sales <= 0`` are dropped,
        matching the same filtering applied during model training.
    """

    def __init__(self, training: bool = False) -> None:
        self.training = training

    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:  # noqa: N803
        return prepare_features(X, training=self.training)


# ──────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ──────────────────────────────────────────────────────────────────────────────

def _safe_week(series: pd.Series) -> pd.Series:
    return series.dt.isocalendar().week.astype("int64")


def _column_or_default(df: pd.DataFrame, column: str, default) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series(default, index=df.index)


# ──────────────────────────────────────────────────────────────────────────────
# Core transformation pipeline
# ──────────────────────────────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame, training: bool = False) -> pd.DataFrame:
    """Apply all feature-engineering steps to a raw Rossmann DataFrame.

    Steps
    -----
    1. Normalize column names to snake_case.
    2. Parse dates and derive calendar features (year, month, day, week).
    3. Encode cyclical features with sine/cosine to preserve periodicity
       (a technique recommended in the book for time-aware models).
    4. Compute competition and promo duration in months / weeks.
    5. Map categorical codes to human-readable labels.
    6. Filter closed stores and, during training, zero-sales rows.
    7. Guarantee all model features are present in the output frame.
    """
    result = normalize_columns(df)

    if "date" not in result.columns:
        raise KeyError("Input data must contain a 'date' column.")

    # ── 1. Date parsing ────────────────────────────────────────────────────────
    result["date"] = pd.to_datetime(result["date"])

    if "day_of_week" not in result.columns:
        result["day_of_week"] = result["date"].dt.dayofweek + 1

    # ── 2. Fill missing operational columns ───────────────────────────────────
    for col, default in {"open": 1, "promo": 0, "school_holiday": 0, "promo2": 0}.items():
        if col not in result.columns:
            result[col] = default

    # ── 3. Competition features ────────────────────────────────────────────────
    result["competition_distance"] = (
        _column_or_default(result, "competition_distance", np.nan).fillna(200_000.0)
    )
    result["competition_open_since_month"] = (
        _column_or_default(result, "competition_open_since_month", np.nan)
        .fillna(result["date"].dt.month)
        .astype(int)
    )
    result["competition_open_since_year"] = (
        _column_or_default(result, "competition_open_since_year", np.nan)
        .fillna(result["date"].dt.year)
        .astype(int)
    )
    result["competition_since"] = result.apply(
        lambda row: dt.datetime(
            int(row["competition_open_since_year"]),
            int(row["competition_open_since_month"]),
            1,
        ),
        axis=1,
    )
    result["competition_time_month"] = (
        ((result["date"] - result["competition_since"]) / 30).dt.days.astype(int)
    )

    # ── 4. Promo2 features ────────────────────────────────────────────────────
    result["promo2_since_week"] = (
        _column_or_default(result, "promo2_since_week", np.nan)
        .fillna(_safe_week(result["date"]))
        .astype(int)
    )
    result["promo2_since_year"] = (
        _column_or_default(result, "promo2_since_year", np.nan)
        .fillna(result["date"].dt.year)
        .astype(int)
    )
    result["promo_interval"] = (
        _column_or_default(result, "promo_interval", "0").fillna("0").astype(str)
    )
    result["month_map"] = result["date"].dt.month.map(MONTH_MAP)
    result["is_promo"] = result.apply(
        lambda row: 0
        if row["promo_interval"] in {"0", "nan", ""}
        else int(row["month_map"] in row["promo_interval"].split(",")),
        axis=1,
    )
    promo_str = result["promo2_since_year"].astype(str) + "-" + result["promo2_since_week"].astype(str) + "-1"
    result["promo_since"] = promo_str.apply(
        lambda v: dt.datetime.strptime(v, "%Y-%W-%w") - dt.timedelta(days=7)
    )
    result["promo_time_week"] = (
        ((result["date"] - result["promo_since"]) / 7).dt.days.astype(int)
    )

    # ── 5. Calendar features ──────────────────────────────────────────────────
    result["year"] = result["date"].dt.year
    result["month"] = result["date"].dt.month
    result["day"] = result["date"].dt.day
    result["week_of_year"] = _safe_week(result["date"])

    # ── 6. Cyclical encoding (sine/cosine) ────────────────────────────────────
    #   Preserves the circular nature of periodic features so the model sees
    #   Monday and Sunday as adjacent, not at opposite ends of a linear scale.
    result["day_of_week_sin"] = np.sin(result["day_of_week"] * (2 * np.pi / 7))
    result["day_of_week_cos"] = np.cos(result["day_of_week"] * (2 * np.pi / 7))
    result["month_sin"] = np.sin(result["month"] * (2 * np.pi / 12))
    result["month_cos"] = np.cos(result["month"] * (2 * np.pi / 12))
    result["day_sin"] = np.sin(result["day"] * (2 * np.pi / 30))
    result["day_cos"] = np.cos(result["day"] * (2 * np.pi / 30))
    result["week_of_year_sin"] = np.sin(result["week_of_year"] * (2 * np.pi / 52))
    result["week_of_year_cos"] = np.cos(result["week_of_year"] * (2 * np.pi / 52))

    # ── 7. Categorical encoding ───────────────────────────────────────────────
    result["assortment"] = (
        _column_or_default(result, "assortment", "a")
        .replace(ASSORTMENT_MAP)
        .fillna("basic")
    )
    result["state_holiday"] = (
        _column_or_default(result, "state_holiday", "0")
        .replace(HOLIDAY_MAP)
        .fillna("regular_day")
    )
    result["store_type"] = (
        _column_or_default(result, "store_type", "a").fillna("a").astype(str)
    )

    # ── 8. Filter rows ────────────────────────────────────────────────────────
    result = result[result["open"].fillna(0).astype(int) != 0]
    if training and "sales" in result.columns:
        result = result[result["sales"] > 0]

    # ── 9. Guarantee all model features exist ─────────────────────────────────
    for col in FEATURE_COLUMNS:
        if col not in result.columns:
            result[col] = 0

    return result


def model_frame(
    df: pd.DataFrame, training: bool = False
) -> tuple[pd.DataFrame, pd.Series | None, pd.DataFrame]:
    """Return ``(X, y, prepared_df)`` ready for a Scikit-Learn estimator."""
    prepared = prepare_features(df, training=training)
    X = prepared[FEATURE_COLUMNS].copy()
    y = prepared["sales"].copy() if "sales" in prepared.columns else None
    return X, y, prepared
