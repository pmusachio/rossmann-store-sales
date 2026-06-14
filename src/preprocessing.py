"""Transformation layer. Date-derived features, competition tenure and active-promo
flags live in a custom first pipeline step so training and serving share the
identical transform. Customers (a post-hoc field) is dropped to prevent leakage.
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config

logger = logging.getLogger(__name__)

_MONTHS = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
          7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    d = pd.to_datetime(out[config.DATE_COL], errors="coerce")
    out["year"] = d.dt.year
    out["month"] = d.dt.month
    out["day"] = d.dt.day
    out["week_of_year"] = d.dt.isocalendar().week.astype("float")
    dow = pd.to_numeric(out.get("DayOfWeek"), errors="coerce")
    out["day_of_week_sin"] = np.sin(2 * np.pi * dow / 7)
    out["day_of_week_cos"] = np.cos(2 * np.pi * dow / 7)
    cy = pd.to_numeric(out.get("CompetitionOpenSinceYear"), errors="coerce")
    cm = pd.to_numeric(out.get("CompetitionOpenSinceMonth"), errors="coerce")
    out["competition_open_months"] = ((out["year"] - cy) * 12 + (out["month"] - cm)).clip(lower=0).fillna(0)
    out["CompetitionDistance"] = pd.to_numeric(out.get("CompetitionDistance"), errors="coerce")
    promo2 = pd.to_numeric(out.get("Promo2"), errors="coerce").fillna(0)
    interval = out.get("PromoInterval", "").fillna("") if "PromoInterval" in out else pd.Series("", index=out.index)
    month_abbr = out["month"].map(_MONTHS)
    in_interval = np.array([m in s.split(",") if s else False for m, s in zip(month_abbr, interval)])
    out["promo2_active"] = ((promo2 == 1).to_numpy() & in_interval).astype(float)
    sh = out.get("StateHoliday")
    if sh is not None:
        out["StateHoliday"] = sh.astype(str).replace({"0": "none", "0.0": "none"})
    return out


class FeaturePrep(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X) -> pd.DataFrame:
        df = engineer(pd.DataFrame(X).copy())
        cols = list(config.NUMERIC_FEATURES) + list(config.CATEGORICAL_FEATURES)
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan if c in config.NUMERIC_FEATURES else "none"
        return df[cols]


def build_column_transformer() -> ColumnTransformer:
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer(
        [("num", numeric_pipe, list(config.NUMERIC_FEATURES)),
         ("cat", cat_pipe, list(config.CATEGORICAL_FEATURES))], remainder="drop")


class Preprocessor:
    def __init__(self, processed_path=config.PROCESSED_PATH) -> None:
        self.processed_path = processed_path

    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        d = df.copy()
        d = d.drop(columns=[c for c in config.LEAKAGE_COLS if c in d.columns], errors="ignore")
        if "Open" in d.columns:
            d = d[d["Open"] == 1]
        d = d[pd.to_numeric(d[config.TARGET], errors="coerce") > 0]
        dates = pd.to_datetime(d[config.DATE_COL], errors="coerce")
        order = dates.sort_values().index
        d = d.loc[order].reset_index(drop=True)
        dates = dates.loc[order].reset_index(drop=True)
        y = pd.to_numeric(d[config.TARGET], errors="coerce")
        X = d.drop(columns=[config.TARGET])
        self.processed_path.parent.mkdir(parents=True, exist_ok=True)
        engineer(d).head(50000).to_parquet(self.processed_path, index=False)
        logger.info("Prepared %d open-day records (chronological)", len(d))
        return X, y, dates
