"""Smoke tests for the data contract, leakage guarantees and the serving surface."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.predict import Predictor  # noqa: E402
from src.preprocessing import FeaturePrep, Preprocessor, engineer  # noqa: E402

STORE = {"Store": 1, "DayOfWeek": 4, "Date": "2015-08-01", "Open": 1, "Promo": 1,
         "StateHoliday": "0", "SchoolHoliday": 1, "StoreType": "c", "Assortment": "a",
         "CompetitionDistance": 1270, "CompetitionOpenSinceMonth": 9, "CompetitionOpenSinceYear": 2008,
         "Promo2": 0, "Promo2SinceWeek": None, "Promo2SinceYear": None, "PromoInterval": ""}


@pytest.fixture(scope="module")
def sample():
    return pd.read_csv(config.SAMPLE_PATH, low_memory=False)


def test_leakage_column_dropped(sample):
    X, y, dates = Preprocessor().run(sample)
    for col in config.LEAKAGE_COLS:
        assert col not in X.columns
    assert (y > 0).all()  # closed-day / zero-sale rows removed


def test_engineering_present(sample):
    eng = engineer(sample.head(200))
    for c in config.ENGINEERED_FEATURES:
        assert c in eng.columns


def test_feature_prep_fixed_columns(sample):
    a = FeaturePrep().fit_transform(sample.head(20))
    b = FeaturePrep().transform(sample.head(5))
    assert list(a.columns) == list(b.columns)


def test_predictor_contract():
    pred = Predictor()
    s = pred.predict_one(STORE)
    assert s > 0
    assert len(pred.top_features(5)) >= 1


def test_promo_raises_predicted_sales():
    pred = Predictor()
    with_promo = pred.predict_one({**STORE, "Promo": 1})
    without_promo = pred.predict_one({**STORE, "Promo": 0})
    assert with_promo > without_promo
