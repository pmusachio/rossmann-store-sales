import pandas as pd

from rossmann_store_sales.features import FEATURE_COLUMNS, prepare_features
from rossmann_store_sales.models import regression_metrics


def test_prepare_features_contract():
    df = pd.DataFrame(
        [
            {
                "Store": 1,
                "DayOfWeek": 4,
                "Date": "2015-08-01",
                "Open": 1,
                "Promo": 1,
                "StateHoliday": "0",
                "SchoolHoliday": 1,
                "StoreType": "c",
                "Assortment": "a",
                "CompetitionDistance": 1270,
                "CompetitionOpenSinceMonth": 9,
                "CompetitionOpenSinceYear": 2008,
                "Promo2": 0,
                "Promo2SinceWeek": None,
                "Promo2SinceYear": None,
                "PromoInterval": None,
            }
        ]
    )
    prepared = prepare_features(df)
    assert set(FEATURE_COLUMNS).issubset(prepared.columns)
    assert len(prepared) == 1


def test_regression_metrics_contract():
    metrics = regression_metrics([100, 200, 300], [90, 220, 330])

    assert set(metrics) == {"mae", "rmse", "mape", "rmspe"}
    assert metrics["mae"] > 0
    assert metrics["rmse"] > 0
