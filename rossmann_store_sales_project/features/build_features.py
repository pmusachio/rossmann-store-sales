import datetime
import math

import inflection
import numpy as np
import pandas as pd


class DataPreprocessor:
    """Transforms raw dataframe to features."""

    def __init__(self):
        self.cols_to_drop = [
            "open",
            "promo_interval",
            "month_map",
            "week_of_year",
            "day",
            "month",
            "day_of_week",
            "promo_since",
            "competition_since",
        ]

    def _clean_names(self, df):
        cols_old = df.columns.tolist()

        def snakecase(x):
            return inflection.underscore(x)

        df.columns = list(map(snakecase, cols_old))
        return df

    def _fill_na(self, df):
        df["competition_distance"] = df["competition_distance"].apply(
            lambda x: 200000.0 if math.isnan(x) else x
        )
        df["date"] = pd.to_datetime(df["date"])

        df["competition_open_since_month"] = df.apply(
            lambda x: (
                x["date"].month
                if math.isnan(x["competition_open_since_month"])
                else x["competition_open_since_month"]
            ),
            axis=1,
        )
        df["competition_open_since_year"] = df.apply(
            lambda x: (
                x["date"].year
                if math.isnan(x["competition_open_since_year"])
                else x["competition_open_since_year"]
            ),
            axis=1,
        )

        df["promo2_since_week"] = df.apply(
            lambda x: (
                x["date"].week if math.isnan(x["promo2_since_week"]) else x["promo2_since_week"]
            ),
            axis=1,
        )
        df["promo2_since_year"] = df.apply(
            lambda x: (
                x["date"].year if math.isnan(x["promo2_since_year"]) else x["promo2_since_year"]
            ),
            axis=1,
        )

        month_map = {
            1: "Jan",
            2: "Fev",
            3: "Mar",
            4: "Apr",
            5: "May",
            6: "Jun",
            7: "Jul",
            8: "Aug",
            9: "Sep",
            10: "Oct",
            11: "Nov",
            12: "Dec",
        }
        df["promo_interval"].fillna(0, inplace=True)
        df["month_map"] = df["date"].dt.month.map(month_map)
        df["is_promo"] = df[["promo_interval", "month_map"]].apply(
            lambda x: (
                0
                if x["promo_interval"] == 0
                else 1
                if x["month_map"] in x["promo_interval"].split(",")
                else 0
            ),
            axis=1,
        )

        df["competition_open_since_month"] = df["competition_open_since_month"].astype(int)
        df["competition_open_since_year"] = df["competition_open_since_year"].astype(int)
        df["promo2_since_week"] = df["promo2_since_week"].astype(int)
        df["promo2_since_year"] = df["promo2_since_year"].astype(int)
        return df

    def _feature_engineering(self, X):
        X["year"] = X["date"].dt.year
        X["month"] = X["date"].dt.month
        X["day"] = X["date"].dt.day
        X["week_of_year"] = X["date"].dt.isocalendar().week.astype(int)

        X["competition_since"] = X.apply(
            lambda x: datetime.datetime(
                year=x["competition_open_since_year"],
                month=x["competition_open_since_month"],
                day=1,
            ),
            axis=1,
        )
        X["competition_time_month"] = (
            ((X["date"] - X["competition_since"]) / 30).apply(lambda x: x.days).astype(int)
        )

        X["promo_since"] = (
            X["promo2_since_year"].astype(str) + "-" + X["promo2_since_week"].astype(str)
        )
        X["promo_since"] = X["promo_since"].apply(
            lambda x: datetime.datetime.strptime(x + "-1", "%Y-%W-%w") - datetime.timedelta(days=7)
        )
        X["promo_time_week"] = (
            ((X["date"] - X["promo_since"]) / 7).apply(lambda x: x.days).astype(int)
        )

        X["assortment"] = X["assortment"].apply(
            lambda x: "basic" if x == "a" else "extra" if x == "b" else "extended"
        )
        X["state_holiday"] = X["state_holiday"].apply(
            lambda x: (
                "public_holiday"
                if x == "a"
                else "easter_holiday"
                if x == "b"
                else "christmas"
                if x == "c"
                else "regular_day"
            )
        )

        X = X[(X["open"] != 0)]
        if "sales" in X.columns:
            X = X[(X["sales"] > 0)]

        X["day_of_week_sin"] = X["day_of_week"].apply(lambda x: np.sin(x * (2.0 * np.pi / 7)))
        X["day_of_week_cos"] = X["day_of_week"].apply(lambda x: np.cos(x * (2.0 * np.pi / 7)))
        X["month_sin"] = X["month"].apply(lambda x: np.sin(x * (2.0 * np.pi / 12)))
        X["month_cos"] = X["month"].apply(lambda x: np.cos(x * (2.0 * np.pi / 12)))
        X["day_sin"] = X["day"].apply(lambda x: np.sin(x * (2.0 * np.pi / 30)))
        X["day_cos"] = X["day"].apply(lambda x: np.cos(x * (2.0 * np.pi / 30)))
        X["week_of_year_sin"] = X["week_of_year"].apply(lambda x: np.sin(x * (2.0 * np.pi / 52)))
        X["week_of_year_cos"] = X["week_of_year"].apply(lambda x: np.cos(x * (2.0 * np.pi / 52)))

        X = X.drop(self.cols_to_drop, axis=1)
        return X

    def process(self, df_sales, df_store):
        df_raw = pd.merge(df_sales, df_store, how="left", on="Store")
        df_cleaned = self._clean_names(df_raw)
        df_filled = self._fill_na(df_cleaned)
        df_features = self._feature_engineering(df_filled)

        cols_selected = [
            "store",
            "promo",
            "store_type",
            "assortment",
            "competition_distance",
            "competition_open_since_month",
            "competition_open_since_year",
            "promo2",
            "promo2_since_week",
            "promo2_since_year",
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
            "date",
        ]

        if "sales" in df_features.columns:
            cols_selected.append("sales")

        return df_features[cols_selected]
