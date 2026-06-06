"""Training and prediction routines."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import load_config, resolve_project_path
from .data import load_scoring_frame, load_training_frame, merge_store, normalize_columns, read_csv
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, model_frame


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_estimator(config: dict):
    modeling = config.get("modeling", {})
    if modeling.get("algorithm", "xgboost").lower() == "xgboost":
        try:
            from xgboost import XGBRegressor

            return XGBRegressor(
                objective="reg:squarederror",
                n_estimators=int(modeling.get("n_estimators", 450)),
                learning_rate=float(modeling.get("learning_rate", 0.05)),
                max_depth=int(modeling.get("max_depth", 8)),
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=int(modeling.get("random_state", 42)),
                n_jobs=-1,
            )
        except ImportError:
            pass

    return HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.06,
        random_state=int(modeling.get("random_state", 42)),
        l2_regularization=0.01,
    )


def build_pipeline(config: dict) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", _one_hot_encoder(), CATEGORICAL_FEATURES),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", build_estimator(config))])


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.where(y_true == 0, np.nan, y_true)
    mape = np.nanmean(np.abs((y_true - y_pred) / denominator))
    rmspe = np.sqrt(np.nanmean(np.square((y_true - y_pred) / denominator)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": float(mape),
        "rmspe": float(rmspe),
    }


def time_split(df: pd.DataFrame, weeks: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    cutoff = df["date"].max() - pd.Timedelta(weeks=int(weeks))
    train = df[df["date"] < cutoff]
    valid = df[df["date"] >= cutoff]
    return train, valid


def business_scenarios(predictions: pd.DataFrame, mape: float) -> dict:
    total = float(predictions["prediction"].sum())
    return {
        "predicted_sales": total,
        "worst_scenario": total * (1 - float(mape)),
        "best_scenario": total * (1 + float(mape)),
        "mape_used": float(mape),
    }


def train(config_path: str | Path | None = None) -> dict:
    config = load_config(config_path)
    df = load_training_frame(config)
    train_df, valid_df = time_split(df, config.get("modeling", {}).get("validation_weeks", 6))

    X_train, y_train, _ = model_frame(train_df, training=True)
    X_valid, y_valid, valid_prepared = model_frame(valid_df, training=True)

    pipeline = build_pipeline(config)
    pipeline.fit(X_train, np.log1p(y_train))

    pred_valid = np.expm1(pipeline.predict(X_valid)).clip(min=0)
    metrics = regression_metrics(y_valid, pred_valid)

    model_path = resolve_project_path(config, config.get("modeling", {}).get("model_file", "models/model.joblib"))
    metrics_path = resolve_project_path(config, "reports/metrics.json")
    predictions_path = resolve_project_path(config, "reports/validation_predictions.csv")
    scenarios_path = resolve_project_path(config, "reports/business_scenarios.json")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    metrics_payload = {
        **metrics,
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_valid)),
        "model_path": str(model_path),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    out = valid_prepared[["store", "date", "sales"]].copy()
    out["prediction"] = pred_valid
    out.to_csv(predictions_path, index=False)
    scenarios_path.write_text(json.dumps(business_scenarios(out, metrics["mape"]), indent=2), encoding="utf-8")

    return metrics_payload


def load_model(config: dict):
    model_path = resolve_project_path(config, config.get("modeling", {}).get("model_file", "models/model.joblib"))
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run `PYTHONPATH=src python -m rossmann_store_sales.cli train` first.")
    return joblib.load(model_path)


def predict_records(records: list[dict], config_path: str | Path | None = None) -> list[dict]:
    config = load_config(config_path)
    model = load_model(config)
    raw = pd.DataFrame(records)
    X, _, prepared = model_frame(raw, training=False)
    if X.empty:
        return []
    predictions = np.expm1(model.predict(X)).clip(min=0)
    response = prepared.copy()
    response["prediction"] = predictions
    return json.loads(response.to_json(orient="records", date_format="iso"))


def predict_file(input_path: str | Path, output_path: str | Path | None = None, config_path: str | Path | None = None) -> dict:
    config = load_config(config_path)
    raw = read_csv(input_path)
    store_path = resolve_project_path(config, config["data"]["store_file"])
    normalized_columns = set(normalize_columns(raw.head(0)).columns)
    if store_path.exists() and "store_type" not in normalized_columns:
        store = read_csv(store_path)
        raw = merge_store(raw, store)
    predictions = predict_records(raw.to_dict(orient="records"), config_path)
    output = Path(output_path) if output_path else resolve_project_path(config, "data/processed/predictions.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions).to_csv(output, index=False)
    return {"rows": len(predictions), "output": str(output)}


def forecast_test(config_path: str | Path | None = None) -> pd.DataFrame:
    config = load_config(config_path)
    records = load_scoring_frame(config).to_dict(orient="records")
    return pd.DataFrame(predict_records(records, config_path))
