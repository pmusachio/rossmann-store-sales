"""Modeling layer: linear baseline, cross-validated gradient-boosting with tuning,
chronological holdout evaluation (RMSE / MAE / R2 / RMSPE) and slices, promo-uplift
business translation, and serialization of a self-contained pipeline.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from src import config
from src.preprocessing import FeaturePrep, build_column_transformer

logger = logging.getLogger(__name__)
SCHEMA_VERSION = "1.0"
N_JOBS = 1


def rmspe(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > 0
    return float(np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2)))


def _metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "rmspe": round(rmspe(y_true, y_pred), 4),
    }


@dataclass
class TrainingResult:
    baseline: Dict[str, Any] = field(default_factory=dict)
    best_params: Dict[str, Any] = field(default_factory=dict)
    holdout: Dict[str, Any] = field(default_factory=dict)
    business: Dict[str, Any] = field(default_factory=dict)
    importances: list = field(default_factory=list)


def _pipeline(estimator) -> Pipeline:
    return Pipeline([("prep", FeaturePrep()), ("ct", build_column_transformer()), ("model", estimator)])


class ModelTrainer:
    def __init__(self, X: pd.DataFrame, y: pd.Series, dates: pd.Series,
                 data_source: Path | None = None) -> None:
        self.data_source = data_source
        cut = int(len(X) * (1 - config.HOLDOUT_FRACTION))
        self.X_train, self.X_holdout = X.iloc[:cut], X.iloc[cut:]
        self.y_train, self.y_holdout = y.iloc[:cut], y.iloc[cut:]
        self.median_sales = float(y.median())
        self.result = TrainingResult()

    def fit_baseline(self) -> Dict[str, Any]:
        pipe = _pipeline(LinearRegression()).fit(self.X_train, self.y_train)
        pred = np.maximum(pipe.predict(self.X_holdout), 0)
        self.result.baseline = {"model": "LinearRegression", **_metrics(self.y_holdout, pred)}
        logger.info("Baseline RMSPE=%.4f R2=%.4f", self.result.baseline["rmspe"], self.result.baseline["r2"])
        return self.result.baseline

    def fit(self) -> Pipeline:
        sub = self.X_train.tail(config.N_SELECT)
        suby = self.y_train.tail(config.N_SELECT)
        tscv = TimeSeriesSplit(n_splits=config.CV_FOLDS)
        params = {
            "model__learning_rate": np.logspace(-2, -0.3, 12),
            "model__max_leaf_nodes": [31, 63, 127, 255],
            "model__max_depth": [None, 6, 10],
            "model__l2_regularization": [0.0, 0.1, 1.0],
            "model__max_iter": [300, 500, 800],
        }
        search = RandomizedSearchCV(
            _pipeline(HistGradientBoostingRegressor(random_state=config.SEED)), params,
            n_iter=config.TUNING_ITERS, scoring="neg_root_mean_squared_error", cv=tscv,
            n_jobs=N_JOBS, random_state=config.SEED, refit=False).fit(sub, suby)
        self.result.best_params = {k: _j(v) for k, v in search.best_params_.items()}
        logger.info("Tuned HGB best CV RMSE=%.1f", -search.best_score_)
        self.final_pipeline = _pipeline(HistGradientBoostingRegressor(random_state=config.SEED)).set_params(
            **search.best_params_).fit(self.X_train, self.y_train)
        return self.final_pipeline

    def evaluate(self) -> Dict[str, Any]:
        pred = np.maximum(self.final_pipeline.predict(self.X_holdout), 0)
        m = _metrics(self.y_holdout, pred)
        slices = {}
        for col in ("StoreType", "Promo"):
            if col in self.X_holdout.columns:
                s = self.X_holdout[col].reset_index(drop=True)
                yv = self.y_holdout.reset_index(drop=True)
                for val in sorted(pd.Series(s).dropna().unique())[:4]:
                    mask = (s == val).to_numpy()
                    if mask.sum() > 200:
                        slices[f"{col}={val}"] = round(rmspe(yv[mask], pred[mask]), 4)
        self.result.holdout = {**m, "slice_rmspe": slices,
                               "n_holdout": int(len(self.y_holdout))}
        logger.info("Holdout RMSPE=%.4f R2=%.4f RMSE=%.0f", m["rmspe"], m["r2"], m["rmse"])
        return self.result.holdout

    def to_business_metrics(self) -> Dict[str, Any]:
        h = self.X_holdout.reset_index(drop=True)
        y = self.y_holdout.reset_index(drop=True)
        uplift = None
        if "Promo" in h.columns:
            promo = pd.to_numeric(h["Promo"], errors="coerce")
            with_p, without_p = y[promo == 1].mean(), y[promo == 0].mean()
            if without_p:
                uplift = round(100 * (with_p - without_p) / without_p, 1)
        self.result.business = {
            "headline": (f"Daily store sales are predicted within {self.result.holdout['rmspe']*100:.1f}% "
                         f"(RMSPE); promotions lift average sales by about {uplift}% in the holdout."),
            "rmspe_pct": round(self.result.holdout["rmspe"] * 100, 1),
            "promo_uplift_pct": uplift}
        return self.result.business

    def compute_importances(self) -> list:
        n = min(20000, len(self.X_holdout))
        Xs, ys = self.X_holdout.iloc[-n:], self.y_holdout.iloc[-n:]
        r = permutation_importance(self.final_pipeline, Xs, ys, n_repeats=3,
                                   random_state=config.SEED, scoring="neg_root_mean_squared_error", n_jobs=N_JOBS)
        cols = list(Xs.columns)
        self.result.importances = sorted(
            [{"feature": cols[i], "importance": round(float(r.importances_mean[i]), 2)} for i in range(len(cols))],
            key=lambda d: d["importance"], reverse=True)[:12]
        return self.result.importances

    def save(self) -> None:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({"schema_version": SCHEMA_VERSION, "pipeline": self.final_pipeline,
                     "best_model": "HistGradientBoostingRegressor", "median_sales": self.median_sales,
                     "importances": self.result.importances}, config.PIPELINE_PATH)
        logger.info("Pipeline artifact written to %s", config.PIPELINE_PATH)
        card = {"schema_version": SCHEMA_VERSION,
                "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "dataset": config.KAGGLE_DATASET, "data_sha256": self._hash(),
                "target": config.TARGET, "problem": "daily store sales forecast (time-series regression)",
                "best_model": "HistGradientBoostingRegressor", "best_params": self.result.best_params,
                "baseline": self.result.baseline, "holdout": self.result.holdout,
                "business": self.result.business, "top_features": self.result.importances[:8]}
        config.MODEL_CARD_PATH.write_text(json.dumps(card, indent=2))
        logger.info("Model card written to %s", config.MODEL_CARD_PATH)

    def _hash(self) -> str:
        src = self.data_source or config.SAMPLE_PATH
        return hashlib.sha256(Path(src).read_bytes()).hexdigest() if src and Path(src).exists() else "unknown"


def _j(v):
    if isinstance(v, np.floating): return float(v)
    if isinstance(v, np.integer): return int(v)
    return v
