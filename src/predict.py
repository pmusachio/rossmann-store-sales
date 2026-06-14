"""Serving layer: load the serialized pipeline and predict daily store sales."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self, artifact_path: Path = config.PIPELINE_PATH) -> None:
        import joblib

        if not Path(artifact_path).exists():
            raise FileNotFoundError(f"No artifact at {artifact_path}. Run `python -m src.pipeline` first.")
        art = joblib.load(artifact_path)
        self.pipeline = art["pipeline"]
        self.importances: List[Dict[str, Any]] = art.get("importances", [])
        self.best_model: str = art.get("best_model", "")
        self.median_sales: float = art.get("median_sales", 0.0)

    def predict(self, records: pd.DataFrame) -> np.ndarray:
        return np.maximum(self.pipeline.predict(records), 0.0)

    def predict_one(self, features: Dict[str, Any]) -> float:
        return float(self.predict(pd.DataFrame([features]))[0])

    def top_features(self, n: int = 6) -> List[Dict[str, Any]]:
        return self.importances[:n]
