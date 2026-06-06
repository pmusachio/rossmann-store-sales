"""Data loading and profiling utilities."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .config import load_config, resolve_project_path


def to_snake(name: str) -> str:
    name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", str(name))
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.columns = [to_snake(col) for col in result.columns]
    return result


def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, **kwargs)


def load_train(config: dict | None = None) -> pd.DataFrame:
    config = config or load_config()
    path = resolve_project_path(config, config["data"]["train_file"])
    return normalize_columns(read_csv(path))


def load_test(config: dict | None = None) -> pd.DataFrame:
    config = config or load_config()
    path = resolve_project_path(config, config["data"]["test_file"])
    return normalize_columns(read_csv(path))


def load_store(config: dict | None = None) -> pd.DataFrame:
    config = config or load_config()
    path = resolve_project_path(config, config["data"]["store_file"])
    return normalize_columns(read_csv(path))


def merge_store(df: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
    df_norm = normalize_columns(df)
    store_norm = normalize_columns(store)
    if "store" not in df_norm.columns or "store" not in store_norm.columns:
        raise KeyError("Both frames must contain a store column.")
    return df_norm.merge(store_norm, on="store", how="left")


def load_training_frame(config: dict | None = None) -> pd.DataFrame:
    config = config or load_config()
    return merge_store(load_train(config), load_store(config))


def load_scoring_frame(config: dict | None = None) -> pd.DataFrame:
    config = config or load_config()
    return merge_store(load_test(config), load_store(config))


def profile(config_path: str | Path | None = None) -> dict:
    config = load_config(config_path)
    train = load_train(config)
    store = load_store(config)
    date_col = config["data"].get("date_column", "date")
    train[date_col] = pd.to_datetime(train[date_col])
    summary = {
        "train_rows": int(train.shape[0]),
        "train_columns": int(train.shape[1]),
        "stores": int(store["store"].nunique()),
        "date_min": train[date_col].min().date().isoformat(),
        "date_max": train[date_col].max().date().isoformat(),
        "target": config["data"].get("target", "sales"),
        "missing_by_column": train.isna().sum().sort_values(ascending=False).head(20).to_dict(),
    }
    output = resolve_project_path(config, "reports/data_profile.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
