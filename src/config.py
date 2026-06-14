"""Central configuration: paths, dataset identity, modeling constants and the
Dracula palette shared by the pipeline, the serving layer and the dashboard.
"""
from __future__ import annotations

from pathlib import Path

BASE_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = BASE_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
SAMPLE_DIR: Path = DATA_DIR / "sample"
MODELS_DIR: Path = BASE_DIR / "models"

PIPELINE_PATH: Path = MODELS_DIR / "pipeline.joblib"
MODEL_CARD_PATH: Path = MODELS_DIR / "model_card.json"
PROCESSED_PATH: Path = PROCESSED_DIR / "train.parquet"

SAMPLE_FILENAME: str = "rossmann_sample.csv"
SAMPLE_PATH: Path = SAMPLE_DIR / SAMPLE_FILENAME

# Public mirror of the Rossmann Store Sales competition data.
KAGGLE_DATASET: str = "pratyushakar/rossmann-store-sales"
TRAIN_FILENAME: str = "train.csv"
STORE_FILENAME: str = "store.csv"

TARGET: str = "Sales"
DATE_COL: str = "Date"
STORE_COL: str = "Store"
# Customers is realized only after the day occurs, so it is not available at
# prediction time and is dropped to prevent target leakage.
LEAKAGE_COLS: tuple[str, ...] = ("Customers",)

NUMERIC_FEATURES: tuple[str, ...] = (
    "DayOfWeek", "Promo", "SchoolHoliday", "CompetitionDistance",
    "competition_open_months", "promo2_active", "year", "month", "day",
    "week_of_year", "day_of_week_sin", "day_of_week_cos",
)
CATEGORICAL_FEATURES: tuple[str, ...] = ("StoreType", "Assortment", "StateHoliday")
ENGINEERED_FEATURES: tuple[str, ...] = (
    "year", "month", "day", "week_of_year", "day_of_week_sin", "day_of_week_cos",
    "competition_open_months", "promo2_active",
)

HOLDOUT_FRACTION: float = 0.15   # most recent dates held out (chronological)
N_SELECT: int = 150_000
SEED: int = 42
CV_FOLDS: int = 3
TUNING_ITERS: int = 10

DRACULA = {
    "background": "#282a36", "current_line": "#44475a", "foreground": "#f8f8f2",
    "comment": "#6272a4", "cyan": "#8be9fd", "green": "#50fa7b", "orange": "#ffb86c",
    "pink": "#ff79c6", "purple": "#bd93f9", "red": "#ff5555", "yellow": "#f1fa8c",
}
