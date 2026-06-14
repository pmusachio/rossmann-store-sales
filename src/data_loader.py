"""Acquisition layer: pull the Rossmann train and store tables from a public Kaggle
mirror, merge them, with the versioned sample as an offline fallback.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pandas as pd

from src import config

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, raw_dir: Path = config.RAW_DIR, sample_path: Path = config.SAMPLE_PATH,
                 dataset: str = config.KAGGLE_DATASET) -> None:
        self.raw_dir = raw_dir
        self.sample_path = sample_path
        self.dataset = dataset
        self.train_path = raw_dir / config.TRAIN_FILENAME
        self.store_path = raw_dir / config.STORE_FILENAME

    def download(self, force: bool = False) -> Path:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        if self.train_path.exists() and self.store_path.exists() and not force:
            return self.train_path
        try:
            import kagglehub

            logger.info("Downloading %s from Kaggle", self.dataset)
            cache = Path(kagglehub.dataset_download(self.dataset))
            shutil.copyfile(next(cache.rglob(config.TRAIN_FILENAME)), self.train_path)
            shutil.copyfile(next(cache.rglob(config.STORE_FILENAME)), self.store_path)
            logger.info("Raw tables written to %s", self.raw_dir)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Kaggle download unavailable (%s); using versioned sample", exc)
            if not self.sample_path.exists():
                raise FileNotFoundError(f"No Kaggle access and no sample at {self.sample_path}")
            shutil.copyfile(self.sample_path, self.train_path)
        return self.train_path

    def load(self) -> pd.DataFrame:
        self.download()
        if self.store_path.exists():
            train = pd.read_csv(self.train_path, low_memory=False)
            store = pd.read_csv(self.store_path)
            df = train.merge(store, how="left", on=config.STORE_COL)
        else:  # sample is already merged
            df = pd.read_csv(self.train_path, low_memory=False)
        logger.info("Loaded %d rows x %d cols", df.shape[0], df.shape[1])
        return df
