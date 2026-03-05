from pathlib import Path

import joblib
from loguru import logger
import pandas as pd

from rossmann_store_sales_project.features.build_features import DataPreprocessor
from rossmann_store_sales_project.models.train_model import ModelTrainer


def train_and_save_model():
    """Trains the model on processed data and saves the complete pipeline."""
    logger.info("Loading processed data...")
    # Paths according to CCDS
    base_dir = Path(__file__).resolve().parents[2]
    processed_dir = base_dir / "data" / "processed"
    processed_train_path = processed_dir / "processed_train.csv"

    if not processed_train_path.exists():
        logger.error(f"Processed data not found at {processed_train_path}. Please run make_dataset.py first.")
        return

    df_processed = pd.read_csv(processed_train_path, low_memory=False)

    logger.info("Splitting data and training Model...")
    trainer = ModelTrainer()
    X_train, y_train, X_test, y_test = trainer.split_data(df_processed)

    # Train the model
    trainer.train(X_train, y_train)

    # Save the trained model pipeline
    model_path = base_dir / "models" / "model.pkl"
    logger.info(f"Saving model to {model_path}...")
    joblib.dump(trainer, model_path)
    logger.success("Model trained and saved successfully!")


if __name__ == "__main__":
    train_and_save_model()
