from pathlib import Path
import pandas as pd
from loguru import logger

from rossmann_store_sales_project.features.build_features import DataPreprocessor

def main():
    """Builds the treated dataset and saves it for modeling."""
    base_dir = Path(__file__).resolve().parents[1]
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"

    train_path = raw_dir / "train.csv"
    store_path = raw_dir / "store.csv"

    if not train_path.exists() or not store_path.exists():
        logger.error(f"Raw data not found! Ensure {train_path} and {store_path} exist.")
        return

    logger.info("Loading raw data...")
    df_sales = pd.read_csv(train_path, low_memory=False)
    df_store = pd.read_csv(store_path, low_memory=False)

    logger.info("Initializing DataPreprocessor...")
    preprocessor = DataPreprocessor()
    
    logger.info("Building features (this may take a few moments)...")
    # Processing and generating features
    df_processed = preprocessor.process(df_sales, df_store)

    # Make sure output directory exists
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = processed_dir / "processed_train.csv"
    logger.info(f"Saving processed dataset to {output_path}...")
    df_processed.to_csv(output_path, index=False)
    
    logger.success("Dataset successfully created and saved!")

if __name__ == "__main__":
    main()
