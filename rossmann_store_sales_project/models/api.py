from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Rossmann Store Sales Predictor API")

# Load model here (requires saving model first, which occurs in train.py usually)
from pathlib import Path

base_dir = Path(__file__).resolve().parents[2]
model_path = base_dir / "models" / "model.pkl"

try:
    model_pipeline = joblib.load(model_path)
except Exception:
    model_pipeline = None


# Input Schema matching raw json data
class StoreSalesInput(BaseModel):
    Store: int
    DayOfWeek: int
    Date: str
    Customers: int
    Open: int
    Promo: int
    StateHoliday: str
    SchoolHoliday: int
    StoreType: str
    Assortment: str
    CompetitionDistance: float = None
    CompetitionOpenSinceMonth: float = None
    CompetitionOpenSinceYear: float = None
    Promo2: int
    Promo2SinceWeek: float = None
    Promo2SinceYear: float = None
    PromoInterval: str = None


@app.get("/")
def home():
    return {"message": "Rossmann Store Sales Prediction API is running."}


@app.post("/predict")
def predict(data: list[StoreSalesInput]):
    if not model_pipeline:
        return {"error": "Model not loaded. Please train and save the model as 'model.pkl' first."}

    df_raw = pd.DataFrame([item.dict() for item in data])

    # Process features
    from rossmann_store_sales_project.features.build_features import DataPreprocessor

    fe = DataPreprocessor()
    df_clean = fe._clean_names(df_raw)
    df_filled = fe._fill_na(df_clean)
    df_features = fe._feature_engineering(df_filled)

    # Predict
    import numpy as np

    predictions = model_pipeline.predict(df_features)
    predictions_exp = np.expm1(predictions)

    df_raw["Prediction"] = predictions_exp
    return df_raw.to_dict(orient="records")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
