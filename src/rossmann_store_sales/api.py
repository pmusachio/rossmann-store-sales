"""FastAPI service for Rossmann sales forecasts."""

from __future__ import annotations

from fastapi import Body, FastAPI, HTTPException

from .models import predict_records


app = FastAPI(title="Rossmann Store Sales API")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _records_from_payload(payload) -> list[dict]:
    if isinstance(payload, dict) and "records" in payload:
        return payload["records"]
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise HTTPException(status_code=400, detail="Payload must be a record, a list of records or {'records': [...]}.")


@app.post("/rossmann/predict")
def rossmann_predict(payload=Body(...)):
    try:
        return predict_records(_records_from_payload(payload))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict")
def predict_alias(payload=Body(...)):
    return rossmann_predict(payload)
