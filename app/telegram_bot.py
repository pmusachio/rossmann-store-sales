"""Telegram webhook for Rossmann store forecasts.

Environment variables:
- TELEGRAM_TOKEN: token created with BotFather.
- ROSSMANN_API_URL: API endpoint, default http://127.0.0.1:8000/rossmann/predict.
- ROSSMANN_TEST_PATH: path to Kaggle test.csv, default data/raw/test.csv.
- ROSSMANN_STORE_PATH: path to Kaggle store.csv, default data/raw/store.csv.
"""

from __future__ import annotations

import json
import os
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from flask import Flask, Response, request


ROOT = Path(__file__).resolve().parents[1]
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
ROSSMANN_API_URL = os.environ.get("ROSSMANN_API_URL", "http://127.0.0.1:8000/rossmann/predict")
TEST_PATH = Path(os.environ.get("ROSSMANN_TEST_PATH", ROOT / "data" / "raw" / "test.csv"))
STORE_PATH = Path(os.environ.get("ROSSMANN_STORE_PATH", ROOT / "data" / "raw" / "store.csv"))

app = Flask(__name__)


def send_message(chat_id: int, text: str) -> None:
    if not TELEGRAM_TOKEN:
        print(f"[telegram disabled] {chat_id}: {text}")
        return
    requests.post(f"{TELEGRAM_API}/sendMessage", json={"chat_id": chat_id, "text": text}, timeout=20)


def send_photo(chat_id: int, image: BytesIO, caption: str) -> None:
    if not TELEGRAM_TOKEN:
        print(f"[telegram disabled] chart for {chat_id}: {caption}")
        return
    image.seek(0)
    files = {"photo": ("forecast.png", image, "image/png")}
    data = {"chat_id": chat_id, "caption": caption}
    requests.post(f"{TELEGRAM_API}/sendPhoto", data=data, files=files, timeout=30)


def parse_store_id(text: str) -> int | None:
    text = str(text or "").strip()
    if text in {"/start", "start", "/help", "help"}:
        return None
    text = text.replace("/", "").strip()
    try:
        return int(text)
    except ValueError:
        return -1


def load_store_records(store_id: int) -> list[dict]:
    if not TEST_PATH.exists() or not STORE_PATH.exists():
        raise FileNotFoundError("Missing data/raw/test.csv or data/raw/store.csv. Download the Kaggle files before running the bot.")
    test = pd.read_csv(TEST_PATH, low_memory=False)
    store = pd.read_csv(STORE_PATH, low_memory=False)
    data = test.merge(store, how="left", on="Store")
    data = data[data["Store"] == store_id]
    if data.empty:
        return []
    data = data[data["Open"].fillna(0).astype(int) != 0]
    data = data.drop(columns=[col for col in ["Id"] if col in data.columns])
    return data.to_dict(orient="records")


def request_forecast(records: list[dict]) -> pd.DataFrame:
    response = requests.post(ROSSMANN_API_URL, json={"records": records}, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and "predictions" in payload:
        payload = payload["predictions"]
    return pd.DataFrame(payload)


def make_chart(predictions: pd.DataFrame, store_id: int) -> BytesIO | None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return None
    if predictions.empty or "date" not in predictions or "prediction" not in predictions:
        return None
    plot_df = predictions.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"])
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.lineplot(data=plot_df, x="date", y="prediction", marker="o", ax=ax)
    ax.set_title(f"Store {store_id} - daily sales forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Predicted sales")
    fig.autofmt_xdate()
    buffer = BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=150)
    plt.close(fig)
    buffer.seek(0)
    return buffer


def summarize_forecast(predictions: pd.DataFrame, store_id: int) -> str:
    if predictions.empty:
        return f"Store {store_id} is not available for prediction."
    total = predictions["prediction"].sum()
    days = len(predictions)
    daily_average = predictions["prediction"].mean()
    predictions["date"] = pd.to_datetime(predictions["date"])
    week_summary = predictions.assign(year_week=predictions["date"].dt.strftime("%Y-%W")).groupby("year_week")["prediction"].sum()
    best_week = week_summary.idxmax()
    best_week_value = week_summary.max()
    return (
        f"Forecast for store {store_id}\n"
        f"Next {days} open days: {total:,.2f}\n"
        f"Average daily sales: {daily_average:,.2f}\n"
        f"Best week: {best_week} ({best_week_value:,.2f})"
    )


@app.get("/")
def index():
    return "Rossmann Telegram Bot is running."


@app.post("/rossmann/bot")
def webhook():
    update = request.get_json(force=True)
    message = update.get("message", {})
    chat_id = message.get("chat", {}).get("id")
    text = message.get("text", "")
    if not chat_id:
        return Response("ok", status=200)

    store_id = parse_store_id(text)
    if store_id is None:
        send_message(chat_id, "Send a store number like /1 to receive the next weeks sales forecast.")
        return Response("ok", status=200)
    if store_id == -1:
        send_message(chat_id, "Invalid store id. Send only a number, for example /1.")
        return Response("ok", status=200)

    try:
        send_message(chat_id, f"Generating forecast for store {store_id}...")
        records = load_store_records(store_id)
        predictions = request_forecast(records) if records else pd.DataFrame()
        send_message(chat_id, summarize_forecast(predictions, store_id))
        chart = make_chart(predictions, store_id)
        if chart is not None:
            send_photo(chat_id, chart, f"Daily forecast for store {store_id}")
    except Exception as exc:  # noqa: BLE001
        send_message(chat_id, f"Could not generate forecast: {exc}")

    return Response("ok", status=200)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
