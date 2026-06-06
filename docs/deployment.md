# Deployment Guide

## Architecture

```
┌──────────────┐    POST /rossmann/predict    ┌──────────────────────┐
│ Telegram Bot │ ───────────────────────────► │   FastAPI (api.py)   │
│  (Flask)     │ ◄─────────────── JSON ─────  │                      │
└──────────────┘                              │  loads model.joblib  │
                                              │  → XGBoost Pipeline  │
┌──────────────┐    POST /rossmann/predict    │                      │
│  Streamlit   │ ───────────────────────────► │                      │
└──────────────┘                              └──────────────────────┘
```

---

## Local Development

### 1. Start the API

```bash
make api
# Swagger UI: http://127.0.0.1:8000/docs
```

### 2. Test the API

```bash
python scripts/sample_api_request.py
```

Example request body:
```json
{
  "records": [
    {
      "Store": 1,
      "Date": "2015-08-01",
      "DayOfWeek": 6,
      "Open": 1,
      "Promo": 0,
      "StateHoliday": "0",
      "SchoolHoliday": 0
    }
  ]
}
```

### 3. Start the Telegram bot

```bash
cp .env.example .env   # fill in TELEGRAM_TOKEN
make bot
```

### 4. Simulate a Telegram message locally

```bash
python scripts/sample_telegram_payload.py --store /1
```

---

## Cloud Deployment

The repo includes `Procfile.api` and `Procfile.bot` for separate services:

```
# Procfile.api
web: PYTHONPATH=src uvicorn rossmann_store_sales.api:app --host 0.0.0.0 --port $PORT

# Procfile.bot
web: python app/telegram_bot.py
```

**Environment variables to set in your cloud dashboard:**

| Variable | Value |
|---|---|
| `TELEGRAM_TOKEN` | From @BotFather |
| `ROSSMANN_API_URL` | Public URL of your API + `/rossmann/predict` |
| `ROSSMANN_TEST_PATH` | `data/raw/test.csv` |
| `ROSSMANN_STORE_PATH` | `data/raw/store.csv` |

### Register the Telegram webhook

```bash
make webhook WEBHOOK_URL=https://your-bot-app.onrender.com
```

Verify:
```bash
curl https://api.telegram.org/bot<YOUR_TOKEN>/getWebhookInfo
```

---

## Requirements Files

| File | Purpose |
|---|---|
| `requirements.txt` | Core ML pipeline (pandas, sklearn, xgboost, joblib) |
| `requirements-api.txt` | FastAPI + uvicorn |
| `requirements-app.txt` | Streamlit + Flask + matplotlib |
| `requirements-dev.txt` | pytest, ruff |
