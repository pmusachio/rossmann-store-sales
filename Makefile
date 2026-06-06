.PHONY: install setup profile train api streamlit bot webhook test lint help

# ──────────────────────────────────────────────
#  Environment
# ──────────────────────────────────────────────
install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt -r requirements-api.txt -r requirements-app.txt -r requirements-dev.txt

setup: install
	@echo "Copy .env.example to .env and fill in your TELEGRAM_TOKEN before running 'make bot'."
	@test -f .env || cp .env.example .env

# ──────────────────────────────────────────────
#  Data  (requires kaggle CLI: pip install kaggle)
# ──────────────────────────────────────────────
data:
	mkdir -p data/raw
	kaggle competitions download -c rossmann-store-sales -p data/raw
	unzip -o data/raw/rossmann-store-sales.zip -d data/raw

# ──────────────────────────────────────────────
#  ML pipeline
# ──────────────────────────────────────────────
profile:
	PYTHONPATH=src python -m rossmann_store_sales.cli profile

train:
	PYTHONPATH=src python -m rossmann_store_sales.cli train

predict:
	PYTHONPATH=src python -m rossmann_store_sales.cli predict \
		--input data/raw/test.csv \
		--output data/processed/predictions.csv

# ──────────────────────────────────────────────
#  Services
# ──────────────────────────────────────────────
api:
	PYTHONPATH=src uvicorn rossmann_store_sales.api:app --reload --port 8000

streamlit:
	PYTHONPATH=src streamlit run app/streamlit_app.py

bot:
	@test -f .env && export $$(grep -v '^#' .env | xargs) || true
	PYTHONPATH=src python app/telegram_bot.py

# Configure the Telegram webhook (set WEBHOOK_URL before calling)
webhook:
	python scripts/set_telegram_webhook.py --url "$(WEBHOOK_URL)/rossmann/bot"

# ──────────────────────────────────────────────
#  Quality
# ──────────────────────────────────────────────
test:
	python -m pytest -v

lint:
	python -m ruff check src app scripts tests

# ──────────────────────────────────────────────
#  Help
# ──────────────────────────────────────────────
help:
	@echo ""
	@echo "  make install    Install all Python dependencies"
	@echo "  make setup      Install deps + create .env from template"
	@echo "  make data       Download Kaggle dataset (needs kaggle CLI configured)"
	@echo "  make profile    Profile raw data and save summary to reports/"
	@echo "  make train      Train XGBoost pipeline and save model + metrics"
	@echo "  make predict    Run batch prediction on test.csv"
	@echo "  make api        Start FastAPI server on port 8000"
	@echo "  make streamlit  Open Streamlit dashboard"
	@echo "  make bot        Start Telegram bot (needs .env with TELEGRAM_TOKEN)"
	@echo "  make webhook    Register WEBHOOK_URL with Telegram"
	@echo "  make test       Run test suite with pytest"
	@echo ""
