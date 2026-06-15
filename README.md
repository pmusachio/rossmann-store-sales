# Rossmann Store Sales — Daily Sales Forecast

> Time-series regression · Gradient boosting · Demand forecasting · RMSPE

## Business Problem

Rossmann store managers need a daily sales forecast to plan staffing, stock and promotions. The
decision the model informs is operational: **how much to prepare for tomorrow** at each store.

The cost of error is two-sided. Over-forecasting wastes labour and inventory; under-forecasting
loses sales and frustrates customers. Because errors should scale with a store's size, the model is
judged on **RMSPE** (root mean squared *percentage* error), the competition's own metric, rather
than absolute error that would be dominated by the largest stores. Crucially, the model is
evaluated on a **chronological holdout** — the most recent weeks — because a forecast that only
works in-sample is worthless.

A flat "same as last period" rule was rejected: it cannot anticipate promotions, holidays or the
strong day-of-week pattern that drive most of the variation.

## Dataset

[Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales) (loaded from a
public Kaggle mirror), joining daily sales to a store master table.

| Property | Value |
|----------|-------|
| Records | 1,017,209 store-days (844,338 open days used) |
| Target | `Sales` (daily, per store) |
| Fields | day, promo, holidays, store type, assortment, competition, Promo2 |
| Split | chronological — most recent 15% held out |

## Solution Strategy

1. **Acquisition** — pull the train and store tables from Kaggle and join them; a versioned sample backs an offline run.
2. **Leakage control** — `Customers` is removed: it is only known after the day happens, so using it would leak the outcome.
3. **Scope** — train on open days with positive sales; closed days are a trivial zero and excluded.
4. **Feature engineering** — calendar parts, cyclical day-of-week, competition tenure and an active-Promo2 flag, inside the model `Pipeline` so serving reuses the exact transform.
5. **Model selection** — a linear baseline versus histogram gradient boosting, tuned with `RandomizedSearchCV` over a `TimeSeriesSplit` so validation never sees the future.
6. **Evaluation** — RMSPE, RMSE, MAE and R-squared on the chronological holdout, plus RMSPE by store type and promo state.

## Top Insights & Hypotheses

- **Competition distance is the strongest single driver** of sales level, ahead of store attributes.
- **Store type and assortment** materially shift the baseline, so a single global average would mislead.
- **Promotions lift average sales by about 40%** in the holdout — the largest controllable lever.
- **The day-of-week pattern is strong and cyclical**, captured by the sine/cosine encoding.

## Engineered Features

| Feature | Definition | Business signal |
|---------|-----------|-----------------|
| year / month / day / week_of_year | calendar parts of the date | seasonality and trend |
| day_of_week_sin / _cos | cyclical encoding of the weekday | smooth weekly pattern |
| competition_open_months | months since a competitor opened nearby | competitive pressure ramp |
| promo2_active | whether the store's recurring Promo2 is active this month | continuous-promotion effect |

## Model

A histogram gradient boosting regressor tuned over a time-series split, inside a `Pipeline` that
owns the engineering and encoding. The linear baseline sets the bar.

| Model | RMSPE | RMSE | MAE | R-squared |
|-------|------:|-----:|----:|----------:|
| Linear baseline | 0.461 | 2,755 | 1,989 | 0.22 |
| **Hist gradient boosting (final)** | **0.247** | **1,620** | **1,053** | **0.73** |

## Business Results

On the most recent weeks (126,651 store-days held out), the model predicts daily sales within
**24.7% RMSPE**, nearly halving the linear baseline's error, and explains **73%** of the variance.
It quantifies the promotion lever directly: running a promotion raises predicted daily sales by
about **40%**, which the app lets a manager toggle to weigh promo cost against the sales gain.

## How to Run

1. **Clone**
   ```
   git clone https://github.com/pmusachio/rossmann-store-sales.git
   cd rossmann-store-sales
   ```
2. **Environment**
   ```
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Kaggle access** — place a Kaggle API token at `~/.kaggle/`; the pipeline falls back to the versioned sample if none is present.
4. **Run the pipeline**
   ```
   python -m src.pipeline
   ```
5. **Tests**
   ```
   pytest tests/
   ```
6. **App (local)**
   ```
   streamlit run app/streamlit_app.py
   ```
7. **Live app** — [rossmann-store-sales-50ob.onrender.com](https://rossmann-store-sales-50ob.onrender.com) — forecast a store-day and weigh a promotion.

## Next Steps

- Add lagged and rolling sales features (last week, last year, trailing average), the largest
  remaining lever for cutting RMSPE toward the competition's best results; deferred here to keep the
  serving contract a single store-day without a history buffer.
- Model per-store effects explicitly (store embeddings or a hierarchical model) instead of relying
  on store attributes alone.
- Forecast a horizon (next N days) with prediction intervals, since planning needs uncertainty, not
  just a point estimate.
