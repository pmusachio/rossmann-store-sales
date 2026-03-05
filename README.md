# Rossmann Store Sales Project - Refactored for Portfolio

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>
<a target="_blank" href="https://fastapi.tiangolo.com/">
    <img src="https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi" />
</a>
<a target="_blank" href="https://streamlit.io/">
    <img src="https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit" />
</a>

This repository contains a professional End-to-End Data Science project refactored from a data analysis class to predict Rossmann store sales up to 6 weeks in advance. 

## 1. Business Problem
Rossmann operates over 3,000 drug stores in 7 European countries. Currently, Rossmann store managers are tasked with predicting their daily sales for up to six weeks in advance. Store sales are influenced by many factors, including promotions, competition, school and state holidays, seasonality, and locality. With thousands of individual managers predicting sales based on their unique circumstances, the accuracy of results can be quite varied. 

**The Goal:** Provide a robust automated forecasting tool so the CFO can accurately predict the revenue of all stores for the next 6 weeks, which will be used to base budget planning for store renovations.

## 2. Solution Strategy
The project was structured following the **Cookiecutter Data Science** logic, applying **Clean Code** and **Object-Oriented Programming (OOP)** for code reusability.

1. **Data Understanding & Cleaning**: Exploratory Data Analysis (EDA) of the `train.csv` and `store.csv` historical data.
2. **Feature Engineering**: Created temporal variables (weeks, cyclical variables for months and days), and competitor timespans to capture seasonal patterns.
3. **Data Preparation Strategy (Leakage Fixed)**: Created a robust Scikit-Learn `Pipeline` (`ColumnTransformer`) avoiding data leakage by strictly applying `.fit()` on training sets. 
4. **Machine Learning Model**: Tested Linear Regression, Lasso, Random Forest, and XGBoost regressor. XGBoost was chosen due to its favorable performance and footprint trade-off over Random Forest.
5. **Hyperparameter Tuning**: Ran a Bayesian Optimization using `Optuna` which drastically improved the baseline MAPE (Mean Absolute Percentage Error) from **~26%** down to **11.5%**.
6. **Deploy**: Code was refactored into Python classes (`DataPreprocessor`, `ModelTrainer`) and served through a `FastAPI` instance. An interactive `Streamlit` dashboard was also created for users to consume the API visually.

## 3. Top Data Insights

* **Insight 1:** Stores open during the Christmas holidays unexpectedly sell *less* on average relative to non-holidays, likely due to specific customer purchasing behaviors heavily centered *before* the holiday.
* **Insight 2:** Extending promotions for too long has diminishing returns. Initial promotions spark high sales, but prolonged concurrent "promo2" statuses drop baseline sales.
* **Insight 3:** Distance mapping revealed that stores with *closer competitors* actually sell more, highlighting the effect of dense commercial centers.

## 4. Project Structure (Cookiecutter Data Science v2)
The repository is strictly organized to separate concerns, making it highly maintainable and reproducible:
```text
├── data/                  <- Raw, interim, and processed data assets.
├── notebooks/             <- Jupyter notebooks for exploratory analysis and tuning.
├── rossmann_store_sales_project/
│   ├── features/          <- Object-oriented preprocessors and feature engineering logic.
│   ├── models/            <- Model training scripts and the FastAPI application.
│   ├── app.py             <- Streamlit Dashboard script.
│   └── config.py          <- Global configuration and path management.
├── tests/                 <- Pytest unit tests.
├── requirements.txt       <- Frozen environment dependencies.
└── Makefile               <- Terminal commands orchestrator.
```

## 5. Machine Learning Model Performance
After rigorous cross-validation and Optuna Bayesian tuning, the XGBoost model achieved:
- **MAPE:** 11.57% (Predictions deviate ~11% from actual values on average)
- **Business Translation:** The model allows the CFO to project a 6-week total expected revenue with best and worst case margins driven by the MAE, replacing empirical guesswork with a measurable data asset.

## 6. How to Run Locally

### Requirements
- Python 3.12+

### Setup
1. Clone the repository and navigate to the project directory:
   ```bash
   git clone git@github.com:pmusachio/rossmann-store-sales.git
   cd rossmann-store-sales
   ```
2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy raw data to `data/raw/` (You need the Kaggle dataset files `train.csv` and `store.csv`). Note: The `data/` folder contents are `.gitignore`d to prevent committing large files.
4. Run the API:
   ```bash
   python -m uvicorn rossmann_store_sales_project.models.api:app --reload
   ```
5. In another terminal, run the Streamlit dashboard:
   ```bash
   python -m streamlit run rossmann_store_sales_project/app.py
   ```
