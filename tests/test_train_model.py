import pytest
import pandas as pd
import numpy as np
from rossmann_store_sales_project.models.train_model import ModelTrainer, ModelEvaluator

@pytest.fixture
def sample_train_data():
    df = pd.DataFrame({
        'date': pd.date_range(start='2015-06-01', periods=30), # 18 days train, 12 days test
        'store': np.random.randint(1, 100, 30),
        'promo': np.random.randint(0, 2, 30),
        'store_type': np.random.choice(['a', 'b', 'c', 'd'], 30),
        'assortment': np.random.choice(['basic', 'extra', 'extended'], 30),
        'competition_distance': np.random.uniform(100, 10000, 30),
        'competition_open_since_month': np.random.randint(1, 13, 30),
        'competition_open_since_year': np.random.randint(2000, 2015, 30),
        'promo2': np.random.randint(0, 2, 30),
        'promo2_since_week': np.random.randint(1, 52, 30),
        'promo2_since_year': np.random.randint(2009, 2015, 30),
        'competition_time_month': np.random.randint(0, 100, 30),
        'promo_time_week': np.random.randint(0, 200, 30),
        'day_of_week_sin': np.random.uniform(-1, 1, 30),
        'day_of_week_cos': np.random.uniform(-1, 1, 30),
        'month_sin': np.random.uniform(-1, 1, 30),
        'month_cos': np.random.uniform(-1, 1, 30),
        'day_sin': np.random.uniform(-1, 1, 30),
        'day_cos': np.random.uniform(-1, 1, 30),
        'week_of_year_sin': np.random.uniform(-1, 1, 30),
        'week_of_year_cos': np.random.uniform(-1, 1, 30),
        'sales': np.random.randint(1000, 10000, 30)
    })
    return df

def test_split_data(sample_train_data):
    trainer = ModelTrainer()
    X_train, y_train, X_test, y_test = trainer.split_data(sample_train_data)
    
    # Check lengths
    assert len(X_train) == 18 # Total 30 days, < '2015-06-19' implies dates 01 to 18
    assert len(X_test) == 12  # dates 19 to 30
    
    # Check if target values are transformed with log1p
    # Just check if it's generally valid mapping
    assert np.allclose(y_train, np.log1p(sample_train_data.head(18)['sales']))
    assert 'date' not in X_train.columns
    assert 'sales' not in X_train.columns

def test_model_evaluator():
    # True sales around 5000 (log1p ~ 8.5)
    y_true = np.log1p(np.array([5000, 6000, 7000]))
    y_pred = np.log1p(np.array([5500, 5800, 7200]))
    
    metrics = ModelEvaluator.evaluate("Test Model", y_true, y_pred)
    
    assert metrics['Model Name'] == "Test Model"
    assert 'MAE' in metrics
    assert 'MAPE' in metrics
    assert 'RMSE' in metrics
    
    # Exponentiated predictions are approx what we passed
    assert metrics['MAE'] > 0
    assert metrics['MAPE'] > 0
    assert metrics['RMSE'] > 0
