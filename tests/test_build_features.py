import pytest
import pandas as pd
import numpy as np
from rossmann_store_sales_project.features.build_features import DataPreprocessor

@pytest.fixture
def sample_sales_df():
    return pd.DataFrame({
        'Store': [1, 2],
        'DayOfWeek': [5, 5],
        'Date': ['2015-07-31', '2015-07-31'],
        'Sales': [5263, 6064],
        'Customers': [555, 625],
        'Open': [1, 1],
        'Promo': [1, 1],
        'StateHoliday': ['0', '0'],
        'SchoolHoliday': [1, 1]
    })

@pytest.fixture
def sample_store_df():
    return pd.DataFrame({
        'Store': [1, 2],
        'StoreType': ['c', 'a'],
        'Assortment': ['a', 'a'],
        'CompetitionDistance': [1270.0, 570.0],
        'CompetitionOpenSinceMonth': [9.0, 11.0],
        'CompetitionOpenSinceYear': [2008.0, 2007.0],
        'Promo2': [0, 1],
        'Promo2SinceWeek': [np.nan, 13.0],
        'Promo2SinceYear': [np.nan, 2010.0],
        'PromoInterval': [np.nan, 'Jan,Apr,Jul,Oct']
    })

def test_clean_names(sample_sales_df):
    preprocessor = DataPreprocessor()
    df_clean = preprocessor._clean_names(sample_sales_df.copy())
    
    assert 'Store' not in df_clean.columns
    assert 'store' in df_clean.columns
    assert 'day_of_week' in df_clean.columns

def test_process(sample_sales_df, sample_store_df):
    preprocessor = DataPreprocessor()
    
    # Process the data
    df_processed = preprocessor.process(sample_sales_df, sample_store_df)
    
    # Assert column names after feature engineering
    expected_columns = ['store', 'promo', 'store_type', 'assortment', 'competition_distance', 
                        'competition_open_since_month', 'competition_open_since_year', 'promo2', 
                        'promo2_since_week', 'promo2_since_year', 'competition_time_month', 
                        'promo_time_week', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 
                        'month_cos', 'day_sin', 'day_cos', 'week_of_year_sin', 'week_of_year_cos', 
                        'date', 'sales']
    
    assert all(col in df_processed.columns for col in expected_columns)
    assert len(df_processed) == 2
    
    # Assortment should be encoded as strings according to the logic
    assert df_processed['assortment'].tolist() == ['basic', 'basic']
