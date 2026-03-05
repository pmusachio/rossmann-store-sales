import math
import numpy as np
import pandas as pd
import datetime
import warnings
import inflection
import optuna
import xgboost as xgb
import pickle

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, MinMaxScaler, OneHotEncoder, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings('ignore')

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        X['year'] = X['date'].dt.year
        X['month'] = X['date'].dt.month
        X['day'] = X['date'].dt.day
        X['week_of_year'] = X['date'].dt.isocalendar().week.astype(int)
        
        X['competition_since'] = X.apply(lambda x: datetime.datetime(year=x['competition_open_since_year'], month=x['competition_open_since_month'], day=1), axis=1)
        X['competition_time_month'] = ((X['date'] - X['competition_since']) / 30).apply(lambda x: x.days).astype(int)
        
        X['promo_since'] = X['promo2_since_year'].astype(str) + '-' + X['promo2_since_week'].astype(str)
        X['promo_since'] = X['promo_since'].apply(lambda x: datetime.datetime.strptime(x + '-1', '%Y-%W-%w') - datetime.timedelta(days=7))
        X['promo_time_week'] = ((X['date'] - X['promo_since']) / 7).apply(lambda x: x.days).astype(int)
        
        X['assortment'] = X['assortment'].apply(lambda x: 'basic' if x == 'a' else 'extra' if x == 'b' else 'extended')
        X['state_holiday'] = X['state_holiday'].apply(lambda x: 'public_holiday' if x == 'a' else 'easter_holiday' if x == 'b' else 'christmas' if x == 'c' else 'regular_day')
        
        X = X[(X['open'] != 0)]
        
        X['day_of_week_sin'] = X['day_of_week'].apply(lambda x: np.sin(x * (2. * np.pi/7)))
        X['day_of_week_cos'] = X['day_of_week'].apply(lambda x: np.cos(x * (2. * np.pi/7)))
        X['month_sin'] = X['month'].apply(lambda x: np.sin(x * (2. * np.pi/12)))
        X['month_cos'] = X['month'].apply(lambda x: np.cos(x * (2. * np.pi/12)))
        X['day_sin'] = X['day'].apply(lambda x: np.sin(x * (2. * np.pi/30)))
        X['day_cos'] = X['day'].apply(lambda x: np.cos(x * (2. * np.pi/30)))
        X['week_of_year_sin'] = X['week_of_year'].apply(lambda x: np.sin(x * (2. * np.pi/52)))
        X['week_of_year_cos'] = X['week_of_year'].apply(lambda x: np.cos(x * (2. * np.pi/52)))
        
        cols_drop = ['open', 'promo_interval', 'month_map', 'week_of_year', 'day', 'month', 'day_of_week', 'promo_since', 'competition_since']
        X = X.drop(cols_drop, axis=1)
        
        return X

def load_and_clean_data(train_path, store_path):
    df_sales_raw = pd.read_csv(train_path, low_memory=False)
    df_store_raw = pd.read_csv(store_path, low_memory=False)
    df_raw = pd.merge(df_sales_raw, df_store_raw, how='left', on='Store')
    
    cols_old = ['Store', 'DayOfWeek', 'Date', 'Sales', 'Customers', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday', 
                'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
                'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']
    snakecase = lambda x: inflection.underscore(x)
    df_raw.columns = list(map(snakecase, cols_old))
    
    df_raw = df_raw[df_raw['sales'] > 0]
    
    df_raw['competition_distance'] = df_raw['competition_distance'].apply(lambda x: 200000.0 if math.isnan(x) else x)
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    df_raw['competition_open_since_month'] = df_raw.apply(lambda x: x['date'].month if math.isnan(x['competition_open_since_month']) else x['competition_open_since_month'], axis=1)
    df_raw['competition_open_since_year'] = df_raw.apply(lambda x: x['date'].year if math.isnan(x['competition_open_since_year']) else x['competition_open_since_year'], axis=1)
    df_raw['promo2_since_week'] = df_raw.apply(lambda x: x['date'].week if math.isnan(x['promo2_since_week']) else x['promo2_since_week'], axis=1)
    df_raw['promo2_since_year'] = df_raw.apply(lambda x: x['date'].year if math.isnan(x['promo2_since_year']) else x['promo2_since_year'], axis=1)
    
    month_map = {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    df_raw['promo_interval'].fillna(0, inplace=True)
    df_raw['month_map'] = df_raw['date'].dt.month.map(month_map)
    df_raw['is_promo'] = df_raw[['promo_interval', 'month_map']].apply(lambda x: 0 if x['promo_interval'] == 0 else 1 if x['month_map'] in x['promo_interval'].split(',') else 0, axis=1)
    
    df_raw['competition_open_since_month'] = df_raw['competition_open_since_month'].astype(int)
    df_raw['competition_open_since_year'] = df_raw['competition_open_since_year'].astype(int)
    df_raw['promo2_since_week'] = df_raw['promo2_since_week'].astype(int)
    df_raw['promo2_since_year'] = df_raw['promo2_since_year'].astype(int)
    
    return df_raw

if __name__ == "__main__":
    df = load_and_clean_data('/Users/pmusachio/REPOS/cds-rossmann-store-sales-main/data/train.csv', '/Users/pmusachio/REPOS/cds-rossmann-store-sales-main/data/store.csv')
    
    fe = FeatureEngineering()
    df_fe = fe.transform(df)
    
    cols_selected = ['store', 'promo', 'store_type', 'assortment', 'competition_distance', 'competition_open_since_month', 
                     'competition_open_since_year', 'promo2', 'promo2_since_week', 'promo2_since_year', 'competition_time_month', 
                     'promo_time_week', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 
                     'week_of_year_sin', 'week_of_year_cos', 'date', 'sales']
    
    df_fe = df_fe[cols_selected]
    
    df_train = df_fe[df_fe['date'] < '2015-06-19']
    df_test = df_fe[df_fe['date'] >= '2015-06-19']
    
    X_train = df_train.drop(['date', 'sales'], axis=1)
    y_train = np.log1p(df_train['sales'])
    
    X_test = df_test.drop(['date', 'sales'], axis=1)
    y_test = np.log1p(df_test['sales'])
    
    numeric_robust_features = ['competition_distance', 'competition_time_month']
    numeric_minmax_features = ['promo_time_week']
    categorical_features = ['store_type', 'assortment']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('robust', RobustScaler(), numeric_robust_features),
            ('minmax', MinMaxScaler(), numeric_minmax_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ], remainder='passthrough')
    
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)
    
    # Optuna tuning
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500), # Menos arvores pra ser rapido (estudo rapido)
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 15)
        }
        
        m_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42, **param)
        m_xgb.fit(X_train_preprocessed, y_train)
        yhat = m_xgb.predict(X_test_preprocessed)
        mape = mean_absolute_percentage_error(np.expm1(y_test), np.expm1(yhat))
        return mape

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10) # 10 iteracoes pra nao demorar mto
    print("Best params:", study.best_params)
    print("Best MAPE:", study.best_value)

