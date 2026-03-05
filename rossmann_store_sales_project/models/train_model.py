import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler
import xgboost as xgb


class ModelTrainer:
    """Handles data splitting, preprocessing, and training."""

    def __init__(self, model_params=None):
        if model_params is None:
            self.model_params = {
                "n_estimators": 411,
                "max_depth": 10,
                "learning_rate": 0.0279,
                "subsample": 0.666,
                "colsample_bytree": 0.882,
                "min_child_weight": 7,
                "objective": "reg:squarederror",
                "n_jobs": -1,
                "random_state": 42,
            }
        else:
            self.model_params = model_params

        self.preprocessor = None
        self.model = None

    def _build_preprocessor(self):
        numeric_robust_features = ["competition_distance", "competition_time_month"]
        numeric_minmax_features = ["promo_time_week"]
        categorical_features = ["store_type", "assortment"]

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("robust", RobustScaler(), numeric_robust_features),
                ("minmax", MinMaxScaler(), numeric_minmax_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ],
            remainder="passthrough",
        )

    def split_data(self, df):
        df_train = df[df["date"] < "2015-06-19"]
        df_test = df[df["date"] >= "2015-06-19"]

        X_train = df_train.drop(["date", "sales"], axis=1)
        y_train = np.log1p(df_train["sales"])

        X_test = df_test.drop(["date", "sales"], axis=1)
        y_test = np.log1p(df_test["sales"])

        return X_train, y_train, X_test, y_test

    def train(self, X_train, y_train):
        self._build_preprocessor()
        X_train_processed = self.preprocessor.fit_transform(X_train)

        self.model = xgb.XGBRegressor(**self.model_params)
        self.model.fit(X_train_processed, y_train)
        return self

    def predict(self, X):
        X_processed = self.preprocessor.transform(X)
        return self.model.predict(X_processed)


class ModelEvaluator:
    """Evaluates the model metrics."""

    @staticmethod
    def evaluate(model_name, y_true, y_pred):
        # Exponentiating because inputs are typically log1p transformed
        y_true_exp = np.expm1(y_true)
        y_pred_exp = np.expm1(y_pred)

        mae = mean_absolute_error(y_true_exp, y_pred_exp)
        mape = mean_absolute_percentage_error(y_true_exp, y_pred_exp)
        rmse = np.sqrt(mean_squared_error(y_true_exp, y_pred_exp))

        return {"Model Name": model_name, "MAE": mae, "MAPE": mape, "RMSE": rmse}
