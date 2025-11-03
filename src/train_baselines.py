# src/train_baselines.py
import mlflow, mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from src.data import load_raw, basic_clean, add_targets, split_data
from src.features import build_preprocessor
import os
import numpy as np

MLFLOW_EXPERIMENT = "midpoint_bikeshare"
mlflow.set_experiment(MLFLOW_EXPERIMENT)

def train_and_log_classification(X_train, X_val, X_test, y_train, y_val, y_test):
    models = {
        'logreg': LogisticRegression(max_iter=200),
        'rf': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    results = {}
    pre = build_preprocessor(X_train)
    for name, model in models.items():
        with mlflow.start_run(run_name=f"cls_{name}"):
            pipe = Pipeline([('pre', pre), ('clf', model)])
            pipe.fit(X_train, y_train)
            for split_name, X_, y_ in [('val', X_val, y_val), ('test', X_test, y_test)]:
                pred = pipe.predict(X_)
                acc = accuracy_score(y_, pred)
                f1 = f1_score(y_, pred, average='weighted', zero_division=0)
                mlflow.log_metric(f'{split_name}_acc', acc)
                mlflow.log_metric(f'{split_name}_f1_w', f1)
            mlflow.sklearn.log_model(pipe, f"cls_{name}")
            results[name] = pipe
    return results

def train_and_log_regression(X_train, X_val, X_test, y_train, y_val, y_test):
    models = {
        'linreg': LinearRegression(),
        'rfreg': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    results = {}
    pre = build_preprocessor(X_train)
    for name, model in models.items():
        with mlflow.start_run(run_name=f"reg_{name}"):
            pipe = Pipeline([('pre', pre), ('reg', model)])
            pipe.fit(X_train, y_train)
            for split_name, X_, y_ in [('val', X_val, y_val), ('test', X_test, y_test)]:
                pred = pipe.predict(X_)
                mae = mean_absolute_error(y_, pred)
                rmse = np.sqrt(mean_squared_error(y_, pred))
                mlflow.log_metric(f'{split_name}_mae', mae)
                mlflow.log_metric(f'{split_name}_rmse', rmse)
            mlflow.sklearn.log_model(pipe, f"reg_{name}")
            results[name] = pipe
    return results

if __name__ == "__main__":
    raw_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'train.csv')
    df = load_raw(raw_path)
    df = basic_clean(df)
    df = add_targets(df)
    X_train, X_val, X_test, y_train_cls, y_val_cls, y_test_cls, y_train_reg, y_val_reg, y_test_reg = split_data(df, stratify_col='is_peak')
    train_and_log_classification(X_train, X_val, X_test, y_train_cls, y_val_cls, y_test_cls)
    train_and_log_regression(X_train, X_val, X_test, y_train_reg, y_val_reg, y_test_reg)
