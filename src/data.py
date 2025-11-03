# src/data.py
import os, pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_raw(path):
    df = pd.read_csv(path, parse_dates=['datetime'])
    return df

def basic_clean(df):
    # example cleaning steps
    df = df.copy()
    df['hour'] = df['datetime'].dt.hour
    df['weekday'] = df['datetime'].dt.weekday
    # drop obvious duplicates if any
    df = df.drop_duplicates()
    # no. of missing? drop or impute if present
    df = df.dropna()
    return df

def add_targets(df):
    df = df.copy()
    # Regression target: count
    # Classification target: peak vs off-peak (method 1: median threshold)
    median_count = df['count'].median()
    df['is_peak'] = (df['count'] >= median_count).astype(int)
    # alternative classification: rush hours (7-9,16-18)
    df['is_rush'] = df['hour'].isin([7,8,9,16,17,18]).astype(int)
    return df

def split_data(df, stratify_col='is_peak'):
    X = df.drop(columns=['count', 'is_peak','is_rush'])
    y_cls = df[stratify_col]
    y_reg = df['count']
    # 60/40 first split
    X_train, X_temp, y_train_cls, y_temp_cls, y_train_reg, y_temp_reg = train_test_split(
        X, y_cls, y_reg, test_size=0.4, stratify=y_cls, random_state=RANDOM_STATE)
    # split temp 50/50 to get 20/20
    X_val, X_test, y_val_cls, y_test_cls, y_val_reg, y_test_reg = train_test_split(
        X_temp, y_temp_cls, y_temp_reg, test_size=0.5, stratify=y_temp_cls, random_state=RANDOM_STATE)
    return (X_train, X_val, X_test, y_train_cls, y_val_cls, y_test_cls, y_train_reg, y_val_reg, y_test_reg)

if __name__ == "__main__":
    raw_path = os.path.join(DATA_DIR, 'raw', 'train.csv')  # adjust filename
    df = load_raw(raw_path)
    df = basic_clean(df)
    df = add_targets(df)
    splits = split_data(df, stratify_col='is_peak')
    print("Done splitting. Train rows:", len(splits[0]))
