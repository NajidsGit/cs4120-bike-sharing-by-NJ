# src/features.py
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def build_preprocessor(X):
    num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()
    # Remove any ID-like columns, datetime handled already
    # Example: drop 'datetime' if present
    if 'datetime' in num_cols: num_cols.remove('datetime')

    num_transform = StandardScaler()
    cat_transform = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transform, num_cols),
        ('cat', cat_transform, cat_cols),
    ], remainder='drop')
    return preprocessor
