# src/evaluate.py
import joblib, os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.data import load_raw, basic_clean, add_targets, split_data
from src.train_baselines import train_and_log_classification, train_and_log_regression

def make_target_distribution(df, out='fig_target_distribution.png'):
    import seaborn as sns
    sns.countplot(x='is_peak', data=df)
    plt.title('Target distribution (is_peak)')
    plt.savefig(out, bbox_inches='tight', dpi=200)
    plt.clf()

def make_corr_heatmap(X, out='fig_corr_heatmap.png'):
    corr = X.select_dtypes(include=['number']).corr()
    plt.figure(figsize=(6,6))
    plt.imshow(corr, aspect='auto')
    plt.colorbar()
    plt.title('Correlation heatmap')
    plt.savefig(out, bbox_inches='tight', dpi=200)
    plt.clf()

def make_confusion_matrix(pipe, X_test, y_test, out='fig_confusion_matrix_best_cls.png'):
    disp = ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test)
    disp.figure_.suptitle('Confusion matrix — best classification baseline')
    disp.figure_.savefig(out, bbox_inches='tight', dpi=200)
    plt.clf()

def make_residuals_plot(pipe, X_test, y_test, out='fig_residuals_vs_pred_best_reg.png'):
    preds = pipe.predict(X_test)
    res = y_test - preds
    plt.scatter(preds, res, s=6)
    plt.axhline(0, linestyle='--', color='k')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted — best regression baseline')
    plt.savefig(out, bbox_inches='tight', dpi=200)
    plt.clf()

if __name__ == "__main__":
    raw_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'train.csv')
    df = load_raw(raw_path)
    df = basic_clean(df)
    df = add_targets(df)
    X_train, X_val, X_test, y_train_cls, y_val_cls, y_test_cls, y_train_reg, y_val_reg, y_test_reg = split_data(df, stratify_col='is_peak')

    # Create plot 1 & 2 (target distribution + corr)
    make_target_distribution(df, out='fig_target_distribution.png')
    make_corr_heatmap(X_train, out='fig_corr_heatmap.png')

    # For confusion matrix & residuals pick the best model (we retrain quickly here or load saved one)
    # Quick retrain to get pipelines (or load from mlflow artifacts)
    from src.features import build_preprocessor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.pipeline import Pipeline
    cls_pipe = Pipeline([('pre', build_preprocessor(X_train)), ('clf', RandomForestClassifier(n_estimators=100, random_state=42))])
    cls_pipe.fit(X_train, y_train_cls)
    make_confusion_matrix(cls_pipe, X_test, y_test_cls, out='fig_confusion_matrix_best_cls.png')

    reg_pipe = Pipeline([('pre', build_preprocessor(X_train)), ('reg', RandomForestRegressor(n_estimators=100, random_state=42))])
    reg_pipe.fit(X_train, y_train_reg)
    make_residuals_plot(reg_pipe, X_test, y_test_reg, out='fig_residuals_vs_pred_best_reg.png')

    # Build metric tables (save CSV)
    # Simple example: compute metrics for all baseline models (retrain or load)
    import numpy as np
    cls_rows = []
    for name, model in [('logreg', 'LogisticRegression'), ('rf', 'RandomForest')]:
        # compute metrics on val & test (fill with your previously logged metrics)
        cls_rows.append({'model':name, 'val_acc':None, 'val_f1':None, 'test_acc':None, 'test_f1':None})
    pd.DataFrame(cls_rows).to_csv('table_classification_metrics.csv', index=False)

    reg_rows = []
    for name in ['linreg', 'rfreg']:
        reg_rows.append({'model':name, 'val_mae':None, 'val_rmse':None, 'test_mae':None, 'test_rmse':None})
    pd.DataFrame(reg_rows).to_csv('table_regression_metrics.csv', index=False)
