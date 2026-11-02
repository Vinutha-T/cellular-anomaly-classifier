import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

TARGET = "Unusual"
TIME_COL = "Time"
CELL_COL = "CellName"
KPI_COLS = [
    "PRBUsageUL",
    "PRBUsageDL",
    "meanThr_DL",
    "meanThr_UL",
    "maxThr_DL",
    "maxThr_UL",
    "meanUE_DL",
    "meanUE_UL",
    "maxUE_DL",
    "maxUE_UL",
    "maxUE_UL+DL",
]
FEATURE_COLS = [TIME_COL, CELL_COL] + KPI_COLS

MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "KNN",
    "naive_bayes": "Naive Bayes",
    "svm": "Support Vector Machine",
}


def time_to_cyclic(time_frame):
    # "HH:MM" -> hour of day on a circle, so 23:45 and 00:00 end up close together
    parts = time_frame.iloc[:, 0].astype(str).str.split(":", expand=True).astype(float)
    hours = parts[0] + parts[1] / 60.0
    angle = 2 * np.pi * hours / 24.0
    return np.column_stack([np.sin(angle), np.cos(angle)])


def load_kpi_csv(source):
    # Kaggle file is ';' separated, a re-saved file may be ',' separated
    df = pd.read_csv(source, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    return df


def anomaly_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def compute_metrics(y_true, y_pred, y_score):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
