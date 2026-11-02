import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from features import (
    CELL_COL,
    FEATURE_COLS,
    KPI_COLS,
    MODEL_NAMES,
    TARGET,
    TIME_COL,
    anomaly_scores,
    compute_metrics,
    load_kpi_csv,
    time_to_cyclic,
)

ROOT = Path(__file__).resolve().parent
RAW_CSV = ROOT / "data" / "ML-MATT-CompetitionQT2021_train.csv"
MODEL_DIR = ROOT / "model"
TEST_CSV = ROOT / "test_data.csv"
SEED = 2026
TEST_FRACTION = 0.20


def build_preprocessor():
    # KPIs are non-negative with long tails, log1p before scaling tames the outliers
    kpi_pipe = make_pipeline(FunctionTransformer(np.log1p, feature_names_out="one-to-one"), StandardScaler())
    return ColumnTransformer(
        [
            ("hour", FunctionTransformer(time_to_cyclic), [TIME_COL]),
            ("cell", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [CELL_COL]),
            ("kpi", kpi_pipe, KPI_COLS),
        ]
    )


def candidate_models():
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=3000),
            {"clf__C": [0.1, 1.0, 10.0]},
        ),
        "decision_tree": (
            DecisionTreeClassifier(random_state=SEED),
            {"clf__max_depth": [6, 10, 15, 20, None], "clf__min_samples_leaf": [1, 5, 10]},
        ),
        "knn": (
            KNeighborsClassifier(),
            {"clf__n_neighbors": [3, 5, 9, 15], "clf__weights": ["uniform", "distance"]},
        ),
        "naive_bayes": (
            GaussianNB(),
            {"clf__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3]},
        ),
        "svm": (
            SVC(kernel="rbf"),
            {"clf__C": [1.0, 10.0], "clf__gamma": ["scale"]},
        ),
    }


def main():
    df = load_kpi_csv(RAW_CSV)
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Loaded {before} rows, {before - len(df)} duplicates dropped -> {len(df)} rows")

    X = df[FEATURE_COLS]
    y = df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_FRACTION, stratify=y, random_state=SEED
    )
    print(f"Train {len(X_train)} rows, test {len(X_test)} rows, anomaly share train={y_train.mean():.3f} test={y_test.mean():.3f}")

    test_df = X_test.copy()
    test_df[TARGET] = y_test
    test_df.to_csv(TEST_CSV, index=False)

    MODEL_DIR.mkdir(exist_ok=True)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    results = {}

    for key, (clf, grid) in candidate_models().items():
        start = time.time()
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        search = GridSearchCV(pipe, grid, scoring="matthews_corrcoef", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_

        y_pred = best.predict(X_test)
        metrics = compute_metrics(y_test, y_pred, anomaly_scores(best, X_test))
        joblib.dump(best, MODEL_DIR / f"{key}.joblib", compress=3)

        best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
        results[key] = {
            "name": MODEL_NAMES[key],
            "best_params": best_params,
            "cv_mcc": round(search.best_score_, 4),
            "test_metrics": {m: round(float(v), 4) for m, v in metrics.items()},
            "train_seconds": round(time.time() - start, 1),
        }
        print(f"{MODEL_NAMES[key]:<24} params={best_params} cv_mcc={search.best_score_:.4f} ({results[key]['train_seconds']}s)")

    with open(MODEL_DIR / "metrics.json", "w") as fh:
        json.dump(results, fh, indent=2)

    table = pd.DataFrame({r["name"]: r["test_metrics"] for r in results.values()}).T
    table.index.name = "ML Model Name"
    print("\nTest-set metrics:")
    print(table.to_string())
    print("\nMarkdown table for README:")
    print("| ML Model Name | " + " | ".join(table.columns) + " |")
    print("|---" * (len(table.columns) + 1) + "|")
    for name, row in table.iterrows():
        print(f"| {name} | " + " | ".join(f"{v:.4f}" for v in row) + " |")


if __name__ == "__main__":
    main()
