import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from features import (
    FEATURE_COLS,
    MODEL_NAMES,
    TARGET,
    anomaly_scores,
    compute_metrics,
    load_kpi_csv,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
BUNDLED_TEST = ROOT / "test_data.csv"
CLASS_LABELS = ["Normal", "Anomaly"]
METRIC_HELP = {
    "Accuracy": "Share of 15-min cell samples classified correctly",
    "AUC": "How well the model ranks anomalous samples above normal ones",
    "Precision": "Of the alarms raised, how many were real anomalies",
    "Recall": "Of the real anomalies, how many raised an alarm",
    "F1": "Balance of precision and recall",
    "MCC": "Correlation between prediction and truth, robust to the 72/28 class imbalance",
}

st.set_page_config(page_title="LTE Cell Anomaly Monitor", page_icon="📡", layout="wide")


@st.cache_resource
def load_models():
    return {key: joblib.load(MODEL_DIR / f"{key}.joblib") for key in MODEL_NAMES}


@st.cache_data
def load_training_summary():
    with open(MODEL_DIR / "metrics.json") as fh:
        return json.load(fh)


def read_uploaded(upload):
    if upload is not None:
        return load_kpi_csv(upload), upload.name
    return load_kpi_csv(BUNDLED_TEST), "Bundled test_data.csv"


def check_columns(df):
    missing = [c for c in FEATURE_COLS + [TARGET] if c not in df.columns]
    return missing


def evaluate(model, X, y):
    y_pred = model.predict(X)
    return y_pred, compute_metrics(y, y_pred, anomaly_scores(model, X))


def plot_confusion(cm, title):
    # correct predictions shaded green, errors (missed anomalies / false alarms) shaded red
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    cell_colours = np.zeros((2, 2, 4))
    shares = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            shares[i, j] = cm[i, j] / cm[i].sum() if cm[i].sum() else 0
            cmap = plt.get_cmap("Greens" if i == j else "Reds")
            cell_colours[i, j] = cmap(0.15 + 0.7 * shares[i, j])
    ax.imshow(cell_colours)
    for i in range(2):
        for j in range(2):
            colour = "white" if shares[i, j] > 0.5 else "#1f2a44"
            ax.text(j, i, f"{cm[i, j]}\n({shares[i, j]:.1%})", ha="center", va="center", color=colour, fontsize=11)
    ax.set_xticks([0, 1], labels=CLASS_LABELS)
    ax.set_yticks([0, 1], labels=CLASS_LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


models = load_models()
training_summary = load_training_summary()

st.title("📡 LTE Cell Anomaly Monitor")
st.caption(
    "Classifies 15-minute LTE cell KPI samples (PRB usage, throughput, active UEs) as normal or anomalous "
    "using five pre-trained models. Upload labelled test data; models are only loaded, never retrained."
)

with st.sidebar:
    st.header("1 · Test data")
    upload = st.file_uploader("Upload test CSV (features + Unusual column)", type=["csv"])
    st.caption("No file? The bundled hold-out set (7,361 samples) is used.")
    st.header("2 · Model")
    model_key = st.selectbox("Classifier", list(MODEL_NAMES), format_func=MODEL_NAMES.get)
    params = training_summary[model_key]["best_params"]
    st.caption("Tuned hyper-parameters: " + ", ".join(f"`{k}={v}`" for k, v in params.items()))

try:
    data, source_name = read_uploaded(upload)
except Exception as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

missing = check_columns(data)
if missing:
    st.error("The uploaded file is missing required columns: " + ", ".join(f"`{c}`" for c in missing))
    st.info("Expected columns: " + ", ".join(f"`{c}`" for c in FEATURE_COLS + [TARGET]))
    st.stop()

data = data.dropna(subset=FEATURE_COLS + [TARGET])
X = data[FEATURE_COLS]
y = data[TARGET].astype(int)

if y.nunique() < 2:
    st.warning("The test data contains only one class, so AUC and MCC cannot be computed meaningfully.")
    st.stop()

src_col, rows_col, cells_col, anom_col = st.columns(4)
src_col.metric("Source", source_name if len(source_name) < 28 else source_name[:25] + "…")
rows_col.metric("Samples", f"{len(data):,}")
cells_col.metric("Cells", data["CellName"].nunique())
anom_col.metric("Actual anomaly share", f"{y.mean():.1%}")

tab_model, tab_compare, tab_data = st.tabs(["🔍 Selected model", "📊 Compare all models", "🗂 Data overview"])

with tab_model:
    model = models[model_key]
    y_pred, metrics = evaluate(model, X, y)
    st.subheader(f"{MODEL_NAMES[model_key]} on {len(data):,} samples")

    for col, (name, value) in zip(st.columns(6), metrics.items()):
        col.metric(name, f"{value:.4f}", help=METRIC_HELP[name])

    cm_col, rep_col = st.columns([1, 1.2])
    with cm_col:
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        st.pyplot(plot_confusion(cm, "Confusion matrix"), clear_figure=True)
        missed, false_alarms = cm[1, 0], cm[0, 1]
        st.caption(f"Missed anomalies: **{missed}** · False alarms: **{false_alarms}**")
    with rep_col:
        st.markdown("**Classification report**")
        report = classification_report(y, y_pred, labels=[0, 1], target_names=CLASS_LABELS, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(4), width="stretch")

    with st.expander("Per-cell alarm view"):
        per_cell = (
            pd.DataFrame({"CellName": X["CellName"].values, "actual": y.values, "predicted": y_pred})
            .groupby("CellName")
            .agg(samples=("actual", "size"), actual_anomalies=("actual", "sum"), predicted_anomalies=("predicted", "sum"))
            .sort_values("actual_anomalies", ascending=False)
        )
        st.dataframe(per_cell, width="stretch")

    predictions = data.copy()
    predictions["Predicted"] = y_pred
    st.download_button(
        "Download predictions CSV",
        predictions.to_csv(index=False).encode(),
        file_name=f"predictions_{model_key}.csv",
        mime="text/csv",
    )

with tab_compare:
    st.subheader("All five models on the same test data")
    rows = {}
    for key, mdl in models.items():
        rows[MODEL_NAMES[key]] = evaluate(mdl, X, y)[1]
    table = pd.DataFrame(rows).T
    table.index.name = "ML Model Name"
    st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#cfe8d5"), width="stretch")

    best = table["MCC"].idxmax()
    st.success(f"Best model on this data by MCC: **{best}** (MCC {table.loc[best, 'MCC']:.4f}, F1 {table.loc[best, 'F1']:.4f})")

    chosen = st.multiselect("Metrics to chart", list(table.columns), default=["F1", "MCC", "AUC"])
    if chosen:
        fig, ax = plt.subplots(figsize=(8, 3.6))
        width = 0.8 / len(chosen)
        positions = np.arange(len(table))
        for i, metric in enumerate(chosen):
            ax.bar(positions + i * width, table[metric], width, label=metric)
        ax.set_xticks(positions + width * (len(chosen) - 1) / 2, labels=table.index, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(ncol=len(chosen), fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

with tab_data:
    st.subheader("Uploaded data")
    st.dataframe(data.head(200), width="stretch")
    left, right = st.columns(2)
    with left:
        st.markdown("**Class distribution**")
        st.bar_chart(y.map({0: "Normal", 1: "Anomaly"}).value_counts())
    with right:
        st.markdown("**Anomaly rate by hour of day**")
        hours = X["Time"].astype(str).str.split(":").str[0].astype(int)
        st.line_chart(y.groupby(hours).mean().rename("anomaly share"))
