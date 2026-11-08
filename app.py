"""
Student Outcome Predictor - Streamlit front-end for ML Assignment 2
Author : Chaitanya Rai (2026ak05673), M.Tech AIML, BITS Pilani WILP

Loads the five pre-trained classifiers from ./model and evaluates them on an
uploaded test CSV. No model is retrained inside the app.
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"
SAMPLE_CSV = APP_DIR / "test_data.csv"
TARGET = "Target"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
CLASS_COLOURS = {"Dropout": "#d1495b", "Enrolled": "#edae49", "Graduate": "#00798c"}

st.set_page_config(page_title="Student Outcome Predictor", page_icon="🎓", layout="wide")


# ---------------------------------------------------------------- loaders
@st.cache_resource
def load_artifacts():
    models = {name: joblib.load(MODEL_DIR / f) for name, f in MODEL_FILES.items()}
    encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
    features = json.loads((MODEL_DIR / "feature_columns.json").read_text())
    return models, encoder, features


@st.cache_data
def read_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]
    return df


def score_model(model, X, y_true, n_classes):
    y_pred = model.predict(X)
    proba = model.predict_proba(X)
    present = np.unique(y_true)
    # AUC needs at least two classes in the uploaded file
    if len(present) < 2:
        auc = np.nan
    elif n_classes == 2:
        auc = roc_auc_score(y_true, proba[:, 1])
    else:
        auc = roc_auc_score(y_true, proba[:, present] / proba[:, present].sum(1, keepdims=True),
                            multi_class="ovr", average="weighted", labels=present)
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    return metrics, y_pred, proba


# ---------------------------------------------------------------- sidebar
models, encoder, feature_cols = load_artifacts()
class_names = list(encoder.classes_)

with st.sidebar:
    st.markdown("### 🎓 Student Outcome Predictor")
    st.caption("ML Assignment 2 · Chaitanya Rai · 2026ak05673")
    st.divider()

    st.markdown("**1. Test data**")
    uploaded = st.file_uploader("Upload a test CSV (features + `Target`)", type=["csv"])
    use_sample = st.toggle("Use the bundled test_data.csv instead", value=uploaded is None)
    with open(SAMPLE_CSV, "rb") as fh:
        st.download_button("⬇️ Download sample test_data.csv", fh, "test_data.csv",
                           mime="text/csv", width="stretch")

    st.markdown("**2. Model**")
    chosen = st.selectbox("Choose a classifier", list(MODEL_FILES), index=0)
    st.divider()
    st.caption("Models were trained offline (`train_models.py`) on an 80/20 stratified "
               "split of the UCI *Predict Students' Dropout and Academic Success* dataset. "
               "The app only runs inference.")

# ---------------------------------------------------------------- header
st.title("Will the student drop out, stay enrolled, or graduate?")
st.write("Evaluate five pre-trained classifiers on held-out student records from a Portuguese "
         "higher-education institution (36 academic, demographic and socio-economic features).")

source = SAMPLE_CSV if (use_sample or uploaded is None) else uploaded
if uploaded is None and not use_sample:
    st.info("Upload a CSV in the sidebar, or switch on the bundled test data.")
    st.stop()

data = read_csv(source)
missing = [c for c in feature_cols + [TARGET] if c not in data.columns]
if missing:
    st.error(f"The uploaded file is missing {len(missing)} required column(s): "
             + ", ".join(f"`{m}`" for m in missing[:8]) + (" …" if len(missing) > 8 else ""))
    st.stop()

unknown = sorted(set(data[TARGET].astype(str)) - set(class_names))
if unknown:
    st.error(f"Unknown target labels {unknown}. Expected one of {class_names}.")
    st.stop()

X = data[feature_cols]
y = encoder.transform(data[TARGET].astype(str))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows evaluated", f"{len(data):,}")
c2.metric("Input features", len(feature_cols))
counts = data[TARGET].value_counts()
c3.metric("Graduate share", f"{counts.get('Graduate', 0) / len(data):.0%}")
c4.metric("Dropout share", f"{counts.get('Dropout', 0) / len(data):.0%}")

with st.expander("Preview the test data"):
    st.dataframe(data.head(50), width="stretch")

tab_single, tab_compare = st.tabs([f"🔍 {chosen}", "📊 Compare all five models"])

# ---------------------------------------------------------------- single model
with tab_single:
    metrics, y_pred, proba = score_model(models[chosen], X, y, len(class_names))
    st.subheader(f"Evaluation metrics: {chosen}")
    cols = st.columns(6)
    for col, (k, v) in zip(cols, metrics.items()):
        col.metric(k, "n/a" if pd.isna(v) else f"{v:.3f}")

    left, right = st.columns([1, 1.15])
    with left:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y, y_pred, labels=range(len(class_names)))
        fig, ax = plt.subplots(figsize=(4.6, 3.9))
        sns.heatmap(cm, annot=True, fmt="d", cmap="crest", cbar=False,
                    xticklabels=class_names, yticklabels=class_names, ax=ax,
                    annot_kws={"size": 12})
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("**Classification report**")
        report = classification_report(y, y_pred, labels=range(len(class_names)),
                                       target_names=class_names, output_dict=True,
                                       zero_division=0)
        rep_df = pd.DataFrame(report).T
        rep_df["support"] = rep_df["support"].astype(int)
        st.dataframe(rep_df.style.format({"precision": "{:.3f}", "recall": "{:.3f}",
                                          "f1-score": "{:.3f}"}),
                     width="stretch")

        st.markdown("**Predicted vs actual class distribution**")
        dist = pd.DataFrame({
            "Actual": pd.Series(encoder.inverse_transform(y)).value_counts(),
            "Predicted": pd.Series(encoder.inverse_transform(y_pred)).value_counts(),
        }).reindex(class_names).fillna(0).astype(int)
        st.bar_chart(dist, height=200)

    preds = data.copy()
    preds.insert(0, "Predicted", encoder.inverse_transform(y_pred))
    preds.insert(1, "Confidence", proba.max(axis=1).round(3))
    preds.insert(2, "Correct", preds["Predicted"] == preds[TARGET])
    st.markdown("**Row-level predictions**")
    only_wrong = st.checkbox("Show misclassified rows only")
    view = preds[~preds["Correct"]] if only_wrong else preds
    st.dataframe(view[["Predicted", TARGET, "Confidence", "Correct"] + feature_cols[:6]],
                 width="stretch", height=260)
    st.download_button("⬇️ Download predictions", preds.to_csv(index=False).encode(),
                       f"predictions_{chosen.lower().replace(' ', '_')}.csv", "text/csv")

# ---------------------------------------------------------------- comparison
with tab_compare:
    st.subheader("All models on this test data")
    rows = {name: score_model(m, X, y, len(class_names))[0] for name, m in models.items()}
    table = pd.DataFrame(rows).T
    st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#cfe8e4"),
                 width="stretch")
    best = table["MCC"].idxmax()
    st.success(f"Best model on this file by MCC: **{best}** "
               f"(accuracy {table.loc[best, 'Accuracy']:.3f}, MCC {table.loc[best, 'MCC']:.3f})")

    metric = st.radio("Metric to plot", list(table.columns), horizontal=True, index=5)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    order = table[metric].sort_values()
    ax.barh(order.index, order.values,
            color=["#00798c" if n == order.idxmax() else "#9bb7c0" for n in order.index])
    for i, v in enumerate(order.values):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, min(1.0, order.max() + 0.12))
    ax.set_xlabel(metric)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
