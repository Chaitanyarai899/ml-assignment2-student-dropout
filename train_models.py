"""
Training script - ML Assignment 2 (BITS WILP, M.Tech AIML)
Author : Chaitanya Rai (2026ak05673)

Dataset: "Predict Students' Dropout and Academic Success" (UCI ML Repository, id 697)
Task   : 3-class classification -> Dropout / Enrolled / Graduate

Trains five classifiers on the same 80/20 stratified split, evaluates each on the
held-out split and saves:
  * model/<name>.joblib      - fitted sklearn Pipelines (preprocessing + estimator)
  * model/metrics.json       - evaluation metrics for every model
  * model/label_encoder.joblib, model/feature_columns.json - preprocessing artifacts
  * test_data.csv            - the held-out split (features + target) used by the app
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
RAW_CSV = ROOT / "data" / "student_dropout_full.csv"
MODEL_DIR = ROOT / "model"
TARGET = "Target"
SEED = 5673  # last digits of my BITS ID, used for a reproducible split


def load_dataset(path: Path) -> pd.DataFrame:
    """UCI ships this file ';'-separated with a BOM and a stray tab in one header."""
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df.columns = [c.strip().replace("\t", "") for c in df.columns]
    # Fix the original typo so the column name is readable in the app
    df = df.rename(columns={"Nacionality": "Nationality"})
    return df


def build_models() -> dict:
    """Each model is a Pipeline so the scaler travels with the estimator."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, C=0.5, random_state=SEED)),
        ]),
        "Decision Tree": Pipeline([
            # trees do not need scaling; depth/leaf limits curb overfitting
            ("clf", DecisionTreeClassifier(max_depth=7, min_samples_leaf=12,
                                           random_state=SEED)),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=15, weights="distance")),
        ]),
        "Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GaussianNB()),
        ]),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            # SVC has no native probabilities; calibrate it so AUC can be computed
            ("clf", CalibratedClassifierCV(
                SVC(kernel="rbf", C=1.5, gamma="scale", random_state=SEED),
                ensemble=False)),
        ]),
    }


def evaluate(model, X, y_true) -> dict:
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        # multi-class AUC: one-vs-rest, weighted by class support
        "AUC": roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def main():
    df = load_dataset(RAW_CSV)
    print(f"Dataset shape: {df.shape}")
    print(df[TARGET].value_counts().to_string(), "\n")

    feature_cols = [c for c in df.columns if c != TARGET]
    encoder = LabelEncoder().fit(df[TARGET])

    train_df, test_df = train_test_split(
        df, test_size=0.20, stratify=df[TARGET], random_state=SEED)

    X_train, y_train = train_df[feature_cols], encoder.transform(train_df[TARGET])
    X_test, y_test = test_df[feature_cols], encoder.transform(test_df[TARGET])

    MODEL_DIR.mkdir(exist_ok=True)
    results = {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        results[name] = evaluate(model, X_test, y_test)
        joblib.dump(model, MODEL_DIR / f"{slug(name)}.joblib")
        print(f"{name:<24} " + "  ".join(f"{k}={v:.4f}" for k, v in results[name].items()))

    joblib.dump(encoder, MODEL_DIR / "label_encoder.joblib")
    (MODEL_DIR / "feature_columns.json").write_text(json.dumps(feature_cols, indent=2))
    (MODEL_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    test_df.to_csv(ROOT / "test_data.csv", index=False)
    print(f"\nSaved models to {MODEL_DIR} and test split ({len(test_df)} rows) to test_data.csv")


if __name__ == "__main__":
    main()
