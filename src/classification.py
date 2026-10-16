"""
classification.py
-----------------
Trains three binary classifiers (IsHighValue, IsFrequent, IsChurning)
and returns predicted probabilities for each customer.

These probability scores feed directly into the clustering step — we cluster
customers in "behavioral prediction space" rather than raw feature space.

Why this two-step approach?
  Raw features (Recency, Monetary, etc.) are skewed and multi-scale.
  Probability scores are already on [0, 1], share the same unit of meaning,
  and encode the classifier's learned non-linear relationships.
  Clustering on probability scores = clustering on learned behavior, not just numbers.

Model candidates:
  - LogisticRegression  : linear baseline, fast, interpretable
  - RandomForest        : handles non-linearity, built-in feature importance
  - XGBoost             : state-of-the-art for tabular data, gradient boosting

Selection metric: PR AUC (Precision-Recall AUC)
  Why PR AUC over ROC AUC?
  Our labels are imbalanced (e.g., only ~25% IsHighValue).
  ROC AUC is optimistic on imbalanced data because it counts true negatives
  heavily. PR AUC focuses only on the positive class — the one we care about.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    classification_report,
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
RANDOM_STATE = 42
LABEL_COLUMNS = ["IsHighValue", "IsFrequent", "IsChurning"]

# Base behavioral features (basket, temporal, diversity) — safe for all labels
BEHAVIORAL_FEATURES = [
    "AvgBasketSize", "AvgOrderValue", "MaxOrderValue",
    "PurchaseDaySpread", "AvgDaysBetweenOrders",
    "PreferredHour", "PreferredDayOfWeek",
    "UniqueProducts",
]

# Per-label feature sets:
# We EXCLUDE the RFM dimension that directly defines each label to prevent leakage.
#   IsHighValue is defined from Monetary -> exclude Monetary, keep Recency + Frequency
#   IsFrequent  is defined from Frequency -> exclude Frequency, keep Recency + Monetary
#   IsChurning  is defined from Recency   -> exclude Recency, keep Frequency + Monetary
LABEL_FEATURES: dict[str, list[str]] = {
    "IsHighValue": BEHAVIORAL_FEATURES + ["Recency", "Frequency"],
    "IsFrequent":  BEHAVIORAL_FEATURES + ["Recency", "Monetary"],
    "IsChurning":  BEHAVIORAL_FEATURES + ["Frequency", "Monetary"],
}


def _get_candidate_models() -> dict:
    """
    Return candidate model pipelines.

    Each pipeline includes StandardScaler because LogisticRegression is
    sensitive to feature scale. XGBoost and RandomForest are scale-invariant
    but we scale anyway for consistency across the comparison.
    """
    candidates = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE,
            )),
        ]),
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
    }
    if HAS_XGB:
        candidates["xgboost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("model", XGBClassifier(
                n_estimators=200,
                scale_pos_weight=3,   # handles imbalance
                eval_metric="aucpr",
                random_state=RANDOM_STATE,
                verbosity=0,
            )),
        ])
    return candidates


def compare_models(X: pd.DataFrame, y: pd.Series, label_name: str) -> pd.DataFrame:
    """
    Cross-validate all candidate models for a single label.
    Returns a DataFrame with mean PR AUC and ROC AUC for each model.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, pipeline in _get_candidate_models().items():
        scores = cross_validate(
            pipeline, X, y, cv=cv,
            scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc"},
            n_jobs=1,
        )
        rows.append({
            "label": label_name,
            "model": name,
            "pr_auc_mean": scores["test_pr_auc"].mean(),
            "pr_auc_std": scores["test_pr_auc"].std(),
            "roc_auc_mean": scores["test_roc_auc"].mean(),
        })
    return pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False)


def train_best_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    comparison: pd.DataFrame,
) -> Pipeline:
    """
    Retrain the best model (by PR AUC) on the full training set.
    The best model name is read from the comparison table.
    """
    best_name = comparison.iloc[0]["model"]
    print(f"    Best model: {best_name} (PR AUC = {comparison.iloc[0]['pr_auc_mean']:.3f})")
    pipeline = _get_candidate_models()[best_name]
    pipeline.fit(X_train, y_train)
    return pipeline


def run_classification(features: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Full classification pipeline for all three labels.

    Returns:
        scores_df  : DataFrame with Customer ID + 3 probability columns
        metrics    : dict with test metrics for each label
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    all_probs = {"Customer ID": features["Customer ID"].values}
    all_metrics = {}
    all_comparisons = []

    for label in LABEL_COLUMNS:
        print(f"\n--- Training classifier for: {label} ---")
        feature_cols = LABEL_FEATURES[label]
        print(f"    Using {len(feature_cols)} features (excluding '{label}' source column).")
        X = features[feature_cols]
        y = features[label]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
        )

        comparison = compare_models(X_train, y_train, label)
        all_comparisons.append(comparison)

        model = train_best_model(X_train, y_train, comparison)
        probs = model.predict_proba(X_test)[:, 1]

        # Evaluate on test set
        pr_auc = average_precision_score(y_test, probs)
        roc_auc = roc_auc_score(y_test, probs)
        all_metrics[label] = {
            "test_pr_auc": round(pr_auc, 4),
            "test_roc_auc": round(roc_auc, 4),
            "positive_rate": round(float(y.mean()), 4),
            "features_used": feature_cols,
        }
        print(f"    Test PR AUC: {pr_auc:.4f} | ROC AUC: {roc_auc:.4f}")

        # Refit on ALL data for final probability scores
        model.fit(X, y)
        all_probs[f"p_{label.lower()}"] = model.predict_proba(X)[:, 1]

        # Save model
        joblib.dump(model, MODELS_DIR / f"clf_{label.lower()}.joblib")

    # Combine comparisons and save
    comparison_df = pd.concat(all_comparisons, ignore_index=True)
    comparison_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    # Save test metrics
    with open(REPORTS_DIR / "classification_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    scores_df = pd.DataFrame(all_probs)
    scores_df.to_csv(
        Path(__file__).resolve().parents[1] / "data" / "processed" / "classification_scores.csv",
        index=False,
    )
    print("\nClassification complete. All models saved.")
    return scores_df, all_metrics
