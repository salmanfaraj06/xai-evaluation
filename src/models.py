"""Model training and evaluation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier


def _make_model(config: dict):
    """Instantiate a model from config."""
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "xgboost").lower()
    params = model_cfg.get("params", {})

    if model_type == "xgboost":
        return XGBClassifier(**params)
    if model_type == "random_forest":
        return RandomForestClassifier(**params)
    if model_type == "logistic_regression":
        return LogisticRegression(max_iter=1000, **params)

    raise ValueError(f"Unsupported model type '{model_type}'.")


def train_credit_model(config: dict, X_train: np.ndarray, y_train: np.ndarray):
    """Train the configured model."""
    model = _make_model(config)
    model.fit(X_train, y_train)
    return model


def evaluate_model_performance(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Compute standard Layer-1 metrics."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_proba),
    }


def save_artifact(
    model,
    preprocessor,
    feature_names,
    path: str | Path,
    default_threshold: float = 0.5,
):
    """Persist model + preprocessor bundle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "default_threshold": default_threshold,
    }
    joblib.dump(artifact, path)
    return artifact


def load_artifact(path: str | Path) -> Dict[str, Any]:
    """Load model bundle from disk."""
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError("Artifact does not include a 'model' key.")
    return artifact
