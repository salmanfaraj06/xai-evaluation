"""Feature preprocessing utilities."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def infer_feature_types(X: pd.DataFrame, config: dict) -> Tuple[List[str], List[str], List[str]]:
    """Infer numeric, binary, and categorical columns from config + dtypes."""
    data_cfg = config.get("data", {})

    categorical = list(data_cfg.get("categorical", []))
    binary = list(data_cfg.get("binary", []))
    numeric = list(data_cfg.get("numeric", []))

    if not numeric:
        numeric = [
            c for c in X.columns
            if pd.api.types.is_numeric_dtype(X[c]) and c not in binary
        ]

    if not categorical:
        categorical = [
            c for c in X.columns
            if c not in numeric and c not in binary
        ]

    # Keep only existing cols
    categorical = [c for c in categorical if c in X.columns]
    binary = [c for c in binary if c in X.columns]
    numeric = [c for c in numeric if c in X.columns]

    return numeric, binary, categorical


def build_preprocessor(config: dict, X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str]]:
    """Create and fit a preprocessing pipeline; return transformer and feature names."""
    scale_numeric = config.get("preprocessing", {}).get("scale_numeric", True)
    impute_strategy = config.get("preprocessing", {}).get("impute_strategy", "median")

    numeric, binary, categorical = infer_feature_types(X, config)

    numeric_cols = numeric + binary

    num_steps = [("imputer", SimpleImputer(strategy=impute_strategy))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(num_steps)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    preprocessor.fit(X)
    feature_names = preprocessor.get_feature_names_out().tolist()

    return preprocessor, feature_names


def transform_features(preprocessor: ColumnTransformer, X: pd.DataFrame) -> np.ndarray:
    """Apply a fitted preprocessor to a DataFrame."""
    return preprocessor.transform(X)


def build_feature_metadata(
    X: pd.DataFrame,
    preprocessor: ColumnTransformer,
    config: dict,
) -> Dict[str, List[str]]:
    """Return metadata for explainers (especially DiCE)."""
    numeric, binary, categorical = infer_feature_types(X, config)
    return {
        "continuous": numeric,
        "binary": binary,
        "categorical": categorical,
        "feature_names": preprocessor.get_feature_names_out().tolist(),
    }


def one_hot_encode(
    X: pd.DataFrame,
    categorical: List[str],
    drop_first: bool = True,
) -> pd.DataFrame:
    """One-hot encode categorical columns with a deterministic column order."""
    categorical = [c for c in categorical if c in X.columns]
    return pd.get_dummies(X, columns=categorical, drop_first=drop_first)


def align_columns(X: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """Align a DataFrame to the expected feature order, adding missing cols as 0."""
    aligned = X.copy()
    for col in feature_names:
        if col not in aligned.columns:
            aligned[col] = 0.0
    extra = [c for c in aligned.columns if c not in feature_names]
    if extra:
        aligned = aligned.drop(columns=extra)
    return aligned[feature_names]
