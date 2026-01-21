"""Data handling utilities for HEXEval.

Generic tabular data processing for any CSV dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

LOG = logging.getLogger(__name__)


def load_data(
    path: str | Path,
    target_column: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict:
    """
    Load tabular data from CSV and auto-detect feature types.
    
    Parameters
    ----------
    path : str or Path
        Path to CSV file
    target_column : str, optional
        Name of target/label column. If None, no labels are extracted.
    test_size : float, default=0.2
        Proportion of data to use for test set
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'X_train': Training features (DataFrame)
        - 'X_test': Test features (DataFrame)
        - 'y_train': Training labels (Series or None)
        - 'y_test': Test labels (Series or None)
        - 'feature_names': List of feature column names
        - 'categorical_features': List of categorical column names
        - 'numeric_features': List of numeric column names
    
    Examples
    --------
    >>> data = load_data("dataset.csv", target_column="default")
    >>> X_train, y_train = data['X_train'], data['y_train']
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    
    LOG.info(f"Loading data from {path}")
    
<<<<<<< Updated upstream
    df = pd.read_csv(path)
    LOG.info(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    
    if target_column:
        if target_column not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in dataset. "
                f"Available columns: {df.columns.tolist()}"
            )
=======
    # Load CSV
    df = pd.read_csv(path)
    LOG.info(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Separate features and target
    if target_column and target_column in df.columns:
>>>>>>> Stashed changes
        y = df[target_column]
        X = df.drop(columns=[target_column])
        LOG.info(f"  Target column: '{target_column}'")
        LOG.info(f"  Class distribution: {y.value_counts().to_dict()}")
    else:
        y = None
        X = df
<<<<<<< Updated upstream
        LOG.warning("No target column specified - loading features only")
=======
        if target_column:
            LOG.warning(f"Target column '{target_column}' not found in dataset")
>>>>>>> Stashed changes
    
    # Auto-detect feature types
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    
    LOG.info(f"  Categorical features: {len(categorical_features)}")
    LOG.info(f"  Numeric features: {len(numeric_features)}")
    
    # Split train/test
    if y is not None:
        stratify = y if len(y.unique()) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
    else:
        X_train, X_test = train_test_split(
            X, test_size=test_size, random_state=random_state
        )
        y_train, y_test = None, None
    
    LOG.info(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": X.columns.tolist(),
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
    }


def get_data_summary(data: Dict) -> str:
    """
    Get human-readable data summary.
    
    Parameters
    ----------
    data : dict
        Data dictionary from load_data()
    
    Returns
    -------
    str
        Summary of dataset properties
    """
    n_train = len(data["X_train"])
    n_test = len(data["X_test"])
    n_features = len(data["feature_names"])
    n_categorical = len(data["categorical_features"])
    n_numeric = len(data["numeric_features"])
    has_target = data["y_train"] is not None
    
    summary = f"""
Dataset Summary:
  Total Features: {n_features}
    - Categorical: {n_categorical}
    - Numeric: {n_numeric}
  Training Samples: {n_train}
  Test Samples: {n_test}
  Target Labels: {'Yes' if has_target else 'No'}
    """.strip()
    
    return summary


def preprocess_for_model(
    X: pd.DataFrame,
    preprocessor=None,
    feature_names: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Apply preprocessing to features.
    
    Parameters
    ----------
    X : pd.DataFrame
        Raw features
    preprocessor : sklearn transformer, optional
        Fitted preprocessing pipeline
    feature_names : list, optional
        Expected feature names after preprocessing
    
    Returns
    -------
    np.ndarray
        Preprocessed features ready for model
    """
    if preprocessor is not None:
        X_proc = preprocessor.transform(X)
        return np.asarray(X_proc, dtype=np.float64)
    else:
        # No preprocessing - return as-is
        if feature_names:
            # Ensure column order matches
            missing_cols = set(feature_names) - set(X.columns)
            if missing_cols:
                raise ValueError(f"Missing expected features: {missing_cols}")
            X = X[feature_names]
<<<<<<< Updated upstream
        
        # Try converting to float for efficiency
        try:
            return X.values.astype(np.float64)
        except ValueError:
            # Fallback for mixed types (strings/categories)
            # This allows pipelines with internal encoders to function
            LOG.debug("Input contains non-numeric data, keeping as object dtype for pipeline processing")
            return X.values.astype(object)
=======
        return X.values.astype(np.float64)
>>>>>>> Stashed changes
