"""Model loading utilities for HEXEval.

Supports loading sklearn and XGBoost models from .pkl or .joblib files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import joblib

LOG = logging.getLogger(__name__)


def load_model(path: str | Path) -> Dict[str, Any]:
    """
    Load a trained model from disk.
    
    Supports:
    - sklearn models (.pkl, .joblib)
    - XGBoost models (.pkl, .joblib)
    - Model artifacts (dict with 'model', 'preprocessor', 'feature_names')
    
    Parameters
    ----------
    path : str or Path
        Path to model file (.pkl or .joblib)
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'model': The trained model
        - 'preprocessor': Optional preprocessing pipeline
        - 'feature_names': Optional list of feature names
        - 'model_type': Model class name
        - 'threshold': Classification threshold (default 0.5)
    
    Raises
    ------
    FileNotFoundError
        If model file doesn't exist
    ValueError
        If model doesn't have predict_proba method
    
    Examples
    --------
    >>> artifact = load_model("my_model.pkl")
    >>> model = artifact['model']
    >>> predictions = model.predict_proba(X_test)
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    
    LOG.info(f"Loading model from {path}")
    
    # Load artifact
    try:
        artifact = joblib.load(path)
    except Exception as e:
        raise ValueError(f"Failed to load model: {e}")
    
    # Handle different artifact formats
    if isinstance(artifact, dict):
        # Already in artifact format
        model = artifact.get("model")
        preprocessor = artifact.get("preprocessor")
        feature_names = artifact.get("feature_names")
        threshold = artifact.get("default_threshold", 0.5)
    else:
        # Raw model object
        model = artifact
        preprocessor = None
        feature_names = None
        threshold = 0.5
    
    # Validate model has predict_proba
    if not hasattr(model, "predict_proba"):
        raise ValueError(
            f"Model of type {type(model).__name__} must have 'predict_proba' method. "
            "HEXEval requires probability estimates for evaluation."
        )
    
    # Extract model type
    model_type = type(model).__name__
    
    LOG.info(f"Loaded {model_type} model successfully")
    if feature_names:
        LOG.info(f"  Features: {len(feature_names)}")
    if preprocessor:
        LOG.info(f"  Preprocessor: {type(preprocessor).__name__}")
    
    return {
        "model": model,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "model_type": model_type,
        "threshold": float(threshold),
    }


def get_model_info(artifact: Dict[str, Any]) -> str:
    """
    Get human-readable model information.
    
    Parameters
    ----------
    artifact : dict
        Model artifact from load_model()
    
    Returns
    -------
    str
        Summary of model properties
    """
    model_type = artifact["model_type"]
    n_features = len(artifact["feature_names"]) if artifact["feature_names"] else "Unknown"
    has_preprocessor = artifact["preprocessor"] is not None
    threshold = artifact["threshold"]
    
    info = f"""
Model Type: {model_type}
Features: {n_features}
Preprocessor: {'Yes' if has_preprocessor else 'No'}
Classification Threshold: {threshold}
    """.strip()
    
    return info
