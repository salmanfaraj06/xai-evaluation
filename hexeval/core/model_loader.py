"""Model loading utilities for HEXEval.

Supports loading sklearn and XGBoost models from .pkl or .joblib files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import joblib

<<<<<<< Updated upstream
from hexeval.core.wrapper import ModelWrapper

LOG = logging.getLogger(__name__)


def load_model(path: str | Path) -> ModelWrapper:
    """
    Load a trained model from disk and wrap it.
=======
LOG = logging.getLogger(__name__)


def load_model(path: str | Path) -> Dict[str, Any]:
    """
    Load a trained model from disk.
>>>>>>> Stashed changes
    
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
<<<<<<< Updated upstream
    ModelWrapper
        Wrapped model instance ready for evaluation
=======
    dict
        Dictionary containing:
        - 'model': The trained model
        - 'preprocessor': Optional preprocessing pipeline
        - 'feature_names': Optional list of feature names
        - 'model_type': Model class name
        - 'threshold': Classification threshold (default 0.5)
>>>>>>> Stashed changes
    
    Raises
    ------
    FileNotFoundError
        If model file doesn't exist
    ValueError
        If model doesn't have predict_proba method
<<<<<<< Updated upstream
=======
    
    Examples
    --------
    >>> artifact = load_model("my_model.pkl")
    >>> model = artifact['model']
    >>> predictions = model.predict_proba(X_test)
>>>>>>> Stashed changes
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    
    LOG.info(f"Loading model from {path}")
    
<<<<<<< Updated upstream
=======
    # Load artifact
>>>>>>> Stashed changes
    try:
        artifact = joblib.load(path)
    except Exception as e:
        raise ValueError(f"Failed to load model: {e}")
    
<<<<<<< Updated upstream
=======
    # Handle different artifact formats
>>>>>>> Stashed changes
    if isinstance(artifact, dict):
        # Already in artifact format
        model = artifact.get("model")
        preprocessor = artifact.get("preprocessor")
        feature_names = artifact.get("feature_names")
<<<<<<< Updated upstream
        threshold = artifact.get("threshold", 0.5)
=======
        threshold = artifact.get("default_threshold", 0.5)
>>>>>>> Stashed changes
    else:
        # Raw model object
        model = artifact
        preprocessor = None
        feature_names = None
        threshold = 0.5
    
<<<<<<< Updated upstream
    # Check if the loaded object is already a ModelWrapper (or creates one)
    if hasattr(model, "predict_proba") and not isinstance(model, ModelWrapper):
        wrapper = ModelWrapper(
            model=model,
            preprocessor=preprocessor,
            threshold=float(threshold)
        )
        
        # Verify predict_proba exists on the underlying model
        if not hasattr(model, "predict_proba"):
             raise ValueError(f"Model {type(model)} missing predict_proba. HEXEval requires probabilistic models.")
             
    elif isinstance(model, ModelWrapper):
        wrapper = model
    else:
        # Last attempt - maybe the artifact IS the model wrapper
        if isinstance(artifact, ModelWrapper):
            wrapper = artifact
        # Or maybe it's a raw model but without predict_proba (which we catch later)
        else:
             wrapper = ModelWrapper(model=artifact)

    # Validate model has predict_proba via wrapper
    try:
        # We don't call it here, just inspect wrapped model
        if not hasattr(wrapper.model, "predict_proba"):
             raise ValueError(f"Model {type(wrapper.model)} missing predict_proba")
    except Exception:
         # Some custom models might implement predict_proba dynamically, so this check is soft
         pass
    
    model_type = type(wrapper.model).__name__
    
    if wrapper.preprocessor:
        LOG.info(f"Loaded {model_type} with attached preprocessor")
    else:
        LOG.info(f"Loaded {model_type} (no preprocessor found)")
        
    return wrapper


def get_model_info(wrapper: ModelWrapper) -> str:
=======
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
>>>>>>> Stashed changes
    """
    Get human-readable model information.
    
    Parameters
    ----------
<<<<<<< Updated upstream
    wrapper : ModelWrapper
        Loaded model wrapper
=======
    artifact : dict
        Model artifact from load_model()
>>>>>>> Stashed changes
    
    Returns
    -------
    str
        Summary of model properties
    """
<<<<<<< Updated upstream
    info_dict = wrapper.get_model_info()
    
    info = f"""
Model Type: {info_dict['model_type']}
Features: {info_dict['n_features']}
Preprocessor: {'Yes' if info_dict['has_preprocessor'] else 'No'}
Classes: {info_dict['classes']}
=======
    model_type = artifact["model_type"]
    n_features = len(artifact["feature_names"]) if artifact["feature_names"] else "Unknown"
    has_preprocessor = artifact["preprocessor"] is not None
    threshold = artifact["threshold"]
    
    info = f"""
Model Type: {model_type}
Features: {n_features}
Preprocessor: {'Yes' if has_preprocessor else 'No'}
Classification Threshold: {threshold}
>>>>>>> Stashed changes
    """.strip()
    
    return info
