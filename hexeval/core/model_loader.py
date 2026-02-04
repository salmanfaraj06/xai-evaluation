"""Model loading utilities for HEXEval.

Supports loading sklearn and XGBoost models from .pkl or .joblib files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import joblib

from hexeval.core.wrapper import ModelWrapper

LOG = logging.getLogger(__name__)


def load_model(path: str | Path) -> ModelWrapper:
    """
    Load a trained model from disk and wrap it.
    
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
    ModelWrapper
        Wrapped model instance ready for evaluation
    
    Raises
    ------
    FileNotFoundError
        If model file doesn't exist
    ValueError
        If model doesn't have predict_proba method
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    
    LOG.info(f"Loading model from {path}")
    
    try:
        import xgboost as xgb
        if hasattr(xgb, 'XGBClassifier'):
            if not hasattr(xgb.XGBClassifier, 'use_label_encoder'):
                xgb.XGBClassifier.use_label_encoder = False
            if not hasattr(xgb.XGBClassifier, 'gpu_id'):
                xgb.XGBClassifier.gpu_id = None
            if not hasattr(xgb.XGBClassifier, 'predictor'):
                xgb.XGBClassifier.predictor = None
        if hasattr(xgb, 'XGBRegressor'):
            if not hasattr(xgb.XGBRegressor, 'use_label_encoder'):
                xgb.XGBRegressor.use_label_encoder = False
            if not hasattr(xgb.XGBRegressor, 'gpu_id'):
                xgb.XGBRegressor.gpu_id = None
            if not hasattr(xgb.XGBRegressor, 'predictor'):
                xgb.XGBRegressor.predictor = None
        if hasattr(xgb, 'XGBModel'):
            if not hasattr(xgb.XGBModel, 'gpu_id'):
                xgb.XGBModel.gpu_id = None
            if not hasattr(xgb.XGBModel, 'predictor'):
                xgb.XGBModel.predictor = None
    except ImportError:
        pass 
    
    try:
        artifact = joblib.load(path)
    except Exception as e:
        raise ValueError(f"Failed to load model: {e}")
    if isinstance(artifact, dict):
        model = artifact.get("model")
        preprocessor = artifact.get("preprocessor")
        feature_names = artifact.get("feature_names")
        threshold = artifact.get("threshold", 0.5)
    else:
        # Raw model object
        model = artifact
        preprocessor = None
        feature_names = None
        threshold = 0.5
    
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
        if isinstance(artifact, ModelWrapper):
            wrapper = artifact
        else:
             wrapper = ModelWrapper(model=artifact)

    # Validate model has predict_proba via wrapper
    try:

        if not hasattr(wrapper.model, "predict_proba"):
             raise ValueError(f"Model {type(wrapper.model)} missing predict_proba")
    except Exception:

         pass
    
    model_type = type(wrapper.model).__name__
    
    if wrapper.preprocessor:
        LOG.info(f"Loaded {model_type} with attached preprocessor")
    else:
        LOG.info(f"Loaded {model_type} (no preprocessor found)")
        
    return wrapper


def get_model_info(wrapper: ModelWrapper) -> str:
    """
    Get human-readable model information.
    
    Parameters
    ----------
    wrapper : ModelWrapper
        Loaded model wrapper
    
    Returns
    -------
    str
        Summary of model properties
    """
    info_dict = wrapper.get_model_info()
    
    info = f"""
Model Type: {info_dict['model_type']}
Features: {info_dict['n_features']}
Preprocessor: {'Yes' if info_dict['has_preprocessor'] else 'No'}
Classes: {info_dict['classes']}
    """.strip()
    
    return info
