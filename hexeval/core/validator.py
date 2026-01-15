"""Validation utilities for HEXEval.

Ensures model and data compatibility.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)


def validate_model_data_compatibility(
    model_artifact: Dict[str, Any],
    data: Dict,
) -> Dict[str, str]:
    """
    Validate that model and data are compatible.
    
    Parameters
    ----------
    model_artifact : dict
        Model artifact from load_model()
    data : dict
        Data dictionary from load_data()
    
    Returns
    -------
    dict
        Validation results with keys:
        - 'status': 'valid' or 'invalid'
        - 'warnings': List of warning messages
        - 'errors': List of error messages
    
    Examples
    --------
    >>> result = validate_model_data_compatibility(model_artifact, data)
    >>> if result['status'] == 'invalid':
    ...     print("Errors:", result['errors'])
    """
    warnings = []
    errors = []
    
    model = model_artifact["model"]
    preprocessor = model_artifact.get("preprocessor")
    expected_features = model_artifact.get("feature_names")
    
    X_sample = data["X_train"].iloc[:1]
    
    # Check 1: Model has predict_proba
    if not hasattr(model, "predict_proba"):
        errors.append("Model must have 'predict_proba' method")
    
    # Check 2: Feature compatibility
    if expected_features:
        actual_features = set(data["feature_names"])
        expected_set = set(expected_features) if isinstance(expected_features, list) else set()
        
        missing_features = expected_set - actual_features
        if missing_features:
            # Check if these are one-hot encoded features (categorical columns)
            # These are OK to be missing - preprocessor will handle it
            onehot_missing = [f for f in missing_features if any(sep in f for sep in ['_', '-'])]
            real_missing = [f for f in missing_features if f not in onehot_missing]
            
            if real_missing:
                errors.append(f"Missing required numeric features: {list(real_missing)[:5]}")
            
            if onehot_missing:
                warnings.append(f"Some categorical values missing (OK if using preprocessor): {list(onehot_missing)[:3]}")
        
        extra_features = actual_features - expected_set
        if extra_features:
            warnings.append(f"Extra features in data (will be ignored): {list(extra_features)[:5]}")
    
    # Check 3: Can we make a prediction?
    try:
        if preprocessor:
            X_proc = preprocessor.transform(X_sample)
        else:
            if expected_features:
                X_proc = X_sample[expected_features].values
            else:
                X_proc = X_sample.values
        
        pred = model.predict_proba(X_proc)
        
        # Validate prediction shape
        if pred.shape[0] != 1:
            errors.append(f"Unexpected prediction shape: {pred.shape}")
        
        if pred.shape[1] != 2:
            warnings.append(f"Model has {pred.shape[1]} classes (expected binary classification)")
        
        LOG.info("✓ Model can make predictions on data")
        
    except Exception as e:
        errors.append(f"Failed to make prediction: {str(e)}")
    
    # Determine status
    status = "valid" if len(errors) == 0 else "invalid"
    
    if warnings:
        LOG.warning(f"Validation warnings: {warnings}")
    if errors:
        LOG.error(f"Validation errors: {errors}")
    
    return {
        "status": status,
        "warnings": warnings,
        "errors": errors,
    }


def check_explainer_requirements(data: Dict) -> Dict[str, bool]:
    """
    Check which explainers can work with the given data.
    
    Parameters
    ----------
    data : dict
        Data dictionary from load_data()
    
    Returns
    -------
    dict
        Boolean flags for each explainer:
        - 'shap': Can use SHAP
        - 'lime': Can use LIME
        - 'anchor': Can use Anchor
        - 'dice': Can use DiCE
    """
    n_samples_train = len(data["X_train"])
    n_samples_test = len(data["X_test"])
    
    # SHAP: needs enough background samples
    can_shap = n_samples_train >= 100
    
    # LIME: needs enough samples for perturbation
    can_lime = n_samples_train >= 50
    
    # Anchor: works with small datasets
    can_anchor = n_samples_test >= 10
    
    # DiCE: needs enough samples for counterfactual search
    can_dice = n_samples_train >= 100
    
    results = {
        "shap": can_shap,
        "lime": can_lime,
        "anchor": can_anchor,
        "dice": can_dice,
    }
    
    if not all(results.values()):
        LOG.warning("Some explainers may not work well with this dataset:")
        for explainer, ok in results.items():
            if not ok:
                LOG.warning(f"  - {explainer.upper()}: insufficient data")
    
    return results
