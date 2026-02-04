"""Model wrapper for HEXEval.

Provides a consistent interface for models, handling preprocessing and metadata.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)


class BaseModelWrapper(ABC):
    """Abstract base class for model wrappers."""

    def __init__(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        preprocessor: Any = None,
    ):
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.preprocessor = preprocessor

    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions."""
        pass

    @abstractmethod
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class probabilities."""
        pass

    def _preprocess(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Apply preprocessing if available."""
        if self.preprocessor:
            return self.preprocessor.transform(X)
        
        # If no preprocessor, convert DataFrame to numpy if needed
        # but respect feature names if possible (some models handle DataFrames directly)
        if isinstance(X, pd.DataFrame):
            # If the model handles dataframes natively, pass it through
            # Otherwise, convert to numpy
            if hasattr(self.model, "predict_proba"):
                # Most sklearn models handle dataframes but strip names internally
                # XGBoost/LightGBM might use names
                pass
            return X.values if not hasattr(self.model, "feature_names_in_") else X
            
        return X


class ModelWrapper(BaseModelWrapper):
    """
    Standard wrapper for sklearn-compatible models.
    
    Handles:
    - Preprocessing (if provided)
    - Input conversion (pandas -> numpy)
    - Metadata access
    """

    def __init__(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        preprocessor: Any = None,
        threshold: float = 0.5,
    ):
        super().__init__(model, feature_names, class_names, preprocessor)
        self.threshold = threshold
        
        # Try to infer feature names if not provided
        if not self.feature_names:
            if hasattr(model, "feature_names_in_"):
                self.feature_names = list(model.feature_names_in_)
            elif hasattr(preprocessor, "get_feature_names_out"):
                try:
                    self.feature_names = list(preprocessor.get_feature_names_out())
                except Exception:
                    pass

        # Try to infer class names if not provided
        if not self.class_names and hasattr(model, "classes_"):
            self.class_names = list(map(str, model.classes_))

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions on input data.
        
        Parameters
        ----------
        X : array-like or DataFrame
            Input features
            
        Returns
        -------
        numpy.ndarray
            Predictions
        """
        X_proc = self._preprocess(X)
        return self.model.predict(X_proc)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class probabilities for input data.
        
        Parameters
        ----------
        X : array-like or DataFrame
            Input features
            
        Returns
        -------
        numpy.ndarray
            Class probabilities (n_samples, n_classes)
        """
        X_proc = self._preprocess(X)
        
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_proc)
        else:
            raise NotImplementedError("Model does not support predict_proba")

    def _preprocess(self, X: Union[np.ndarray, pd.DataFrame]) -> Any:
        """Internal preprocessing logic."""
       
        if self.preprocessor:
            
            # We assume user provides compatible input relative to preprocessor
            X = self.preprocessor.transform(X)
            return X # Usually returns numpy array
            
        # 2. If no preprocessor, handle format
        if isinstance(X, pd.DataFrame):
            
            if not hasattr(self.model, "feature_names_in_"):
                return X.values
                
        return X

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores if available.
        
        Returns
        -------
        dict
            Dictionary mapping feature names to importance scores
        """
        if not self.feature_names:
            return {}
            
        importance = None
        
        # Tree-based
        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
            
        # Linear models
        elif hasattr(self.model, "coef_"):
         
            if self.model.coef_.ndim > 1:
                importance = np.abs(self.model.coef_).mean(axis=0)
            else:
                importance = np.abs(self.model.coef_)
                
        if importance is not None:
            if len(importance) == len(self.feature_names):
                return dict(zip(self.feature_names, importance))
                
        return {}

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get metadata about the model.
        
        Returns
        -------
        dict
            Model information
        """
        return {
            "model_type": type(self.model).__name__,
            "n_features": len(self.feature_names) if self.feature_names else "Unknown",
            "has_preprocessor": self.preprocessor is not None,
            "classes": self.class_names,
            "threshold": self.threshold
        }
