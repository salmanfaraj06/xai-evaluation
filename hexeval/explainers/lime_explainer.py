"""LIME wrapper."""

from __future__ import annotations

import numpy as np


class LimeExplainer:
    def __init__(
        self,
        training_data: np.ndarray,
        feature_names,
        class_names,
        predict_fn,
        random_state: int = 42,
    ):
        try:
            from lime import lime_tabular  
        except ImportError as exc:  
            raise ImportError("Install lime to use LimeExplainer") from exc

        self.feature_names = feature_names
        self.predict_fn = predict_fn
        self.explainer = lime_tabular.LimeTabularExplainer(
            training_data=training_data,
            feature_names=feature_names,
            class_names=class_names,
            mode="classification",
            discretize_continuous=False, 
            random_state=random_state,
        )

    def explain_instance(
        self,
        x_row: np.ndarray,
        num_features: int = 10,
        num_samples: int = 2000,
    ):
        """Return LIME explanation object for one instance."""
        return self.explainer.explain_instance(
            data_row=x_row,
            predict_fn=self.predict_fn,
            num_features=num_features,
            num_samples=num_samples,
        )

    def as_importance_vector(
        self,
        x_row: np.ndarray,
        num_features: int,
        num_samples: int,
    ) -> np.ndarray:
        """Return weights aligned to feature_names length."""
        exp = self.explain_instance(x_row, num_features=num_features, num_samples=num_samples)
        weights = np.zeros(len(self.feature_names), dtype=float)
        for idx, w in exp.as_map()[1]:
            if idx < len(weights):
                weights[idx] = w
        return weights
