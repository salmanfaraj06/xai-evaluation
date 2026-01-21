"""Thin SHAP wrapper."""

from __future__ import annotations

from typing import Optional

import numpy as np


class ShapExplainer:
    def __init__(
        self,
        model,
        background: np.ndarray,
        feature_names,
        class_index: int = 1,
    ):
        try:
            import shap  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError("Install shap to use ShapExplainer") from exc

        self.shap = shap
        self.class_index = class_index
        self.feature_names = feature_names
<<<<<<< Updated upstream
        try:
            self.explainer = shap.Explainer(model, background)
        except Exception:
            # Fallback for Pipelines or unsupported objects: use predict_proba
            if hasattr(model, "predict_proba"):
                self.explainer = shap.Explainer(model.predict_proba, background)
            else:
                raise
=======
        self.explainer = shap.Explainer(model, background)
>>>>>>> Stashed changes

    def _reduce(self, shap_values) -> np.ndarray:
        """Extract class-specific values as (n_samples, n_features)."""
        values = shap_values.values if hasattr(shap_values, "values") else shap_values
        if values.ndim == 3:
            return values[:, self.class_index, :]
        return values

    def explain_instance(self, x_row: np.ndarray) -> np.ndarray:
        """Return SHAP values for a single instance."""
        sv = self.explainer(x_row.reshape(1, -1))
        return self._reduce(sv)[0]

    def explain_dataset(self, X: np.ndarray) -> np.ndarray:
        """Return SHAP values for a batch of instances."""
        sv = self.explainer(X)
        return self._reduce(sv)
