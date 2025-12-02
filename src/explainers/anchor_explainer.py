"""Anchor explainer wrapper."""

from __future__ import annotations

import numpy as np


class AnchorExplainer:
    def __init__(
        self,
        X_train_raw,
        feature_names,
        predict_fn,
        class_names=None,
        categorical_names=None,
    ):
        try:
            from anchor import anchor_tabular  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError("Install anchor-exp to use AnchorExplainer") from exc

        self.predict_fn = predict_fn
        self.feature_names = feature_names
        self.explainer = anchor_tabular.AnchorTabularExplainer(
            class_names=class_names or ["No Default", "Default"],
            feature_names=feature_names,
            train_data=np.asarray(X_train_raw),
            categorical_names=categorical_names or {},
        )

    def explain_instance(self, x_row: np.ndarray, threshold: float = 0.9):
        """Generate an anchor explanation object."""
        return self.explainer.explain_instance(
            x_row,
            self.predict_fn,
            threshold=threshold,
        )
