"""Parsimony and coverage metrics."""

from __future__ import annotations

import numpy as np


def sparsity_from_importances(importances: np.ndarray, threshold: float = 0.0) -> float:
    """Average count of non-zero (or above threshold) features per instance."""
    mask = np.abs(importances) > threshold
    return float(mask.sum(axis=1).mean())


def anchor_parsimony_and_coverage(anchor_exp) -> dict:
    """Extract rule length, precision, and coverage from an anchor explanation."""
    return {
        "n_conditions": len(anchor_exp.names()),
        "precision": float(anchor_exp.precision()),
        "coverage": float(anchor_exp.coverage()),
    }
