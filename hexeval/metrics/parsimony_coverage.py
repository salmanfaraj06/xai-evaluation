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


def counterfactual_sparsity(
    x_row: np.ndarray,
    cf_rows: np.ndarray,
    tolerance: float = 1e-6,
) -> float:
    """Average number of features changed between an instance and its counterfactuals."""
    if cf_rows.size == 0:
        return float("nan")
    cf_arr = np.atleast_2d(cf_rows)
    diffs = np.abs(cf_arr - x_row)
    changed = diffs > tolerance
    return float(changed.sum(axis=1).mean())
