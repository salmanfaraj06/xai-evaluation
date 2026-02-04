"""Fidelity metrics aligned with VXAI (insertion/deletion AUC)."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def _predict(model, X: np.ndarray) -> np.ndarray:
    """Predict class-1 probabilities."""
    proba = model.predict_proba(X)
    return proba[:, 1] if proba.ndim > 1 else proba


def insertion_deletion_auc(
    model,
    X: np.ndarray,
    importances: np.ndarray,
    baseline_vec: np.ndarray,
    steps: int = 50,
    return_per_instance: bool = False,
) -> Dict[str, float | np.ndarray]:
    """
    Compute insertion and deletion AUCs following Covert & Lundberg (2021).

    Parameters
    ----------
    model : fitted classifier with predict_proba
    X : np.ndarray of shape (n_samples, n_features)
    importances : np.ndarray of same shape as X
    baseline_vec : 1D np.ndarray baseline values for masking
    steps : number of feature removal/addition steps
    return_per_instance : whether to include per-instance AUC arrays
    """
    n_samples, n_features = importances.shape

    deletion_auc_list = []
    insertion_auc_list = []

    grid = np.linspace(0, 1, steps + 1)

    step_size = max(1, n_features // steps)

    for i in range(n_samples):
        row = np.array(X[i], dtype=float)
        imp = importances[i]

        order = np.argsort(-np.abs(imp))

        del_scores = []
        ins_scores = []

        # initial points
        del_scores.append(float(_predict(model, row.reshape(1, -1))[0]))
        ins_scores.append(float(_predict(model, baseline_vec.reshape(1, -1))[0]))

        for k in range(step_size, n_features + step_size, step_size):
            k = min(k, n_features)
            idx_subset = order[:k]

            x_del = row.copy()
            x_del[idx_subset] = baseline_vec[idx_subset]
            del_scores.append(float(_predict(model, x_del.reshape(1, -1))[0]))

            x_ins = baseline_vec.copy()
            x_ins[idx_subset] = row[idx_subset]
            ins_scores.append(float(_predict(model, x_ins.reshape(1, -1))[0]))

            if k == n_features:
                break

        # ensure same length as grid
        del_scores = np.interp(grid, np.linspace(0, 1, len(del_scores)), del_scores)
        ins_scores = np.interp(grid, np.linspace(0, 1, len(ins_scores)), ins_scores)

        deletion_auc_list.append(float(np.trapz(del_scores, grid)))
        insertion_auc_list.append(float(np.trapz(ins_scores, grid)))

    results: Dict[str, float | np.ndarray] = {
        "deletion_auc": float(np.nanmean(deletion_auc_list)),
        "insertion_auc": float(np.nanmean(insertion_auc_list)),
    }
    if return_per_instance:
        results["deletion_auc_per_instance"] = np.array(deletion_auc_list)
        results["insertion_auc_per_instance"] = np.array(insertion_auc_list)
    return results
