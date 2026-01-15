"""Robustness / stability proxy metrics."""

from __future__ import annotations

import numpy as np


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def explanation_stability(
    explain_fn,
    x_row: np.ndarray,
    noise_std: float = 0.01,
    n_repeats: int = 5,
) -> float:
    """
    Approximate robustness: cosine similarity between original and perturbed explanations.

    explain_fn: callable that returns a 1D importance vector for a given x_row.
    """
    base = explain_fn(x_row)
    sims = []
    for _ in range(n_repeats):
        noise = np.random.normal(scale=noise_std, size=x_row.shape)
        perturbed = x_row + noise
        exp = explain_fn(perturbed)
        sims.append(_cosine_similarity(base, exp))
    return float(np.mean(sims))
