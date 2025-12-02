"""Optional plotting helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_bar(metrics_df: pd.DataFrame, metric: str, out_dir: str | Path):
    """Save a simple bar plot comparing methods for a metric."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    sns.barplot(data=metrics_df, x="method", y=metric, palette="viridis")
    plt.title(metric)
    plt.tight_layout()
    path = out_dir / f"{metric}.png"
    plt.savefig(path)
    plt.close()
    return path
