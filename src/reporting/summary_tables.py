"""Reporting helpers for VXAI metrics."""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def save_summary_table(metrics_df: pd.DataFrame, out_dir: str | Path):
    """Save metrics and simple summary to CSV."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "technical_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    return metrics_path
