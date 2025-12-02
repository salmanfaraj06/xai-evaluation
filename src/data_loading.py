"""Data loading utilities for the credit risk VXAI PoC."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_yaml_config(path: str | Path) -> dict:
    """Load a YAML config file into a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _normalize_binary_column(series: pd.Series) -> pd.Series:
    """Map common yes/no style values to 0/1, filling missing with mode."""
    s = series
    if pd.api.types.is_numeric_dtype(s):
        return (
            s.fillna(0)
            .round()
            .clip(0, 1)
            .astype(int)
        )

    mapping = {
        "yes": 1,
        "no": 0,
        "y": 1,
        "n": 0,
        "1": 1,
        "0": 0,
        "true": 1,
        "false": 0,
        "t": 1,
        "f": 0,
    }
    s_norm = s.astype(str).str.strip().str.lower()
    mapped = s_norm.map(mapping)

    if mapped.isna().any():
        if mapped.notna().any():
            mapped = mapped.fillna(int(mapped.mode()[0]))
        else:
            raise ValueError(
                f"Binary column contains unmappable values. Sample: {s.unique()[:10]}"
            )
    return mapped.astype(int)


def _clean_binary_columns(df: pd.DataFrame, binary_cols: Iterable[str]) -> pd.DataFrame:
    """Return a copy of df with binary columns normalized to 0/1."""
    df_clean = df.copy()
    for col in binary_cols:
        if col in df_clean.columns:
            df_clean[col] = _normalize_binary_column(df_clean[col])
    return df_clean


def load_credit_data(config: dict) -> Tuple[pd.DataFrame, pd.Series]:
    """Load, clean, and split features/target according to config."""
    data_cfg = config.get("data", {})
    path = Path(data_cfg.get("path", "loan_default.csv"))
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at {path}")

    df = pd.read_csv(path)

    # Drop unwanted columns
    drop_cols = data_cfg.get("drop_cols", [])
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Clean binary columns
    df = _clean_binary_columns(df, data_cfg.get("binary", []))

    target_col = data_cfg.get("target", "Default")
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' missing. Columns: {df.columns}")

    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    return X, y


def split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Train/test split with stratification on the target."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
