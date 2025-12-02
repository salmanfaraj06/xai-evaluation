"""Train credit risk model and persist artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data_loading import load_credit_data, load_yaml_config, split_data  # noqa: E402
from src.models import evaluate_model_performance, save_artifact, train_credit_model  # noqa: E402
from src.preprocessing import build_preprocessor, transform_features  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Train credit risk model.")
    parser.add_argument("--config", default="config/config_credit.yaml", help="Path to YAML config.")
    parser.add_argument("--model-out", default=None, help="Optional override for model artifact path.")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    data_cfg = config.get("data", {})
    model_dir = Path(config.get("paths", {}).get("model_dir", "outputs/models"))
    model_dir.mkdir(parents=True, exist_ok=True)
    model_out = Path(args.model_out) if args.model_out else model_dir / "credit_model.joblib"

    X, y = load_credit_data(config)
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=data_cfg.get("test_size", 0.2), random_state=data_cfg.get("random_state", 42)
    )

    preprocessor, feature_names = build_preprocessor(config, X_train)
    X_train_proc = np.asarray(transform_features(preprocessor, X_train))
    X_test_proc = np.asarray(transform_features(preprocessor, X_test))

    model = train_credit_model(config, X_train_proc, y_train.values)
    metrics = evaluate_model_performance(model, X_test_proc, y_test.values)

    artifact = save_artifact(
        model,
        preprocessor,
        feature_names,
        path=model_out,
        default_threshold=0.5,
    )

    metrics_path = model_dir / "model_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    print(f"Model saved to {model_out}")
    print(f"Metrics: {metrics}")
    print(f"Metrics saved to {metrics_path}")
    return artifact


if __name__ == "__main__":
    main()
