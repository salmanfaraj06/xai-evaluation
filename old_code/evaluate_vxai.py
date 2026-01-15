"""CLI entry to run VXAI technical evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.run_technical_eval import run_technical_eval  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run VXAI technical evaluation.")
    parser.add_argument("--config", default="config/config_credit.yaml", help="Path to YAML config.")
    parser.add_argument("--model-path", default=None, help="Path to trained model artifact.")
    args = parser.parse_args()

    results = run_technical_eval(args.config, model_path=args.model_path)
    print(f"Saved metrics to: {results['metrics_path']}")
    if "metrics" in results:
        print(results["metrics"])


if __name__ == "__main__":
    main()
