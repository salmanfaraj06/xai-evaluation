"""CLI to emit human evaluation template."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.run_human_proxy_eval import generate_human_template  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Generate human evaluation template.")
    parser.add_argument("--config", default="config/config_credit.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    out_path = generate_human_template(args.config)
    print(f"Human evaluation template saved to: {out_path}")


if __name__ == "__main__":
    main()
