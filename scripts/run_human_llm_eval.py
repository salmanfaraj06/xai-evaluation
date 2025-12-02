"""CLI to run LLM-based human proxy evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.run_human_llm_eval import run_llm_human_eval  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run LLM-based human proxy evaluation.")
    parser.add_argument("--config", default="config/config_credit.yaml", help="Path to YAML config.")
    args = parser.parse_args()
    res = run_llm_human_eval(args.config)
    print(f"Raw ratings: {res['raw_path']}")
    print(f"Summary: {res['summary_path']}")
    print("LLM used:", res["used_llm"])


if __name__ == "__main__":
    main()
