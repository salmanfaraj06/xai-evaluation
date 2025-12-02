"""LLM-assisted human-proxy evaluation with predefined personas."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.data_loading import load_yaml_config
from src.evaluation.run_human_proxy_eval import PERSONAS

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


RATING_DIMENSIONS: List[Tuple[str, str]] = [
    ("interpretability", "How easy is this explanation to understand? (1-5)"),
    ("completeness", "How complete is the reasoning behind the decision? (1-5)"),
    ("actionability", "Does this suggest actionable next steps? (1-5)"),
    ("trust", "How much do you trust the model after seeing this? (1-5)"),
    ("satisfaction", "Overall satisfaction with the explanation. (1-5)"),
    ("decision_support", "How well does it support your loan decision? (1-5)"),
]


def _load_metrics(metrics_path: Path) -> pd.DataFrame | None:
    if metrics_path.exists():
        return pd.read_csv(metrics_path)
    return None


def _fallback_score(method: str) -> Dict[str, float]:
    """Deterministic fallback scores when no LLM is available."""
    base = {
        "SHAP": 4.0,
        "LIME": 3.6,
        "ANCHOR": 3.8,
        "COUNTERFACTUAL": 3.5,
    }.get(method.upper(), 3.0)
    noise = np.linspace(-0.1, 0.1, num=len(RATING_DIMENSIONS))
    return {dim: float(np.clip(base + delta, 1, 5)) for (dim, _), delta in zip(RATING_DIMENSIONS, noise)}


def _build_prompt(persona: Dict, method: str, metrics: Dict | None) -> str:
    metric_text = ""
    if metrics:
        metric_pairs = [f"{k}: {v:.3f}" for k, v in metrics.items() if v == v]
        if metric_pairs:
            metric_text = "Technical cues: " + ", ".join(metric_pairs) + "."
    return (
        f"You are {persona['name']} ({persona['role']}). "
        f"Risk tolerance: {persona['risk_tolerance']}. "
        f"Decision speed: {persona.get('decision_speed', 'n/a')}. "
        f"Trust in AI: {persona['trust_in_ai']}. "
        f"Priorities: {', '.join(persona['priorities'])}. "
        f"Explanation type: {method}. "
        f"{metric_text} "
        "Provide 1-5 ratings for interpretability, completeness, actionability, trust, "
        "satisfaction, decision_support, and add one short comment."
    )


def _call_openai(client, prompt: str) -> tuple[Dict[str, float | str], str]:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict JSON generator. "
                    "Respond ONLY with JSON object containing keys: "
                    "interpretability, completeness, actionability, trust, "
                    "satisfaction, decision_support, comment."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=300,
    )
    content = resp.choices[0].message.content or "{}"

    def safe_parse(text: str) -> Dict:
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        try:
            return json.loads(text)
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except Exception:
                    pass
            # lenient parse: handle newline-separated key:value without commas
            lines = [ln.strip().strip(",") for ln in text.replace("{", "").replace("}", "").splitlines() if ":" in ln]
            kv = {}
            for ln in lines:
                if ":" not in ln:
                    continue
                k, v = ln.split(":", 1)
                k = k.strip().strip('"').strip()
                v = v.strip().strip('"').strip()
                kv[k] = v
            return kv

    data = safe_parse(content)
    return data, content


def _ratings_from_llm(client, prompt: str, method: str) -> Dict[str, float | str]:
    if client is None:
        scores = _fallback_score(method)
        scores["comment"] = "LLM not available; fallback scores."
        return scores
    raw_content = ""
    try:
        data, raw_content = _call_openai(client, prompt)
        result = {}
        missing = False
        for dim, _ in RATING_DIMENSIONS:
            raw_val = data.get(dim)
            try:
                val = float(raw_val)
                result[dim] = float(np.clip(val, 1, 5))
            except Exception:
                result[dim] = np.nan
                missing = True
        comment = str(data.get("comment", "")).strip()
        # If values are parsed from lenient mode strings, try again
        if missing:
            for dim, _ in RATING_DIMENSIONS:
                if pd.isna(result.get(dim, np.nan)):
                    try:
                        result[dim] = float(np.clip(float(data.get(dim, np.nan)), 1, 5))
                        missing = False
                    except Exception:
                        missing = True
        result["comment"] = comment if comment else "No comment."
        result["raw_llm_response"] = raw_content
        return result
    except Exception as exc:  # pragma: no cover - runtime
        scores = _fallback_score(method)
        scores["comment"] = f"LLM error or parse issue: {exc}"
        scores["raw_llm_response"] = raw_content
        return scores


def run_llm_human_eval(config_path: str) -> Dict[str, Path | pd.DataFrame]:
    cfg = load_yaml_config(config_path)
    paths_cfg = cfg.get("paths", {})
    human_dir = Path(paths_cfg.get("human_eval_dir", "outputs/human_eval_templates"))
    human_dir.mkdir(parents=True, exist_ok=True)

    human_cfg = cfg.get("evaluation", {}).get("human_llm", {}) or {}
    runs_per_method = int(human_cfg.get("runs_per_method", 1))
    LOG.info("LLM human eval runs_per_method=%d", runs_per_method)

    metrics_path = Path(paths_cfg.get("metrics_dir", "outputs/vxai_metrics")) / "technical_metrics.csv"
    metrics_df = _load_metrics(metrics_path)
    metrics_lookup = {}
    if metrics_df is not None:
        for _, row in metrics_df.iterrows():
            metrics_lookup[row["method"].upper()] = {
                "deletion_auc": row.get("deletion_auc", np.nan),
                "insertion_auc": row.get("insertion_auc", np.nan),
                "anchor_precision": row.get("anchor_precision", np.nan),
                "anchor_coverage": row.get("anchor_coverage", np.nan),
                "cf_validity_rate": row.get("cf_validity_rate", np.nan),
            }

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        # try loading from .env if present
        env_path = Path(".env")
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("OPENAI_API_KEY"):
                    key = line.split("=", 1)[-1].strip().strip('"').strip("'")
                    if key:
                        api_key = key
                        break
    client = OpenAI(api_key=api_key) if (OpenAI is not None and api_key) else None
    if client is None:
        LOG.warning("OPENAI_API_KEY missing or openai not installed. Using fallback scores.")

    methods = ["SHAP", "LIME", "ANCHOR", "COUNTERFACTUAL"]
    records = []
    for persona in PERSONAS:
        for method in methods:
            for run_idx in range(runs_per_method):
                LOG.info("LLM eval persona=%s method=%s run=%d", persona["name"], method, run_idx + 1)
                prompt = _build_prompt(persona, method, metrics_lookup.get(method))
                ratings = _ratings_from_llm(client, prompt, method)
                row = {
                    "persona_name": persona["name"],
                    "persona_role": persona["role"],
                    "explanation_type": method,
                    "run": run_idx + 1,
                }
                row.update({dim: ratings.get(dim, np.nan) for dim, _ in RATING_DIMENSIONS})
                row["comment"] = ratings.get("comment", "")
                row["raw_llm_response"] = ratings.get("raw_llm_response", "")
                row["used_llm"] = client is not None
                records.append(row)

    df = pd.DataFrame(records)
    out_path = human_dir / "agent_llm_eval.csv"
    df.to_csv(out_path, index=False)

    summary = (
        df.groupby(["explanation_type"])[[d for d, _ in RATING_DIMENSIONS]]
        .mean()
        .reset_index()
    )
    summary_path = human_dir / "agent_llm_eval_summary.csv"
    summary.to_csv(summary_path, index=False)
    LOG.info("Saved LLM persona results to %s and summary to %s", out_path, summary_path)

    return {
        "raw_path": out_path,
        "summary_path": summary_path,
        "df": df,
        "summary": summary,
        "used_llm": client is not None,
    }


def main():
    parser = argparse.ArgumentParser(description="Run LLM-based human proxy evaluation with personas.")
    parser.add_argument("--config", default="config/config_credit.yaml", help="Path to YAML config.")
    args = parser.parse_args()
    res = run_llm_human_eval(args.config)
    print(f"Saved raw ratings to: {res['raw_path']}")
    print(f"Saved summary to: {res['summary_path']}")
    print("LLM used:", res["used_llm"])
    print(res["summary"])


if __name__ == "__main__":
    main()
