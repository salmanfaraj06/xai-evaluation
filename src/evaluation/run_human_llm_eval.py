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
from src.explainers.shap_explainer import ShapExplainer
from src.explainers.lime_explainer import LimeExplainer
from src.explainers.anchor_explainer import AnchorExplainer
from src.explainers.dice_counterfactuals import DiceExplainer
from src.preprocessing import align_columns, one_hot_encode, transform_features
from src.models import load_artifact
from src.data_loading import load_credit_data, split_data

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


def _prepare_data_for_explanations(config: dict, artifact: dict):
    X, y = load_credit_data(config)
    data_cfg = config.get("data", {})
    test_size = data_cfg.get("test_size", 0.2)
    random_state = data_cfg.get("random_state", 42)
    X_train_raw, X_test_raw, y_train, y_test = split_data(
        X, y, test_size=test_size, random_state=random_state
    )

    preprocessor = artifact.get("preprocessor")
    feature_names = artifact.get("feature_names")
    categorical_cols = config.get("data", {}).get("categorical", [])

    if preprocessor is not None:
        X_train_proc = np.asarray(transform_features(preprocessor, X_train_raw)).astype(np.float64)
        X_test_proc = np.asarray(transform_features(preprocessor, X_test_raw)).astype(np.float64)
        if feature_names is None:
            feature_names = preprocessor.get_feature_names_out().tolist()
    else:
        if feature_names is None:
            raise ValueError("Artifact missing feature_names and preprocessor for LLM explanations.")

        def encode(df: pd.DataFrame) -> pd.DataFrame:
            encoded = one_hot_encode(df, categorical_cols, drop_first=True)
            return align_columns(encoded, feature_names)

        X_train_enc = encode(X_train_raw)
        X_test_enc = encode(X_test_raw)
        X_train_proc = X_train_enc.to_numpy().astype(np.float64)
        X_test_proc = X_test_enc.to_numpy().astype(np.float64)

    return {
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "X_train_proc": X_train_proc,
        "X_test_proc": X_test_proc,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "preprocessor": preprocessor,
    }


def _build_example_explanations(config: dict, artifact: dict):
    """Create representative explanation texts for each method (one test instance)."""
    data = _prepare_data_for_explanations(config, artifact)
    feature_names = data["feature_names"]
    top_k = int(config.get("evaluation", {}).get("human_llm", {}).get("top_k_explanation", 5))
    outcome_name = config.get("data", {}).get("target", "Default")
    n_instances = int(config.get("evaluation", {}).get("human_llm", {}).get("n_instances", 5))
    rng = np.random.default_rng(config.get("data", {}).get("random_state", 42))
    subset = rng.choice(len(data["X_test_proc"]), size=min(n_instances, len(data["X_test_proc"])), replace=False)

    explanations: Dict[int, Dict[str, str]] = {}
    try:
        bg_size = min(100, len(data["X_train_proc"]))
        shap_exp = ShapExplainer(
            artifact["model"],
            background=data["X_train_proc"][:bg_size],
            feature_names=feature_names,
        )
    except Exception as exc:
        shap_exp = None
        shap_error = exc

    try:
        lime = LimeExplainer(
            training_data=data["X_train_proc"],
            feature_names=feature_names,
            class_names=["No Default", "Default"],
            predict_fn=artifact["model"].predict_proba,
            random_state=config.get("data", {}).get("random_state", 42),
        )
    except Exception as exc:
        lime = None
        lime_error = exc

    try:
        anchor = AnchorExplainer(
            data["X_train_proc"],
            feature_names=feature_names,
            predict_fn=lambda arr: (artifact["model"].predict_proba(arr)[:, 1] >= artifact.get("default_threshold", 0.5)).astype(int),
            class_names=["No Default", "Default"],
        )
    except Exception as exc:
        anchor = None
        anchor_error = exc

    try:
        dice = DiceExplainer(
            model=artifact["model"],
            X_train_processed=data["X_train_proc"],
            y_train=data["y_train"],
            feature_names=feature_names,
            outcome_name=outcome_name,
            method="random",
        )
    except Exception as exc:
        dice = None
        dice_error = exc

    for idx in subset:
        explanations[idx] = {}
        instance_proc = data["X_test_proc"][idx]

        # SHAP
        if shap_exp is not None:
            try:
                shap_vals = shap_exp.explain_instance(instance_proc)
                top_idx = np.argsort(-np.abs(shap_vals))[:top_k]
                parts = [f"{feature_names[i]}: {shap_vals[i]:.3f}" for i in top_idx]
                explanations[idx]["SHAP"] = "Top SHAP attributions: " + "; ".join(parts)
            except Exception as exc:
                explanations[idx]["SHAP"] = f"SHAP explanation unavailable ({exc})"
        else:
            explanations[idx]["SHAP"] = f"SHAP explainer unavailable ({shap_error})"

        # LIME
        if lime is not None:
            try:
                exp = lime.explain_instance(instance_proc, num_features=top_k, num_samples=2000)
                parts = [f"{fn}: {w:.3f}" for fn, w in exp.as_list()]
                explanations[idx]["LIME"] = "LIME weights: " + "; ".join(parts)
            except Exception as exc:
                explanations[idx]["LIME"] = f"LIME explanation unavailable ({exc})"
        else:
            explanations[idx]["LIME"] = f"LIME explainer unavailable ({lime_error})"

        # Anchor
        if anchor is not None:
            try:
                anchor_exp = anchor.explain_instance(instance_proc, threshold=0.9)
                rule = " AND ".join(anchor_exp.names())
                explanations[idx]["ANCHOR"] = (
                    f"Rule: IF {rule}; precision={anchor_exp.precision():.3f}, coverage={anchor_exp.coverage():.3f}"
                )
            except Exception as exc:
                explanations[idx]["ANCHOR"] = f"Anchor explanation unavailable ({exc})"
        else:
            explanations[idx]["ANCHOR"] = f"Anchor explainer unavailable ({anchor_error})"

        # Counterfactual
        if dice is not None:
            try:
                cf = dice.generate_counterfactuals(instance_proc, total_cfs=1)
                cf_df = cf.cf_examples_list[0].final_cfs_df[feature_names]
                orig = instance_proc
                changes = []
                for col_i, col in enumerate(feature_names):
                    delta = cf_df.iloc[0][col] - orig[col_i]
                    if abs(delta) > 1e-6:
                        changes.append(f"{col}: change by {delta:.3f}")
                explanations[idx]["COUNTERFACTUAL"] = "Counterfactual suggestion: " + (", ".join(changes) if changes else "no changes")
            except Exception as exc:
                explanations[idx]["COUNTERFACTUAL"] = f"Counterfactual explanation unavailable ({exc})"
        else:
            explanations[idx]["COUNTERFACTUAL"] = f"Counterfactual explainer unavailable ({dice_error})"

    return explanations, subset, data


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


def build_system_prompt(persona: Dict) -> str:
    """
    Enhanced system prompt that uses richer persona attributes
    (loss_aversion, risk_tolerance, heuristics, etc.),
    but is robust if some fields are missing.
    """
    loss_aversion = persona.get("loss_aversion", 1.5)
    experience_years = persona.get("experience_years", "many")
    risk_tolerance = persona.get("risk_tolerance", "Moderate")
    decision_speed = persona.get("decision_speed", "Moderate")
    trust_in_ai = persona.get("trust_in_ai", "Medium")
    priorities = ", ".join(persona.get("priorities", [])) or "N/A"
    heuristics = persona.get("heuristics", [])
    explanation_preferences = persona.get("explanation_preferences", persona.get("description", ""))
    mental_model = persona.get("mental_model", "")
    behavior = persona.get("behavioral_signature", {})

    identity = f"""You are {persona['name']}, a {persona['role']} with {experience_years} years of experience in credit and lending.

BEHAVIORAL PROFILE:
- Loss Aversion (λ): {loss_aversion}  (you weigh potential losses about {loss_aversion}× more than equivalent gains)
- Risk Tolerance: {risk_tolerance}
- Decision Speed: {decision_speed}
- Trust in AI: {trust_in_ai}

YOUR MENTAL MODEL:
{mental_model}

YOUR DECISION HEURISTICS:
"""
    if heuristics:
        identity += "\n".join(f"- {h}" for h in heuristics) + "\n"
    else:
        identity += "- You rely on your experience and judgment.\n"

    identity += f"""
WHAT YOU VALUE IN EXPLANATIONS:
{explanation_preferences}
"""

    rating_guidelines = f"""
RATING GUIDELINES (1–5 for each dimension):

1) INTERPRETABILITY – How easy is this explanation to understand?
   - 1: Completely incomprehensible
   - 3: Understandable with effort
   - 5: Crystal clear and immediately obvious
   YOUR STANDARD: {"Simple, rule-like formats are best; heavy jargon reduces this score."
                    if behavior.get("favors_simplicity")
                    else "Detailed, precise explanations are fine; oversimplification reduces this score."}

2) COMPLETENESS – Does it cover all important factors for the decision?
   - 1: Many important factors are missing
   - 3: Covers main factors but misses some details
   - 5: Very comprehensive
   YOUR STANDARD: {"Only key risk drivers are needed; too much detail is noise."
                    if behavior.get("favors_simplicity")
                    else "All major contributing features should be visible, not just one or two."}

3) ACTIONABILITY – Does it tell you or the borrower what can be done next?
   - 1: No actionable insight
   - 3: Somewhat actionable
   - 5: Very actionable, with clear next steps
   YOUR STANDARD: {"Must highlight realistic borrower actions (e.g., reduce debt, increase savings)."
                    if behavior.get("prefers_actionability")
                    else "Must support your own decision process (approve/reject/monitor), not give advice to the borrower."}

4) TRUST – How much do you trust this explanation?
   - 1: Do not trust at all
   - 3: Some reservations
   - 5: Fully trust it
   YOUR BASELINE: Your default trust in AI is {trust_in_ai}.
   {"You are strongly loss-averse and prefer conservative, risk-avoiding signals."
    if behavior.get("prefers_conservative_errors")
    else "You trust explanations that show technical rigor and consistency with model behavior."}

5) SATISFACTION – Overall satisfaction with this explanation.
   - 1: Very dissatisfied
   - 3: Neutral
   - 5: Very satisfied
   YOUR STANDARD: Reflects how well this explanation fits your priorities: {priorities}.

6) DECISION SUPPORT – How much does this help you make a better decision?
   - 1: Not helpful at all
   - 3: Somewhat helpful
   - 5: Extremely helpful
   YOUR STANDARD: {"Must be defensible for risk/compliance and usable in an audit trail."
                    if behavior.get("values_documentation")
                    else "Must help you make fast, scalable, strategy-aligned decisions."}
"""

    examples = f"""
CALIBRATION EXAMPLES (rough intuition of how YOU might react):

Example A – Simple IF-THEN rule:
  "IF CreditScore < 650 AND DebtToIncome > 0.4 THEN high risk of default."

As {persona['name']}, you would likely see this as:
- High interpretability if you value simple rules.
- Moderate completeness (only 2 features).
- Higher trust if you are conservative and loss-averse.

Example B – Detailed SHAP feature list:
  "Top 10 features with SHAP contributions: CreditScore (-0.15), Income (-0.08), Age (-0.06), ..."

As {persona['name']}, you would likely see this as:
- Lower interpretability if you dislike numbers and complex details.
- Higher completeness if you value fidelity to the model.
- Trust depends on whether you generally trust complex AI explanations.
"""

    task_header = f"""
YOUR TASK:
You will be shown a loan default prediction and ONE explanation (SHAP, LIME, ANCHOR, or COUNTERFACTUAL).

For each explanation:
1. Read it **from YOUR persona's perspective**.
2. Rate it on the 6 dimensions from 1–5.
3. Write a short 2–4 sentence comment, in character, explaining WHY you gave those scores.

IMPORTANT:
- Stay in character as {persona['name']} at all times.
- Different personas SHOULD give different ratings for the same explanation.
- Let your loss aversion (λ = {loss_aversion}) and risk tolerance influence especially your TRUST and DECISION_SUPPORT scores.
- Rate the quality of the *explanation* for YOUR job, not whether the prediction is correct.
"""

    json_instructions = """
OUTPUT FORMAT (STRICT JSON, NO EXTRA TEXT):

{
  "interpretability": <1-5>,
  "completeness": <1-5>,
  "actionability": <1-5>,
  "trust": <1-5>,
  "satisfaction": <1-5>,
  "decision_support": <1-5>,
  "comment": "<2-4 sentences from your persona's perspective>"
}
"""

    return identity + rating_guidelines + examples + task_header + json_instructions


def format_prediction_info_text(pred_info: Dict) -> str:
    cls_map = {0: "No Default", 1: "Default"}
    # Handle cases where actual_class might be missing or different format
    actual_str = f"{pred_info['actual_class']} ({cls_map.get(pred_info['actual_class'], '?')})"
    
    return (
        f"Instance ID: {pred_info.get('instance_index', 'N/A')}\n"
        f"Model predicted probability of DEFAULT = {pred_info['predicted_proba_default']:.3f}\n"
        f"Predicted class: {pred_info['predicted_class']} ({cls_map.get(pred_info['predicted_class'], '?')})\n"
        f"Actual class: {actual_str}\n"
    )


def build_eval_prompt(pred_info: Dict, explanation_text: str, explanation_type: str, metrics: Dict | None = None) -> str:
    """
    Build the user prompt asking the persona to rate a given explanation.
    """
    pred_text = format_prediction_info_text(pred_info)

    dims_text = "\n".join([
        "- interpretability: 1 (very hard to understand) to 5 (very easy to understand)",
        "- completeness: 1 (very incomplete) to 5 (very complete)",
        "- actionability: 1 (not actionable) to 5 (very actionable)",
        "- trust: 1 (do not trust) to 5 (fully trust)",
        "- satisfaction: 1 (very dissatisfied) to 5 (very satisfied)",
        "- decision_support: 1 (not helpful) to 5 (very helpful)",
    ])
    
    metric_text = ""
    if metrics:
        metric_pairs = [f"{k}: {v:.3f}" for k, v in metrics.items() if v == v]
        if metric_pairs:
            metric_text = "\n== Technical Metrics (for context) ==\n" + ", ".join(metric_pairs) + "\n"

    prompt = f"""
You are evaluating an explanation for a loan default prediction model.

== Prediction Information ==
{pred_text}

== Explanation Type ==
{explanation_type}

== Explanation Content ==
{explanation_text}
{metric_text}
Please rate this explanation on the following 6 dimensions, each from 1 (very low) to 5 (very high):

{dims_text}

Also provide a short natural language comment (2–4 sentences) about the explanation quality.

IMPORTANT:
- Base your ratings only on the explanation above and your persona role.
- Return your answer as STRICT JSON with exactly these keys:
  "interpretability", "completeness", "actionability",
  "trust", "satisfaction", "decision_support", "comment"

Example JSON format (do NOT add any extra fields):

{{
  "interpretability": 4,
  "completeness": 3,
  "actionability": 3,
  "trust": 3,
  "satisfaction": 4,
  "decision_support": 4,
  "comment": "Short explanation of why you gave these scores."
}}
"""
    return prompt


def _call_openai(client, system_prompt: str, user_prompt: str) -> tuple[Dict[str, float | str], str]:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": user_prompt},
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


def _ratings_from_llm(client, system_prompt: str, user_prompt: str, method: str) -> Dict[str, float | str]:
    if client is None:
        scores = _fallback_score(method)
        scores["comment"] = "LLM not available; fallback scores."
        return scores
    raw_content = ""
    try:
        data, raw_content = _call_openai(client, system_prompt, user_prompt)
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

    # Build example explanations per method
    example_explanations = {}
    subset_indices = []
    data_dict = {}
    try:
        artifact = load_artifact(paths_cfg.get("pretrained_model", "xgboost_loan_default_research.pkl"))
        example_explanations, subset_indices, data_dict = _build_example_explanations(cfg, artifact)
    except Exception as exc:
        LOG.warning("Failed to build example explanations: %s", exc)

    # Pre-calculate prediction info
    pred_infos = {}
    if data_dict and "X_test_proc" in data_dict:
        model = artifact["model"]
        default_threshold = float(artifact.get("default_threshold", 0.5))
        for idx in subset_indices:
            try:
                x_row = data_dict["X_test_proc"][idx].reshape(1, -1)
                proba = float(model.predict_proba(x_row)[0, 1])
                pred_class = int(proba >= default_threshold)
                actual_class = int(data_dict["y_test"].iloc[idx])
                pred_infos[idx] = {
                    "instance_index": int(idx),
                    "predicted_proba_default": proba,
                    "predicted_class": pred_class,
                    "actual_class": actual_class,
                }
            except Exception as e:
                LOG.warning(f"Could not calculate prediction info for index {idx}: {e}")
                pred_infos[idx] = {}

    methods = ["SHAP", "LIME", "ANCHOR", "COUNTERFACTUAL"]
    records = []
    for persona in PERSONAS:
        system_prompt = build_system_prompt(persona)
        for idx in subset_indices:
            pred_info = pred_infos.get(idx, {})
            for method in methods:
                for run_idx in range(runs_per_method):
                    LOG.info("LLM eval persona=%s method=%s instance=%d run=%d", persona["name"], method, idx, run_idx + 1)
                    expl_text = example_explanations.get(idx, {}).get(method, "Explanation unavailable.")
                    
                    user_prompt = build_eval_prompt(pred_info, expl_text, method, metrics_lookup.get(method))
                    ratings = _ratings_from_llm(client, system_prompt, user_prompt, method)
                    
                    row = {
                        "persona_name": persona["name"],
                        "persona_role": persona["role"],
                        "explanation_type": method,
                        "instance_index": int(idx),
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
