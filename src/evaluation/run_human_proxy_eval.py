"""Generate human-centred evaluation templates (Layer 3)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from src.data_loading import load_yaml_config


PERSONAS = [
    {
        "name": "Margaret Chen",
        "role": "Conservative Loan Officer",
        "experience_years": 18,
        "loss_aversion": 2.5,
        "risk_tolerance": "Very Low",
        "decision_speed": "Slow (methodical)",
        "trust_in_ai": "Low (prefers human oversight)",
        "priorities": [
            "Actionability",
            "Trust",
            "Clear rules / thresholds",
        ],
        "mental_model": (
            "Credit score and payment history are paramount. Any hint of instability "
            "(short employment, high debt-to-income) is a red flag. Defaults are catastrophic."
        ),
        "heuristics": [
            "If CreditScore < 650, lean heavily toward reject.",
            "Employment < 12 months is concerning.",
            "Multiple recent credit inquiries suggest desperation.",
        ],
        "explanation_preferences": (
            "Prefers simple, rule-based explanations (IF-THEN). Distrusts complex statistical methods. "
            "Wants clear thresholds and bright-line rules."
        ),
        "behavioral_signature": {
            "favors_simplicity": True,
            "prefers_conservative_errors": True,
            "values_precedent": True,
            "skeptical_of_novelty": True,
        },
    },
    {
        "name": "David Rodriguez",
        "role": "Data-Driven Analyst",
        "experience_years": 5,
        "loss_aversion": 1.5,
        "risk_tolerance": "Moderate",
        "decision_speed": "Moderate (analytical)",
        "trust_in_ai": "Medium (trusts validated models)",
        "priorities": [
            "Fidelity",
            "Completeness",
            "Consistency with model behavior",
        ],
        "mental_model": (
            "Statistical models capture patterns humans miss. Focus on performance metrics. "
            "Data quality and feature engineering are key."
        ),
        "heuristics": [
            "Look for feature importance alignment with business logic.",
            "Check for overfitting (train vs test performance).",
            "Validate explanations against sensitivity analysis.",
        ],
        "explanation_preferences": (
            "Prefers comprehensive, quantitative explanations. Values fidelity to the underlying model "
            "over simplicity. Wants to see all important features."
        ),
        "behavioral_signature": {
            "favors_completeness": True,
            "prefers_technical_rigor": True,
            "values_consistency": True,
            "comfortable_with_complexity": True,
        },
    },
    {
        "name": "Patricia Williams",
        "role": "Risk Manager",
        "experience_years": 22,
        "loss_aversion": 3.0,
        "risk_tolerance": "Very Low",
        "decision_speed": "Slow (compliance-focused)",
        "trust_in_ai": "Very Low (skeptical of automation)",
        "priorities": [
            "Risk control",
            "Regulatory defensibility",
            "Fair and robust decisions",
        ],
        "mental_model": (
            "Portfolio risk trumps individual accuracy. One bad loan can harm metrics. "
            "Regulatory compliance is non-negotiable."
        ),
        "heuristics": [
            "Focus on worst-case scenarios.",
            "Require multiple independent signals before approval.",
            "Document everything for audit trail.",
        ],
        "explanation_preferences": (
            "Needs explanations that satisfy regulators and internal audit. "
            "Must explicitly call out risk factors and be defensible."
        ),
        "behavioral_signature": {
            "favors_defensibility": True,
            "prefers_conservative_errors": True,
            "values_documentation": True,
            "fears_systemic_risk": True,
        },
    },
    {
        "name": "James Thompson",
        "role": "Customer Relationship Manager",
        "experience_years": 8,
        "loss_aversion": 1.2,
        "risk_tolerance": "Moderate-High",
        "decision_speed": "Fast (relationship-focused)",
        "trust_in_ai": "High (trusts combined human-AI judgment)",
        "priorities": [
            "Interpretability to non-experts",
            "Simplicity",
            "Communication to borrowers",
        ],
        "mental_model": (
            "People are more than numbers. Context matters, and life circumstances change. "
            "Relationships have long-term value beyond a single loan."
        ),
        "heuristics": [
            "Look for positive trajectory (improving credit over time).",
            "Consider extenuating circumstances.",
            "Balance short-term risk with long-term relationship value.",
        ],
        "explanation_preferences": (
            "Needs explanations that can be communicated to customers. Should highlight actionable steps "
            "to improve outcomes and avoid heavy jargon."
        ),
        "behavioral_signature": {
            "favors_empathy": True,
            "prefers_actionability": True,
            "values_communication": True,
            "optimistic_bias": True,
        },
    },
    {
        "name": "Sarah Martinez",
        "role": "Executive Decision Maker",
        "experience_years": 15,
        "loss_aversion": 1.8,
        "risk_tolerance": "Moderate",
        "decision_speed": "Very Fast (strategic focus)",
        "trust_in_ai": "Medium-High (trusts proven systems)",
        "priorities": [
            "Strategic impact",
            "Alignment with policy",
            "High-level clarity",
        ],
        "mental_model": (
            "Needs scalable, efficient decisions. Focus on portfolio-level metrics, not individual loans. "
            "Explanations must support business strategy."
        ),
        "heuristics": [
            "Time is valuable: prioritize high-impact decisions.",
            "Delegate details to domain experts.",
            "Focus on systemic patterns rather than edge cases.",
        ],
        "explanation_preferences": (
            "Needs high-level summaries that align with business objectives and support strategic planning. "
            "Details can be delegated."
        ),
        "behavioral_signature": {
            "favors_efficiency": True,
            "prefers_strategic_view": True,
            "values_scalability": True,
            "delegates_details": True,
        },
    },
]


DIMENSIONS = [
    ("Interpretability", "How easy was this explanation to understand?"),
    ("Completeness", "Does it cover the important reasons for the model's decision?"),
    ("Decision support", "Did this help you make a better loan decision?"),
    ("Trust calibration", "Does this help you know when to rely on the model?"),
]


def _template_rows(personas: List[dict]) -> str:
    rows = []
    for persona in personas:
        rows.append(f"### Persona: {persona['name']} – {persona['role']}")
        rows.append(f"- Experience: {persona['experience_years']} years")
        rows.append(f"- Loss aversion: {persona.get('loss_aversion', 'n/a')}")
        rows.append(f"- Risk tolerance: {persona['risk_tolerance']}")
        rows.append(f"- Decision speed: {persona.get('decision_speed', 'n/a')}")
        rows.append(f"- Trust in AI: {persona['trust_in_ai']}")
        rows.append(f"- Priorities: {', '.join(persona['priorities'])}")
        rows.append(f"- Mental model: {persona['mental_model']}")
        rows.append("- Heuristics:")
        for h in persona["heuristics"]:
            rows.append(f"  - {h}")
        rows.append(f"- Explanation preferences: {persona['explanation_preferences']}")
        rows.append("- Behavioral signature:")
        for k, v in persona["behavioral_signature"].items():
            rows.append(f"  - {k.replace('_', ' ')}: {v}")
        rows.append("\n#### Rating grid\n")
        rows.append("| Construct | Question | 1 (low) – 5 (high) |")
        rows.append("| --- | --- | --- |")
        for name, question in DIMENSIONS:
            rows.append(f"| {name} | {question} | [1] [2] [3] [4] [5] |")
        rows.append("\n---\n")
    return "\n".join(rows)


def generate_human_template(config_path: str):
    config = load_yaml_config(config_path)
    out_dir = Path(config.get("paths", {}).get("human_eval_dir", "outputs/human_eval_templates"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "human_eval_template.md"

    header = "# Human-Centred Evaluation Template (Credit Risk XAI)\n"
    context = (
        "This template pairs VXAI technical evaluation with a human-grounded check. "
        "Use the personas below to collect Likert-scale ratings for SHAP, LIME, ANCHOR, "
        "and counterfactual explanations on a small set of loan cases.\n\n"
    )

    table_section = _template_rows(PERSONAS)
    output_text = header + "\n" + context + table_section
    out_path.write_text(output_text, encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate human evaluation template.")
    parser.add_argument("--config", default="config/config_credit.yaml", help="Path to YAML config.")
    args = parser.parse_args()
    out_path = generate_human_template(args.config)
    print(f"Human evaluation template saved to {out_path}")


if __name__ == "__main__":
    main()
