"""
<<<<<<< Updated upstream
Utilities for loading persona definitions from configuration.
"""
from pathlib import Path
from typing import List, Dict
import yaml
import logging

LOG = logging.getLogger(__name__)

def load_personas_from_file(path: str | Path) -> List[Dict]:
    """
    Load personas from a YAML file.
    
    Parameters
    ----------
    path : str or Path
        Path to the YAML file containing persona definitions.
        
    Returns
    -------
    List[Dict]
        List of persona dictionaries.
    """
    path = Path(path)
    if not path.exists():
        # Fallback to looking in config directory relative to project root
        project_root = Path(__file__).parent.parent.parent
        possible_path = project_root / path
        if possible_path.exists():
            path = possible_path
        else:
            raise FileNotFoundError(f"Personas file not found at {path} or {possible_path}")
            
    with open(path, "r") as f:
        personas = yaml.safe_load(f)
        
    if not isinstance(personas, list):
        raise ValueError(f"Personas file must contain a list of persona dictionaries, got {type(personas)}")
        
    LOG.info(f"Loaded {len(personas)} personas from {path.name}")
    return personas
=======
Stakeholder personas for human-centered XAI evaluation.

These personas represent different types of decision-makers who interact with
loan default prediction systems.
"""

from typing import List, Dict


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
    {
        "name": "Alex Johnson",
        "role": "Loan Applicant (End User)",
        "experience_years": 0,  # Not a professional, just a customer
        "loss_aversion": 3.0,  # High - loan rejection is personally devastating
        "risk_tolerance": "N/A - Customer Perspective",
        "decision_speed": "Immediate",
        "trust_in_ai": "Low",  # Skeptical of automated systems making life decisions
        "priorities": [
            "Simplicity - Can I understand this without financial jargon?",
            "Actionability - What can I do to improve my chances?",
            "Fairness - Is this decision fair and unbiased?",
            "Transparency - Why was I rejected/approved?",
        ],
        "mental_model": (
            "I applied for a loan to improve my life (buy a home, start a business, consolidate debt). "
            "I don't understand financial formulas or statistical models. I just want to know: "
            "1) Why was this decision made? 2) Is it fair? 3) What can I do about it? "
            "Technical jargon confuses and frustrates me. I need explanations in plain English that "
            "respect my intelligence but don't assume financial expertise."
        ),
        "heuristics": [
            "If I can't understand the explanation, I assume the system is hiding something.",
            "If it tells me specific actions I can take, I trust it more.",
            "Credit score matters, but I don't know exactly how or why.",
            "Complex numbers and statistics make me anxious and confused.",
            "I want to feel respected, not talked down to.",
        ],
        "explanation_preferences": (
            "I need simple, jargon-free explanations in everyday language. "
            "Tell me the 2-3 main reasons for the decision. "
            "Most importantly, tell me what I can DO about it - can I improve my credit score? "
            "Should I save more? Do I need a co-signer? "
            "I don't care about SHAP values or statistical weights - I care about understanding "
            "the decision and having a path forward."
        ),
        "behavioral_signature": {
            "values_plain_language": True,
            "needs_actionable_steps": True,
            "low_financial_literacy": True,
            "high_anxiety_about_rejection": True,
            "prefers_simple_rules": True,
            "distrusts_black_box_decisions": True,
        },
    },
]

# PERSONAS is the main export - used by persona_evaluator.py
__all__ = ["PERSONAS"]
>>>>>>> Stashed changes
