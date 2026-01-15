"""
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
        "risk_profile": "Highly risk-averse, conservative decision-maker who feels responsible for every approval",
        "decision_style": "Slow and methodical, prefers established rules over statistical models",
        "ai_comfort": "Low - prefers human oversight and doesn't fully trust automated systems",
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
    },
    {
        "name": "David Rodriguez",
        "role": "Data-Driven Analyst",
        "experience_years": 5,
        "risk_profile": "Moderate risk tolerance, data-driven and analytical",
        "decision_style": "Methodical analysis of quantitative evidence and model performance",
        "ai_comfort": "Medium - trusts validated models but verifies their behavior",
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
    },
    {
        "name": "Patricia Williams",
        "role": "Risk Manager",
        "experience_years": 22,
        "risk_profile": "Extremely risk-averse, focused on portfolio-level risk management",
        "decision_style": "Slow and compliance-focused, prioritizes worst-case scenarios",
        "ai_comfort": "Very Low - highly skeptical of automation, requires extensive validation",
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
        "experience_years": 12,
        "risk_profile": "Moderate risk tolerance with optimistic outlook on customer potential",
        "decision_style": "Fast and relationship-focused, considers human context over pure numbers",
        "ai_comfort": "High - trusts AI as a tool to enhance human judgment, not replace it",
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

    },
    {
        "name": "Sarah Martinez",
        "role": "Executive Decision Maker",
        "experience_years": 15,
        "risk_profile": "Moderate risk tolerance, balances innovation with stability",
        "decision_style": "Very fast and strategic, delegates details to focus on high-level patterns",
        "ai_comfort": "Medium-High - trusts proven systems that align with business objectives",
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
