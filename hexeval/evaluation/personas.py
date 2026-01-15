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
        "experience_years": 0,
        "risk_profile": "Highly anxious about rejection, feels vulnerable and judged by the system",
        "decision_style": "Not a decision-maker - seeking to understand and respond to the decision",
        "ai_comfort": "Low - unfamiliar with AI, needs plain-language explanations",
        "priorities": [
            "Understanding why decision was made",
            "Fairness and transparency",
            "Clear next steps to improve",
        ],
        "mental_model": (
            "This loan decision affects my entire life. I need to understand what went wrong "
            "and what I can realistically do to improve my chances. The system should explain "
            "in terms I can understand, not financial jargon."
        ),
        "heuristics": [
            "Look for things I can actually control and change.",
            "Check if the decision seems fair compared to others.",
            "Understand if there's a path forward for me.",
        ],
        "explanation_preferences": (
            "Needs plain-language explanations without jargon. Wants to understand the 'why' in simple terms "
            "and get actionable steps to improve their application. Cares deeply about fairness."
        ),
    },
]

# PERSONAS is the main export - used by persona_evaluator.py
__all__ = ["PERSONAS"]
