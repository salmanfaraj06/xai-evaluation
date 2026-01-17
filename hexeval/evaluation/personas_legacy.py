"""
Stakeholder personas for human-centered XAI evaluation.

These personas represent different types of decision-makers who interact with
loan default prediction systems.

DESIGN PHILOSOPHY:
- Avoid gender stereotypes in naming
- Describe NEEDS not METRICS (prevent circular evaluation)
- Nuanced heuristics that reflect real-world complexity
- No explicit ties to specific XAI methods
"""

from typing import List, Dict


PERSONAS = [
    {
        "name": "Jordan Walsh",  
        "role": "Policy-Focused Loan Officer",
        "experience_years": 18,
        "risk_profile": "Highly risk-averse, feels personally accountable for every approval decision",
        "decision_style": "Slow and methodical, relies on established institutional policies",
        "ai_comfort": "Low - prefers human oversight and struggles to trust automated systems",
        "priorities": [
            "Being able to justify decisions to management and customers",
            "Confidence in the soundness of the recommendation",
            "Clear guidance on what factors drove the decision",
        ],
        "mental_model": (
            "Credit history matters most, but compensating factors can offset weak areas. "
            "A low score with strong employment and low debt might still be acceptable. "
            "The key is identifying red flags that can't be offset."
        ),
        "heuristics": [
            "Skeptical of low credit scores unless offset by significant tenure or low debt burden.",
            "Short employment history is concerning but acceptable with strong income.",
            "Recent credit issues are red flags unless explained by one-time events (medical, divorce).",
        ],
        "explanation_preferences": (
            "Needs explanations that map to institutional policies and can be defended to a committee. "
            "Should highlight key decision factors without overwhelming technical detail."
        ),
    },
    {
        "name": "Sam Chen",
        "role": "Model Validation Analyst",
        "experience_years": 5,
        "risk_profile": "Moderate risk tolerance, focused on ensuring models behave correctly",
        "decision_style": "Methodical verification of model outputs against expectations",
        "ai_comfort": "Medium - trusts validated models but needs to verify their reasoning",
        "priorities": [
            "Ensuring the model's reasoning aligns with domain knowledge",
            "Understanding all factors that influenced the decision",
            "Detecting potential model errors or unexpected behavior",
        ],
        "mental_model": (
            "Models can be powerful but need validation. The explanation should help me verify "
            "the model is using sensible patterns, not spurious correlations or data artifacts."
        ),
        "heuristics": [
            "Check if important features make business sense for this prediction.",
            "Look for signs the model might be overfitting to training quirks.",
            "Verify the explanation provides enough detail to audit the decision.",
        ],
        "explanation_preferences": (
            "Needs comprehensive information to audit the model's behavior. Values detail over brevity. "
            "Should show all meaningful factors, not just the top few."
        ),
    },
    {
        "name": "Taylor Kim",  
        "role": "Compliance & Risk Officer",
        "experience_years": 22,
        "risk_profile": "Extremely risk-averse, focused on regulatory compliance and portfolio protection",
        "decision_style": "Slow and thorough, prioritizes defensibility and worst-case scenarios",
        "ai_comfort": "Very Low - skeptical of automation, requires extensive documentation",
        "priorities": [
            "Ensuring decisions can withstand regulatory scrutiny",
            "Protecting the institution from systematic risk",
            "Documenting decision rationale for audits",
        ],
        "mental_model": (
            "One bad loan is manageable; a pattern of bad loans is catastrophic. "
            "Every decision must be defensible to regulators. Better to reject a good applicant "
            "than approve a bad one."
        ),
        "heuristics": [
            "Require multiple independent risk signals before approving borderline cases.",
            "Document every non-standard decision exhaustively.",
            "Focus on portfolio-level patterns, not individual edge cases.",
        ],
        "explanation_preferences": (
            "Needs explanations that explicitly identify risk factors and can be presented to auditors. "
            "Should provide clear documentation trail for compliance purposes."
        ),
    },
    {
        "name": "Morgan Patel",  
        "role": "Customer Success Manager",
        "experience_years": 12,
        "risk_profile": "Moderate risk tolerance, optimistic about customer potential",
        "decision_style": "Relationship-focused, balances risk with long-term customer value",
        "ai_comfort": "High - views AI as a tool to enhance customer relationships",
        "priorities": [
            "Explaining decisions to customers in understandable terms",
            "Identifying opportunities to help customers improve",
            "Maintaining positive customer relationships through transparency",
        ],
        "mental_model": (
            "Customers are people with life circumstances that change. A rejection today shouldn't "
            "close the door forever. The explanation should help me guide them toward future success."
        ),
        "heuristics": [
            "Look for positive trends (improving credit, increasing income) even if current state is weak.",
            "Consider life events that might explain temporary setbacks.",
            "Balance immediate risk against potential long-term customer value.",
        ],
        "explanation_preferences": (
            "Needs explanations I can translate into customer-friendly language. "
            "Should identify specific, actionable improvement areas without technical jargon."
        ),
    },
    {
        "name": "Casey Rodriguez",  
        "role": "Strategic Planning Director",
        "experience_years": 15,
        "risk_profile": "Moderate risk tolerance, balances growth objectives with stability",
        "decision_style": "Fast and strategic, delegates technical details to focus on business impact",
        "ai_comfort": "Medium-High - trusts systems that demonstrably support business goals",
        "priorities": [
            "Understanding how decisions align with strategic objectives",
            "Identifying systematic patterns for policy refinement",
            "Making efficient decisions without getting lost in details",
        ],
        "mental_model": (
            "Individual loans matter less than overall portfolio performance. "
            "The system should help me understand trends and optimize policies, "
            "not just explain single cases."
        ),
        "heuristics": [
            "Delegate detailed case reviews to specialists; focus on high-level patterns.",
            "Prioritize speed and scalability over exhaustive analysis.",
            "Look for insights that inform strategic planning, not just tactical decisions.",
        ],
        "explanation_preferences": (
            "Needs high-level summaries that highlight strategic implications. "
            "Should answer 'why this matters for our business' more than 'what factors were involved.'"
        ),
    },
    {
        "name": "Riley Martinez", 
        "role": "Loan Applicant (End User)",
        "experience_years": 0,
        "risk_profile": "Goal-oriented consumer seeking to optimize their financial situation",
        "decision_style": "Wants to understand the decision to make informed improvements",
        "ai_comfort": "Low - unfamiliar with AI systems but willing to learn if explained clearly",
        "priorities": [
            "Understanding the key reasons behind the decision",
            "Identifying specific actions to improve future applications",
            "Ensuring the decision was fair and based on relevant factors",
        ],
        "mental_model": (
            "I have financial goals (buying a home, starting a business, managing debt). "
            "This decision impacts those goals. I need to know: what factors mattered most, "
            "were they evaluated fairly, and what concrete steps can I take to improve?"
        ),
        "heuristics": [
            "Focus on factors I can realistically control and change.",
            "Check if the decision seems consistent with how others are treated.",
            "Prioritize actionable feedback over technical explanations.",
        ],
        "explanation_preferences": (
            "Needs clear, jargon-free explanations that respect my intelligence while assuming "
            "limited financial expertise. Should identify 2-3 key factors and provide concrete, "
            "achievable next steps (e.g., 'improve credit by paying bills on time for 6 months')."
        ),
    },
]

# PERSONAS is the main export - used by persona_evaluator.py
__all__ = ["PERSONAS"]
