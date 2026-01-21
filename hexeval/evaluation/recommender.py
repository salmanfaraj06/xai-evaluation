"""Recommendation engine for HEXEval.

Maps stakeholder profiles to best explanation methods.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd
import numpy as np

LOG = logging.getLogger(__name__)


# Stakeholder profile → preferred characteristics
STAKEHOLDER_PREFERENCES = {
    "Conservative": {
        "preferred_traits": ["rule_based", "simple", "high_precision"],
        "avoid_traits": ["complex", "many_features"],
        "priority_dimensions": ["trust", "actionability"],
    },
    "Technical": {
        "preferred_traits": ["comprehensive", "faithful", "consistent"],
        "avoid_traits": ["oversimplified"],
        "priority_dimensions": ["completeness", "decision_support"],
    },
    "Executive": {
        "preferred_traits": ["simple", "strategic", "efficient"],
        "avoid_traits": ["too_detailed"],
        "priority_dimensions": ["satisfaction", "decision_support"],
    },
    "Customer-Facing": {
        "preferred_traits": ["actionable", "simple", "communicable"],
        "avoid_traits": ["technical_jargon"],
        "priority_dimensions": ["actionability", "interpretability"],
    },
    "Risk-Averse": {
        "preferred_traits": ["defensive", "conservative", "rule_based"],
        "avoid_traits": ["uncertain"],
        "priority_dimensions": ["trust", "satisfaction"],
    },
}


# Method characteristics
METHOD_TRAITS = {
    "SHAP": ["comprehensive", "faithful", "many_features", "complex"],
    "LIME": ["balanced", "interpretable", "moderate_features"],
    "Anchor": ["rule_based", "simple", "high_precision", "defensive"],
    "DiCE": ["actionable", "counterfactual", "communicable"],
    "COUNTERFACTUAL": ["actionable", "counterfactual", "communicable"],  
}


def generate_recommendations(
    technical_metrics: pd.DataFrame,
    persona_ratings: pd.DataFrame,
    config: Dict = None,
) -> Dict[str, Dict]:
    """
    Generate method recommendations for different stakeholder types.
    
    Parameters
    ----------
    technical_metrics : pd.DataFrame
        Technical evaluation results (from run_technical_evaluation)
    persona_ratings : pd.DataFrame
        Persona ratings (from run_persona_evaluation)
    config : dict, optional
        Recommendation configuration with weights
    
    Returns
    -------
    dict
        Recommendations structured as:
        {
            "stakeholder_type": {
                "recommended_method": str,
                "score": float,
                "reasoning": str,
                "technical_strengths": dict,
                "persona_feedback": str
            }
        }
    
    Examples
    --------
    >>> recs = generate_recommendations(tech_df, persona_df)
    >>> print(recs["Conservative"]["recommended_method"])
    "Anchor"
    """
    config = config or {}
    weights = config.get("weights", {
        "technical_fidelity": 0.3,
        "technical_parsimony": 0.2,
        "persona_trust": 0.3,
        "persona_satisfaction": 0.2,
    })
    
    recommendations = {}
    
    # Group persona ratings by role
    persona_grouped = persona_ratings.groupby(["persona_role", "explanation_type"]).agg({
        "trust": "mean",
        "satisfaction": "mean",
        "actionability": "mean",
        "interpretability": "mean",
        "completeness": "mean",
        "decision_support": "mean",
    }).reset_index()
    
    # Extract unique stakeholder types
    stakeholder_types = persona_grouped["persona_role"].unique()
    
    for stakeholder in stakeholder_types:
        stakeholder_data = persona_grouped[persona_grouped["persona_role"] == stakeholder]
        
        # Calculate scores for each method
        method_scores = {}
        
        for _, row in stakeholder_data.iterrows():
            method = row["explanation_type"]
            
            # Get technical metrics for this method
            tech_row = technical_metrics[technical_metrics["method"] == method]
            
            if tech_row.empty:
                continue
            
            tech_row = tech_row.iloc[0]
            
            # METHOD-SPECIFIC TECHNICAL SCORES (normalized to 0-1)
            # This fixes the bias bug - each method is scored on its own strengths
            
            if method == "SHAP" or method == "LIME":
                # Use fidelity & parsimony for attribution methods
                fidelity_score = 0
                if pd.notna(tech_row.get("fidelity_deletion")):
                    # Lower deletion AUC = better fidelity
                    fidelity_score = 1 - tech_row["fidelity_deletion"]
                
                parsimony_score = 0
                if pd.notna(tech_row.get("num_important_features")):
                    # Fewer features = better parsimony
                    # Normalize: 5 features=1.0, 20 features=0.25
                    sparsity = tech_row["num_important_features"]
                    parsimony_score = min(1.0, 10 / max(sparsity, 1))
                
                # Average the two scores for SHAP/LIME
                technical_score = (fidelity_score + parsimony_score) / 2
            
            elif method == "Anchor":
                # Use precision (accuracy) & coverage for rule-based
                precision_score = 0
                if pd.notna(tech_row.get("rule_accuracy")):
                    # Higher precision = better (already 0-1)
                    precision_score = tech_row["rule_accuracy"]
                
                coverage_score = 0
                if pd.notna(tech_row.get("rule_applicability")):
                    # Higher coverage = better (already 0-1)
                    coverage_score = tech_row["rule_applicability"]
                
                # Precision matters more than coverage (80/20 split)
                technical_score = 0.8 * precision_score + 0.2 * coverage_score
            
            elif method == "DiCE" or method == "COUNTERFACTUAL":
                # Use success rate for counterfactuals
                success_score = 0
                if pd.notna(tech_row.get("counterfactual_success")):
                    # Higher success = better (already 0-1)
                    success_score = tech_row["counterfactual_success"]
                
                # Only one metric for DiCE currently
                technical_score = success_score
            
            else:
                # Unknown method - default to 0
                technical_score = 0
            
            # Persona score (already 1-5, normalize to 0-1)
            persona_score = (row["trust"] + row["satisfaction"]) / 10
            
            # Combined weighted score
            combined_score = (
                weights["technical_fidelity"] * technical_score +
                weights["technical_parsimony"] * technical_score +  # Both use technical_score now
                weights["persona_trust"] * (row["trust"] / 5) +
                weights["persona_satisfaction"] * (row["satisfaction"] / 5)
            )
            
            method_scores[method] = {
                "score": combined_score,
                "trust": row["trust"],
                "satisfaction": row["satisfaction"],
                "actionability": row["actionability"],
                "technical_score": technical_score,  # Store for transparency
            }
        
        # Find best method
        if not method_scores:
            continue
        
        best_method = max(method_scores.keys(), key=lambda m: method_scores[m]["score"])
        best_scores = method_scores[best_method]
        
        # Generate reasoning
        reasoning_parts = []
        
        if best_scores["trust"] >= 4.0:
            reasoning_parts.append(f"high stakeholder trust ({best_scores['trust']:.1f}/5)")
        
        if best_scores["satisfaction"] >= 4.0:
            reasoning_parts.append(f"strong satisfaction ({best_scores['satisfaction']:.1f}/5)")
        
        if best_scores["technical_score"] >= 0.7:
            reasoning_parts.append(f"strong technical performance ({best_scores['technical_score']:.2f})")
        
        if not reasoning_parts:
            # If no standout scores, mention best available
            reasoning_parts.append(f"best combined score across metrics")
        
        reasoning = f"{best_method} recommended due to: " + ", ".join(reasoning_parts)
        
        # Get technical details
        tech_row = technical_metrics[technical_metrics["method"] == best_method].iloc[0]
        technical_strengths = tech_row.to_dict()
        
        # Get persona feedback
        trust_adj = "highly" if best_scores['trust'] >= 4 else "moderately" if best_scores['trust'] >= 2.5 else "poorly"
        sat_adj = "highly" if best_scores['satisfaction'] >= 4 else "moderately" if best_scores['satisfaction'] >= 2.5 else "poorly"
        
        persona_feedback = (
            f"This stakeholder type ({stakeholder}) rated {best_method} {trust_adj} on "
            f"trust ({best_scores['trust']:.1f}/5) and {sat_adj} on satisfaction ({best_scores['satisfaction']:.1f}/5)."
        )
        
        recommendations[stakeholder] = {
            "recommended_method": best_method,
            "score": float(best_scores["score"]),
            "reasoning": reasoning,
            "technical_strengths": {k: str(v) for k, v in technical_strengths.items()},
            "persona_feedback": persona_feedback,
            # BUG FIX: Add alternatives for UI display
            "alternatives": {
                method: float(scores["score"])
                for method, scores in method_scores.items()
                if method != best_method
            }
        }
    
    LOG.info(f"Generated recommendations for {len(recommendations)} stakeholder types")
    
    return recommendations


def get_method_summary(method: str) -> str:
    """Get human-readable summary of explanation method."""
    summaries = {
        "SHAP": "Feature attribution using Shapley values - comprehensive but complex",
        "LIME": "Local linear approximation - balanced interpretability and fidelity",
        "Anchor": "Rule-based explanations (IF-THEN) - simple and precise",
        "DiCE": "Counterfactual examples - actionable recourse for users",
        "COUNTERFACTUAL": "Counterfactual examples - actionable recourse for users",
    }
    return summaries.get(method, "Unknown method")
