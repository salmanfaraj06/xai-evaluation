"""
Complete LLM Persona Evaluator for HEXEval.

Implements full persona-based evaluation using OpenAI's models (GPT-4, GPT-4o, o1/o3-mini reasoning models).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from hexeval.core.data_handler import preprocess_for_model
from hexeval.explainers.shap_explainer import ShapExplainer
from hexeval.explainers.lime_explainer import LimeExplainer
from hexeval.explainers.anchor_explainer import AnchorExplainer
from hexeval.explainers.dice_counterfactuals import DiceExplainer

try:
    from openai import OpenAI
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    OpenAI = None

LOG = logging.getLogger(__name__)

from hexeval.evaluation.personas import load_personas_from_file

RATING_DIMENSIONS: List[Tuple[str, str]] = [
    ("interpretability", "How easy is this explanation to understand? (1-5)"),
    ("completeness", "How complete is the reasoning behind the decision? (1-5)"),
    ("actionability", "Does this suggest actionable next steps? (1-5)"),
    ("trust", "How much do you trust the model after seeing this? (1-5)"),
    ("satisfaction", "Overall satisfaction with the explanation. (1-5)"),
    ("decision_support", "How well does it support your loan decision? (1-5)"),
]


def run_persona_evaluation(model_wrapper: Any, data: Dict, config: Dict) -> pd.DataFrame | None:
    """
    Run LLM-based persona evaluation.
    
    Parameters
    ----------
    model_wrapper : ModelWrapper
        Wrapped model object
    data : dict
        Data dictionary from load_data()
    config : dict
        Full configuration including persona settings and domain context
    
    Returns
    -------
    pd.DataFrame or None
        Persona ratings with columns:
        - persona_name, persona_role
        - explanation_type
        - instance_index, run
        - trust, satisfaction, actionability, interpretability, completeness, decision_support
        - comment
        - raw_llm_response, used_llm
    """
    persona_config = config.get("personas", {})
    
    if not persona_config.get("enabled", False):
        LOG.info("Persona evaluation disabled in config")
        return None
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        LOG.warning("OPENAI_API_KEY not found. Persona evaluation requires OpenAI API access.")
        LOG.warning("Set environment variable: export OPENAI_API_KEY='your-key'")
        return None
    
    if OpenAI is None:
        LOG.error("openai package not installed. Install with: pip install openai")
        return None
    
    client = OpenAI(api_key=api_key)
    llm_model = persona_config.get("llm_model", "gpt-4o")
    
    # Extract domain config for prompts
    domain_config = config.get("domain", {})
    
    LOG.info(f"Running persona evaluation with {llm_model}")
    LOG.info(f"Domain: {domain_config.get('name', 'Generic')}")
    
    # Load personas from config file
    personas_file = persona_config.get("file", "hexeval/config/personas_healthcare.yaml") 
    try:
        personas = load_personas_from_file(personas_file)
    except Exception as e:
        LOG.error(f"Failed to load personas from {personas_file}: {e}")
        return None

    # Generate explanations
    explanations, instance_indices = _generate_explanations(
        model_wrapper, data, config
    )
    
    # Run LLM evaluation with domain context
    results = _evaluate_with_llm(
        client,
        llm_model,
        explanations,
        instance_indices,
        data,
        personas,  
        persona_config,
        domain_config,  
    )
    
    return pd.DataFrame(results)


def _generate_explanations(model_wrapper: Any, data: Dict, config: Dict) -> Tuple[Dict, List[int]]:
    """Generate explanations for sample instances using all 4 methods."""
    
    model = model_wrapper.model
    preprocessor = model_wrapper.preprocessor
    feature_names = model_wrapper.feature_names or data["feature_names"]
    
    # Preprocess data
    X_train = preprocess_for_model(data["X_train"], preprocessor, feature_names)
    X_test = preprocess_for_model(data["X_test"], preprocessor, feature_names)
    
    # Sample instances
    n_instances = config.get("personas", {}).get("sample_instances", 5)
    n_instances = min(n_instances, len(X_test))
    
    rng = np.random.default_rng(config.get("evaluation", {}).get("random_state", 42))
    instance_indices = rng.choice(len(X_test), size=n_instances, replace=False).tolist()
    
    top_k = config.get("personas", {}).get("top_k_features", 5)
    
    explanations = {}  
    
    LOG.info(f"Generating explanations for {n_instances} instances...")
    
    # Initialize explainers
    shap_exp = ShapExplainer(model, X_train[:min(500, len(X_train))], feature_names)
    
    # Get class names from domain config
    domain = config.get("domain", {})
    positive = domain.get("positive_outcome", "Class 0")
    negative = domain.get("negative_outcome", "Class 1")
    
    lime_exp = LimeExplainer(
        X_train, feature_names,
        class_names=[positive, negative],
        predict_fn=model.predict_proba
    )
    
    threshold = model_wrapper.threshold
    anchor_exp = AnchorExplainer(
        X_train,
        feature_names=feature_names,
        predict_fn=lambda arr: (model.predict_proba(arr)[:, 1] >= threshold).astype(int)
    )
    
    dice_exp = DiceExplainer(
        model, X_train, data["y_train"],
        feature_names, outcome_name="target"
    )
    
    # Generate explanations
    for idx in tqdm(instance_indices, desc="Generating explanations"):
        instance = X_test[idx]
        explanations[idx] = {}
        
        # SHAP
        try:
            shap_vals = shap_exp.explain_instance(instance)
            top_idx = np.argsort(-np.abs(shap_vals))[:top_k]
            parts = [f"{feature_names[i]}: {shap_vals[i]:.3f}" for i in top_idx]
            explanations[idx]["SHAP"] = "Top SHAP values: " + "; ".join(parts)
        except Exception as e:
            explanations[idx]["SHAP"] = f"SHAP unavailable: {e}"
        
        # LIME
        try:
            exp = lime_exp.explain_instance(instance, num_features=top_k, num_samples=2000)
            parts = [f"{fn}: {w:.3f}" for fn, w in exp.as_list()]
            explanations[idx]["LIME"] = "LIME weights: " + "; ".join(parts)
        except Exception as e:
            explanations[idx]["LIME"] = f"LIME unavailable: {e}"
        
        # Anchor
        try:
            anchor_res = anchor_exp.explain_instance(instance, threshold=0.9)
            rule = " AND ".join(anchor_res.names())
            explanations[idx]["ANCHOR"] = f"Rule: IF {rule}; precision={anchor_res.precision():.2f}, coverage={anchor_res.coverage():.2f}"
        except Exception as e:
            explanations[idx]["ANCHOR"] = f"Anchor unavailable: {e}"
        
        # DiCE
        try:
            cf = dice_exp.generate_counterfactuals(instance, total_cfs=1)
            cf_df = cf.cf_examples_list[0].final_cfs_df[feature_names]
            changes = []
            for col_i, col in enumerate(feature_names):
                delta = cf_df.iloc[0][col] - instance[col_i]
                if abs(delta) > 1e-6:
                    changes.append(f"{col}: change by {delta:.2f}")
            explanations[idx]["COUNTERFACTUAL"] = "To flip prediction: " + (", ".join(changes[:5]) if changes else "no changes")
        except Exception as e:
            explanations[idx]["COUNTERFACTUAL"] = f"Counterfactual unavailable: {e}"
    
    return explanations, instance_indices


def _evaluate_with_llm(
    client,
    llm_model: str,
    explanations: Dict,
    instance_indices: List[int],
    data: Dict,
    personas: List[Dict],
    persona_config: Dict,
    domain_config: Dict,  
) -> List[Dict]:
    """Evaluate explanations using LLM personas."""
    
    results = []
    runs_per_method = persona_config.get("runs_per_method", 2)
    
    LOG.info("Running LLM persona evaluation...")
    
    total_calls = len(personas) * 4 * len(instance_indices) * runs_per_method
    LOG.info(f"Total LLM calls: {total_calls}")
    
    with tqdm(total=total_calls, desc="LLM evaluation") as pbar:
        for persona in personas:
            system_prompt = _build_system_prompt(persona, domain_config)  
            
            for idx in instance_indices:
                pred_info = {
                    "instance_index": idx,
                    "predicted_proba_default": 0.5,  
                    "predicted_class": 1,
                    "actual_class": data["y_test"].iloc[idx] if data["y_test"] is not None else None,
                    "role": persona["role"],  
                }
                
                for method, explanation_text in explanations[idx].items():
                    for run in range(runs_per_method):
                        user_prompt = _build_eval_prompt(pred_info, explanation_text, method, domain_config)  # Pass domain
                        
                        try:
                            ratings, raw_response = _call_llm(client, llm_model, system_prompt, user_prompt)
                            
                            result = {
                                "persona_name": persona["name"],
                                "persona_role": persona["role"],
                                "explanation_type": method,
                                "instance_index": idx,
                                "run": run,
                                **ratings,
                                "raw_llm_response": raw_response,
                                "used_llm": llm_model,
                            }
                            results.append(result)
                            
                        except Exception as e:
                            LOG.warning(f"LLM call failed for {persona['name']}/{method}/{idx}: {e}")
                            # Fallback scores
                            results.append({
                                "persona_name": persona["name"],
                                "persona_role": persona["role"],
                                "explanation_type": method,
                                "instance_index": idx,
                                "run": run,
                                **_fallback_scores(method),
                                "raw_llm_response": str(e),
                                "used_llm": "fallback",
                            })
                        
                        pbar.update(1)
    
    return results


def _build_system_prompt(persona: Dict, domain_config: Dict) -> str:
    """Build rich, context-heavy system prompt that makes LLM deeply embody the persona."""
    
    # Extract persona details
    name = persona['name']
    role = persona['role']
    years = persona.get('experience_years', 'many')
    risk_profile = persona.get('risk_profile', 'Balanced approach to risk')
    decision_style = persona.get('decision_style', 'Methodical')
    ai_comfort = persona.get('ai_comfort', 'Medium')
    priorities = persona.get('priorities', [])
    mental_model = persona.get('mental_model', '')
    heuristics = persona.get('heuristics', [])
    preferences = persona.get('explanation_preferences', '')
    
    # Extract domain context (NEW: makes it reusable)
    domain_name = domain_config.get('name', 'Machine Learning')
    prediction_task = domain_config.get('prediction_task', 'prediction')
    decision_verb = domain_config.get('decision_verb', 'make a decision')
    decision_noun = domain_config.get('decision_noun', 'case')
    stakeholder_ctx = domain_config.get('stakeholder_context', '')
    
    # Adapt intro based on whether this is an end-user or stakeholder
    is_end_user = "End User" in role or "Applicant" in role or "Customer" in role
    
    if is_end_user:
        intro = f"""You are {name}, a person {domain_config.get('end_user_context', 'using this AI system')}.

YOUR SITUATION:
You are affected by an AI prediction about you. The decision impacts your life directly.
- A positive outcome ({domain_config.get('positive_outcome', 'approval')}) can change your future
- A negative outcome ({domain_config.get('negative_outcome', 'rejection')}) feels personal and devastating
- You don't have technical expertise - you're just a regular person trying to understand"""
    else:
        intro = f"""You are {name}, a {role} {stakeholder_ctx} with {years} years of experience.

YOUR IDENTITY & BACKGROUND:
You make critical decisions about {prediction_task}. Each decision impacts:
- The people affected by your decisions (their outcomes matter)
- Your organization's objectives (accuracy, fairness, compliance)
- Your own professional reputation and career"""
    
    return f"""{intro}

YOUR PROFILE:
• Risk Profile: {risk_profile}
• Decision Style: {decision_style}
• AI Comfort Level: {ai_comfort}

YOUR MENTAL MODEL:
{mental_model}

YOUR DECISION-MAKING APPROACH:
{chr(10).join(f'• {h}' for h in heuristics)}

WHAT YOU VALUE IN EXPLANATIONS:
{preferences}

YOUR TOP PRIORITIES (in order):
{chr(10).join(f'{i+1}. {p}' for i, p in enumerate(priorities[:4]))}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

YOUR TASK TODAY:
You're reviewing AI predictions for {prediction_task}. The AI has provided an explanation for its prediction.

You need to rate HOW USEFUL this explanation is for YOUR needs as a {role}.

Rate on 6 dimensions (1-5 scale):

1. **interpretability** - Can you easily understand what the AI is saying?
   - 1 = Completely confusing
   - 5 = Crystal clear

2. **completeness** - Does it cover all the factors you need to know?
   - 1 = Missing critical information
   - 5 = Covers everything needed

3. **actionability** - Does it suggest what to DO next?
   - 1 = No guidance on next steps
   - 5 = Clear, specific actions

4. **trust** - Do you trust this explanation enough to use it?
   - 1 = Don't trust at all
   - 5 = Fully trust it

5. **satisfaction** - Overall, how satisfied are you with this?
   - 1 = Very dissatisfied
   - 5 = Very satisfied

6. **decision_support** - Does this help you with your needs?
   - 1 = Doesn't help at all
   - 5 = Extremely helpful

IMPORTANT: Rate from YOUR perspective as {name}, not as a generic evaluator. Your {risk_profile.lower()} should influence your ratings.

Respond in TOML format:
interpretability = <1-5>
completeness = <1-5>
actionability = <1-5>
trust = <1-5>
satisfaction = <1-5>
decision_support = <1-5>
comment = "<2-3 sentences explaining your ratings FROM YOUR PERSPECTIVE as a {role}>"
"""


def _build_eval_prompt(pred_info: Dict, explanation_text: str, method: str, domain_config: Dict) -> str:
    """Build realistic evaluation prompt with rich scenario context."""
    
    instance_id = pred_info['instance_index']
    decision_noun = domain_config.get('decision_noun', 'case')
    decision_verb = domain_config.get('decision_verb', 'make a decision')
    prediction_task = domain_config.get('prediction_task', 'prediction')
    
    # Build realistic scenario
    scenario = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REVIEW - Case #{instance_id}

You are reviewing {decision_noun}. The AI system has analyzed this case and made a prediction about {prediction_task}.

AI SYSTEM OUTPUT:
The system used the "{method}" explanation method to show you WHY it made this prediction.

Here's what the AI is telling you:

{explanation_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

YOUR EVALUATION:
Imagine you're looking at this explanation in your real work. You need to {decision_verb}, and this AI explanation is supposed to help you.

Rate this explanation on the 6 dimensions (1-5).
Think about:
- Would this actually help you?
- Does it match how YOU think about this problem?
- Can you defend/explain this to others if needed?
- Does it give you confidence in the decision?

Provide your ratings in TOML format AS {pred_info.get('role', 'your role')}:
"""
    
    return scenario


def _call_llm(client, model: str, system_prompt: str, user_prompt: str) -> Tuple[Dict, str]:
    """Call OpenAI API and parse TOML response."""
    
    is_reasoning = model.startswith("o1") or model.startswith("o3")
    
    if is_reasoning:
        messages = [{"role": "user", "content": system_prompt + "\n\n" + user_prompt}]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=400 if not is_reasoning else None,  
    )
    
    content = response.choices[0].message.content or ""
    
    try:
        import toml
        data = toml.loads(content)
    except:

        data = {}
        for line in content.split('\n'):
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, val = line.split('=', 1)
                key = key.strip()
                val = val.strip().strip('"\'')
                
              
                try:
                    data[key] = float(val)
                except:
                    data[key] = val
    
    # Extract ratings with fallback
    ratings = {}
    for dim, _ in RATING_DIMENSIONS:
        raw_val = data.get(dim, 3.0)
        try:
            ratings[dim] = float(np.clip(float(raw_val), 1, 5))
        except:
            ratings[dim] = 3.0
    
    ratings["comment"] = str(data.get("comment", "No comment provided"))
    
    return ratings, content


def _fallback_scores(method: str) -> Dict:
    """Deterministic fallback when LLM unavailable."""
    base = {"SHAP": 4.0, "LIME": 3.6, "ANCHOR": 3.8, "COUNTERFACTUAL": 3.5}.get(method.upper(), 3.0)
    noise = np.linspace(-0.1, 0.1, num=len(RATING_DIMENSIONS))
    scores = {dim: float(np.clip(base + delta, 1, 5)) for (dim, _), delta in zip(RATING_DIMENSIONS, noise)}
    scores["comment"] = "LLM unavailable; fallback scores."
    return scores
