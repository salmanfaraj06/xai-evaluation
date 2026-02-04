"""Technical evaluation module for HEXEval.

Orchestrates SHAP, LIME, Anchor, and DiCE evaluations with technical metrics.
Removes VXAI jargon and uses practitioner-friendly terminology.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from hexeval.core.data_handler import preprocess_for_model
from hexeval.explainers.shap_explainer import ShapExplainer
from hexeval.explainers.lime_explainer import LimeExplainer
from hexeval.explainers.anchor_explainer import AnchorExplainer
from hexeval.explainers.dice_counterfactuals import DiceExplainer
from hexeval.metrics.fidelity import insertion_deletion_auc
from hexeval.metrics.parsimony_coverage import (
    sparsity_from_importances,
    anchor_parsimony_and_coverage,
    counterfactual_sparsity,
)
from hexeval.metrics.robustness import explanation_stability

LOG = logging.getLogger(__name__)


def run_technical_evaluation(
    model_wrapper: Any,  
    data: Dict,
    config: Dict,
) -> pd.DataFrame:
    """
    Run technical evaluation on all enabled explanation methods.
    
    Evaluates SHAP, LIME, Anchor, and DiCE.
    """
    # Defensive config access
    eval_config = config if "explainers" in config else {"explainers": {}, "fidelity": {}, "stability": {}}
    
    # Defaults for explainers if missing
    if "shap" not in eval_config.get("explainers", {}):
        eval_config.setdefault("explainers", {})["shap"] = {"enabled": False}
    if "lime" not in eval_config["explainers"]:
        eval_config["explainers"]["lime"] = {"enabled": False}
    if "anchor" not in eval_config["explainers"]:
        eval_config["explainers"]["anchor"] = {"enabled": False}
    if "dice" not in eval_config["explainers"]:
        eval_config["explainers"]["dice"] = {"enabled": False}
        
    LOG.info("Starting technical evaluation...")
    
    model = model_wrapper.model
    preprocessor = model_wrapper.preprocessor
    feature_names = model_wrapper.feature_names or data["feature_names"]
    
    # Prepare data
    X_train_proc = preprocess_for_model(data["X_train"], preprocessor, feature_names)
    X_test_proc = preprocess_for_model(data["X_test"], preprocessor, feature_names)
    
    # Sample for evaluation
    sample_size = min(config.get("sample_size", 150), len(X_test_proc))
    rng = np.random.default_rng(config.get("random_state", 42))
    sample_idx = rng.choice(len(X_test_proc), size=sample_size, replace=False)
    X_sample = X_test_proc[sample_idx]
    baseline = X_train_proc.mean(axis=0)
    
    results = []
    
    
    if config["explainers"]["shap"]["enabled"]:
        LOG.info("Evaluating SHAP...")
        try:
            shap_metrics = _evaluate_shap(
                model, X_train_proc, X_sample, baseline, feature_names, config
            )
            results.append({"method": "SHAP", **shap_metrics})
            LOG.info(f"  ✓ SHAP complete")
        except Exception as e:
            LOG.error(f"  ✗ SHAP failed: {e}")
            results.append({"method": "SHAP", "error": str(e)})
    
            results.append({"method": "SHAP", "error": str(e)})
    
    if config["explainers"]["lime"]["enabled"]:
        LOG.info("Evaluating LIME...")
        try:
            lime_metrics = _evaluate_lime(
                model, X_train_proc, X_sample, baseline, feature_names, config
            )
            results.append({"method": "LIME", **lime_metrics})
            LOG.info(f"  ✓ LIME complete")
        except Exception as e:
            LOG.error(f"  ✗ LIME failed: {e}")
            results.append({"method": "LIME", "error": str(e)})
    
            results.append({"method": "LIME", "error": str(e)})
    
    if config["explainers"]["anchor"]["enabled"]:
        LOG.info("Evaluating Anchor...")
        try:
            anchor_metrics = _evaluate_anchor(
                model, X_train_proc, X_sample, feature_names, config, model_wrapper
            )
            results.append({"method": "Anchor", **anchor_metrics})
            LOG.info(f"  ✓ Anchor complete")
        except Exception as e:
            LOG.error(f"  ✗ Anchor failed: {e}")
            results.append({"method": "Anchor", "error": str(e)})
    
            results.append({"method": "Anchor", "error": str(e)})
    
    if config["explainers"]["dice"]["enabled"]:
        LOG.info("Evaluating DiCE...")
        try:
            dice_metrics = _evaluate_dice(
                model, X_train_proc, X_sample, data["y_train"], feature_names, config
            )
            results.append({"method": "DiCE", **dice_metrics})
            LOG.info(f"  ✓ DiCE complete")
        except Exception as e:
            LOG.error(f"  ✗ DiCE failed: {e}")
            results.append({"method": "DiCE", "error": str(e)})
    
    return pd.DataFrame(results)


def _evaluate_shap(model, X_train, X_sample, baseline, feature_names, config) -> Dict:
    """Evaluate SHAP explainer."""
    bg_size = min(config["explainers"]["shap"]["background_size"], len(X_train))
    background = X_train[:bg_size]
    
    explainer = ShapExplainer(model, background, feature_names)
    shap_values = explainer.explain_dataset(X_sample)
    
    # Fidelity metrics
    fidelity = insertion_deletion_auc(
        model, X_sample, shap_values, baseline,
        steps=config["fidelity"]["steps"]
    )
    
    # Parsimony
    sparsity = sparsity_from_importances(shap_values)
    
    return {
        "fidelity_deletion": fidelity["deletion_auc"],
        "fidelity_insertion": fidelity["insertion_auc"],
        "num_important_features": sparsity,
        "rule_accuracy": np.nan,
        "rule_applicability": np.nan,
        "rule_length": np.nan,
        "counterfactual_success": np.nan,
        "counterfactual_sparsity": np.nan,
        "stability": np.nan,
    }


def _evaluate_lime(model, X_train, X_sample, baseline, feature_names, config) -> Dict:
    """Evaluate LIME explainer."""
    domain_cfg = config.get("domain", {})
    class_names = [
        domain_cfg.get("positive_outcome", "Class 0"),
        domain_cfg.get("negative_outcome", "Class 1")
    ]
    
    LOG.info(f"LIME Debug: X_train type: {type(X_train)}, shape: {X_train.shape if hasattr(X_train, 'shape') else 'N/A'}")
    LOG.info(f"LIME Debug: X_train dtype: {X_train.dtype if hasattr(X_train, 'dtype') else 'N/A'}")
    
    if not isinstance(X_train, np.ndarray):
        X_train = np.asarray(X_train, dtype=np.float64)
    else:
        X_train = X_train.astype(np.float64)
    
    LOG.info(f"LIME Debug: After conversion - X_train has NaN: {np.any(np.isnan(X_train))}")
    LOG.info(f"LIME Debug: NaN count: {np.sum(np.isnan(X_train))}")
    
    X_train_jittered = X_train.copy()
    
    variances = np.var(X_train_jittered, axis=0)
    min_variance = 1e-8  
    
    LOG.info(f"LIME Debug: Min variance: {np.min(variances)}, Max variance: {np.max(variances)}")
    LOG.info(f"LIME Debug: Features with low variance: {np.sum(variances < min_variance)}")
    
    for i in range(X_train_jittered.shape[1]):
        if variances[i] < min_variance:
            mean_val = np.abs(np.mean(X_train_jittered[:, i]))
            noise_scale = max(mean_val * 0.01, 0.01) 
            noise = np.random.RandomState(42 + i).normal(0, noise_scale, X_train_jittered.shape[0])
            X_train_jittered[:, i] += noise
            LOG.info(f"LIME Debug: Added jitter to feature {i} (variance was {variances[i]:.2e})")
    
    # Check for and fill any NaN values
    if np.any(np.isnan(X_train_jittered)):
        nan_count = np.sum(np.isnan(X_train_jittered))
        LOG.warning(f"LIME Debug: Found {nan_count} NaN values in X_train, filling with 0.0")
        X_train_jittered = np.nan_to_num(X_train_jittered, nan=0.0, copy=False)
        if np.any(np.isnan(X_train_jittered)):
            LOG.error("LIME Debug: NaN still present after nan_to_num!")
        else:
            LOG.info("LIME Debug: Successfully removed all NaN values from X_train")
    
    LOG.info(f"LIME Debug: Final X_train_jittered has NaN: {np.any(np.isnan(X_train_jittered))}")
    LOG.info(f"LIME Debug: Final shape: {X_train_jittered.shape}, dtype: {X_train_jittered.dtype}")
    
    X_train_clean = X_train_jittered.copy()

    explainer = LimeExplainer(
        training_data=X_train_clean,
        feature_names=feature_names,
        class_names=class_names,
        predict_fn=model.predict_proba,
    )
    
    # Generate explanations
    lime_cfg = config["explainers"]["lime"]
    importances = []
    
    # Check and fix NaN in X_sample as well
    if isinstance(X_sample, np.ndarray):
        if np.any(np.isnan(X_sample)):
            nan_count = np.sum(np.isnan(X_sample))
            LOG.warning(f"LIME Debug: Found {nan_count} NaN values in X_sample, filling with 0.0")
            X_sample = np.nan_to_num(X_sample, nan=0.0)
    
    for i in range(len(X_sample)):
        weights = explainer.as_importance_vector(
            X_sample[i],
            num_features=lime_cfg["num_features"],
            num_samples=lime_cfg["num_samples"],
        )
        importances.append(weights)
    importances = np.vstack(importances)
    
    # Fidelity
    fidelity = insertion_deletion_auc(
        model, X_sample, importances, baseline,
        steps=config["fidelity"]["steps"]
    )
    
    # Parsimony
    sparsity = sparsity_from_importances(importances)
    
    # Stability
    stability_score = np.nan
    if lime_cfg.get("stability_test", False):
        subset_size = min(lime_cfg.get("stability_subset", 30), len(X_sample))
        stability_scores = []
        for i in range(subset_size):
            score = explanation_stability(
                lambda x: explainer.as_importance_vector(
                    x, lime_cfg["num_features"], lime_cfg["num_samples"]
                ),
                X_sample[i],
                noise_std=config["stability"]["noise_std"],
                n_repeats=config["stability"]["repeats"],
            )
            stability_scores.append(score)
        stability_score = np.mean(stability_scores)
    
    return {
        "fidelity_deletion": fidelity["deletion_auc"],
        "fidelity_insertion": fidelity["insertion_auc"],
        "num_important_features": sparsity,
        "rule_accuracy": np.nan,
        "rule_applicability": np.nan,
        "rule_length": np.nan,
        "counterfactual_success": np.nan,
        "counterfactual_sparsity": np.nan,
        "stability": stability_score,
    }


def _evaluate_anchor(model, X_train, X_sample, feature_names, config, model_wrapper) -> Dict:
    """Evaluate Anchor explainer."""
    threshold = model_wrapper.threshold
    
    def predict_fn(X):
        proba = model.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)
    
    explainer = AnchorExplainer(
        X_train,
        feature_names=feature_names,
        predict_fn=predict_fn,
    )
    
    # Generate anchors
    anchor_cfg = config["explainers"]["anchor"]
    max_instances = min(anchor_cfg["max_instances"], len(X_sample))
    
    anchor_results = []
    for i in range(max_instances):
        try:
            exp = explainer.explain_instance(
                X_sample[i],
                threshold=anchor_cfg["precision_threshold"]
            )
            metrics = anchor_parsimony_and_coverage(exp)
            anchor_results.append(metrics)
        except Exception:
            continue
    
    if not anchor_results:
        raise ValueError("No anchor explanations generated")
    
    # Aggregate
    df = pd.DataFrame(anchor_results)
    
    return {
        "fidelity_deletion": np.nan,
        "fidelity_insertion": np.nan,
        "num_important_features": np.nan,
        "rule_accuracy": df["precision"].mean(),
        "rule_applicability": df["coverage"].mean(),
        "rule_length": df["n_conditions"].mean(),
        "counterfactual_success": np.nan,
        "counterfactual_sparsity": np.nan,
        "stability": np.nan,
    }


def _evaluate_dice(model, X_train, X_sample, y_train, feature_names, config) -> Dict:
    """Evaluate DiCE explainer."""
    dice_cfg = config["explainers"]["dice"]
    
    explainer = DiceExplainer(
        model=model,
        X_train_processed=X_train.astype(np.float64),
        y_train=y_train,
        feature_names=feature_names,
        outcome_name="target",
        method=dice_cfg.get("method", "random"),
    )
    
    max_instances = min(dice_cfg["max_instances"], len(X_sample))
    
    validity_scores = []
    sparsity_scores = []
    tolerance = dice_cfg.get("sparsity_epsilon", 1e-6)
    for i in range(max_instances):
        try:
            cf_exp = explainer.generate_counterfactuals(
                X_sample[i].astype(np.float64),
                total_cfs=dice_cfg["num_counterfactuals"],
            )
            
            # Check validity
            cf_df = cf_exp.cf_examples_list[0].final_cfs_df[feature_names]
            cf_arr = cf_df.values
            
            base_pred = model.predict(X_sample[i].reshape(1, -1))[0]
            cf_preds = model.predict(cf_arr)
            
            validity = np.mean(cf_preds != base_pred)
            validity_scores.append(validity)

            sparsity = counterfactual_sparsity(
                X_sample[i].astype(np.float64),
                cf_arr.astype(np.float64),
                tolerance=tolerance,
            )
            if not np.isnan(sparsity):
                sparsity_scores.append(sparsity)
        except Exception:
            continue
    
    if not validity_scores:
        raise ValueError("No counterfactuals generated")
    
    return {
        "fidelity_deletion": np.nan,
        "fidelity_insertion": np.nan,
        "num_important_features": np.nan,
        "rule_accuracy": np.nan,
        "rule_applicability": np.nan,
        "rule_length": np.nan,
        "counterfactual_success": np.mean(validity_scores),
        "counterfactual_sparsity": float(np.mean(sparsity_scores)) if sparsity_scores else np.nan,
        "stability": np.nan,
    }
