"""Main evaluation orchestrator for HEXEval.

Coordinates technical metrics and human proxy evaluation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import yaml
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*X does not have valid feature names.*")

from hexeval.core import load_model, load_data, validate_model_data_compatibility
from hexeval.evaluation.technical_evaluator import run_technical_evaluation
from hexeval.evaluation.persona_evaluator import run_persona_evaluation
from hexeval.evaluation.recommender import generate_recommendations

LOG = logging.getLogger(__name__)


def load_config(config_path: str | None = None) -> Dict:
    """Load evaluation configuration."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "eval_config.yaml"
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def evaluate(
    model_path: str,
    data_path: str,
    target_column: str | None = None,
    config_path: str | None = None,
    output_dir: str | None = None,
    config_overrides: Dict[str, Any] | None = None,
    progress_callback: callable | None = None,
) -> Dict[str, Any]:
    """
    Run complete HEXEval evaluation pipeline.
    
    This is the main entry point for the framework.
    
    Parameters
    ----------
    model_path : str
        Path to trained model (.pkl or .joblib)
    data_path : str
        Path to CSV dataset
    target_column : str, optional
        Name of target column in dataset
    config_path : str, optional
        Path to config YAML (uses default if None)
    output_dir : str, optional
        Directory to save results (uses config default if None)
    
    Returns
    -------
    dict
        Evaluation results containing:
        - 'technical_metrics': DataFrame with technical evaluation
        - 'persona_ratings': DataFrame with LLM persona ratings
        - 'recommendations': Dict mapping stakeholders to recommended methods
        - 'model_info': Model metadata
        - 'data_info': Data metadata
    
    Examples
    --------
    >>> results = evaluate(
    ...     model_path="my_model.pkl",
    ...     data_path="my_data.csv",
    ...     target_column="target"
    ... )
    >>> print(results['recommendations'])
    """
    LOG.info("=" * 60)
    LOG.info("HEXEval - Holistic Explanation Evaluation")
    LOG.info("=" * 60)
    
    def report_progress(percent: float, message: str):
        """Helper to report progress if callback provided."""
        if progress_callback:
            progress_callback(percent, message)
    
    report_progress(0, "Starting evaluation...")
    
    config = load_config(config_path)
    if config_overrides:
        # Simple recursive update or flat update? 
        # For now, handle specific top-level keys used by UI
        if "personas" in config_overrides:
            config.setdefault("personas", {}).update(config_overrides["personas"])
        if "evaluation" in config_overrides:
             config.setdefault("evaluation", {}).update(config_overrides["evaluation"])
             
    LOG.info("✓ Loaded configuration")
    report_progress(5, "Configuration loaded")
    
    report_progress(10, "Loading model...")
    model_wrapper = load_model(model_path)
    model_info = model_wrapper.get_model_info()
    LOG.info(f"✓ Loaded model: {model_info['model_type']}")
    report_progress(15, f"Model loaded: {model_info['model_type']}")
    
    # Load data
    report_progress(20, "Loading data...")
    data = load_data(
        data_path,
        target_column=target_column,
        test_size=0.2,
        random_state=config["evaluation"]["random_state"],
    )
    LOG.info(f"✓ Loaded data: {len(data['X_train'])} train, {len(data['X_test'])} test")
    report_progress(25, f"Data loaded: {len(data['X_train'])} samples")
    
    report_progress(28, "Validating model-data compatibility...")
    validation = validate_model_data_compatibility(model_wrapper, data)
    if validation["status"] == "invalid":
        raise ValueError(f"Model-data validation failed: {validation['errors']}")
    LOG.info("✓ Validated model-data compatibility")
    report_progress(30, "Validation complete")
    
    # Run technical evaluation
    LOG.info("\n" + "=" * 60)
    LOG.info("Running Technical Evaluation")
    LOG.info("=" * 60)
    report_progress(35, "Running technical evaluation (SHAP, LIME, Anchor, DiCE)...")
    
    technical_results = run_technical_evaluation(
        model_wrapper=model_wrapper,
        data=data,
        config=config["evaluation"],
    )
    LOG.info(f"✓ Technical evaluation complete ({len(technical_results)} methods)")
    report_progress(60, "Technical evaluation complete")
    
    # Persona Evaluation (LLM-based)
    persona_results = None
    if config.get("personas", {}).get("enabled", False):
        LOG.info("\n" + "=" * 60)
        LOG.info("Running Persona Evaluation (LLM)")
        LOG.info("=" * 60)
        report_progress(65, "Running persona evaluation (LLM-based)...")
        try:
            persona_results = run_persona_evaluation(
                model_wrapper=model_wrapper,
                data=data,
                config=config,
            )
            LOG.info("✓ Persona evaluation complete")
            report_progress(85, "Persona evaluation complete")
        except Exception as e:
            LOG.warning(f"Persona evaluation failed: {e}")
            LOG.warning("Continuing without persona ratings...")
            report_progress(85, "Persona evaluation skipped")
    else:
        LOG.info("Persona evaluation disabled in config")
        report_progress(85, "Persona evaluation skipped (disabled)")
    
    # Generate recommendations
    recommendations = None
    if config.get("recommendations", {}).get("enabled", True):
        import json
        if persona_results is not None:
            report_progress(90, "Generating recommendations...")
            recommendations = generate_recommendations(
                technical_metrics=technical_results,
                persona_ratings=persona_results,
                config=config.get("recommendations", {}),
            )
            LOG.info("✓ Generated recommendations")
            report_progress(92, "Recommendations generated")
    
    # Save results
    report_progress(95, "Saving results...")
    # User requested flattened output structure, no domain splitting
    final_output_dir = output_dir if output_dir else config.get("output", {}).get("dir", "outputs/hexeval_results")
    output_path = Path(final_output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    technical_results.to_csv(output_path / "technical_metrics.csv", index=False)
    LOG.info(f"  Saved: technical_metrics.csv")
    
    if persona_results is not None:
        persona_results.to_csv(output_path / "persona_ratings.csv", index=False)
        LOG.info(f"  Saved: persona_ratings.csv")
    
    if recommendations is not None:
        with open(output_path / "recommendations.json", "w") as f:
            json.dump(recommendations, f, indent=2)
        LOG.info(f"  Saved: recommendations.json")
    
    LOG.info(f"\n✓ Results saved to: {output_path}")
    report_progress(100, "Evaluation complete!")
    
    return {
        "technical_metrics": technical_results,
        "persona_ratings": persona_results,
        "recommendations": recommendations,
        "model_info": model_info,
        "data_info": data,
        "output_path": str(output_path),
    }
