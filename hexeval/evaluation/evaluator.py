"""Main evaluation orchestrator for HEXEval.

Coordinates technical metrics and human proxy evaluation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import yaml

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
    
    # Load configuration
    config = load_config(config_path)
    LOG.info("✓ Loaded configuration")
    
    # Load model
    model_artifact = load_model(model_path)
    LOG.info(f"✓ Loaded model: {model_artifact['model_type']}")
    
    # Load data
    data = load_data(
        data_path,
        target_column=target_column,
        test_size=0.2,
        random_state=config["evaluation"]["random_state"],
    )
    LOG.info(f"✓ Loaded data: {len(data['X_train'])} train, {len(data['X_test'])} test")
    
    # Validate compatibility
    validation = validate_model_data_compatibility(model_artifact, data)
    if validation["status"] == "invalid":
        raise ValueError(f"Model-data validation failed: {validation['errors']}")
    LOG.info("✓ Validated model-data compatibility")
    
    # Run technical evaluation
    LOG.info("\n" + "=" * 60)
    LOG.info("Running Technical Evaluation")
    LOG.info("=" * 60)
    
    technical_results = run_technical_evaluation(
        model_artifact=model_artifact,
        data=data,
        config=config["evaluation"],
    )
    LOG.info(f"✓ Technical evaluation complete ({len(technical_results)} methods)")
    
    # Run persona evaluation
    persona_results = None
    if config["personas"]["enabled"]:
        LOG.info("\n" + "=" * 60)
        LOG.info("Running Persona Evaluation (LLM)")
        LOG.info("=" * 60)
        
        try:
            persona_results = run_persona_evaluation(
                model_artifact=model_artifact,
                data=data,
                config=config,
            )
            LOG.info(f"✓ Persona evaluation complete")
        except Exception as e:
            LOG.warning(f"Persona evaluation failed: {e}")
            LOG.warning("Continuing without persona ratings...")
    
    # Generate recommendations
    recommendations = None
    if config.get("recommendations", {}).get("enabled", True):
        if persona_results is not None:
            recommendations = generate_recommendations(
                technical_metrics=technical_results,
                persona_ratings=persona_results,
                config=config.get("recommendations", {}),
            )
            LOG.info("✓ Generated recommendations")
    
    # Save results
    output_path = Path(output_dir) if output_dir else Path(config["output"]["dir"])
    output_path.mkdir(parents=True, exist_ok=True)
    
    technical_results.to_csv(output_path / "technical_metrics.csv", index=False)
    LOG.info(f"  Saved: technical_metrics.csv")
    
    if persona_results is not None:
        persona_results.to_csv(output_path / "persona_ratings.csv", index=False)
        LOG.info(f"  Saved: persona_ratings.csv")
    
    if recommendations is not None:
        import json
        with open(output_path / "recommendations.json", "w") as f:
            json.dump(recommendations, f, indent=2)
        LOG.info(f"  Saved: recommendations.json")
    
    LOG.info(f"\n✓ Results saved to: {output_path}")
    
    return {
        "technical_metrics": technical_results,
        "persona_ratings": persona_results,
        "recommendations": recommendations,
        "model_info": model_artifact,
        "data_info": data,
        "output_path": str(output_path),
    }
