"""Unit tests for HEXEval core modules."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


def test_sample_data_fixture(sample_data):
    """Test that sample data fixture works correctly."""
    assert isinstance(sample_data, pd.DataFrame)
    assert len(sample_data) == 100
    assert 'target' in sample_data.columns
    assert set(sample_data['target'].unique()) == {0, 1}


def test_sample_model_fixture(sample_model):
    """Test that sample model fixture works correctly."""
    assert hasattr(sample_model, 'predict_proba')
    assert hasattr(sample_model, 'predict')
    
    # Test prediction
    X_test = np.random.randn(10, 3)
    predictions = sample_model.predict_proba(X_test)
    
    assert predictions.shape == (10, 2)
    assert np.allclose(predictions.sum(axis=1), 1.0)


def test_project_structure(project_root):
    """Test that required project directories exist."""
    assert (project_root / "hexeval").exists()
    assert (project_root / "hexeval" / "core").exists()
    assert (project_root / "hexeval" / "explainers").exists()
    assert (project_root / "hexeval" / "metrics").exists()
    assert (project_root / "hexeval" / "evaluation").exists()
    assert (project_root / "hexeval" / "ui").exists()


def test_usecases_exist(usecases_dir):
    """Test that demo use case files exist."""
    assert usecases_dir.exists()
    
    # Check for demo data files
    assert (usecases_dir / "heart.csv").exists()
    assert (usecases_dir / "credit_risk_dataset.csv").exists()
    
    # Check for model files
    assert (usecases_dir / "heart_disease_pipeline.pkl").exists()
    assert (usecases_dir / "xgboost_credit_risk_new.pkl").exists()


def test_config_files_exist(project_root):
    """Test that configuration files exist."""
    config_dir = project_root / "hexeval" / "config"
    
    assert (config_dir / "eval_config.yaml").exists()
    assert (config_dir / "eval_config_credit_risk.yaml").exists()
    assert (config_dir / "personas_healthcare.yaml").exists()
    assert (config_dir / "personas_credit_risk.yaml").exists()


def test_requirements_file(project_root):
    """Test that requirements.txt exists and has key dependencies."""
    req_file = project_root / "requirements.txt"
    assert req_file.exists()
    
    content = req_file.read_text()
    
    # Check for critical dependencies
    assert "streamlit" in content
    assert "shap" in content
    assert "lime" in content
    assert "dice-ml" in content
    assert "openai" in content
    assert "pytest" in content


def test_env_example_exists(project_root):
    """Test that .env.example exists for deployment guidance."""
    assert (project_root / ".env.example").exists()


def test_streamlit_config_exists(project_root):
    """Test that Streamlit configuration exists."""
    streamlit_dir = project_root / ".streamlit"
    assert streamlit_dir.exists()
    assert (streamlit_dir / "config.toml").exists()
    assert (streamlit_dir / "secrets.toml.example").exists()
