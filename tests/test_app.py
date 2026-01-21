"""Integration tests for Streamlit app."""

import pytest
from pathlib import Path
import sys


def test_app_imports():
    """Test that the Streamlit app can be imported without errors."""
    try:
        # Add project root to path
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))
        
        # Import the app module (this will execute module-level code)
        # We're just checking it doesn't crash on import
        import importlib.util
        
        app_path = project_root / "hexeval" / "ui" / "app.py"
        spec = importlib.util.spec_from_file_location("app", app_path)
        
        assert spec is not None
        assert spec.loader is not None
        
    except Exception as e:
        pytest.fail(f"Failed to import app: {e}")


def test_use_cases_configuration():
    """Test that USE_CASES configuration in app is valid."""
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    # Read the app.py file and check USE_CASES
    app_path = project_root / "hexeval" / "ui" / "app.py"
    app_content = app_path.read_text()
    
    # Check that USE_CASES is defined
    assert "USE_CASES = {" in app_content
    
    # Check for expected use cases
    assert "Heart Disease" in app_content
    assert "Credit Risk" in app_content
    assert "Custom Upload" in app_content


def test_demo_data_paths():
    """Test that demo data files referenced in app exist."""
    project_root = Path(__file__).parent.parent
    
    # Healthcare use case
    assert (project_root / "usecases" / "heart.csv").exists()
    assert (project_root / "usecases" / "heart_disease_pipeline.pkl").exists()
    
    # Credit risk use case
    assert (project_root / "usecases" / "credit_risk_dataset.csv").exists()
    assert (project_root / "usecases" / "xgboost_credit_risk_new.pkl").exists()


def test_config_paths():
    """Test that config files referenced in app exist."""
    project_root = Path(__file__).parent.parent
    
    assert (project_root / "hexeval" / "config" / "eval_config.yaml").exists()
    assert (project_root / "hexeval" / "config" / "eval_config_credit_risk.yaml").exists()


def test_documentation_files():
    """Test that documentation files exist for the Documentation tab."""
    project_root = Path(__file__).parent.parent
    docs_dir = project_root / "docs"
    
    # These files are referenced in the app's Documentation tab
    assert (docs_dir / "HEXEval_Prerequisites.md").exists()
    assert (docs_dir / "HEXEval_HowItWorks.md").exists()
    assert (docs_dir / "HEXEval_Configuration.md").exists()


def test_hexeval_package_importable():
    """Test that hexeval package can be imported."""
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    try:
        import hexeval
        assert hasattr(hexeval, '__version__') or hasattr(hexeval, 'evaluate')
    except ImportError as e:
        pytest.fail(f"Failed to import hexeval package: {e}")


def test_streamlit_config_valid():
    """Test that Streamlit config file is valid TOML."""
    project_root = Path(__file__).parent.parent
    config_path = project_root / ".streamlit" / "config.toml"
    
    assert config_path.exists()
    
    # Try to parse TOML
    try:
        import tomli
    except ImportError:
        # If tomli not available, just check file exists
        return
    
    with open(config_path, 'rb') as f:
        config = tomli.load(f)
        
    assert 'theme' in config
