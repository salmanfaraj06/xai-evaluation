"""Pytest fixtures and shared test utilities."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_data():
    """Generate sample tabular data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    data = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
        'target': np.random.choice([0, 1], n_samples)
    })
    
    return data


@pytest.fixture
def sample_model():
    """Create a simple trained model for testing."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd
    import numpy as np
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    
    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.choice(['A', 'B', 'C'], n_samples)
    })
    
    y = np.random.choice([0, 1], n_samples)
    
    # Encode categorical features
    le = LabelEncoder()
    X['feature3'] = le.fit_transform(X['feature3'])
    
    # Train model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    return model


@pytest.fixture
def project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def usecases_dir(project_root):
    """Get the usecases directory."""
    return project_root / "usecases"


@pytest.fixture
def mock_openai_key(monkeypatch):
    """Mock OpenAI API key for testing."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")
