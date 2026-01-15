"""
Quick script to train a matching model for testing HEXEval.

This trains an XGBoost model on credit_risk_dataset.csv so we can
test the HEXEval framework end-to-end.
"""

import pandas as pd
from hexeval.core import load_data
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib

print("Training test model for HEXEval...")

# Load data
data = load_data(
    "credit_risk_dataset.csv",
    target_column="loan_status",
    test_size=0.2,
    random_state=42,
)

X_train = data["X_train"]
y_train = data["y_train"]
categorical_features = data["categorical_features"]
numeric_features = data["numeric_features"]

print(f"Training on {len(X_train)} samples")
print(f"Categorical features: {categorical_features}")
print(f"Numeric features: {numeric_features}")

# Build preprocessor
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    verbose_feature_names_out=False
)

# Fit and transform
X_train_proc = preprocessor.fit_transform(X_train)
feature_names = preprocessor.get_feature_names_out().tolist()

print(f"Transformed to {len(feature_names)} features")

# Train model
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
)

model.fit(X_train_proc, y_train)

# Save artifact
artifact = {
    "model": model,
    "preprocessor": preprocessor,
    "feature_names": feature_names,
    "default_threshold": 0.5,
}

joblib.dump(artifact, "hexeval_test_model.pkl")

print("✓ Model saved to hexeval_test_model.pkl")
print("\nNow you can test HEXEval with:")
print("  python hexeval_cli.py validate hexeval_test_model.pkl credit_risk_dataset.csv --target loan_status")
