import joblib
import pickle
import sys
from pathlib import Path

path = Path("xgboost_loan_default_research_v2.pkl")
if not path.exists():
    # Try the one in config if relative path fails
    path = Path("outputs/models/xgboost_loan_default_research_v2.pkl")

if not path.exists():
    print(f"File not found: {path}")
    sys.exit(1)

try:
    with open(path, "rb") as f:
        artifact = pickle.load(f)
except Exception:
    artifact = joblib.load(path)

print(f"Type: {type(artifact)}")
if isinstance(artifact, dict):
    print(f"Keys: {artifact.keys()}")
    if "preprocessor" in artifact:
        print(f"Preprocessor: {artifact['preprocessor']}")
    else:
        print("Preprocessor: MISSING")
else:
    print("Artifact is not a dict (likely just the model).")
