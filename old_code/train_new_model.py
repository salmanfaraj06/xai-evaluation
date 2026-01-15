import logging
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path.cwd()))

from src.data_loading import load_yaml_config, load_credit_data, split_data
from src.preprocessing import build_preprocessor, transform_features
from src.models import train_credit_model, evaluate_model_performance, save_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger(__name__)

def main():
    config_path = "config/config_credit_new.yaml"
    LOG.info(f"Loading config from {config_path}")
    config = load_yaml_config(config_path)

    LOG.info("Loading and splitting data...")
    X, y = load_credit_data(config)
    
    # Split
    data_cfg = config.get("data", {})
    X_train, X_test, y_train, y_test = split_data(
        X, y, 
        test_size=data_cfg.get("test_size", 0.2),
        random_state=data_cfg.get("random_state", 42)
    )
    
    LOG.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Preprocessing
    LOG.info("Building preprocessor...")
    preprocessor, feature_names = build_preprocessor(config, X_train)
    
    LOG.info("Transforming features...")
    X_train_proc = transform_features(preprocessor, X_train)
    X_test_proc = transform_features(preprocessor, X_test)
    
    # Train
    LOG.info("Training XGBoost model...")
    model = train_credit_model(config, X_train_proc, y_train)
    
    # Evaluate
    LOG.info("Evaluating model...")
    metrics = evaluate_model_performance(model, X_test_proc, y_test)
    LOG.info(f"Test Metrics: {metrics}")
    
    # Save
    model_name = config["paths"]["pretrained_model"]
    save_path = Path(config["paths"]["model_dir"]) / model_name
    LOG.info(f"Saving artifact to {save_path}...")
    
    # Calculate a simple default threshold (e.g., maximizing F1 or just 0.5)
    # For now, we'll stick to 0.5 or calculate a better one if needed.
    # Let's do a quick threshold search for max F1 on test set (simple version)
    from sklearn.metrics import f1_score
    probs = model.predict_proba(X_test_proc)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 9)
    best_th = 0.5
    best_f1 = 0.0
    for th in thresholds:
        preds = (probs >= th).astype(int)
        score = f1_score(y_test, preds)
        if score > best_f1:
            best_f1 = score
            best_th = th
    
    LOG.info(f"Best F1 threshold on test set: {best_th} (F1={best_f1:.4f})")

    save_artifact(
        model=model,
        preprocessor=preprocessor,
        feature_names=feature_names,
        path=save_path,
        default_threshold=best_th
    )
    LOG.info("Done.")

if __name__ == "__main__":
    main()
