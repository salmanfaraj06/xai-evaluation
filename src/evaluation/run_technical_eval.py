"""VXAI-aligned technical evaluation runner."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.data_loading import load_credit_data, load_yaml_config, split_data
from src.explainers.anchor_explainer import AnchorExplainer
from src.explainers.dice_counterfactuals import DiceExplainer
from src.explainers.lime_explainer import LimeExplainer
from src.explainers.shap_explainer import ShapExplainer
from src.metrics.fidelity import insertion_deletion_auc
from src.metrics.parsimony_coverage import anchor_parsimony_and_coverage, sparsity_from_importances
from src.metrics.robustness import explanation_stability
from src.models import load_artifact
from src.preprocessing import align_columns, one_hot_encode, transform_features


LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _get_paths(config: dict):
    paths = config.get("paths", {})
    model_dir = Path(paths.get("model_dir", "outputs/models"))
    metrics_dir = Path(paths.get("metrics_dir", "outputs/vxai_metrics"))
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return model_dir, metrics_dir


def _prepare_data(config: dict):
    X, y = load_credit_data(config)
    data_cfg = config.get("data", {})
    test_size = data_cfg.get("test_size", 0.2)
    random_state = data_cfg.get("random_state", 42)
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=test_size, random_state=random_state
    )
    return {
        "X_train_raw": X_train,
        "X_test_raw": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def _resolve_artifact_path(config: dict, model_path: str | None, model_dir: Path) -> Path:
    if model_path:
        return Path(model_path)
    cfg_path = config.get("paths", {}).get("pretrained_model")
    if cfg_path:
        return Path(cfg_path)
    return model_dir / "credit_model.joblib"


def _safe_anchor_eval(anchor_explainer, x_row, predict_fn, threshold):
    try:
        exp = anchor_explainer.explain_instance(x_row, threshold=threshold)
        metrics = anchor_parsimony_and_coverage(exp)
        return metrics
    except Exception:
        return None


def _dice_metrics(model, x_row_proc, cf_exp, feature_names):
    cf_df = cf_exp.cf_examples_list[0].final_cfs_df[feature_names]
    cf_arr = cf_df.values
    base_pred = int(model.predict(x_row_proc.reshape(1, -1))[0])
    cf_preds = model.predict(cf_arr)
    validity = np.mean(cf_preds != base_pred)

    deltas = cf_arr - x_row_proc
    l1 = np.abs(deltas).sum(axis=1).mean()
    l2 = np.sqrt((deltas**2).sum(axis=1)).mean()
    sparsity = (np.abs(deltas) > 1e-6).sum(axis=1).mean()

    return {
        "cf_validity_rate": float(validity),
        "cf_mean_l1_distance": float(l1),
        "cf_mean_l2_distance": float(l2),
        "cf_mean_sparsity": float(sparsity),
    }


def run_technical_eval(config_path: str, model_path: str | None = None) -> Dict[str, pd.DataFrame]:
    config = load_yaml_config(config_path)
    data = _prepare_data(config)
    model_dir, metrics_dir = _get_paths(config)

    artifact_path = _resolve_artifact_path(config, model_path, model_dir)
    LOG.info("Loading model artifact from %s", artifact_path)
    artifact = load_artifact(artifact_path)
    model = artifact["model"]
    preprocessor = artifact.get("preprocessor")
    feature_names = artifact.get("feature_names")
    categorical_cols = config.get("data", {}).get("categorical", [])

    if preprocessor is not None:
        X_train_proc = np.asarray(transform_features(preprocessor, data["X_train_raw"]))
        X_test_proc = np.asarray(transform_features(preprocessor, data["X_test_raw"]))
        if feature_names is None:
            feature_names = preprocessor.get_feature_names_out().tolist()
    else:
        if feature_names is None:
            raise ValueError(
                "Artifact missing feature_names and preprocessor. Cannot align data."
            )

        def encode(df: pd.DataFrame) -> pd.DataFrame:
            encoded = one_hot_encode(df, categorical_cols, drop_first=True)
            return align_columns(encoded, feature_names)

        X_train_enc = encode(data["X_train_raw"])
        X_test_enc = encode(data["X_test_raw"])
        X_train_proc = X_train_enc.to_numpy()
        X_test_proc = X_test_enc.to_numpy()

    X_train_proc = X_train_proc.astype(np.float32)
    X_test_proc = X_test_proc.astype(np.float32)

    default_threshold = float(artifact.get("default_threshold", 0.5))

    def predict_proba_from_raw(raw_df: pd.DataFrame) -> np.ndarray:
        """Predict probabilities from raw dataframe rows."""
        if preprocessor is not None:
            return model.predict_proba(preprocessor.transform(raw_df))
        encoded = one_hot_encode(raw_df, categorical_cols, drop_first=True)
        encoded = align_columns(encoded, feature_names)
        return model.predict_proba(encoded)

    eval_cfg = config.get("evaluation", {})
    fidelity_cfg = eval_cfg.get("fidelity", {})
    lime_cfg = eval_cfg.get("lime", {})
    shap_cfg = eval_cfg.get("shap", {})
    anchor_cfg = eval_cfg.get("anchor", {})
    dice_cfg = eval_cfg.get("dice", {})

    rng = np.random.default_rng(config.get("data", {}).get("random_state", 42))

    # Sample for fidelity
    subset_size = min(fidelity_cfg.get("subset_size", 50), len(X_test_proc))
    subset_idx = rng.choice(len(X_test_proc), size=subset_size, replace=False)
    X_subset = X_test_proc[subset_idx]
    baseline_vec = X_train_proc.mean(axis=0)

    metrics_records: List[dict] = []

    # === SHAP ===
    try:
        bg_size = min(shap_cfg.get("background_size", 500), len(X_train_proc))
        background = X_train_proc[:bg_size]
        LOG.info("Running SHAP with background_size=%d on %d instances", bg_size, len(X_subset))
        shap_explainer = ShapExplainer(model, background=background, feature_names=feature_names)
        shap_vals = shap_explainer.explain_dataset(X_subset)
        shap_id = insertion_deletion_auc(
            model,
            X_subset,
            shap_vals,
            baseline_vec=baseline_vec,
            steps=fidelity_cfg.get("steps", 50),
            return_per_instance=True,
        )
        metrics_records.append({
            "method": "SHAP",
            "deletion_auc": shap_id["deletion_auc"],
            "insertion_auc": shap_id["insertion_auc"],
            "sparsity": float(sparsity_from_importances(shap_vals)),
            "anchor_precision": np.nan,
            "anchor_coverage": np.nan,
            "anchor_rule_length": np.nan,
            "cf_validity_rate": np.nan,
            "cf_mean_l1_distance": np.nan,
            "cf_mean_l2_distance": np.nan,
            "cf_mean_sparsity": np.nan,
            "stability": np.nan,
        })
        LOG.info(
            "SHAP done: deletion_auc=%.4f insertion_auc=%.4f",
            shap_id["deletion_auc"],
            shap_id["insertion_auc"],
        )
    except Exception as exc:
        metrics_records.append({
            "method": "SHAP",
            "error": str(exc),
        })

    # === LIME ===
    try:
        LOG.info("Running LIME on %d instances (num_samples=%s)", len(subset_idx), lime_cfg.get("num_samples", 2000))
        lime_explainer = LimeExplainer(
            training_data=X_train_proc,
            feature_names=feature_names,
            class_names=["No Default", "Default"],
            predict_fn=model.predict_proba,
            random_state=config.get("data", {}).get("random_state", 42),
        )
        lime_importances = []
        for idx in subset_idx:
            row = X_test_proc[idx]
            weights = lime_explainer.as_importance_vector(
                row,
                num_features=lime_cfg.get("num_features", 10),
                num_samples=lime_cfg.get("num_samples", 2000),
            )
            lime_importances.append(weights)
        lime_importances = np.vstack(lime_importances)

        lime_id = insertion_deletion_auc(
            model,
            X_subset,
            lime_importances,
            baseline_vec=baseline_vec,
            steps=fidelity_cfg.get("steps", 50),
            return_per_instance=True,
        )

        stability_scores = []
        stability_subset_size = min(lime_cfg.get("stability_subset_size", 30), len(subset_idx))
        for idx in subset_idx[:stability_subset_size]:  # lightweight stability probe
            row = X_test_proc[idx]
            stability_scores.append(
                explanation_stability(
                    lambda r: lime_explainer.as_importance_vector(
                        r,
                        num_features=lime_cfg.get("num_features", 10),
                        num_samples=lime_cfg.get("num_samples", 2000),
                    ),
                    row,
                    noise_std=0.02,
                    n_repeats=5,
                )
            )

        metrics_records.append({
            "method": "LIME",
            "deletion_auc": lime_id["deletion_auc"],
            "insertion_auc": lime_id["insertion_auc"],
            "sparsity": float(sparsity_from_importances(lime_importances)),
            "anchor_precision": np.nan,
            "anchor_coverage": np.nan,
            "anchor_rule_length": np.nan,
            "cf_validity_rate": np.nan,
            "cf_mean_l1_distance": np.nan,
            "cf_mean_l2_distance": np.nan,
            "cf_mean_sparsity": np.nan,
            "stability": float(np.mean(stability_scores)) if stability_scores else np.nan,
        })
        LOG.info(
            "LIME done: deletion_auc=%.4f insertion_auc=%.4f stability=%.3f",
            lime_id["deletion_auc"],
            lime_id["insertion_auc"],
            float(np.mean(stability_scores)) if stability_scores else float("nan"),
        )
    except Exception as exc:
        metrics_records.append({
            "method": "LIME",
            "error": str(exc),
        })

    # === ANCHOR ===
    try:
        def predict_fn_anchor_proc(np_arr):
            proba = model.predict_proba(np_arr)
            return (proba[:, 1] >= default_threshold).astype(int)

        anchor_explainer = AnchorExplainer(
            X_train_proc,
            feature_names=feature_names,
            predict_fn=predict_fn_anchor_proc,
            class_names=["No Default", "Default"],
        )
        anchor_subset_size = min(anchor_cfg.get("subset_size", 30), len(data["X_test_raw"]))
        LOG.info("Running Anchor on %d instances (threshold=%.2f)", anchor_subset_size, anchor_cfg.get("threshold", 0.9))
        anchor_indices = subset_idx[:anchor_subset_size]
        anchor_metrics = []
        for idx in anchor_indices:
            res = _safe_anchor_eval(
                anchor_explainer,
                X_test_proc[idx],
                predict_fn_anchor_proc,
                threshold=anchor_cfg.get("threshold", 0.9),
            )
            if res:
                anchor_metrics.append(res)

        if anchor_metrics:
            anchor_df = pd.DataFrame(anchor_metrics)
            metrics_records.append({
                "method": "ANCHOR",
                "deletion_auc": np.nan,
                "insertion_auc": np.nan,
                "sparsity": np.nan,
                "anchor_precision": anchor_df["precision"].mean(),
                "anchor_coverage": anchor_df["coverage"].mean(),
                "anchor_rule_length": anchor_df["n_conditions"].mean(),
                "cf_validity_rate": np.nan,
                "cf_mean_l1_distance": np.nan,
                "cf_mean_l2_distance": np.nan,
                "cf_mean_sparsity": np.nan,
                "stability": np.nan,
            })
            LOG.info(
                "Anchor done: precision=%.3f coverage=%.3f rule_len=%.2f",
                anchor_df["precision"].mean(),
                anchor_df["coverage"].mean(),
                anchor_df["n_conditions"].mean(),
            )
        else:
            metrics_records.append({"method": "ANCHOR", "error": "No anchor explanations generated"})
    except Exception as exc:
        metrics_records.append({
            "method": "ANCHOR",
            "error": str(exc),
        })

    # === DiCE ===
    try:
        dice_subset_size = min(dice_cfg.get("subset_size", 10), len(subset_idx))
        LOG.info("Running DiCE on %d instances (total_cfs=%d)", dice_subset_size, dice_cfg.get("total_cfs", 3))
        dice_explainer = DiceExplainer(
            model=model,
            X_train_processed=X_train_proc.astype(np.float64),
            y_train=data["y_train"],
            feature_names=feature_names,
            outcome_name=config.get("data", {}).get("target", "Default"),
            method="random",
        )
        dice_subset_size = min(dice_cfg.get("subset_size", 10), len(subset_idx))
        dice_indices = subset_idx[:dice_subset_size]

        cf_records = []
        for idx in dice_indices:
            x_row_proc = X_test_proc[idx].astype(np.float64)
            cf_exp = dice_explainer.generate_counterfactuals(
                x_row_proc,
                total_cfs=dice_cfg.get("total_cfs", 3),
            )
            cf_records.append(_dice_metrics(model, x_row_proc, cf_exp, feature_names))

        if cf_records:
            cf_df = pd.DataFrame(cf_records)
            metrics_records.append({
                "method": "DiCE",
                "deletion_auc": np.nan,
                "insertion_auc": np.nan,
                "sparsity": np.nan,
                "anchor_precision": np.nan,
                "anchor_coverage": np.nan,
                "anchor_rule_length": np.nan,
                "cf_validity_rate": cf_df["cf_validity_rate"].mean(),
                "cf_mean_l1_distance": cf_df["cf_mean_l1_distance"].mean(),
                "cf_mean_l2_distance": cf_df["cf_mean_l2_distance"].mean(),
                "cf_mean_sparsity": cf_df["cf_mean_sparsity"].mean(),
                "stability": np.nan,
            })
            LOG.info(
                "DiCE done: validity=%.3f l1=%.2f l2=%.2f sparsity=%.2f",
                cf_df["cf_validity_rate"].mean(),
                cf_df["cf_mean_l1_distance"].mean(),
                cf_df["cf_mean_l2_distance"].mean(),
                cf_df["cf_mean_sparsity"].mean(),
            )
        else:
            metrics_records.append({"method": "DiCE", "error": "No counterfactuals generated"})
    except Exception as exc:
        metrics_records.append({
            "method": "DiCE",
            "error": str(exc),
        })

    metrics_df = pd.DataFrame(metrics_records)
    metrics_path = metrics_dir / "technical_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    LOG.info("Saved technical metrics to %s", metrics_path)
    return {"metrics": metrics_df, "metrics_path": metrics_path}


def main():
    parser = argparse.ArgumentParser(description="Run VXAI-aligned technical evaluation.")
    parser.add_argument("--config", default="config/config_credit.yaml", help="Path to YAML config.")
    parser.add_argument("--model-path", default=None, help="Override model artifact path.")
    args = parser.parse_args()

    results = run_technical_eval(args.config, model_path=args.model_path)
    print(f"Saved metrics to: {results['metrics_path']}")
    print(results["metrics"])


if __name__ == "__main__":
    main()
