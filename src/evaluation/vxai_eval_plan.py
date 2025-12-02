"""VXAI evaluation mapping for this PoC."""

VXAI_EVAL_PLAN = {
    "SHAP": {
        "explanation_type": "feature_attribution",
        "vxai_contextuality_level": "II-III",
        "desiderata": ["fidelity", "parsimony"],
        "metrics": ["deletion_auc", "insertion_auc", "sparsity"],
    },
    "LIME": {
        "explanation_type": "feature_attribution",
        "vxai_contextuality_level": "II-III",
        "desiderata": ["fidelity", "parsimony", "robustness"],
        "metrics": ["deletion_auc", "insertion_auc", "sparsity", "stability"],
    },
    "ANCHOR": {
        "explanation_type": "rule_based_local_surrogate",
        "vxai_contextuality_level": "II",
        "desiderata": ["fidelity", "coverage", "parsimony"],
        "metrics": ["anchor_precision", "anchor_coverage", "n_conditions"],
    },
    "DiCE": {
        "explanation_type": "counterfactual",
        "vxai_contextuality_level": "II-III",
        "desiderata": ["fidelity", "parsimony", "plausibility"],
        "metrics": ["cf_validity", "cf_proximity", "cf_sparsity"],
    },
}
