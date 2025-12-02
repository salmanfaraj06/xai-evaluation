# VXAI-Guided Evaluation Framework for Credit Risk XAI

*A modular implementation guide for our PoC*

## 1. Project Overview

This repository implements a **VXAI-guided evaluation pipeline** for a **credit risk (loan default) classifier**. The goal is to move from an anecdotal XAI demo to a **framework-driven evaluation design** that is:

* **Aligned with VXAI** (eValuation of Explainable AI) for technical metrics
* **Connected to human-centred evaluation constructs** (interpretability, task utility, trust calibration)
* **Designed to plug into governance-aware risk assessment** (AI risk & documentation)

The system:

* Trains a **credit risk prediction model** (e.g. XGBoost) on tabular loan data.
* Generates **local explanations** using:

  * SHAP (feature attributions)
  * LIME (local surrogate attributions)
  * ANCHOR (rule-based local surrogate)
  * DiCE (counterfactual recourse)
* Evaluates these explanations using **VXAI-aligned metrics**:

  * Fidelity (insertion/deletion AUC, etc.)
  * Coverage & parsimony (Anchor rule length, coverage, feature sparsity)
  * (Optional) robustness/consistency under perturbations
* Provides a **human-centred evaluation template** (loan officer personas + Likert-scale constructs).

This PoC is intended as a **case study** to demonstrate how a concrete XAI evaluation can be:

> *“Guided by a principled framework (VXAI) instead of ad hoc metric choices.”*

---

## 2. Conceptual Framework

### 2.1 VXAI Dimensions

We follow the **VXAI framework** in structuring our evaluation:

1. **Explanation Type**
   What kind of explanation are we evaluating?

   * Feature attributions: SHAP, LIME, ANCHOR rules (as sparse feature conditions)
   * Counterfactual explanations: DiCE recourse

2. **Evaluation Contextuality**
   How deeply does the metric interact with the model and data?

   * Level II–III (roughly): model + input interventions

     * e.g. insertion/deletion tests, perturbation-based fidelity

3. **Explanation Quality Desiderata**
   Which quality aspects are we measuring?

   * **Fidelity** – does the explanation reflect the model’s behaviour?
   * **Parsimony** – is the explanation simple/sparse?
   * **Coverage** – how broadly applicable is the explanation (e.g. anchor coverage)?
   * **(Optional) Robustness/Consistency** – stability under small input changes
   * We treat parsimony & plausibility as **interpretability-oriented**, and fidelity, continuity, consistency, coverage, efficiency as **technical**.

### 2.2 Multi-Layer Evaluation Perspective

Beyond VXAI, we embed the PoC in a **multi-layer evaluation pipeline**:

1. **Layer 1 – Model performance & risk**

   * Accuracy, AUC, calibration, robustness, class imbalance

2. **Layer 2 – Technical XAI metrics (VXAI)**

   * Fidelity, parsimony, coverage, (optionally) robustness

3. **Layer 3 – Human-centred evaluation (template only)**

   * Interpretability, perceived completeness, decision support, trust calibration

4. **Layer 4 – Governance & documentation (conceptual hooks)**

   * How this evaluation could feed into AI risk management (EU AI Act, NIST AI RMF, etc.)

The PoC **fully implements Layers 1–2**, provides a **template for Layer 3**, and **discusses** Layer 4.

---

## 3. Repository Structure

A suggested modular structure:

```text
credit-risk-xai-vxai-poc/
│
├─ README_VXAI_CreditRisk.md       # This document
├─ requirements.txt                # Python dependencies
├─ config/
│   └─ config_credit.yaml          # Data paths, model params, VXAI eval plan
│
├─ data/
│   ├─ raw/                        # Original CSV(s)
│   └─ processed/                  # Cleaned/encoded datasets
│
├─ notebooks/
│   └─ exploration.ipynb           # Optional EDA / scratch
│
├─ src/
│   ├─ data_loading.py             # Load, clean, split data
│   ├─ preprocessing.py            # Encoding, scaling, feature engineering
│   ├─ models.py                   # Model train/save/load
│   │
│   ├─ explainers/
│   │   ├─ shap_explainer.py       # SHAP wrapper
│   │   ├─ lime_explainer.py       # LIME wrapper
│   │   ├─ anchor_explainer.py     # ANCHOR wrapper
│   │   └─ dice_counterfactuals.py # DiCE wrapper
│   │
│   ├─ metrics/
│   │   ├─ fidelity.py             # insertion/deletion AUC, etc.
│   │   ├─ parsimony_coverage.py   # sparsity, rule length, precision, coverage
│   │   └─ robustness.py           # (optional) stability metrics
│   │
│   ├─ evaluation/
│   │   ├─ vxai_eval_plan.py       # VXAI-aligned config for methods & metrics
│   │   ├─ run_technical_eval.py   # Main script to run Layer 2 metrics
│   │   └─ run_human_proxy_eval.py # Template generator for Layer 3 prompts
│   │
│   └─ reporting/
│       ├─ summary_tables.py       # Combine metric results into tables
│       └─ plots.py                # Optional plots (bar charts, etc.)
│
└─ scripts/
    ├─ train_model.py              # CLI entry: train & save model
    ├─ evaluate_vxai.py            # CLI entry: run technical VXAI metrics
    └─ generate_human_template.py  # CLI entry: produce human evaluation task sheets
```

You do **not** have to exactly match this, but the idea is:

* **Each concern = its own module**
* **VXAI logic = explicit in config + evaluation modules**
* Easy to show a reviewer: “here is where we align with VXAI”.

---

## 4. Data & Modelling Workflow

### 4.1 Data loading (`data_loading.py`)

Responsibilities:

* Load raw CSV (e.g. `credit_risk.csv`).
* Drop/rename obvious ID columns (e.g. `LoanID`).
* Handle missing values & basic cleaning.
* Split into train/validation/test sets (with proper random seed).

Key function:

```python
def load_credit_data(config) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load and split the credit risk dataset according to config.
    Returns: X_train, X_test, y_train, y_test.
    """
```

### 4.2 Preprocessing (`preprocessing.py`)

Responsibilities:

* Identify numeric, categorical, and binary columns.
* One-hot/ordinal encode categorical features.
* Ensure binary columns are 0/1.
* Optionally scale numeric features.

Key function:

```python
def build_preprocessor(config, X: pd.DataFrame):
    """
    Fit preprocessing pipeline (encoders/scalers) on X.
    Returns a fitted transformer and list of feature names.
    """
```

This is where you **freeze your feature space** for XAI.

### 4.3 Model training (`models.py` + `train_model.py`)

Responsibilities:

* Train classifier (e.g. XGBoost, LightGBM, or RandomForest).
* Evaluate standard metrics (accuracy, F1, AUC).
* Save trained model + preprocessing pipeline to disk.

Key functions:

```python
def train_credit_model(config, X_train, y_train):
    """
    Train the chosen model type using config hyperparameters.
    Returns the fitted model.
    """

def evaluate_model_performance(model, X_test, y_test):
    """
    Compute Layer-1 metrics (accuracy, AUC, etc.).
    """
```

Command-line script example (`scripts/train_model.py`):

```bash
python scripts/train_model.py --config config/config_credit.yaml
```

---

## 5. Explanation Methods (src/explainers/)

Each explainer gets its own thin wrapper so you can swap and compare methods easily.

### 5.1 SHAP (`shap_explainer.py`)

```python
class ShapExplainer:
    def __init__(self, model, feature_names, config):
        """
        Initialise SHAP explainer (e.g. TreeExplainer or KernelExplainer).
        """
    def explain_instance(self, x: np.ndarray) -> np.ndarray:
        """
        Return SHAP values for a single instance as a 1D array.
        """
```

### 5.2 LIME (`lime_explainer.py`)

```python
class LimeExplainer:
    def __init__(self, model, feature_names, config):
        """
        Initialise a LIME TabularExplainer.
        """
    def explain_instance(self, x: np.ndarray) -> np.ndarray:
        """
        Return local feature importance weights for a single instance.
        """
```

### 5.3 ANCHOR (`anchor_explainer.py`)

```python
class AnchorExplainer:
    def __init__(self, model, feature_names, config):
        """
        Initialise AnchorTabular explainer.
        """
    def explain_instance(self, x: np.ndarray):
        """
        Return Anchor explanation object with precision, coverage, and rule conditions.
        """
```

### 5.4 DiCE (`dice_counterfactuals.py`)

```python
class DiceExplainer:
    def __init__(self, model, preprocessor, feature_metadata, config):
        """
        Wrap DiCE for counterfactual generation.
        """
    def generate_counterfactuals(self, x: pd.DataFrame, n: int = 5):
        """
        Return n valid counterfactual examples for given instance x.
        """
```

---

## 6. VXAI-Guided Evaluation Design (`evaluation/`)

### 6.1 VXAI Evaluation Plan (`vxai_eval_plan.py`)

Here we define **in code** how each explainer maps to VXAI dimensions.

Example:

```python
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
        "desiderata": ["fidelity", "parsimony"],
        "metrics": ["deletion_auc", "insertion_auc", "sparsity"],
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
        "metrics": ["cf_validity", "cf_proximity", "cf_sparsity"],  # optional
    },
}
```

This **replaces anecdotal choices** with a clear framework.

### 6.2 Fidelity metrics (`metrics/fidelity.py`)

Implement metrics like **deletion/insertion AUC**:

```python
def insertion_deletion_auc(
    model, X_test, feature_importances, baseline="zeros", steps=50
):
    """
    Compute insertion and deletion curves + AUC for a set of instances.

    VXAI mapping:
    - Desideratum: fidelity
    - Contextuality: Level II-III (input + model interventions)
    """
```

Used for SHAP & LIME explanations.

### 6.3 Parsimony & coverage metrics (`metrics/parsimony_coverage.py`)

```python
def anchor_parsimony_and_coverage(anchor_exp):
    """
    Returns dict with:
    - n_conditions: number of feature conditions in the rule (parsimony)
    - precision: empirical accuracy when rule fires (coverage+fidelity)
    - coverage: proportion of dataset where rule applies (coverage)
    """

def sparsity_from_importances(importances: np.ndarray, threshold=0.0):
    """
    Count number of non-zero (or above threshold) feature attributions.
    Lower = more parsimonious.
    """
```

### 6.4 Robustness / consistency (optional) (`metrics/robustness.py`)

```python
def explanation_stability(explainer, X_test, noise_std=0.01, n_repeats=5):
    """
    Approximate explanation robustness by measuring cosine similarity
    between original and perturbed explanations.

    VXAI mapping:
    - Desideratum: consistency/continuity
    - Contextuality: Level III (input perturbations)
    """
```

### 6.5 Running technical evaluation (`run_technical_eval.py`)

Main responsibilities:

* Load model + data.
* Instantiate explainers.
* For a subset of test instances:

  * Generate explanations per method.
  * Compute VXAI metrics via `metrics/` functions.
* Aggregate into summary tables (mean ± std, etc.).
* Save results as CSV/JSON for reporting.

CLI example:

```bash
python scripts/evaluate_vxai.py --config config/config_credit.yaml
```

---

## 7. Human-Centred Evaluation Template (`run_human_proxy_eval.py`)

We **design**, but do not fully run, a human study aligned to Doshi-Velez & Kim + your constructs.

### 7.1 Target constructs

We define 4 constructs:

* **Interpretability** – how easy is it to understand the explanation?
* **Perceived completeness** – does it cover the important reasons?
* **Decision support / task utility** – does it help make a better loan decision?
* **Trust calibration / appropriate reliance** – does it help know when to follow or override the model?

### 7.2 Generated template (persona + rating form)

`run_human_proxy_eval.py` can output a **text or Markdown file** describing:

* A **loan officer persona** (experience, risk tolerance).
* A small set of **loan cases**, each with:

  * The model decision + probability
  * Explanation from SHAP, LIME, ANCHOR, or DiCE
* A rating grid:

| Construct         | Question example                                                 | 1 (low) – 5 (high) |
| ----------------- | ---------------------------------------------------------------- | ------------------ |
| Interpretability  | How easy was this explanation to understand?                     | 1–5                |
| Completeness      | How completely does this explanation cover the decision factors? | 1–5                |
| Decision support  | How much did this help you decide on the loan outcome?           | 1–5                |
| Trust calibration | How much does this help you tell when to trust the model?        | 1–5                |

The script only **generates the materials**; a real user study would be future work.

---

## 8. Reporting & Visualisation (`reporting/`)

### 8.1 Summary tables (`summary_tables.py`)

* Combine VXAI metrics per method into one table:

| Method | Deletion AUC | Insertion AUC | Sparsity | Anchor precision | Anchor coverage | … |
| ------ | ------------ | ------------- | -------- | ---------------- | --------------- | - |

### 8.2 Plots (`plots.py`, optional)

* Simple bar plots comparing SHAP vs LIME vs ANCHOR on:

  * Fidelity metrics
  * Parsimony (sparsity, rule length)
  * Coverage (for ANCHOR)

These can be used in thesis / slides.

---

## 9. How to Run the Pipeline (Example)

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Train model**

```bash
python scripts/train_model.py --config config/config_credit.yaml
```

3. **Run VXAI technical evaluation**

```bash
python scripts/evaluate_vxai.py --config config/config_credit.yaml
```

4. **Generate human evaluation templates**

```bash
python scripts/generate_human_template.py --config config/config_credit.yaml
```

5. **Inspect outputs**

* `outputs/vxai_metrics/` – CSV/JSON metric summaries
* `outputs/human_eval_templates/` – markdown/PDF of human study prompts
* `outputs/plots/` – optional figures

---

## 10. Positioning in the Thesis / Paper

In your thesis/paper, you can now say:

> “Our credit risk PoC implements a VXAI-aligned evaluation pipeline. We consider feature-attribution and counterfactual explanation types, evaluate them using functionality-grounded fidelity, parsimony and coverage metrics at contextuality levels II–III, and design a human-grounded evaluation template for interpretability, completeness, decision support and trust calibration. The codebase is modularised so that each explainer and metric is explicitly mapped to VXAI desiderata and contextuality assumptions.”

This Markdown file is your **bridge** between:

* the **conceptual review** (five phases, VXAI, MXAI, human-centred & governance), and
* the **practical PoC** (credit risk model + code).

You can now refactor your existing `financial_credit_risk_xai_evaluation.ipynb` into this structure step by step, without losing any of the work you’ve already done.
