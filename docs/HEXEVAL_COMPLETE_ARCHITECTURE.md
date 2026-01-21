# HEXEval - Complete Framework Architecture & Pipeline

**The Single Source of Truth for the HEXEval Framework**

**Last Updated:** 2026-01-21  
**Version:** 2.1  
**Author:** Salman Faraj  
**Project Type:** Final Year Research Project (FYP)  
**License:** MIT

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Research Context & Motivation](#research-context--motivation)
3. [Installation & Setup](#installation--setup)
4. [Project Structure](#project-structure)
5. [System Architecture](#system-architecture)
6. [Complete Data Flow](#complete-data-flow)
7. [Module-by-Module Breakdown](#module-by-module-breakdown)
8. [Evaluation Pipeline Details](#evaluation-pipeline-details)
9. [Configuration System](#configuration-system)
10. [Persona System](#persona-system)
11. [Streamlit UI](#streamlit-ui)
12. [CLI Usage](#cli-usage)
13. [API Reference](#api-reference)
14. [Key Algorithms](#key-algorithms)
15. [Usage Examples](#usage-examples)
16. [Performance & Scalability](#performance--scalability)
17. [Limitations & Constraints](#limitations--constraints)
18. [Testing Strategy](#testing-strategy)
19. [Troubleshooting](#troubleshooting)
20. [Design Decisions](#design-decisions)
21. [Research Contributions](#research-contributions)
22. [Related Work](#related-work)
23. [Future Extensions](#future-extensions)
24. [Citation & Academic Use](#citation--academic-use)

---

## Executive Summary

**HEXEval** (Holistic Explanation Evaluation) is a production-grade framework for evaluating explainable AI (XAI) methods on tabular classification models. It combines:

- **Technical Metrics:** Fidelity (insertion/deletion AUC), parsimony (sparsity), stability
- **Human-Centered Evaluation:** LLM-simulated personas rating explanations on 6 dimensions
- **Domain-Agnostic Design:** Configurable for any ML prediction task via YAML config
- **4 XAI Methods:** SHAP, LIME, Anchor, DiCE counterfactuals

**Key Innovation:** Dual evaluation (technical + persona-based) reveals the fidelity-interpretability gap—methods with high technical accuracy may still be unusable for stakeholders.

### Codebase Statistics

| Module | Files | Lines | Purpose |
|--------|-------|-------|---------|
| Core (`hexeval/core/`) | 5 | ~701 | Model loading, data handling, validation, wrapper |
| Explainers (`hexeval/explainers/`) | 5 | ~209 | SHAP, LIME, Anchor, DiCE wrappers |
| Metrics (`hexeval/metrics/`) | 4 | ~94 | Fidelity, parsimony, robustness |
| Evaluation (`hexeval/evaluation/`) | 7 | ~1,523 | Technical + Persona evaluation, recommendations |
| UI (`hexeval/ui/`) | 2 | ~574 | Streamlit interface |
| Config (`hexeval/config/`) | 4 | ~402 | YAML configurations + persona files |
| Scripts (`scripts/`) | 2 | ~180 | CLI interface, model training utilities |
| **Total** | **29** | **~3,683** | |

### Research Problem Statement

Traditional XAI evaluation focuses solely on technical metrics (fidelity, stability), assuming that mathematically sound explanations automatically translate to human understanding. However, real-world deployment reveals a critical gap: **explanations that score well on technical metrics often fail to meet stakeholder needs** in terms of interpretability, actionability, and trust.

**Research Question:** How can we systematically evaluate XAI methods from both technical rigor and human-centered perspectives to identify the best explanation method for specific stakeholder groups?

**Hypothesis:** A dual evaluation framework combining technical metrics with LLM-simulated stakeholder personas will reveal method-stakeholder fit patterns that pure technical evaluation cannot capture.

---

## Research Context & Motivation

### The Explainability Gap

As machine learning models are deployed in high-stakes domains (healthcare, finance, legal), the need for explainability has become critical. However, there exists a fundamental disconnect:

1. **Technical Excellence ≠ Human Usability**: Methods like SHAP achieve high fidelity scores (0.11 deletion AUC) but receive poor stakeholder ratings (1.9-2.4/5 trust).

2. **One-Size-Fits-All Fails**: Different stakeholders (loan officers, data analysts, end-users) have vastly different needs. A single explanation format cannot satisfy all.

3. **Evaluation Gap**: Current evaluation frameworks focus on technical metrics, ignoring human factors that determine real-world adoption.

### Research Objectives

1. **Develop a holistic evaluation framework** that combines technical metrics with human-centered assessment
2. **Quantify the fidelity-interpretability trade-off** across multiple XAI methods
3. **Enable stakeholder-specific method selection** through persona-based evaluation
4. **Provide actionable recommendations** for practitioners deploying XAI systems

### Target Domains

- **Healthcare**: Heart disease prediction, patient risk assessment
- **Finance**: Credit risk evaluation, loan approval systems
- **General**: Any binary classification task on tabular data

### Key Findings (Preliminary)

From initial evaluations:
- **Technical metrics are strong**: Fidelity AUC 0.11-0.13, Anchor precision 94.9%
- **Human ratings are poor**: Average trust 2.1/5, actionability 1.3-1.7/5
- **Persona differentiation exists**: 2.5-point variance in ratings across stakeholder types
- **No method excels universally**: Each method has strengths for specific personas

---

## Installation & Setup

### Prerequisites

- **Python**: 3.8 or higher
- **Operating System**: macOS, Linux, or Windows (WSL recommended)
- **Memory**: Minimum 4GB RAM (8GB+ recommended for large datasets)
- **Storage**: ~500MB for dependencies

### Installation Steps

#### 1. Clone or Download the Repository

```bash
cd /path/to/your/project
# Repository should contain hexeval/ directory
```

#### 2. Create Virtual Environment (Recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
# OR
pip install -e .  # If installing as package
```

#### 4. Verify Installation

```bash
python -c "from hexeval import evaluate; print('✓ HEXEval installed successfully')"
```

### Optional: OpenAI API Key Setup

For persona evaluation (LLM-based), set your OpenAI API key:

```bash
# Option 1: Environment variable
export OPENAI_API_KEY="your-key-here"

# Option 2: .env file (recommended)
echo "OPENAI_API_KEY=your-key-here" > .env
```

**Note:** Persona evaluation is optional. Technical evaluation works without API key.

### Quick Test

```bash
# Test with sample data
python scripts/hexeval_cli.py validate \
    usecases/heart_disease_pipeline.pkl \
    usecases/heart.csv \
    --target target
```

### Common Installation Issues

**Issue:** `anchor-exp` installation fails  
**Solution:** Install from source: `pip install git+https://github.com/marcotcr/anchor.git`

**Issue:** `dice-ml` conflicts with other packages  
**Solution:** Use Python 3.8-3.10, or install in isolated environment

**Issue:** SHAP requires specific NumPy version  
**Solution:** `pip install numpy==1.23.5 shap==0.42.0`

---

## Project Structure

```
CODE/
├── hexeval/                          # Main package
│   ├── __init__.py                   # Package exports (evaluate, load_model, load_data)
│   │
│   ├── core/                         # Layer 1: Infrastructure
│   │   ├── __init__.py               # Core exports
│   │   ├── model_loader.py           # Load sklearn/XGBoost models (135 lines)
│   │   ├── data_handler.py           # Load CSV, auto-detect types (189 lines)
│   │   ├── validator.py              # Model-data compatibility checks (176 lines)
│   │   └── wrapper.py                # ModelWrapper class for consistent interface (201 lines)
│   │
│   ├── explainers/                   # Layer 2: XAI Method Wrappers
│   │   ├── __init__.py
│   │   ├── shap_explainer.py         # Kernel SHAP wrapper (51 lines)
│   │   ├── lime_explainer.py         # LIME tabular wrapper (60 lines)
│   │   ├── anchor_explainer.py       # Anchor rules wrapper (38 lines)
│   │   └── dice_counterfactuals.py   # DiCE CF generation (60 lines)
│   │
│   ├── metrics/                      # Technical Metric Functions
│   │   ├── __init__.py
│   │   ├── fidelity.py               # Insertion/deletion AUC
│   │   ├── parsimony_coverage.py     # Sparsity, anchor coverage
│   │   └── robustness.py             # Stability under noise
│   │
│   ├── evaluation/                   # Layer 3: Evaluation Orchestration
│   │   ├── __init__.py               # Exports evaluators
│   │   ├── evaluator.py              # Main orchestrator - entry point (186 lines)
│   │   ├── technical_evaluator.py    # Runs all XAI methods + metrics (325 lines)
│   │   ├── persona_evaluator.py      # LLM-based persona simulation (503 lines)
│   │   ├── personas.py               # Persona loading utilities (43 lines)
│   │   ├── personas_legacy.py        # Legacy embedded personas (deprecated)
│   │   └── recommender.py            # Stakeholder-specific recommendations (269 lines)
│   │
│   ├── config/                       # YAML Configuration Files
│   │   ├── eval_config.yaml          # Default config (Heart Disease) (76 lines)
│   │   ├── eval_config_credit_risk.yaml  # Credit Risk domain config (107 lines)
│   │   ├── personas_healthcare.yaml  # 4 healthcare personas (88 lines)
│   │   └── personas_credit_risk.yaml # 6 credit risk personas (131 lines)
│   │
│   ├── ui/                           # Streamlit Interface
│   │   ├── __init__.py
│   │   └── app.py                    # 5-tab UI (574 lines)
│   │
│   └── reports/                      # (Reserved for future report generation)
│       └── __init__.py
│
├── usecases/                         # Sample Data & Models
│   ├── heart.csv                     # Heart disease dataset (303 samples)
│   ├── heart_disease_pipeline.pkl    # Trained heart disease model
│   ├── heart_disease_prediction.ipynb  # Training notebook
│   ├── credit_risk_dataset.csv       # Credit risk dataset (32,581 samples)
│   └── xgboost_credit_risk_new.pkl   # Trained XGBoost model
│
├── docs/                             # Documentation
│   ├── HEXEVAL_COMPLETE_ARCHITECTURE.md  # This file
│   ├── HEXEval_Prerequisites.md      # Setup guide
│   ├── HEXEval_HowItWorks.md         # Conceptual overview
│   ├── HEXEval_Configuration.md      # Config reference
│   └── ... (additional docs)
│
├── outputs/                          # Evaluation Results (generated)
│   ├── hexeval_results/              # Default output directory
│   ├── heart_disease/                # Heart disease results
│   └── credit_risk/                  # Credit risk results
│
├── scripts/                          # Utility scripts
├── pyproject.toml                    # Package definition
├── requirements.txt                  # Dependencies
└── README_HEXEVAL.md                 # Quick start guide
```

---

## System Architecture

### High-Level Architecture

```
+-------------------------------------------------------------------------+
|                         HEXEval Framework                               |
+-------------------------------------------------------------------------+
|                                                                         |
|  +------------------+     +------------------+     +------------------+ |
|  |   CORE LAYER     |---->|   EXPLAINERS     |---->|   EVALUATION     | |
|  | (Infrastructure) |     |  (XAI Methods)   |     |  (Metrics+LLM)   | |
|  +------------------+     +------------------+     +------------------+ |
|          |                                                   |          |
|          |            +----------------------+               |          |
|          +----------->|     CONFIG (YAML)    |<--------------+          |
|                       +----------------------+                          |
|                                                                         |
|  +-------------------------------------------------------------------+  |
|  |                        STREAMLIT UI                               |  |
|  |  [Configuration] [Use Case Details] [Results] [Recommendations]   |  |
|  +-------------------------------------------------------------------+  |
|                                                                         |
+-------------------------------------------------------------------------+
```

### 3-Layer Design

**Layer 1: Core Infrastructure** (`hexeval/core/`)
- `model_loader.py` - Load sklearn/XGBoost models, return `ModelWrapper`
- `data_handler.py` - Load CSV, auto-detect feature types, train/test split
- `validator.py` - Model-data compatibility checks
- `wrapper.py` - `ModelWrapper` class providing consistent interface

**Layer 2: Explainers** (`hexeval/explainers/`)
- `shap_explainer.py` - Shapley value attribution
- `lime_explainer.py` - Local linear approximations
- `anchor_explainer.py` - Rule-based IF-THEN explanations
- `dice_counterfactuals.py` - Counterfactual generation

**Layer 3: Evaluation** (`hexeval/evaluation/`)
- `evaluator.py` - Main orchestrator (entry point via `evaluate()`)
- `technical_evaluator.py` - Fidelity, parsimony, stability metrics
- `persona_evaluator.py` - LLM-based human simulation
- `personas.py` - Load personas from external YAML files
- `recommender.py` - Stakeholder-specific recommendations

---

## Complete Data Flow

### End-to-End Pipeline

```
+--------------------------------------------------------------------------+
| 1. INPUT                                                                 |
+--------------------------------------------------------------------------+
|  > Model (.pkl/.joblib)    > Data (CSV)    > Config (YAML)               |
|  > Target Column           > OpenAI API Key (optional)                   |
+--------------------------------------------------------------------------+
                                   |
                                   v
+--------------------------------------------------------------------------+
| 2. LOAD & VALIDATE (hexeval/core/)                                       |
+--------------------------------------------------------------------------+
|                                                                          |
|  model_loader.py -> load_model(path)                                     |
|    - Load model artifact with joblib                                     |
|    - Extract: model, preprocessor, feature_names, threshold              |
|    - Wrap in ModelWrapper for consistent interface                       |
|    - Validate: has predict_proba()                                       |
|                                                                          |
|  data_handler.py -> load_data(path, target_column)                       |
|    - Load CSV -> DataFrame                                               |
|    - Auto-detect: categorical vs numeric features                        |
|    - Stratified train/test split (80/20)                                 |
|    - Return: X_train, X_test, y_train, y_test, metadata                  |
|                                                                          |
|  validator.py -> validate_model_data_compatibility(wrapper, data)        |
|    - Check feature compatibility                                         |
|    - Test prediction: wrapper.predict_proba(X_sample)                    |
|    - Status: valid/invalid + warnings/errors                             |
|                                                                          |
+--------------------------------------------------------------------------+
                                   |
                                   v
+--------------------------------------------------------------------------+
| 3. TECHNICAL EVALUATION (hexeval/evaluation/technical_evaluator.py)      |
+--------------------------------------------------------------------------+
|                                                                          |
|  FOR EACH ENABLED METHOD (SHAP, LIME, Anchor, DiCE):                     |
|                                                                          |
|  +--------------------------------------------------------------------+  |
|  | SHAP                                                               |  |
|  +--------------------------------------------------------------------+  |
|  | 1. Create Explainer(model, background)                             |  |
|  | 2. Generate SHAP values for sample_size instances                  |  |
|  | 3. Compute Fidelity:                                               |  |
|  |    - Deletion AUC (remove important features, measure drop)        |  |
|  |    - Insertion AUC (add important features, measure rise)          |  |
|  | 4. Compute Parsimony: Sparsity (avg # important features)          |  |
|  +--------------------------------------------------------------------+  |
|                                                                          |
|  +--------------------------------------------------------------------+  |
|  | LIME                                                               |  |
|  +--------------------------------------------------------------------+  |
|  | 1. Create LimeTabularExplainer                                     |  |
|  | 2. Generate explanations (num_samples perturbations)               |  |
|  | 3. Compute Fidelity (same as SHAP)                                 |  |
|  | 4. Compute Stability: Add noise, measure variance                  |  |
|  +--------------------------------------------------------------------+  |
|                                                                          |
|  +--------------------------------------------------------------------+  |
|  | Anchor                                                             |  |
|  +--------------------------------------------------------------------+  |
|  | 1. Create AnchorTabularExplainer                                   |  |
|  | 2. Generate rules for max_instances samples                        |  |
|  | 3. Compute: precision (rule accuracy), coverage, n_conditions      |  |
|  +--------------------------------------------------------------------+  |
|                                                                          |
|  +--------------------------------------------------------------------+  |
|  | DiCE                                                               |  |
|  +--------------------------------------------------------------------+  |
|  | 1. Create DiCE explainer in processed feature space                |  |
|  | 2. Generate counterfactuals for max_instances samples              |  |
|  | 3. Compute: success_rate (% valid CFs that flip prediction)        |  |
|  +--------------------------------------------------------------------+  |
|                                                                          |
|  OUTPUT: DataFrame -> technical_metrics.csv                              |
|  Columns: method, fidelity_deletion, fidelity_insertion, sparsity, etc.  |
|                                                                          |
+--------------------------------------------------------------------------+
                                   |
                                   v
+--------------------------------------------------------------------------+
| 4. PERSONA EVALUATION (hexeval/evaluation/persona_evaluator.py)          |
+--------------------------------------------------------------------------+
|                                                                          |
|  Step 1: Load Personas from YAML file (personas.yaml via personas.py)    |
|                                                                          |
|  Step 2: Generate Explanations (sample_instances x 4 methods)            |
|  +----------------------------------------------------------+            |
|  | Instance #0:                                             |            |
|  |   - SHAP: "Top SHAP values: age: 0.23, thalach: -0.15"   |            |
|  |   - LIME: "Top LIME: age: 0.19, cp: 0.05"                |            |
|  |   - Anchor: "IF oldpeak > 0.15 AND thal = 2..."          |            |
|  |   - DiCE: "To change outcome: reduce oldpeak by 0.5"     |            |
|  +----------------------------------------------------------+            |
|                                                                          |
|  Step 3: LLM Evaluation Loop                                             |
|  Total API calls = personas x methods x instances x runs                 |
|                                                                          |
|  FOR EACH PERSONA:                                                       |
|  +----------------------------------------------------------+            |
|  | 1. Build System Prompt (rich persona context):           |            |
|  |    - Persona identity (name, role, experience)           |            |
|  |    - Risk profile, decision style, AI comfort            |            |
|  |    - Mental model (how they think about decisions)       |            |
|  |    - Heuristics (decision rules they follow)             |            |
|  |    - Explanation preferences                             |            |
|  |    - Domain context (from config.yaml)                   |            |
|  |                                                          |            |
|  | 2. Build Evaluation Prompt:                              |            |
|  |    - Scenario: "Case #0 - Model prediction: High Risk"   |            |
|  |    - Show explanation text                               |            |
|  |    - Ask: Rate on 6 dimensions (1-5)                     |            |
|  |                                                          |            |
|  | 3. Call OpenAI API:                                      |            |
|  |    - Model: gpt-4o (configurable)                        |            |
|  |    - Parse TOML response                                 |            |
|  |    - Extract: ratings + comment                          |            |
|  +----------------------------------------------------------+            |
|                                                                          |
|  6 Rating Dimensions:                                                    |
|    - interpretability (can you understand it?)                           |
|    - completeness (covers all factors?)                                  |
|    - actionability (what to do next?)                                    |
|    - trust (rely on this?)                                               |
|    - satisfaction (overall quality?)                                     |
|    - decision_support (helps your job?)                                  |
|                                                                          |
|  OUTPUT: DataFrame -> persona_ratings.csv                                |
|  Columns: persona_name, persona_role, explanation_type, instance_index,  |
|           ratings (6 dimensions), comment, raw_llm_response              |
|                                                                          |
+--------------------------------------------------------------------------+
                                   |
                                   v
+--------------------------------------------------------------------------+
| 5. RECOMMENDATIONS (hexeval/evaluation/recommender.py)                   |
+--------------------------------------------------------------------------+
|                                                                          |
|  FOR EACH UNIQUE PERSONA ROLE:                                           |
|                                                                          |
|  1. Get persona ratings for all methods                                  |
|  2. Get technical metrics for all methods                                |
|  3. Calculate Method-Specific Technical Score:                           |
|     - SHAP/LIME: normalize(fidelity + parsimony)                         |
|     - Anchor: 0.8 x precision + 0.2 x coverage                           |
|     - DiCE: success_rate                                                 |
|                                                                          |
|  4. Calculate Combined Score:                                            |
|     score = 0.3 x technical_score + 0.2 x parsimony +                    |
|             0.3 x (trust/5) + 0.2 x (satisfaction/5)                     |
|                                                                          |
|  5. Select Best Method: argmax(score)                                    |
|                                                                          |
|  6. Generate Reasoning:                                                  |
|     "SHAP recommended: high trust (3.5/5), excellent fidelity"           |
|                                                                          |
|  OUTPUT: JSON -> recommendations.json                                    |
|  Structure: {stakeholder: {method, score, reasoning, alternatives}}      |
|                                                                          |
+--------------------------------------------------------------------------+
                                   |
                                   v
+--------------------------------------------------------------------------+
| 6. OUTPUTS                                                               |
+--------------------------------------------------------------------------+
|  outputs/{use_case}/                                                     |
|    - technical_metrics.csv                                               |
|    - persona_ratings.csv                                                 |
|    - recommendations.json                                                |
+--------------------------------------------------------------------------+
```

#### Step 2: Load & Validate Details

**model_loader.py** → `load_model(path)`
- Load model artifact with joblib
- Extract: model, preprocessor, feature_names, threshold
- Wrap in ModelWrapper for consistent interface
- Validate: has predict_proba()

**data_handler.py** → `load_data(path, target_column)`
- Load CSV → DataFrame
- Auto-detect: categorical vs numeric features
- Stratified train/test split (80/20)
- Return: X_train, X_test, y_train, y_test, metadata

**validator.py** → `validate_model_data_compatibility(wrapper, data)`
- Check feature compatibility
- Test prediction: wrapper.predict_proba(X_sample)
- Status: valid/invalid + warnings/errors

#### Step 3: Technical Evaluation Details

| Method | Steps | Metrics |
|--------|-------|---------|
| SHAP | Create Explainer, generate values | fidelity_deletion, fidelity_insertion, sparsity |
| LIME | Create LimeTabularExplainer | fidelity, stability |
| Anchor | Create AnchorTabularExplainer | precision, coverage, n_conditions |
| DiCE | Generate counterfactuals | success_rate |

#### Step 4: Persona Evaluation Details

1. **Load Personas** from YAML file
2. **Generate Explanations** for sample instances × 4 methods
3. **LLM Evaluation Loop**: For each persona, build prompts and call OpenAI API

**6 Rating Dimensions:**
- interpretability, completeness, actionability
- trust, satisfaction, decision_support

#### Step 5: Recommendations Details

For each unique persona role:
1. Get persona ratings for all methods
2. Get technical metrics for all methods
3. Calculate combined score: `0.3×technical + 0.2×parsimony + 0.3×trust + 0.2×satisfaction`
4. Select best method: `argmax(score)`
5. Generate reasoning

---

## Module-by-Module Breakdown

### 1. Core Modules (`hexeval/core/`)

#### `model_loader.py` (135 lines)

**Purpose:** Load trained ML models from disk and wrap them in `ModelWrapper`

**Key Function:**
```python
def load_model(path: str | Path) -> ModelWrapper:
    """
    Load a trained model from disk and wrap it.
    
    Supports:
    - sklearn models (.pkl, .joblib)
    - XGBoost models
    - Model artifacts (dict with 'model', 'preprocessor', 'feature_names')
    
    Returns ModelWrapper with consistent interface.
    """
```

---

#### `data_handler.py` (189 lines)

**Purpose:** Load and prepare tabular data with automatic type detection

**Key Function:**
```python
def load_data(
    path: str | Path,
    target_column: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict:
    """
    Returns:
    {
        "X_train": DataFrame,
        "X_test": DataFrame,
        "y_train": Series,
        "y_test": Series,
        "feature_names": ["age", "income", ...],
        "categorical_features": ["job", "region"],
        "numeric_features": ["age", "income", "debt"]
    }
    """
```

---

#### `wrapper.py` (201 lines)

**Purpose:** Provide consistent interface for any sklearn-compatible model

**Key Class:**
```python
class ModelWrapper(BaseModelWrapper):
    """
    Standard wrapper for sklearn-compatible models.
    
    Handles:
    - Preprocessing (if provided)
    - Input conversion (pandas → numpy)
    - Metadata access (feature names, model type)
    
    Methods:
    - predict(X) → np.ndarray
    - predict_proba(X) → np.ndarray
    - get_feature_importance() → Dict[str, float]
    - get_model_info() → Dict[str, Any]
    """
```

---

#### `validator.py` (176 lines)

**Purpose:** Ensure model and data are compatible before evaluation

**Key Function:**
```python
def validate_model_data_compatibility(model_wrapper, data) -> Dict:
    """
    Returns:
    {
        "status": "valid" | "invalid",
        "warnings": ["Extra features in data: ['col1', 'col2']"],
        "errors": ["Missing required features: ['col3']"]
    }
    """
```

---

### 2. Explainer Modules (`hexeval/explainers/`)

#### `shap_explainer.py` (51 lines)

```python
class ShapExplainer:
    def __init__(self, model, background: np.ndarray, feature_names, class_index=1):
        """Uses shap.Explainer with automatic fallback to predict_proba."""
    
    def explain_instance(self, x_row: np.ndarray) -> np.ndarray:
        """Returns SHAP values for single instance. Shape: (n_features,)"""
    
    def explain_dataset(self, X: np.ndarray) -> np.ndarray:
        """Returns SHAP values for batch. Shape: (n_samples, n_features)"""
```

---

#### `lime_explainer.py` (60 lines)

```python
class LimeExplainer:
    def __init__(self, training_data, feature_names, class_names, predict_fn):
        """Creates LimeTabularExplainer with discretize_continuous=True."""
    
    def explain_instance(self, x_row, num_features=10, num_samples=2000):
        """Returns LIME explanation object."""
    
    def as_importance_vector(self, x_row, num_features, num_samples) -> np.ndarray:
        """Returns weights aligned to all feature_names. Shape: (n_features,)"""
```

---

#### `anchor_explainer.py` (38 lines)

```python
class AnchorExplainer:
    def __init__(self, X_train_raw, feature_names, predict_fn, class_names=None):
        """Creates AnchorTabularExplainer."""
    
    def explain_instance(self, x_row, threshold=0.9):
        """
        Returns Anchor object with:
        - anchor.names(): ["feature1 > 0.5", "feature2 <= 10"]
        - anchor.precision(): 0.95
        - anchor.coverage(): 0.32
        """
```

---

#### `dice_counterfactuals.py` (60 lines)

```python
class DiceExplainer:
    def __init__(self, model, X_train_processed, y_train, feature_names, outcome_name, method="random"):
        """Creates DiCE explainer in processed feature space."""
    
    def generate_counterfactuals(self, x_row_processed, total_cfs=3):
        """Returns DiCE CF object with final_cfs_df DataFrame of counterfactuals."""
```

---

### 3. Evaluation Modules (`hexeval/evaluation/`)

#### `evaluator.py` (186 lines) - **Main Entry Point**

**Purpose:** Orchestrate the complete evaluation pipeline

```python
def evaluate(
    model_path: str,
    data_path: str,
    target_column: str | None = None,
    config_path: str | None = None,
    output_dir: str | None = None,
    config_overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Run complete HEXEval evaluation pipeline.
    
    Returns:
    {
        "technical_metrics": DataFrame,    # 4 rows (one per method)
        "persona_ratings": DataFrame,      # N rows (personas × methods × instances)
        "recommendations": Dict,           # Stakeholder → recommended method
        "model_info": Dict,
        "data_info": Dict,
        "output_path": str
    }
    """
```

---

#### `technical_evaluator.py` (325 lines)

**Purpose:** Run all 4 XAI methods and compute technical metrics

```python
def run_technical_evaluation(model_wrapper, data, config) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
    - method: SHAP, LIME, Anchor, DiCE
    - fidelity_deletion: Lower is better
    - fidelity_insertion: Higher is better
    - sparsity: Number of important features
    - stability: Variance under noise (LIME only)
    - anchor_precision, anchor_coverage, anchor_n_conditions (Anchor only)
    - dice_success_rate (DiCE only)
    """
```

---

#### `persona_evaluator.py` (503 lines) - **Most Complex Module**

**Purpose:** LLM-based human stakeholder simulation

**Architecture:**
```
run_persona_evaluation()
  ├─ load_personas_from_file()       # Load from YAML
  ├─ _generate_explanations()        # Create text explanations
  └─ _evaluate_with_llm()            # Main loop
      ├─ _build_system_prompt()      # Rich persona context
      ├─ _build_eval_prompt()        # Scenario + rating task
      └─ _call_llm()                 # OpenAI API + TOML parsing
```

**Prompt Engineering:**

System Prompt Structure:
1. **Identity:** Role, experience, decision style
2. **Risk Profile:** Risk tolerance, AI comfort level
3. **Priorities:** What matters most (need-based, not metric names)
4. **Mental Model:** How they think about decisions
5. **Heuristics:** Decision rules they follow
6. **Explanation Preferences:** What format works for them
7. **Domain Context:** Injected from config.yaml

---

#### `personas.py` (43 lines)

**Purpose:** Load persona definitions from external YAML files

```python
def load_personas_from_file(path: str | Path) -> List[Dict]:
    """
    Load personas from a YAML file.
    
    Supports relative paths from project root.
    Validates format (must be list of dicts).
    """
```

---

#### `recommender.py` (269 lines)

**Purpose:** Generate method recommendations per stakeholder

**Algorithm:**
```python
# Stakeholder profiles for matching
STAKEHOLDER_PROFILES = {
    "Technical": {"preferred_traits": ["comprehensive", "faithful"], ...},
    "Customer-Facing": {"preferred_traits": ["actionable", "simple"], ...},
    "Risk-Averse": {"preferred_traits": ["defensive", "rule_based"], ...},
}

# Method characteristics
METHOD_TRAITS = {
    "SHAP": ["comprehensive", "faithful", "many_features", "complex"],
    "LIME": ["balanced", "interpretable", "moderate_features"],
    "Anchor": ["rule_based", "simple", "high_precision", "defensive"],
    "DiCE": ["actionable", "counterfactual", "communicable"],
}
```

---

## Configuration System

### Domain Configuration (`eval_config.yaml`)

The domain section makes HEXEval reusable for any ML prediction task:

```yaml
domain:
  name: "Heart Disease Prediction"
  prediction_task: "heart disease risk assessment"
  decision_verb: "diagnose"
  decision_noun: "patient case"
  stakeholder_context: "at a cardiology clinic"
  end_user_context: "concerned about my heart health"
  positive_outcome: "healthy (no disease)"
  negative_outcome: "diagnosis of heart disease"
  
  terms:
    applicant: "patient"
    application: "medical case"
    risk_factor: "health risk"
    decision_maker: "cardiologist"
```

**Why This Matters:**
- Change these lines → works for any domain (healthcare, finance, fraud, churn)
- Injected into LLM prompts for contextual evaluation
- No code changes required for new domains

---

### Evaluation Settings

```yaml
evaluation:
  sample_size: 100               # Instances for technical metrics
  random_state: 42
  
  fidelity:
    steps: 20                    # Granularity for insertion/deletion
  
  stability:
    noise_std: 0.05
    repeats: 5
  
  explainers:
    shap:
      enabled: true
      background_size: 100       # Background samples for SHAP
    
    lime:
      enabled: true
      num_samples: 500           # Perturbations per explanation
      num_features: 5            # Top features
      stability_test: true
    
    anchor:
      enabled: true
      precision_threshold: 0.9
      max_instances: 10          # Anchor is slow
    
    dice:
      enabled: true
      num_counterfactuals: 3
      max_instances: 5           # DiCE is slow
      method: "random"
```

---

### Persona Configuration

```yaml
personas:
  enabled: true
  file: "hexeval/config/personas_healthcare.yaml"  # External YAML file
  llm_model: "gpt-4o"           # Options: gpt-4, gpt-4-turbo, gpt-4o, o1-mini
  runs_per_method: 1            # Evaluations per method
  sample_instances: 3           # Instances to evaluate
  top_k_features: 5             # Features in explanations
```

**Total LLM Calls:** personas × methods × instances × runs = 4 × 4 × 3 × 1 = 48 calls

---

### Recommendation Weights

```yaml
recommendations:
  enabled: true
  weights:
    technical_fidelity: 0.3
    technical_parsimony: 0.2
    persona_trust: 0.3
    persona_satisfaction: 0.2
```

---

## Persona System

### Design Philosophy

Personas are now **external YAML files** (not embedded in code) with:

1. **Gender-neutral names** - Avoid stereotype bias
2. **Need-based priorities** - Not metric names like "Fidelity"
3. **Nuanced heuristics** - Compensating factors, not rigid rules
4. **Empowered framing** - Goal-oriented, not anxious

### Credit Risk Personas (`personas_credit_risk.yaml`)

| Name | Role | Experience | Key Priorities |
|------|------|------------|----------------|
| Jordan Walsh | Policy-Focused Loan Officer | 18 years | Justify decisions, clear rules |
| Sam Chen | Model Validation Analyst | 5 years | Detect model errors, verify reasoning |
| Taylor Kim | Compliance & Risk Officer | 22 years | Regulatory compliance, audit trail |
| Morgan Patel | Customer Success Manager | 12 years | Explain to customers, identify improvements |
| Casey Rodriguez | Strategic Planning Director | 15 years | Strategic insights, portfolio patterns |
| Riley Martinez | Loan Applicant (End User) | 0 years | Understand decision, actionable next steps |

### Healthcare Personas (`personas_healthcare.yaml`)

| Name | Role | Experience | Key Priorities |
|------|------|------------|----------------|
| Dr. Sarah Jenkins | Lead Cardiologist | 15 years | Clinical validity, patient safety |
| Mark Thompson | Medical Researcher | 8 years | Statistical robustness, bias detection |
| Linda Martinez | Hospital Administrator | 20 years | Resource optimization, triage efficiency |
| David Chen | Patient (End User) | 0 years | Simple explanations, actionable lifestyle advice |

### Persona YAML Structure

```yaml
- name: "Jordan Walsh"
  role: "Policy-Focused Loan Officer"
  experience_years: 18
  risk_profile: "Highly risk-averse, feels personally accountable"
  decision_style: "Slow and methodical, relies on established policies"
  ai_comfort: "Low - prefers human oversight"
  priorities:
    - "Being able to justify decisions to management"
    - "Confidence in the soundness of the recommendation"
    - "Clear guidance on what factors drove the decision"
  mental_model: |
    Credit history matters most, but compensating factors can offset weak areas.
  heuristics:
    - "Skeptical of low credit scores unless offset by tenure/assets"
    - "Short employment history concerning but acceptable with strong income"
  explanation_preferences: |
    Needs explanations that map to institutional policies.
```

---

## Streamlit UI

### 5-Tab Interface (`hexeval/ui/app.py` - 574 lines)

```
+--------------------------------------------------------------------------+
| HEXEval - Holistic Explanation Evaluation                                |
+--------------------------------------------------------------------------+
| [Configuration & Run] [Use Case Details] [Results] [Recommendations]     |
| [Documentation]                                                          |
+--------------------------------------------------------------------------+
```

### Tab 1: Configuration & Run
- **Use Case Selection:** Heart Disease, Credit Risk, Custom Upload
- **Load Existing Results:** One-click load from previous runs
- **Sample Size Slider:** 50-500 instances
- **Enable LLM Personas:** Toggle + API key input
- **Run Evaluation Button:** Triggers full pipeline

### Tab 2: Use Case Details
- **Domain Context:** From config YAML
- **Stakeholder Personas:** Expandable cards with priorities, mental model
- **Full Configuration:** YAML code view

### Tab 3: Results
- **Technical Metrics Table:** All methods + metrics
- **Fidelity Comparison Chart:** Bar chart (deletion vs insertion)
- **Persona Ratings Summary:** Average ratings by method
- **Radar Chart:** 6 dimensions for all methods
- **Persona-Wise Analysis:** Expandable cards per persona with ratings + comments

### Tab 4: Recommendations
- **Per-Stakeholder Cards:** Recommended method, reasoning, score
- **Alternatives Table:** All methods with scores
- **Method Comparison Matrix:** Satisfaction scores heatmap

### Tab 5: Documentation
- **Prerequisites & Setup:** From `docs/HEXEval_Prerequisites.md`
- **How It Works:** From `docs/HEXEval_HowItWorks.md`
- **Configuration Guide:** From `docs/HEXEval_Configuration.md`

### Use Case Configuration

```python
USE_CASES = {
    "Heart Disease (Healthcare)": {
        "config_path": "hexeval/config/eval_config.yaml",
        "data_path": "usecases/heart.csv",
        "model_path": "usecases/heart_disease_pipeline.pkl",
        "target": "target",
        "output_dir": "outputs/heart_disease",
        "default_sample_size": 100
    },
    "Credit Risk (Finance)": {
        "config_path": "hexeval/config/eval_config_credit_risk.yaml",
        "data_path": "usecases/credit_risk_dataset.csv",
        "model_path": "usecases/xgboost_credit_risk_new.pkl",
        "target": "loan_status",
        "output_dir": "outputs/credit_risk",
        "default_sample_size": 150
    },
    "Custom Upload": {...}
}
```

---

## Key Algorithms

### 1. Fidelity: Insertion/Deletion AUC

**Reference:** Covert & Lundberg (2021) - "Explaining by Removing"

```
For each instance x:
  1. Get feature importances: I = [i1, i2, ..., in]
  2. Rank features by |importance|: [f3, f1, f7, ...]
  
  DELETION:
  3. Start with full instance: x_full
  4. Remove top-k features (set to baseline):
     k=1: x_del = x with f3 = baseline[f3]
     k=2: x_del = x with f3, f1 = baseline
  5. Measure prediction drop at each step
  6. Compute AUC of (k, prediction) curve
  
  INSERTION:
  7. Start with baseline: x_base
  8. Add top-k features (set to original):
     k=1: x_ins = baseline with f3 = x[f3]
  9. Measure prediction rise at each step
  10. Compute AUC of curve

Average across all instances
```

**Interpretation:**
- **Deletion AUC ≈ 0.10:** Model drops 10% when removing important features (GOOD)
- **Insertion AUC ≈ 0.25:** Model rises 25% when adding important features (GOOD)
- Deletion should be < Insertion

---

### 2. Method-Specific Technical Scoring

```python
if method in ["SHAP", "LIME"]:
    # Normalize fidelity and parsimony
    tech_score = 0.5 * normalize(fidelity) + 0.5 * normalize(parsimony)
    
elif method == "Anchor":
    # Anchor doesn't have fidelity scores
    tech_score = 0.8 * precision + 0.2 * coverage
    
elif method == "DiCE":
    # DiCE only has success rate
    tech_score = success_rate
```

---

### 3. Persona Evaluation Flow

```
FOR EACH persona in personas_file:                    [4-6 personas]
  system_prompt = build_system_prompt(persona, domain_config)
  
  FOR EACH instance_idx in sample_instances:          [2-3 instances]
    FOR EACH method in [SHAP, LIME, Anchor, DiCE]:    [4 methods]
      FOR EACH run in range(runs_per_method):         [1 run]
        
        explanation = explanations[instance_idx][method]
        user_prompt = build_eval_prompt(instance, explanation, method)
        
        response = openai.chat.completions.create(
          model="gpt-4o",
          messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
          ]
        )
        
        ratings = parse_toml(response.content)
        results.append({...})

TOTAL: 4 × 3 × 4 × 1 = 48 LLM API calls (default)
```

---

## CLI Usage

HEXEval provides a command-line interface for programmatic evaluation.

### Basic Commands

#### 1. Validate Model-Data Compatibility

```bash
python scripts/hexeval_cli.py validate \
    path/to/model.pkl \
    path/to/data.csv \
    --target target_column
```

**Output:**
```
✓ Validation passed!
✓ Model can make predictions on data
```

#### 2. Run Full Evaluation

```bash
python scripts/hexeval_cli.py evaluate \
    path/to/model.pkl \
    path/to/data.csv \
    --target target_column \
    --config path/to/config.yaml \
    --output outputs/my_results/
```

**Output:**
```
============================================================
HEXEval - Starting Evaluation
============================================================
✓ Loaded model: XGBClassifier
✓ Loaded data: 260 train, 65 test
✓ Validated model-data compatibility
============================================================
Running Technical Evaluation
============================================================
Evaluating SHAP...
✓ SHAP complete
Evaluating LIME...
✓ LIME complete
Evaluating Anchor...
✓ Anchor complete
Evaluating DiCE...
✓ DiCE complete
✓ Technical evaluation complete (4 methods)

============================================================
Running Persona Evaluation (LLM)
============================================================
Total LLM calls: 48
100%|████████| 48/48 [01:30<00:00,  1.88it/s]
✓ Persona evaluation complete

✅ Evaluation Complete!
Results saved to: outputs/my_results/
```

### Command-Line Options

| Option | Description | Required | Default |
|--------|-------------|----------|---------|
| `model` | Path to model file (.pkl or .joblib) | Yes | - |
| `data` | Path to CSV dataset | Yes | - |
| `--target` | Target column name | Yes* | - |
| `--config` | Path to config YAML | No | `hexeval/config/eval_config.yaml` |
| `--output` | Output directory | No | `outputs/hexeval_results/` |

*Required for `evaluate`, optional for `validate` (will auto-detect if possible)

### Example Workflows

#### Quick Technical Evaluation Only

```bash
# Disable personas in config
python scripts/hexeval_cli.py evaluate \
    model.pkl data.csv --target outcome \
    --config config_no_personas.yaml
```

#### Batch Evaluation

```bash
# Evaluate multiple models
for model in models/*.pkl; do
    python scripts/hexeval_cli.py evaluate \
        "$model" data.csv --target outcome \
        --output "outputs/$(basename $model .pkl)/"
done
```

---

## API Reference

### Main Entry Point

#### `hexeval.evaluate()`

Run complete evaluation pipeline.

```python
from hexeval import evaluate

results = evaluate(
    model_path: str,
    data_path: str,
    target_column: str | None = None,
    config_path: str | None = None,
    output_dir: str | None = None,
    config_overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]
```

**Parameters:**
- `model_path` (str): Path to pickled model file
- `data_path` (str): Path to CSV dataset
- `target_column` (str, optional): Name of target column. Auto-detected if not provided.
- `config_path` (str, optional): Path to YAML config. Defaults to `hexeval/config/eval_config.yaml`
- `output_dir` (str, optional): Output directory. Defaults to `outputs/hexeval_results/`
- `config_overrides` (dict, optional): Override config values programmatically

**Returns:**
```python
{
    "technical_metrics": pd.DataFrame,      # 4 rows (one per method)
    "persona_ratings": pd.DataFrame | None, # N rows (personas × methods × instances)
    "recommendations": Dict | None,         # Stakeholder → recommended method
    "model_info": Dict,                     # Model metadata
    "data_info": Dict,                      # Dataset metadata
    "output_path": str                     # Path to saved results
}
```

**Example:**
```python
results = evaluate(
    model_path="model.pkl",
    data_path="data.csv",
    target_column="target",
    config_overrides={"evaluation": {"sample_size": 200}}
)
```

### Core Module Functions

#### `hexeval.core.load_model()`

```python
from hexeval.core import load_model

model_wrapper = load_model(path: str | Path) -> ModelWrapper
```

Loads a trained model from disk and wraps it in `ModelWrapper` for consistent interface.

**Supported formats:**
- sklearn Pipeline objects
- Raw sklearn/XGBoost models
- Dictionary artifacts: `{"model": ..., "preprocessor": ..., "feature_names": ...}`

#### `hexeval.core.load_data()`

```python
from hexeval.core import load_data

data = load_data(
    path: str | Path,
    target_column: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict
```

Loads CSV data and performs train/test split.

**Returns:**
```python
{
    "X_train": pd.DataFrame,
    "X_test": pd.DataFrame,
    "y_train": pd.Series,
    "y_test": pd.Series,
    "feature_names": List[str],
    "categorical_features": List[str],
    "numeric_features": List[str]
}
```

#### `hexeval.core.validate_model_data_compatibility()`

```python
from hexeval.core import validate_model_data_compatibility

validation = validate_model_data_compatibility(
    model_wrapper: ModelWrapper,
    data: Dict
) -> Dict
```

Validates that model and data are compatible.

**Returns:**
```python
{
    "status": "valid" | "invalid",
    "warnings": List[str],
    "errors": List[str]
}
```

### Evaluation Module Functions

#### `hexeval.evaluation.run_technical_evaluation()`

```python
from hexeval.evaluation import run_technical_evaluation

tech_results = run_technical_evaluation(
    model_wrapper: ModelWrapper,
    data: Dict,
    config: Dict
) -> pd.DataFrame
```

Runs technical evaluation only (no LLM calls).

**Returns DataFrame with columns:**
- `method`: SHAP, LIME, Anchor, DiCE
- `fidelity_deletion`: Lower is better
- `fidelity_insertion`: Higher is better
- `sparsity`: Number of important features
- `stability`: Variance under noise (LIME only)
- `anchor_precision`, `anchor_coverage` (Anchor only)
- `dice_success_rate` (DiCE only)

#### `hexeval.evaluation.generate_recommendations()`

```python
from hexeval.evaluation import generate_recommendations

recommendations = generate_recommendations(
    technical_metrics: pd.DataFrame,
    persona_ratings: pd.DataFrame,
    config: Dict
) -> Dict
```

Generates stakeholder-specific method recommendations.

**Returns:**
```python
{
    "stakeholder_role": {
        "recommended_method": "SHAP",
        "score": 0.85,
        "reasoning": "High trust (3.5/5), excellent fidelity",
        "alternatives": [...]
    },
    ...
}
```

---

## Usage Examples

### Basic Usage

```python
from hexeval import evaluate

results = evaluate(
    model_path="usecases/xgboost_credit_risk_new.pkl",
    data_path="usecases/credit_risk_dataset.csv",
    target_column="loan_status"
)

print(results["technical_metrics"])
print(results["recommendations"])
```

### Custom Configuration

```python
results = evaluate(
    model_path="my_model.pkl",
    data_path="my_data.csv",
    target_column="target",
    config_path="my_config.yaml",
    output_dir="my_results/"
)
```

### Programmatic Access

```python
from hexeval.core import load_model, load_data, validate_model_data_compatibility
from hexeval.evaluation import run_technical_evaluation

model_wrapper = load_model("model.pkl")
data = load_data("data.csv", target_column="target")

validation = validate_model_data_compatibility(model_wrapper, data)
if validation["status"] == "valid":
    config = {"sample_size": 100, "explainers": {...}}
    tech_results = run_technical_evaluation(model_wrapper, data, config)
```

### Streamlit UI

```bash
streamlit run hexeval/ui/app.py
```

---

## Performance & Scalability

### Runtime Estimates

**Technical Evaluation:**

| Method | Time per Instance | 100 Instances |
|--------|------------------:|-------------:|
| SHAP   | ~0.5s            | ~50s         |
| LIME   | ~1.0s            | ~100s        |
| Anchor | ~2.0s            | ~20s (10 inst)|
| DiCE   | ~3.0s            | ~15s (5 inst)|
| **TOTAL** |               | **~3 minutes**|

**Persona Evaluation (with LLM):**
- 48 API calls × 1-2s per call = **~2 minutes**
- Cost: ~$0.20 with GPT-4o

**Total Runtime:** ~5-7 minutes per evaluation

### Memory Usage

| Component | Memory Usage | Notes |
|-----------|--------------|-------|
| Model Loading | ~50-200MB | Depends on model size |
| Data Loading | ~100-500MB | Depends on dataset size |
| SHAP Background | ~50-200MB | Background dataset size |
| LIME Perturbations | ~100-300MB | Temporary during explanation |
| LLM API Calls | Minimal | No local model storage |

**Recommendations:**
- For datasets >10K samples, reduce `sample_size` in config
- For large models, use `background_size: 50` for SHAP
- Monitor memory during Anchor evaluation (can spike)

### Scalability Considerations

**Current Limitations:**
- Single-threaded execution (no parallel processing)
- LLM API calls are sequential (rate limits apply)
- Anchor/DiCE are computationally expensive (limited to 10-20 instances)

**Optimization Strategies:**
1. **Reduce sample sizes** for faster iteration
2. **Disable expensive methods** (Anchor/DiCE) for quick tests
3. **Use smaller LLM models** (gpt-4o-mini) for persona evaluation
4. **Cache explanations** (not yet implemented) for repeated evaluations

---

## Limitations & Constraints

### Model Requirements

**Supported:**
- ✅ Binary classification models (2 classes)
- ✅ sklearn-compatible models (RandomForest, XGBoost, LogisticRegression, etc.)
- ✅ Models with `predict_proba()` method
- ✅ sklearn Pipelines with preprocessing

**Not Supported:**
- ❌ Multi-class classification (>2 classes)
- ❌ Regression models
- ❌ Deep learning models (PyTorch/TensorFlow)
- ❌ Models without `predict_proba()` (e.g., SVM without `probability=True`)
- ❌ Image/text data (tabular only)

### Data Requirements

**Supported:**
- ✅ Tabular CSV files
- ✅ Mixed categorical + numeric features
- ✅ Missing values (handled via imputation)
- ✅ Binary target variable (0/1 or string labels)

**Not Supported:**
- ❌ Non-tabular data (images, text, time series)
- ❌ Multi-label classification
- ❌ Regression targets
- ❌ Extremely high-dimensional data (>1000 features may be slow)

### XAI Method Limitations

**SHAP:**
- ⚠️ Computationally expensive for large datasets
- ⚠️ Requires background dataset (memory intensive)
- ⚠️ May show too many features (low parsimony)

**LIME:**
- ⚠️ Can be unstable (high variance across runs)
- ⚠️ Local approximations may not reflect global behavior
- ⚠️ Requires careful tuning of `num_samples`

**Anchor:**
- ⚠️ Very slow (2-5 seconds per instance)
- ⚠️ Limited to discrete/categorical features
- ⚠️ May generate overly specific rules (low coverage)

**DiCE:**
- ⚠️ Slowest method (3-10 seconds per instance)
- ⚠️ Requires feasible ranges for all features
- ⚠️ Counterfactuals may be unrealistic

### Persona Evaluation Limitations

**LLM-Based Simulation:**
- ⚠️ **Not validated against real humans** - Personas are simulated, not actual stakeholder feedback
- ⚠️ **Cost**: ~$0.20 per evaluation (48 API calls with GPT-4o)
- ⚠️ **API Dependencies**: Requires OpenAI API key and internet connection
- ⚠️ **Potential Bias**: LLM responses may not capture full range of human confusion/frustration
- ⚠️ **No Ground Truth**: Cannot measure actual decision quality, only perceived usefulness

**Known Issues:**
- Persona ratings may vary slightly across runs (LLM non-determinism)
- Some personas may rate all methods similarly (lack of differentiation)
- End-user personas may not fully capture non-technical user perspectives

### Framework Limitations

**Current Version:**
- ❌ No parallel processing (single-threaded)
- ❌ No explanation caching (recomputes on every run)
- ❌ No support for custom explainers (requires code changes)
- ❌ Limited error recovery (fails fast on validation errors)
- ❌ No progress persistence (cannot resume interrupted evaluations)

**Future Work:**
- Parallel explainer execution
- Explanation caching and persistence
- Plugin system for custom explainers
- Incremental evaluation (resume from checkpoints)
- Real human validation study

---

## Testing Strategy

### Unit Testing

**Current Status:** Limited unit tests (see `docs/TESTING.md`)

**Test Coverage Areas:**
- ✅ Model loading (various formats)
- ✅ Data loading and preprocessing
- ✅ Model-data validation
- ⚠️ Explainer wrappers (partial)
- ❌ Metrics computation (not tested)
- ❌ Persona evaluation (not tested)

### Integration Testing

**Manual Testing Workflow:**

1. **Validation Test:**
   ```bash
   python scripts/hexeval_cli.py validate \
       usecases/heart_disease_pipeline.pkl \
       usecases/heart.csv --target target
   ```

2. **Technical Evaluation Test:**
   ```python
   from hexeval import evaluate
   results = evaluate(
       model_path="usecases/heart_disease_pipeline.pkl",
       data_path="usecases/heart.csv",
       target_column="target"
   )
   assert len(results['technical_metrics']) == 4
   ```

3. **Full Pipeline Test:**
   ```bash
   python scripts/hexeval_cli.py evaluate \
       usecases/xgboost_credit_risk_new.pkl \
       usecases/credit_risk_dataset.csv \
       --target loan_status
   ```

### Test Datasets

**Included Test Cases:**
- `usecases/heart.csv` + `heart_disease_pipeline.pkl` (303 samples, 13 features)
- `usecases/credit_risk_dataset.csv` + `xgboost_credit_risk_new.pkl` (32,581 samples, 11 features)

**Validation:**
- Both datasets have been validated to work end-to-end
- Results are reproducible with `random_state=42`

### Persona Testing

**Test Personas:**
- Healthcare: 4 personas (see `hexeval/config/personas_healthcare.yaml`)
- Credit Risk: 6 personas (see `hexeval/config/personas_credit_risk.yaml`)

**Validation Approach:**
- Manual review of persona definitions
- Check for gender-neutral names
- Verify need-based priorities (not metric names)
- Ensure realistic heuristics and mental models

**Known Issues:**
- Persona ratings not yet validated against real humans
- LLM responses may vary across runs (non-determinism)

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Model Loading Fails

**Error:** `AttributeError: 'Model' object has no attribute 'predict_proba'`

**Cause:** Model doesn't implement `predict_proba()` method.

**Solutions:**
1. **For SVM:** Train with `probability=True`:
   ```python
   from sklearn.svm import SVC
   model = SVC(probability=True)
   ```

2. **For Custom Models:** Implement `predict_proba()`:
   ```python
   def predict_proba(self, X):
       predictions = self.predict(X)
       # Convert to probabilities
       return probabilities
   ```

#### Issue 2: Feature Mismatch

**Error:** `ValueError: Missing required features: ['feature1', 'feature2']`

**Cause:** Model expects features that aren't in the data.

**Solutions:**
1. Check feature names match exactly (case-sensitive)
2. Ensure preprocessing pipeline includes all required transformations
3. Use `validate_model_data_compatibility()` before evaluation

#### Issue 3: SHAP Memory Error

**Error:** `MemoryError` during SHAP evaluation

**Cause:** Background dataset too large for available memory.

**Solutions:**
1. Reduce `background_size` in config:
   ```yaml
   explainers:
     shap:
       background_size: 50  # Default is 100
   ```

2. Reduce `sample_size` for evaluation:
   ```yaml
   evaluation:
     sample_size: 50  # Default is 100
   ```

#### Issue 4: Anchor/DiCE Very Slow

**Symptom:** Evaluation takes >30 minutes

**Cause:** Anchor and DiCE are computationally expensive.

**Solutions:**
1. Reduce `max_instances` in config:
   ```yaml
   explainers:
     anchor:
       max_instances: 5  # Default is 10
     dice:
       max_instances: 3  # Default is 5
   ```

2. Disable expensive methods for quick tests:
   ```yaml
   explainers:
     anchor:
       enabled: false
     dice:
       enabled: false
   ```

#### Issue 5: LLM API Errors

**Error:** `openai.error.AuthenticationError` or `RateLimitError`

**Cause:** Invalid API key or rate limit exceeded.

**Solutions:**
1. Check API key is set correctly:
   ```bash
   echo $OPENAI_API_KEY  # Should show your key
   ```

2. Use smaller LLM model (gpt-4o-mini) or reduce calls:
   ```yaml
   personas:
     llm_model: "gpt-4o-mini"  # Cheaper, faster
     sample_instances: 2  # Reduce from 3
     runs_per_method: 1  # Keep at 1
   ```

3. Add retry logic (not yet implemented in framework)

#### Issue 6: Persona Evaluation Returns None

**Symptom:** `results['persona_ratings']` is `None`

**Cause:** Persona evaluation failed or was disabled.

**Solutions:**
1. Check config has personas enabled:
   ```yaml
   personas:
     enabled: true
   ```

2. Check API key is set (see Issue 5)

3. Review logs for error messages:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

#### Issue 7: Results Directory Not Created

**Error:** `FileNotFoundError` when accessing results

**Cause:** Output directory doesn't exist or permissions issue.

**Solutions:**
1. Create output directory manually:
   ```bash
   mkdir -p outputs/my_results
   ```

2. Check write permissions:
   ```bash
   ls -ld outputs/
   ```

#### Issue 8: Import Errors

**Error:** `ModuleNotFoundError: No module named 'hexeval'`

**Cause:** Package not installed or virtual environment not activated.

**Solutions:**
1. Install package:
   ```bash
   pip install -e .
   ```

2. Activate virtual environment:
   ```bash
   source .venv/bin/activate
   ```

3. Check Python path:
   ```python
   import sys
   print(sys.path)  # Should include project directory
   ```

### Getting Help

**Documentation:**
- `docs/HEXEval_Prerequisites.md` - Setup guide
- `docs/HEXEval_Configuration.md` - Config reference
- `docs/TESTING.md` - Testing guide

**Debug Mode:**
Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Report Issues:**
Include:
- Error message and stack trace
- Config file (sanitized)
- Model type and data shape
- Python version and OS

---

## Design Decisions

### 1. Domain-Agnostic Configuration
**Decision:** Use YAML config to adapt HEXEval to any ML domain  
**Benefit:** Research labs can use for their own domains without code changes

### 2. Dual Evaluation (Technical + Human-Centered)
**Decision:** Combine objective metrics with LLM persona feedback  
**Key Innovation:** Technical metrics alone miss the human interpretability gap

### 3. External Persona YAML Files
**Decision:** Move personas from code to YAML files  
**Benefit:** Easy to add/modify personas, domain-specific personas, no code changes

### 4. Method-Specific Technical Scoring
**Decision:** Score each XAI method on its own strengths  
**Rationale:** Anchor doesn't produce fidelity scores, shouldn't be penalized

### 5. ModelWrapper Pattern
**Decision:** Wrap all models in consistent interface  
**Benefit:** Handles preprocessing, type conversion, metadata access uniformly

### 6. Use Case Switching in UI
**Decision:** Pre-configured use cases + custom upload  
**Benefit:** Quick demos with sample data, flexible for custom models

---

## Research Contributions

### Primary Contributions

1. **Dual Evaluation Framework**
   - First framework to combine technical metrics (fidelity, parsimony, stability) with human-centered evaluation (LLM-simulated personas)
   - Reveals the **fidelity-interpretability gap**: methods with high technical scores may have poor human usability

2. **Stakeholder-Specific Method Selection**
   - Enables data-driven selection of XAI methods based on stakeholder needs
   - Demonstrates that different personas prefer different explanation formats (2.5-point variance in ratings)
   - Provides actionable recommendations: "For conservative loan officers, use Anchor because..."

3. **LLM-Based Persona Simulation**
   - Cost-effective alternative to human studies ($0.20 vs $50-100 per evaluation)
   - Reproducible and scalable persona evaluation
   - Rich qualitative feedback (comments) alongside quantitative ratings

4. **Domain-Agnostic Design**
   - Configurable for any binary classification task via YAML
   - No code changes required for new domains
   - Validated on healthcare and finance domains

5. **Comprehensive Technical Evaluation**
   - Standardized implementation of fidelity (insertion/deletion AUC)
   - Method-specific metrics (Anchor precision, DiCE success rate)
   - Robustness testing (stability under noise)

### Research Findings

**Key Discovery:** Technical excellence does not guarantee human interpretability.

**Quantitative Evidence:**
- Technical metrics: Fidelity AUC 0.11-0.13 (good), Anchor precision 94.9% (excellent)
- Human ratings: Average trust 2.1/5, actionability 1.3-1.7/5 (poor)
- **Gap:** 2-3 point difference between technical and human scores

**Persona Differentiation:**
- Conservative loan officers prefer Anchor (rule-based)
- Data analysts prefer SHAP (comprehensive attribution)
- End-users prefer DiCE (actionable counterfactuals)
- **Variance:** 2.5-point difference in ratings across personas

**Method-Specific Insights:**
- **SHAP:** High fidelity but low interpretability (too many features, technical jargon)
- **LIME:** Balanced but unstable (high variance across runs)
- **Anchor:** High precision but low coverage (overly specific rules)
- **DiCE:** Actionable but slow (computational bottleneck)

### Academic Positioning

**Research Area:** Explainable AI (XAI), Human-Computer Interaction (HCI), Machine Learning Evaluation

**Related Fields:**
- Interpretable Machine Learning
- Human-Centered AI
- Evaluation Methodologies
- Stakeholder-Centric Design

**Novel Aspects:**
- Combines quantitative and qualitative evaluation
- Stakeholder-specific recommendations (not one-size-fits-all)
- LLM-based persona simulation (scalable alternative to human studies)
- Domain-agnostic framework (generalizable across domains)

---

## Related Work

### XAI Evaluation Frameworks

**Quantitative Evaluation:**
- **Quantus** (Hedström et al., 2023): Comprehensive metrics library for XAI evaluation
- **XAI-Bench** (Arya et al., 2019): Benchmark suite for explanation methods
- **Limitation:** Focus on technical metrics only, ignore human factors

**Human-Centered Evaluation:**
- **Human-AI Interaction Studies**: User studies on explanation effectiveness (Miller, 2019)
- **Explanation Quality Surveys**: Subjective ratings from real users (Hoffman et al., 2018)
- **Limitation:** Expensive, not scalable, domain-specific

**Our Contribution:** First framework to combine both approaches systematically.

### Persona-Based Evaluation

**LLM Personas:**
- **LLM-as-Judge** (Zheng et al., 2023): Using LLMs to evaluate model outputs
- **Simulated Users** (Wang et al., 2023): LLM-based user simulation for testing
- **Limitation:** Not validated for XAI evaluation, focus on other domains

**Our Contribution:** First application of LLM personas to XAI method evaluation.

### XAI Methods Evaluated

**SHAP (SHapley Additive exPlanations):**
- Lundberg & Lee (2017): Unified framework for feature attribution
- **Strengths:** Theoretically grounded, comprehensive
- **Weaknesses:** Computationally expensive, low parsimony

**LIME (Local Interpretable Model-agnostic Explanations):**
- Ribeiro et al. (2016): Local linear approximations
- **Strengths:** Model-agnostic, interpretable
- **Weaknesses:** Unstable, may hallucinate

**Anchor:**
- Ribeiro et al. (2018): Rule-based explanations
- **Strengths:** High precision, human-readable rules
- **Weaknesses:** Low coverage, slow

**DiCE (Diverse Counterfactual Explanations):**
- Mothilal et al. (2020): Counterfactual generation
- **Strengths:** Actionable, intuitive
- **Weaknesses:** Computationally expensive, may generate unrealistic CFs

### Evaluation Metrics

**Fidelity Metrics:**
- **Insertion/Deletion AUC:** Covert & Lundberg (2021) - "Explaining by Removing"
- **Our Implementation:** Standardized insertion/deletion AUC computation

**Parsimony:**
- **Sparsity:** Number of features in explanation (Miller, 2019)
- **Our Implementation:** Average number of important features across instances

**Stability:**
- **Robustness:** Explanation consistency under noise (Alvarez-Melis & Jaakkola, 2018)
- **Our Implementation:** Variance of explanations under small perturbations

### Domain Applications

**Healthcare:**
- Caruana et al. (2015): Need for interpretability in medical AI
- **Our Contribution:** Validated framework on heart disease prediction

**Finance:**
- Bussone et al. (2015): Explainability requirements in credit risk
- **Our Contribution:** Validated framework on loan approval systems

### Gaps in Literature

1. **No Unified Framework:** Existing work focuses on either technical OR human evaluation, not both
2. **One-Size-Fits-All:** Most frameworks assume a single "best" explanation method
3. **Limited Scalability:** Human studies are expensive and not reproducible
4. **Domain-Specific:** Most evaluation frameworks are tied to specific domains

**Our Framework Addresses All Four Gaps.**

---

## Future Extensions

### Add New Explainer

```python
# hexeval/explainers/my_explainer.py
class MyExplainer:
    def explain_instance(self, x_row):
        return importance_scores
    
    def explain_dataset(self, X):
        return batch_importances

# Add to technical_evaluator.py
def _evaluate_my_explainer(model, X_train, X_sample, ...):
    explainer = MyExplainer(...)
    # Compute metrics
    return {"method": "MyExplainer", ...}
```

### Add New Persona

Create or edit YAML file:

```yaml
# hexeval/config/personas_my_domain.yaml
- name: "New Persona"
  role: "New Role"
  experience_years: 10
  risk_profile: "..."
  decision_style: "..."
  ai_comfort: "..."
  priorities:
    - "Priority 1"
    - "Priority 2"
  mental_model: |
    How they think about decisions...
  heuristics:
    - "Rule 1"
    - "Rule 2"
  explanation_preferences: |
    What format works for them...
```

Update config to point to new file:
```yaml
personas:
  file: "hexeval/config/personas_my_domain.yaml"
```

### Add New Domain

Create new config file:

```yaml
# hexeval/config/eval_config_my_domain.yaml
domain:
  name: "My Domain"
  prediction_task: "my prediction task"
  decision_verb: "approve or reject"
  stakeholder_context: "at my organization"
  # ...

personas:
  file: "hexeval/config/personas_my_domain.yaml"
```

Add to UI USE_CASES:

```python
USE_CASES = {
    # ...
    "My Domain": {
        "config_path": "hexeval/config/eval_config_my_domain.yaml",
        "data_path": "usecases/my_data.csv",
        "model_path": "usecases/my_model.pkl",
        "target": "target_column",
        "output_dir": "outputs/my_domain",
        "default_sample_size": 100
    }
}
```

---

## Citation & Academic Use

### How to Cite HEXEval

If you use HEXEval in your research, please cite:

```bibtex
@software{hexeval2026,
  title={HEXEval: A Holistic Framework for Evaluating Explainable AI Methods},
  author={Faraj, Salman},
  year={2026},
  version={2.1},
  url={https://github.com/yourusername/hexeval},
  note={Final Year Research Project}
}
```

### Academic Use Cases

**Suitable For:**
- ✅ Evaluating XAI methods on tabular classification models
- ✅ Comparing explanation methods across technical and human-centered dimensions
- ✅ Selecting appropriate XAI methods for specific stakeholder groups
- ✅ Research on fidelity-interpretability trade-offs
- ✅ Domain-agnostic XAI evaluation studies

**Not Suitable For:**
- ❌ Multi-class classification (>2 classes)
- ❌ Regression tasks
- ❌ Non-tabular data (images, text, time series)
- ❌ Deep learning models (PyTorch/TensorFlow)
- ❌ Production deployment without validation

### Research Ethics

**LLM Persona Evaluation:**
- Personas are simulated, not real human participants
- Results should be validated with real humans before deployment
- See `docs/FINDINGS_AND_FUTURE_WORK.md` for proposed validation study

**Data Privacy:**
- Framework does not send data to external services (except OpenAI API for persona evaluation)
- Model predictions are sent to OpenAI API (review OpenAI's data usage policy)
- For sensitive data, consider using local LLM models (future work)

**Reproducibility:**
- All random seeds are fixed (`random_state=42`)
- Config files ensure reproducible evaluation
- Results are saved to CSV/JSON for analysis

### License

MIT License - See LICENSE file for details.

**Academic Use:** Free for research and educational purposes.

**Commercial Use:** Contact author for licensing terms.

---

## Dependencies

From `pyproject.toml`:

```toml
dependencies = [
    "pandas>=1.5.0",
    "numpy>=1.23.0",
    "scikit-learn>=1.2.0",
    "xgboost>=1.7.0",
    "pyyaml>=6.0",
    "shap>=0.42.0",
    "lime>=0.2.0",
    "dice-ml>=0.10.0",
    "anchor-exp>=0.0.2",
    "scipy>=1.10.0",
    "openai>=1.0.0",
    "streamlit>=1.28.0",
    "plotly>=5.17.0",
    "joblib>=1.3.0",
    "tqdm>=4.65.0",
    "python-dotenv>=1.0.0",
]
```

---

---

## Document Metadata

**Document Version:** 2.1  
**Last Updated:** 2026-01-21  
**Author:** Salman Faraj  
**Project:** Final Year Research Project (FYP)  
**Target Audience:** Developers, Researchers, Thesis Reviewers, Academic Examiners

### Document Structure

This document serves as the **single source of truth** for the HEXEval framework. It covers:

- **Technical Documentation:** Architecture, API, algorithms, implementation details
- **Research Documentation:** Motivation, contributions, findings, related work
- **User Documentation:** Installation, usage, troubleshooting, examples
- **Academic Documentation:** Citation, ethics, reproducibility

### Related Documents

**Quick Start:**
- `README_HEXEVAL.md` - Quick start guide
- `docs/HEXEval_Prerequisites.md` - Setup instructions
- `docs/RUN_HEXEVAL.md` - Running evaluations

**Configuration:**
- `docs/HEXEval_Configuration.md` - Config file reference
- `docs/DOMAIN_CONFIG_EXAMPLES.md` - Domain configuration examples

**Research:**
- `docs/FINDINGS_AND_FUTURE_WORK.md` - Research findings and future directions
- `docs/TECHNICAL_AUDIT.md` - Technical evaluation audit
- `docs/CONTEXT_ENGINEERING.md` - Persona prompt engineering

**Testing:**
- `docs/TESTING.md` - Testing guide
- `docs/TEST_PERSONAS.md` - Persona testing guide

### Version History

**v2.1 (2026-01-21):**
- Added research context and motivation
- Added installation and setup guide
- Added CLI usage and API reference
- Added limitations and troubleshooting
- Added research contributions and related work
- Added citation and academic use section
- Improved organization and completeness

**v2.0 (2026-01-21):**
- Initial comprehensive architecture document
- Complete data flow documentation
- Module-by-module breakdown
- Configuration and persona system documentation

### Feedback & Contributions

For questions, issues, or contributions:
- Review existing documentation first
- Check `docs/TROUBLESHOOTING.md` for common issues
- Open an issue with detailed error information
- Include config files and error logs

---

**End of Document**
