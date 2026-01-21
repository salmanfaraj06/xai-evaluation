# HEXEval - Complete Framework Architecture & Pipeline

**Comprehensive Technical Documentation**

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Complete Data Flow](#complete-data-flow)
4. [Module-by-Module Breakdown](#module-by-module-breakdown)
5. [Evaluation Pipeline Details](#evaluation-pipeline-details)
6. [Configuration System](#configuration-system)
7. [Key Algorithms](#key-algorithms)
8. [Usage Examples](#usage-examples)

---

## Executive Summary

**HEXEval** is a framework for holistic evaluation of explainable AI (XAI) methods on tabular classification models. It combines:

- **Technical Metrics:** Fidelity, parsimony, stability
- **Human-Centered Evaluation:** LLM-simulated personas rating explanations
- **Domain-Agnostic Design:** Configurable for any ML prediction task
- **4 XAI Methods:** SHAP, LIME, Anchor, DiCE counterfactuals

**Total Codebase:** 2,584 lines across 23 Python files

**Key Innovation:** Dual evaluation (technical + persona-based) reveals fidelity-interpretability gap.

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      HEXEval Framework                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐   │
│  │              │      │              │      │              │   │
│  │     CORE     │─────▶│  EXPLAINERS  │─────▶│  EVALUATION  │   │
│  │              │      │              │      │              │   │
│  └──────────────┘      └──────────────┘      └──────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   OUTPUTS & UI                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3-Layer Design

**Layer 1: Core Infrastructure**
- `model_loader.py` - Load sklearn/XGBoost models
- `data_handler.py` - Load CSV data, auto-detect types
- `validator.py` - Model-data compatibility checks

**Layer 2: Explainers**
- `shap_explainer.py` - Shapley value attribution
- `lime_explainer.py` - Local linear approximations
- `anchor_explainer.py` - Rule-based explanations
- `dice_counterfactuals.py` - Counterfactual generation

**Layer 3: Evaluation**
- `technical_evaluator.py` - Fidelity, parsimony, stability metrics
- `persona_evaluator.py` - LLM-based human simulation
- `recommender.py` - Stakeholder-specific recommendations

---

## Complete Data Flow

### End-to-End Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│ 1. INPUT                                                         │
├──────────────────────────────────────────────────────────────────┤
│  ▶ Model (.pkl)         ▶ Data (CSV)        ▶ Config (YAML)      │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ 2. LOAD & VALIDATE (core/)                                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  model_loader.py                                                 │
│  ├─ Load model artifact                                          │
│  ├─ Extract: model, preprocessor, feature_names, threshold       │
│  └─ Validate: has predict_proba()                                │
│                                                                  │
│  data_handler.py                                                 │
│  ├─ Load CSV → DataFrame                                         │
│  ├─ Auto-detect: categorical vs numeric features                 │
│  ├─ Split: train/test (80/20, stratified)                        │
│  └─ Return: X_train, X_test, y_train, y_test, metadata           │
│                                                                  │
│  validator.py                                                    │
│  ├─ Check feature compatibility                                  │
│  ├─ Test prediction: model.predict_proba(X_sample)               │
│  └─ Status: valid/invalid + warnings/errors                      │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ 3. TECHNICAL EVALUATION (evaluation/technical_evaluator.py)      │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  FOR EACH METHOD (SHAP, LIME, Anchor, DiCE):                     │
│                                                                  │
│  ┌──────────────────────────────────────────────┐                │
│  │ SHAP                                         │                │
│  ├──────────────────────────────────────────────┤                │
│  │ 1. Create KernelExplainer(model, background) │                │
│  │ 2. Generate SHAP values for 150 instances    │                │
│  │ 3. Compute Fidelity:                         │                │
│  │    ├─ Deletion AUC (remove features)         │                │
│  │    └─ Insertion AUC (add features)           │                │
│  │ 4. Compute Parsimony:                        │                │
│  │    └─ Sparsity (avg # important features)    │                │
│  └──────────────────────────────────────────────┘                │
│                                                                  │
│  ┌──────────────────────────────────────────────┐                │
│  │ LIME                                         │                │
│  ├──────────────────────────────────────────────┤                │
│  │ 1. Create LimeTabularExplainer               │                │
│  │ 2. Generate explanations (2000 samples)      │                │
│  │ 3. Compute Fidelity (same as SHAP)           │                │
│  │ 4. Compute Stability:                        │                │
│  │    └─ Add noise, measure variance            │                │
│  └──────────────────────────────────────────────┘                │
│                                                                  │
│  ┌──────────────────────────────────────────────┐                │
│  │ Anchor                                       │                │
│  ├──────────────────────────────────────────────┤                │
│  │ 1. Create AnchorTabularExplainer             │                │
│  │ 2. Generate rules (30 instances)             │                │
│  │ 3. Compute:                                  │                │
│  │    ├─ Rule Accuracy (precision)              │                │
│  │    ├─ Rule Applicability (coverage)          │                │
│  │    └─ Rule Length (# conditions)             │                │
│  └──────────────────────────────────────────────┘                │
│                                                                  │
│  ┌──────────────────────────────────────────────┐                │
│  │ DiCE                                         │                │
│  ├──────────────────────────────────────────────┤                │
│  │ 1. Create DiCE explainer                     │                │
│  │ 2. Generate counterfactuals (10 instances)   │                │
│  │ 3. Compute:                                  │                │
│  │    └─ Success Rate (% valid CFs that flip)   │                │
│  └──────────────────────────────────────────────┘                │
│                                                                  │
│  OUTPUT: technical_metrics.csv                                   │
│  ├─ method | fidelity_del | fidelity_ins | sparsity |...         │
│  └─ 4 rows (one per method)                                      │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ 4. PERSONA EVALUATION (evaluation/persona_evaluator.py)          │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Step 1: Generate Explanations (2 instances × 4 methods)         │
│  ┌────────────────────────────────────────────────┐              │
│  │ Instance #581:                               │              │
│  │ ├─ SHAP: "credit_score: 0.23, income: -0.15"   │              │
│  │ ├─ LIME: "credit_score: 0.19, income: 0.05"    │              │
│  │ ├─ Anchor: "IF loan_pct > 0.15 AND..."         │              │
│  │ └─ DiCE: "income: change by +8000"             │              │
│  └────────────────────────────────────────────────┘              │
│                                                                  │
│  Step 2: LLM Evaluation (6 personas × 4 methods × 2 instances)   │
│  = 48 LLM calls                                                  │
│                                                                  │
│  FOR EACH PERSONA:                                               │
│  ┌────────────────────────────────────────────────┐              │
│  │ 1. Build System Prompt:                        │              │
│  │    ├─ Persona identity (name, role, years)     │              │
│  │    ├─ Psychological traits (loss aversion,     │              │
│  │    │   risk tolerance, trust in AI)            │              │
│  │    ├─ Mental model (how they think)            │              │
│  │    ├─ Heuristics (decision rules)              │              │
│  │    ├─ Explanation preferences                  │              │
│  │    └─ Domain context (loan application)        │              │
│  │                                                │              │
│  │ 2. Build Evaluation Prompt:                    │              │
│  │    ├─ Scenario: "You're reviewing Case #581" │              │
│  │    ├─ Show explanation text                    │              │
│  │    └─ Ask: Rate on 6 dimensions (1-5)          │              │
│  │                                                │              │
│  │ 3. Call OpenAI API:                            │              │
│  │    ├─ Model: GPT-4o / o1-mini / etc            │              │
│  │    ├─ Parse TOML response                      │              │
│  │    └─ Extract: ratings + comment               │              │
│  └────────────────────────────────────────────────┘              │
│                                                                  │
│  6 Dimensions Rated:                                             │
│  ├─ interpretability (can you understand it?)                    │
│  ├─ completeness (covers all factors?)                           │
│  ├─ actionability (what to do next?)                             │
│  ├─ trust (rely on this?)                                        │
│  ├─ satisfaction (overall quality?)                              │
│  └─ decision_support (helps your job?)                           │
│                                                                  │
│  OUTPUT: persona_ratings.csv                                     │
│  ├─ persona_name | role | method | instance | run |...           │
│  └─ 48 rows (6 personas × 4 methods × 2 instances × 1 run)       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ 5. RECOMMENDATIONS (evaluation/recommender.py)                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  FOR EACH STAKEHOLDER TYPE:                                      │
│                                                                  │
│  1. Calculate Combined Score:                                    │
│     score = 0.3×fidelity + 0.2×parsimony +                       │
│             0.3×trust + 0.2×satisfaction                         │
│                                                                  │
│  2. Select Best Method:                                          │
│     best_method = argmax(score)                                  │
│                                                                  │
│  3. Generate Reasoning:                                          │
│     "SHAP recommended due to: high trust (3.5/5),                │
│      excellent fidelity, comprehensive coverage"                 │
│                                                                  │
│  OUTPUT: recommendations.json                                    │
│  └─ {stakeholder: {method, score, reasoning, feedback}}          │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ 6. OUTPUTS                                                       │
├──────────────────────────────────────────────────────────────────┤
│  ▶ outputs/hexeval_results/                                      │
│    ├─ technical_metrics.csv                                      │
│    ├─ persona_ratings.csv                                        │
│    └─ recommendations.json                                       │
└──────────────────────────────────────────────────────────────────┘
```

---

## Module-by-Module Breakdown

### 1. Core Modules (`hexeval/core/`)

#### `model_loader.py` (133 lines)

**Purpose:** Load trained ML models from disk

**Key Function:**
```python
def load_model(path: str) -> Dict:
    # Returns:
    {
        "model": <fitted model>,
        "preprocessor": <optional pipeline>,
        "feature_names": ["feature1", "feature2", ...],
        "model_type": "XGBClassifier",
        "threshold": 0.5
    }
```

**Supports:**
- sklearn models (.pkl, .joblib)
- XGBoost models
- Model artifacts (dicts with preprocessor)

**Validation:**
- Checks file exists
- Ensures model has `predict_proba()` method
- Extracts metadata

---

#### `data_handler.py` (179 lines)

**Purpose:** Load and prepare tabular data

**Key Function:**
```python
def load_data(path: str, target_column: str) -> Dict:
    # Returns:
    {
        "X_train": DataFrame,
        "X_test": DataFrame,
        "y_train": Series,
        "y_test": Series,
        "feature_names": ["age", "income", ...],
        "categorical_features": ["job", "region"],
        "numeric_features": ["age", "income", "debt"]
    }
```

**Features:**
- Auto-detects categorical vs numeric
- Stratified train/test split (80/20)
- Handles missing target column gracefully

**Preprocessing:**
```python
def preprocess_for_model(X, preprocessor, feature_names):
    # Applies preprocessing pipeline if provided
    # Ensures correct feature order
    # Returns numpy array ready for model
```

---

#### `validator.py` (166 lines)

**Purpose:** Ensure model and data are compatible

**Key Checks:**
1. Model has `predict_proba()`
2. Expected features exist in data
3. Can make a test prediction
4. Prediction shape is correct (binary classification)

**Returns:**
```python
{
    "status": "valid" | "invalid",
    "warnings": ["Extra features in data: ['col1', 'col2']"],
    "errors": ["Missing required features: ['col3']"]
}
```

---

### 2. Explainer Modules (`hexeval/explainers/`)

#### `shap_explainer.py` (44 lines)

**Wrapper around SHAP library**

```python
class ShapExplainer:
    def __init__(self, model, background, feature_names):
        self.explainer = shap.Explainer(model, background)
    
    def explain_instance(self, x_row):
        # Returns: SHAP values for single instance
        # Shape: (n_features,)
    
    def explain_dataset(self, X):
        # Returns: SHAP values for batch
        # Shape: (n_samples, n_features)
```

**Key Details:**
- Uses `Kernel SHAP` (model-agnostic)
- Background data for baseline (500 samples)
- Handles binary classification (extracts class 1)

---

#### `lime_explainer.py` (85 lines)

**Wrapper around LIME library**

```python
class LimeExplainer:
    def __init__(self, training_data, feature_names, predict_fn):
        self.explainer = LimeTabularExplainer(
            training_data,
            feature_names=feature_names,
            class_names=["Reject", "Approve"],
            mode="classification"
        )
    
    def explain_instance(self, x_row, num_features=10):
        # Returns: [(feature_name, weight), ...]
    
    def as_importance_vector(self, x_row):
        # Returns: numpy array of shape (n_features,)
        # All features, zero for non-selected
```

**Configuration:**
- `num_samples=2000` - perturbations
- `num_features=10` - top features to explain

---

#### `anchor_explainer.py` (103 lines)

**Rule-based explanations**

```python
class AnchorExplainer:
    def __init__(self, X_train, feature_names, predict_fn):
        self.explainer = AnchorTabularExplainer(
            class_names=["Reject", "Approve"],
            feature_names=feature_names,
            train_data=X_train
        )
    
    def explain_instance(self, x_row, threshold=0.9):
        # Returns: Anchor object with
        #  - anchor.names(): ["feature1 > 0.5", "feature2 <= 10"]
        #  - anchor.precision(): 0.95
        #  - anchor.coverage(): 0.32
```

**Output:**
- IF-THEN rules (e.g., "IF income > $50K AND debt < 30%")
- Precision: accuracy of rule
- Coverage: % of data rule applies to

---

#### `dice_counterfactuals.py` (138 lines)

**Counterfactual generation**

```python
class DiceExplainer:
    def __init__(self, model, X_train, y_train, feature_names):
        self.dice_exp = dice_ml.Dice(
            data_interface,
            model_interface,
            method="random"  # or "genetic"
        )
    
    def generate_counterfactuals(self, x_row, total_cfs=3):
        # Returns: DiCE CF object with
        #  - final_cfs_df: DataFrame of counterfactuals
        #  - Each CF flips the prediction
```

**Output:**
- "To get approved: increase income by $8000"
- Minimal changes to flip outcome

---

### 3. Metrics Modules (`hexeval/metrics/`)

#### `fidelity.py` (88 lines)

**Insertion/Deletion AUC** (Covert & Lundberg 2021)

```python
def insertion_deletion_auc(model, X, importances, baseline):
    # For each instance:
    #   1. Rank features by |importance|
    #   2. Deletion: Remove features in order, measure drop
    #   3. Insertion: Add features in order, measure rise
    #   4. Compute AUC of curves
    
    # Returns:
    {
        "deletion_auc": 0.108,  # Lower is better
        "insertion_auc": 0.249   # Higher is better
    }
```

**Interpretation:**
- Deletion AUC = 0.108: Model drops 10.8% when top features removed
- Insertion AUC = 0.249: Model rises 24.9% when top features added
- Good fidelity: deletion < 0.15, insertion > 0.20

---

#### `parsimony_coverage.py` (71 lines)

**Sparsity:** How many features used?

```python
def sparsity_from_importances(importances):
    # Count features with |importance| > threshold
    # Average across instances
    # Returns: avg number of important features
```

**Anchor Metrics:**
```python
def anchor_parsimony_and_coverage(anchor_exp):
    # Returns:
    {
        "precision": 0.95,     # Rule accuracy
        "coverage": 0.32,      # % data covered
        "n_conditions": 3      # Rule length
    }
```

---

#### `robustness.py` (62 lines)

**Stability:** Do explanations change with noise?

```python
def explanation_stability(explain_fn, x_row, noise_std=0.02):
    # 1. Generate 5 noisy versions of x_row
    # 2. Get explanation for each
    # 3. Measure variance
    
    # Returns: std deviation of explanations
```

**Interpretation:**
- Stability < 0.1: Robust explanations
- Stability > 0.2: Unstable (LIME often shows this)

---

### 4. Evaluation Modules

#### `technical_evaluator.py` (322 lines)

**Orchestrates technical metrics**

```python
def run_technical_evaluation(model_artifact, data, config):
    # For each method (SHAP, LIME, Anchor, DiCE):
    #   1. Create explainer
    #   2. Generate explanations (150 instances)
    #   3. Compute metrics:
    #      - SHAP/LIME: fidelity, parsimony, stability
    #      - Anchor: precision, coverage, rule length
    #      - DiCE: success rate
    #   4. Return DataFrame row
    
    # Returns: DataFrame with 4 rows (one per method)
```

**Sample Output:**
| method | fidelity_del | fidelity_ins | sparsity | stability |
|--------|--------------|--------------|----------|-----------|
| SHAP   | 0.108        | 0.249        | 24.0     | NaN       |
| LIME   | 0.131        | 0.212        | 10.0     | 0.163     |
| Anchor | NaN          | NaN          | NaN      | NaN       |
| DiCE   | NaN          | NaN          | NaN      | NaN       |

---

#### `persona_evaluator.py` (488 lines) - **MOST COMPLEX MODULE**

**LLM-based human simulation**

**Architecture:**
```
run_persona_evaluation()
  ├─ _generate_explanations()        [115-203]
  │   └─ Creates text explanations for 2 instances × 4 methods
  │
  ├─ _evaluate_with_llm()             [206-273]
  │   └─ Loops: 6 personas × 4 methods × 2 instances × 1 run
  │       ├─ _build_system_prompt()   [276-381]
  │       │   └─ Rich persona context + domain config
  │       │
  │       ├─ _build_eval_prompt()     [384-421]
  │       │   └─ Scenario + explanation + rating questions
  │       │
  │       └─ _call_llm()              [424-478]
  │           └─ OpenAI API call + TOML parsing
  │
  └─ Returns: DataFrame with 48 rows
```

**Prompt Engineering (Lines 276-421):**

**System Prompt Structure:**
1. **Identity** (who you are)
   - Name, role, experience
   - Example: "You are Margaret Chen, Conservative Loan Officer with 18 years..."

2. **Psychological Profile**
   - Loss aversion: 2.5× (pain of bad loan)
   - Risk tolerance: Very Low
   - Trust in AI: Low
   - Decision speed: Slow (methodical)

3. **Mental Model**
   - "Credit score and payment history are paramount..."

4. **Heuristics**
   - "If CreditScore < 650, lean heavily toward reject"
   - "Employment < 12 months is concerning"

5. **Explanation Preferences**
   - "Prefers simple, rule-based (IF-THEN)"
   - "Distrusts complex statistical methods"

6. **Domain Context** (NEW: configurable)
   - Prediction task: "loan default risk"
   - Decision: "approve or reject"
   - Stakeholder context: "at a financial institution"
   - End-user context: "applying for a loan"

7. **Rating Instructions**
   - 6 dimensions: interpretability, completeness, actionability, trust, satisfaction, decision_support
   - 1-5 scale with specific definitions

**User Prompt Structure:**
```
Case #581 - Loan Application Review

AI Prediction: High default risk (0.73 > 0.50 threshold)

Explanation Method: SHAP

Explanation Text:
"Top SHAP values: credit_score: 0.234; income: -0.156; debt_ratio: 0.123"

YOUR TASK:
Rate this explanation on 6 dimensions (1-5) from YOUR perspective as
a Conservative Loan Officer.

Respond in TOML format:
interpretability = <1-5>
...
comment = "<your thoughts>"
```

**LLM Response Parsing:**
```toml
interpretability = 2
completeness = 2
actionability = 1
trust = 1
satisfaction = 2
decision_support = 1
comment = "As a Conservative Loan Officer, the SHAP values are not easy
to interpret. They provide numerical weights without clear context or
thresholds that align with my decision-making parameters."
```

---

#### `recommender.py` (226 lines)

**Generate stakeholder-specific recommendations**

**Algorithm:**
```python
def generate_recommendations(technical_metrics, persona_ratings, config):
    # For each unique stakeholder role:
    
    # 1. Get their ratings for all methods
    # 2. Get technical metrics for all methods
    
    # 3. Calculate combined score:
    score = (
        0.3 × fidelity_score +
        0.2 × parsimony_score +
        0.3 × (trust / 5) +
        0.2 × (satisfaction / 5)
    )
    
    # 4. Select best method: argmax(score)
    
    # 5. Generate reasoning:
    "SHAP recommended due to: high stakeholder trust (3.5/5),
     excellent fidelity (0.108 deletion), comprehensive coverage"
    
    # Returns:
    {
        "Conservative Loan Officer": {
            "recommended_method": "Anchor",
            "score": 0.72,
            "reasoning": "...",
            "technical_strengths": {...},
            "persona_feedback": "...",
            "alternatives": {"LIME": 0.65, "SHAP": 0.58}
        }
    }
```

---

#### `personas.py` (226 lines)

**6 Stakeholder Personas**

**Structure:**
```python
PERSONAS = [
    {
        "name": "Margaret Chen",
        "role": "Conservative Loan Officer",
        "experience_years": 18,
        "loss_aversion": 2.5,
        "risk_tolerance": "Very Low",
        "decision_speed": "Slow (methodical)",
        "trust_in_ai": "Low (prefers human oversight)",
        "priorities": ["Actionability", "Trust", "Clear rules"],
        "mental_model": "Credit score paramount. Defaults catastrophic.",
        "heuristics": [
            "If CreditScore < 650, lean heavily toward reject",
            "Employment < 12 months is concerning"
        ],
        "explanation_preferences": "Simple, rule-based (IF-THEN)",
        "behavioral_signature": {
            "favors_simplicity": True,
            "prefers_conservative_errors": True,
            "values_precedent": True
        }
    },
    # ... 5 more personas
]
```

### **Layer 3: Evaluation** (`hexeval/evaluation/`)
Orchestrates dual evaluation: technical + human-centered

**Key Components:**
- `evaluator.py`: Main orchestrator (load → evaluate → recommend)
- `technical_evaluator.py`: Runs SHAP/LIME/Anchor/DiCE, computes metrics
- `persona_evaluator.py`: LLM-driven persona simulation
- `personas.py`: 6 stakeholder profiles (redesigned Jan 2026)
- `recommender.py`: Maps stakeholder needs → best XAI method

**Personas (Gender-Neutral, Bias-Free):**
1. **Jordan Walsh** - Policy-Focused Loan Officer (18 years, conservative, values clear rules)
2. **Sam Chen** - Model Validation Analyst (5 years, data-driven, validates model behavior)
3. **Taylor Kim** - Compliance & Risk Officer (22 years, extremely risk-averse, regulatory focus)
4. **Morgan Patel** - Customer Success Manager (12 years, relationship-focused, communicates to customers)
5. **Casey Rodriguez** - Strategic Planning Director (15 years, fast/strategic, portfolio-level thinking)
6. **Riley Martinez** - Loan Applicant/End User (0 years, goal-oriented consumer, needs plain language)

**Design Philosophy:**
- **Need-based priorities** (not metric names like "Fidelity")
- **Nuanced heuristics** (compensating factors, not rigid rules)
- **Empowered framing** (goal-oriented, not anxious/vulnerable)
- **No arbitrary numbers** (descriptive traits, not loss_aversion coefficients)

**Evaluation Flow:**
## Configuration System

### `eval_config.yaml` (107 lines)

**Domain Configuration** (Lines 4-20) - **KEY INNOVATION**

```yaml
domain:
  name: "Credit Risk / Loan Default"
  prediction_task: "loan default risk"
  decision_verb: "approve or reject"
  decision_noun: "loan application"
  stakeholder_context: "at a financial institution"
  end_user_context: "applying for a loan to improve my life"
  positive_outcome: "loan approval"
  negative_outcome: "loan rejection"
```

**Why This Matters:**
- **Domain-Agnostic:** Change these 9 lines → works for any domain
- **Persona Prompts:** Injected into LLM system/user prompts
- **Examples:** Can adapt to healthcare, hiring, fraud detection

**Evaluation Settings** (Lines 25-63)

```yaml
evaluation:
  sample_size: 150           # Instances for technical metrics
  random_state: 42
  
  fidelity:
    steps: 50                # Granularity of insertion/deletion
  
  stability:
    noise_std: 0.02
    repeats: 5
  
  explainers:
    shap:
      enabled: true
      background_size: 500   # Background samples for SHAP
    
    lime:
      enabled: true
      num_samples: 2000      # Perturbations per explanation
      num_features: 10       # Top features
      stability_test: true
    
    anchor:
      enabled: true
      precision_threshold: 0.9
      max_instances: 30      # Anchor is slow
    
    dice:
      enabled: true
      num_counterfactuals: 3
      max_instances: 10      # DiCE is slow
```

**Persona Configuration** (Lines 65-88)

```yaml
personas:
  enabled: true
  llm_model: "gpt-4o"        # Fast and cheap
  # Options: gpt-4, gpt-4-turbo, gpt-4o, o1-mini, o3-mini
  
  runs_per_method: 1         # Evaluations per method
  sample_instances: 2        # Instances to evaluate
  top_k_features: 5          # Features in explanations
  
  include:                   # Which personas to use
    - "Conservative Loan Officer"
    - "Data-Driven Analyst"
    - "Risk Manager"
    - "Customer Relationship Manager"
    - "Executive Decision Maker"
    - "Loan Applicant (End User)"
```

**Total LLM Calls:** 6 personas × 4 methods × 2 instances × 1 run = 48 calls

**Recommendation Weights** (Lines 99-106)

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

## Key Algorithms

### 1. Fidelity: Insertion/Deletion AUC

**Reference:** Covert & Lundberg (2021) - "Explaining by Removing"

**Algorithm:**
```
For each instance x:
  1. Get feature importances: I = [i1, i2, ..., in]
  2. Rank features: [f3, f1, f7, ...] (by |importance|)
  
  DELETION:
  3. Start with full instance: x_full
  4. Remove top-k features (set to baseline):
     k=1: x_del = x with f3 = baseline[f3]
     k=2: x_del = x with f3, f1 = baseline
     ...
  5. Measure prediction drop at each step
  6. Plot: (k, prediction)
  7. Compute AUC of curve
  
  INSERTION:
  8. Start with baseline: x_base = [0, 0, ..., 0]
  9. Add top-k features (set to original):
     k=1: x_ins = baseline with f3 = x[f3]
     k=2: x_ins = baseline with f3, f1 = x
     ...
  10. Measure prediction rise at each step
  11. Plot: (k, prediction)
  12. Compute AUC of curve

Average across all instances
```

**Interpretation:**
- **Deletion AUC ≈ 0.10**: Model drops 10% when removing important features (GOOD - features matter!)
- **Insertion AUC ≈ 0.25**: Model rises 25% when adding important features (GOOD - features help!)
- **Deletion should be < Insertion** (more impact from adding than removing)

---

### 2. Persona Evaluation Flow

**Nested Loop Structure:**

```
FOR each persona in PERSONAS:                    [6 personas]
  system_prompt = build_system_prompt(persona)
  
  FOR each instance_idx in sample_instances:    [2 instances]
    FOR each method in [SHAP, LIME, Anchor, DiCE]:  [4 methods]
      FOR each run in range(runs_per_method):   [1 run]
        
        # Build prompts
        explanation = explanations[instance_idx][method]
        user_prompt = build_eval_prompt(instance_idx, explanation, method)
        
        # Call LLM
        response = openai.chat.completions.create(
          model="gpt-4o",
          messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
          ]
        )
        
        # Parse TOML response
        ratings = parse_toml(response.content)
        
        # Store result
        results.append({
          "persona_name": persona["name"],
          "persona_role": persona["role"],
          "explanation_type": method,
          "instance_index": instance_idx,
          "run": run,
          "interpretability": ratings["interpretability"],
          "completeness": ratings["completeness"],
          "actionability": ratings["actionability"],
          "trust": ratings["trust"],
          "satisfaction": ratings["satisfaction"],
          "decision_support": ratings["decision_support"],
          "comment": ratings["comment"]
        })

TOTAL: 6 × 2 × 4 × 1 = 48 LLM API calls
```

---

## Usage Examples

### Basic Usage

```python
from hexeval import evaluate

# Run complete evaluation
results = evaluate(
    model_path="xgboost_credit_risk_new.pkl",
    data_path="credit_risk_dataset.csv",
    target_column="loan_status"
)

# Access results
print(results["technical_metrics"])
print(results["persona_ratings"])
print(results["recommendations"])
```

### Custom Configuration

```python
results = evaluate(
    model_path="my_model.pkl",
    data_path="my_data.csv",
    target_column="target",
    config_path="custom_config.yaml",
    output_dir="my_results/"
)
```

### Programmatic Access

```python
from hexeval.core import load_model, load_data
from hexeval.evaluation import run_technical_evaluation

# Load components
model_artifact = load_model("model.pkl")
data = load_data("data.csv", target_column="target")

# Run only technical evaluation
config = {"evaluation": {...}}
tech_results = run_technical_evaluation(model_artifact, data, config)

print(tech_results[["method", "fidelity_deletion", "fidelity_insertion"]])
```

### Streamlit UI

```bash
streamlit run hexeval/ui/app.py
```

**UI Features:**
- Overview metrics
- Technical results table + explanations
- Persona ratings table + radar chart
- **Persona-wise analysis** (expandable cards with ratings + comments)
- Recommendations with comparison matrix

---

## Performance & Scalability

### Runtime Estimates

**Technical Evaluation:**
| Method | Time per Instance | 150 Instances |
|--------|------------------:|-------------: |
| SHAP   | 0.5s              | ~75s          |
| LIME   | 1.0s              | ~150s         |
| Anchor | 2.0s              | ~60s (30 inst)|
| DiCE   | 3.0s              | ~30s (10 inst)|
| **TOTAL** |                | **~5 minutes**|

**Persona Evaluation (with LLM):**
- 48 API calls × 1-2s per call = **~2 minutes**
- Cost: ~$0.18 with GPT-4o

**Total Runtime:** ~7-8 minutes per evaluation

---

## Code Statistics

**Total Lines:** 2,584
**Total Files:** 23 Python files

**Breakdown by Module:**

| Module | Files | Lines | Purpose |
|--------|-------|-------|---------|
| Core | 3 | 478 | Model/data loading, validation |
| Explainers | 4 | 370 | SHAP, LIME, Anchor, DiCE wrappers |
| Metrics | 3 | 221 | Fidelity, parsimony, stability |
| Evaluation | 4 | 1462 | Technical + Persona evaluation |
| UI | 1 | 290 | Streamlit interface |
| **TOTAL** | **23** | **2584** | |

**Most Complex Modules:**
1. `persona_evaluator.py` - 488 lines (LLM integration)
2. `technical_evaluator.py` - 322 lines (orchestration)
3. `recommender.py` - 226 lines (scoring algorithm)

---

## 📐 Key Design Decisions

### 1. Domain-Agnostic Configuration
**Decision:** Use `eval_config.yaml` to adapt HEXEval to any ML domain  
**Rationale:** Makes the framework reusable beyond credit risk

**Implementation:**
```yaml
domain:
  name: "Credit Risk Analysis"
  prediction_task: "loan default risk"
  positive_outcome: "loan approval"
  negative_outcome: "loan rejection"
  # Change these 4 lines → works for healthcare, fraud, churn, etc.
```

**Benefit:** Research labs can use HEXEval for their own domains without code changes

---

### 2. Dual Evaluation: Technical + Human-Centered
**Decision:** Combine objective metrics (fidelity, parsimony) with LLM-simulated persona feedback  
**Rationale:** Technical metrics alone miss the human interpretability gap

**Key Innovation:** 
- Technical metrics: What XAI methods *can* do (fidelity, stability)
- Persona ratings: What stakeholders *actually* find useful (trust, actionability)
- **The gap reveals**: Methods with high fidelity but low trust = unusable in practice

**Example Finding:**
- SHAP: 0.89 fidelity, but 2.5/5 trust from Margaret (too many features)
- Anchor: Lower fidelity, but 3.5/5 trust (simple IF-THEN rules)

---

### 3. Bias-Free Persona Design (2026-01-15 Redesign)
**Decision:** Use gender-neutral names, need-based priorities, nuanced heuristics  
**Rationale:** Avoid circular evaluation and stereotype hallucination

**Changes Made:**
1. **Name Bias Removed:**
   - ❌ OLD: Margaret Chen (Asian woman = conservative stereotype)
   - ✅ NEW: Jordan Walsh (gender-neutral)

2. **Metric-Hacking Eliminated:**
   - ❌ OLD: "David wants Fidelity" (this is the metric name!)
   - ✅ NEW: "Sam wants to detect model errors" (let LLM decide if SHAP helps)

3. **Nuanced Heuristics:**
   - ❌ OLD: "If CreditScore < 650, reject" (unrealistic)
   - ✅ NEW: "Skeptical of low scores unless offset by tenure/assets" (realistic)

**Impact:** No pre-determined winners, more authentic evaluation

---

### 4. LLM-Based Persona Simulation
**Decision:** Use OpenAI GPT-4/GPT-4o to simulate stakeholder feedback  
**Rationale:** Fast, reproducible, and scalable vs real user studies

**Advantages:**
- **Speed:** 48 LLM calls in ~2 minutes vs weeks of human studies
- **Reproducibility:** Same results every run (temperature=0)
- **Cost:** ~$0.50 vs $1000s for participant recruitment
- **Iteration:** Test 10 personas in 1 hour vs months

**Validation Strategy:**
- Diversity in ratings (1.0-3.5 trust) shows LLM isn't always agreeing
- Future work: Validate LLM personas against real human feedback

---

### 5. Method-Specific Technical Scoring (Bug Fix 2026-01-15)
**Decision:** Score each XAI method on its own strengths, not one-size-fits-all  
**Rationale:** Anchor doesn't produce fidelity scores, shouldn't be penalized

**Implementation:**
```python
if method == "SHAP/LIME":
    score = (fidelity + parsimony) / 2
elif method == "Anchor":
    score = 0.8 × precision + 0.2 × coverage
elif method == "DiCE":
    score = success_rate
```

**Before Fix:** Anchor got 0 on 50% of recommendation score (unfair!)  
**After Fix:** Anchor scores 0.81 (precision 0.95) vs SHAP 0.67 (competitive!)

---

### 6. Rich Context Engineering for LLMs
**Decision:** 3000-character prompts with detailed persona profiles  
**Rationale:** Generic prompts = generic ratings; rich context = authentic simulation

**Prompt Structure:**
1. **Identity:** Role, experience, risk profile (60 lines)
2. **Mental Model:** How this persona thinks about decisions
3. **Heuristics:** Specific decision rules they follow
4. **Priorities:** What matters most in explanations (NOT metric names!)
5. **Task:** Rate 6 dimensions with reasoning

**Example:**
```
You are Jordan Walsh, a Policy-Focused Loan Officer with 18 years experience.

YOUR MENTAL MODEL:
Credit history matters most, but compensating factors can offset weak areas.
A low score with strong employment and low debt might still be acceptable...

PRIORITIES:
1. Being able to justify decisions to management and customers
2. Confidence in the soundness of the recommendation
3. Clear guidance on what factors drove the decision

[NOT: "Fidelity" or "Parsimony" - those are OUR metrics, not their language!]
```

---

### 7. Modular, Extensible Architecture
**Decision:** 3-layer design (Core → Explainers → Evaluation)  
**Rationale:** Easy to add new explainers, metrics, or personas

**Adding New Components:**
- **New Explainer:** Drop file in `hexeval/explainers/`, implement 2 methods, done
- **New Metric:** Add function to `hexeval/metrics/`, import in evaluator
- **New Persona:** Add dict to `personas.py` with 8 fields, auto-integrated
- **New Domain:** Change 9 lines in `eval_config.yaml`

**Extension Example (5min task):**
```python
# Add Integrated Gradients explainer
class IGExplainer:
    def explain_instance(self, X, y):
        # ...implementation
        
# That's it! evaluator.py automatically picks it up
```

---

### 8. Production-Ready Engineering (2026-01-15 Improvements)

**Load Existing Results (UI Enhancement):**
- Detects existing evaluations in `outputs/hexeval_results/`
- One-click load = instant results (no $3 re-run!)
- Perfect for iterative UI testing

**Bug Fixes Applied:**
- ✅ Target column validation (clear error messages)
- ✅ Recommender alternatives (UI now shows all method scores)
- ✅ Method-specific technical scoring (fair comparison)
- ✅ Comprehensive error handling

**Quality Assurance:**
- 16-issue bug audit completed
- All critical bugs fixed
- Import validation, edge case handling
- Ready for thesis/production use

---

## 🚀 Recent Improvements (January 2026)

### Persona Redesign for Methodological Rigor
**What Changed:**
- 6 gender-neutral personas (Jordan, Sam, Taylor, Morgan, Casey, Riley)
- Need-based priorities replace metric names
- Nuanced heuristics with compensating factors
- Empowered end-user (goal-oriented, not anxious)

**Why It Matters:**
- Eliminates evaluation bias
- No pre-determined winners
- Easier to defend in thesis
- More realistic stakeholder simulation

### Bug Fixes & Robustness
**Critical Fixes:**
1. Method-specific scoring (Anchor/DiCE now competitive)
2. Target validation (better error messages)
3. Recommender alternatives (UI displays all scores)

**Impact:**
- More reliable evaluations
- Production-ready code
- Better error messages for debugging

### UI Enhancements
**New Features:**
- Load existing results (avoid costly re-runs)
- Clearer progress indicators
- Alternatives display in recommendations

---

## 📊 Validation & Results

**Diversity in Ratings:** Trust scores range 1.0-3.5 across personas/methods  
**Method Rankings Vary:** No single method dominates all personas  
**Fair Competition:** Anchor (0.81) competitive with SHAP (0.67) after fix

**Example:** Riley Martinez (End User) rated DiCE highest for actionability (counterfactuals show "what to change")

---

## Future Extensions

### 1. Add New Explainer
```python
# hexeval/explainers/my_explainer.py
class MyExplainer:
    def explain_instance(self, x_row):
        # Return importance scores
        pass

# hexeval/evaluation/technical_evaluator.py
def _evaluate_my_method(model, X, config):
    explainer = MyExplainer(...)
    # Compute metrics
    return {...}
```

### 2. Add New Persona
```python
# hexeval/evaluation/personas.py
PERSONAS.append({
    "name": "Your Persona",
    "role": "Your Role",
    "experience_years": 10,
    ...
})
```

### 3. Change Domain
```yaml
# hexeval/config/eval_config.yaml
domain:
  name: "Medical Diagnosis"
  prediction_task: "disease diagnosis"
  decision_verb: "diagnose and treat"
  stakeholder_context: "at a healthcare facility"
```

---

**Document Version:** 1.0  
**Last Updated:** 2026-01-15  
**Total Pages:** ~15  
**Target Audience:** Developers, Researchers, Thesis Reviewers
