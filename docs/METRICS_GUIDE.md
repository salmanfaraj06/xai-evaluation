# HEXEval Metrics Guide

Understanding the technical metrics used in HEXEval evaluation.

## Fidelity Metrics

Measure how well explanations reflect actual model behavior.

### Deletion AUC (Lower is Better)

**What it measures:** How much the prediction changes when important features are removed.

**How it works:**
1. Start with all features
2. Progressively remove features in order of importance
3. Measure prediction change at each step
4. Calculate area under the curve

**Interpretation:**
- **Lower score = Better fidelity**
- Features identified as important actually matter to the model
- Good range: 0.2 - 0.4
- Poor: > 0.6

**Example:**
- SHAP identifies "credit_score" as most important
- Removing it drops prediction from 0.8 to 0.2
- This confirms high fidelity

### Insertion AUC (Higher is Better)

**What it measures:** How much the prediction improves when important features are added.

**How it works:**
1. Start with baseline (mean values)
2. Progressively add features in order of importance
3. Measure prediction change at each step
4. Calculate area under the curve

**Interpretation:**
- **Higher score = Better fidelity**
- Adding important features moves prediction toward true value
- Good range: 0.6 - 0.8
- Poor: < 0.4

---

## Parsimony Metrics

Measure explanation simplicity and comprehensibility.

### Number of Important Features (Lower is Better)

**What it measures:** How many features the explanation highlights.

**Interpretation:**
- **Lower = More parsimonious** (simpler)
- Humans can only process ~5-7 items at once
- Good: < 10 features
- Acceptable: 10-20 features
- Too complex: > 20 features

**Method differences:**
- SHAP: Often highlights many features (10-15)
- LIME: Moderate (5-10)
- Anchor: Very few (2-5 conditions)

### Rule Length (Anchor only)

**What it measures:** Number of conditions in the IF-THEN rule.

**Interpretation:**
- Lower = Simpler rule
- Good: 2-4 conditions
- Acceptable: 5-7 conditions
- Too complex: > 7 conditions

**Example:**
- Good: `IF credit_score < 650 AND income < 50k THEN high_risk`
- Too complex: `IF (credit_score < 650 AND income < 50k) OR (age > 60 AND debt_ratio > 0.4) OR ...`

---

## Coverage Metrics (Anchor only)

### Rule Accuracy (Higher is Better)

**What it measures:** When the rule applies, how often is it correct?

**Interpretation:**
- **Higher = More reliable**
- Good: > 0.85
- Acceptable: 0.75 - 0.85
- Poor: < 0.75

**Example:**
- Rule: `IF credit_score < 650 THEN default`
- Accuracy: 0.92
- Means: 92% of people with credit_score < 650 actually defaulted

### Rule Applicability (Coverage)

**What it measures:** What % of dataset does the rule apply to?

**Interpretation:**
- Higher = More generalizable
- But: Trade-off with accuracy (higher coverage → lower accuracy)
- Good: 0.15 - 0.30 (15-30% of cases)
- Narrow: < 0.10
- Too broad: > 0.50

---

## Counterfactual Metrics (DiCE only)

### Counterfactual Success Rate (Higher is Better)

**What it measures:** % of generated counterfactuals that actually flip the prediction.

**Interpretation:**
- **Higher = More valid**
- Good: > 0.80
- Acceptable: 0.60 - 0.80
- Poor: < 0.60

**Example:**
- Instance: "Reject loan" (prob = 0.8)
- Counterfactual: "Increase income by $10k"
- Success: New prediction = "Approve" (prob = 0.3)

### Counterfactual Sparsity (Lower is Better)

**What it measures:** Average number of features changed in generated counterfactuals.

**Interpretation:**
- **Lower = Simpler, more actionable**
- Good: <= 2
- Acceptable: 3 - 5
- Poor: > 5

**Example:**
- Counterfactual A changes 1 feature (income +$2k) -> sparse, easy to act on
- Counterfactual B changes 12 features -> complex, hard to act on

---

## Stability Metrics (LIME only)

### Stability Score (Higher is Better)

**What it measures:** Do similar inputs get similar explanations?

**How it works:**
1. Generate explanation for instance
2. Add small noise to instance
3. Generate explanation for perturbed instance
4. Measure cosine similarity

**Interpretation:**
- **Higher = More stable/robust**
- Good: > 0.85
- Acceptable: 0.70 - 0.85
- Poor: < 0.70

---

## Method Comparison Summary

| Method | Best For | Fidelity | Parsimony | Special Metrics |
|--------|----------|----------|-----------|-----------------|
| **SHAP** | Technical users | ⭐⭐⭐⭐⭐ | ⭐⭐ | - |
| **LIME** | Balanced use | ⭐⭐⭐⭐ | ⭐⭐⭐ | Stability |
| **Anchor** | Non-technical users | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Rule accuracy, Coverage |
| **DiCE** | Actionable recourse | - | - | Counterfactual success, Counterfactual sparsity |

---

## How to Choose?

**If you need:**
- **Comprehensive technical understanding** → SHAP
- **Balance of accuracy and simplicity** → LIME
- **Simple rules for non-experts** → Anchor
- **Actionable steps for users** → DiCE

**Consider your stakeholders:**
- **Data scientists** → SHAP (high fidelity)
- **Business analysts** → LIME (balanced)
- **Loan officers/decision makers** → Anchor (simple rules)
- **Customer support** → DiCE (actionable guidance)

---

## Typical Good Scores

| Metric | SHAP | LIME | Anchor | DiCE |
|--------|------|------|--------|------|
| Fidelity (deletion) | 0.25 | 0.32 | - | - |
| Fidelity (insertion) | 0.72 | 0.65 | - | - |
| Important features | 12 | 8 | 3 | - |
| Rule accuracy | - | - | 0.88 | - |
| Rule coverage | - | - | 0.22 | - |
| CF success | - | - | - | 0.85 |
| CF sparsity (avg changed features) | - | - | - | 2.0 |
| Stability | - | 0.82 | - | - |

Use these as benchmarks when evaluating your model!
