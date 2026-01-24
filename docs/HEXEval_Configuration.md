# Configuration Guide

This guide explains how to configure HEXEval to evaluate XAI methods for your specific use case. You'll learn how to customize the evaluation context, adjust technical settings, and define stakeholder personas.

---

## Quick Start

A HEXEval configuration file has 4 sections:

1. **Domain** - Your business context (healthcare, finance, etc.)
2. **Evaluation** - Technical settings and which methods to test
3. **Personas** - Stakeholder types to simulate
4. **Recommendations** - How to weight different criteria

---

## 1. Domain: Define Your Context

The domain section tells HEXEval about your use case so it can generate relevant feedback from simulated stakeholders.

### Example: Credit Risk

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
  
  terms:
    applicant: "borrower"
    application: "loan application"
    risk_factor: "default risk"
    decision_maker: "loan officer"
```

### What Each Field Means

- **`name`**: Short name for your use case
- **`prediction_task`**: What your model predicts (e.g., "heart disease risk", "fraud detection")
- **`decision_verb`**: Action taken (e.g., "diagnose", "approve", "flag")
- **`decision_noun`**: What's being decided (e.g., "patient", "loan application", "transaction")
- **`stakeholder_context`**: Where your stakeholders work
- **`end_user_context`**: The end user's perspective
- **`positive_outcome`**: Label for good outcome (e.g., "healthy", "approved")
- **`negative_outcome`**: Label for bad outcome (e.g., "disease", "rejected")

> **Tip**: Be specific! Better context = more relevant persona feedback.

---

## 2. Evaluation: Configure Technical Testing

Control which XAI methods to test and how rigorously to evaluate them.

### Basic Settings

```yaml
evaluation:
  sample_size: 150        # How many data points to test (50-500)
  random_state: 42        # For reproducible results
```

**Recommendations**:
- **Testing**: 50-100 samples
- **Production**: 150-300 samples
- **Research**: 300-500 samples

### Fidelity & Stability

```yaml
  fidelity:
    steps: 50             # Granularity of accuracy tests (20-50)
  
  stability:
    noise_std: 0.02       # How much noise to add (0.01-0.05)
    repeats: 5            # Times to repeat tests (3-10)
```

**What this tests**:
- **Fidelity**: Does the explanation accurately reflect the model?
- **Stability**: Does the explanation change with small input variations?

### Explainer Settings

Enable/disable each method and configure its parameters:

#### SHAP (Feature Attribution)

```yaml
  explainers:
    shap:
      enabled: true
      background_size: 500    # Background samples (100-1000)
```

**When to use**: Comprehensive, mathematically rigorous explanations

#### LIME (Local Explanations)

```yaml
    lime:
      enabled: true
      num_samples: 3000       # Perturbations (500-5000)
      num_features: 10        # Top features to show (5-15)
      stability_test: true    # Test robustness
      stability_subset: 30    # Instances to test (20-50)
```

**When to use**: Faster than SHAP, good balance of speed and accuracy

#### Anchor (Rule-Based)

```yaml
    anchor:
      enabled: true
      precision_threshold: 0.9    # Rule accuracy (0.8-0.95)
      max_instances: 30           # Limit instances (slow)
```

**When to use**: Need simple IF-THEN rules for auditing

#### DiCE (Counterfactuals)

```yaml
    dice:
      enabled: true
      num_counterfactuals: 3      # How many CF examples to generate per instance
      max_instances: 10           # Limit instances (slow)
      method: "random"            # Generation method
```

**When to use**: End users need actionable "what-if" guidance

---

## 3. Personas: Simulate Stakeholders

Define which stakeholder types to simulate using AI.

```yaml
personas:
  enabled: true
  file: "hexeval/config/personas_credit_risk.yaml"
  llm_model: "gpt-4o"
  runs_per_method: 1
  sample_instances: 2
  top_k_features: 5
```

### Settings Explained

- **`enabled`**: Turn persona evaluation on/off
- **`file`**: Path to your persona definitions (see below)
- **`llm_model`**: AI model to use
  - `"gpt-4o"` - Fast and cost-effective (recommended)
  - `"gpt-4"` - More thorough but slower
  - `"gpt-4-turbo"` - Faster GPT-4
- **`runs_per_method`**: How many times to evaluate each method (1-3)
- **`sample_instances`**: Data points to show each persona (2-5)
- **`top_k_features`**: Features to include in explanations (3-7)

### Cost Optimization

**Total AI calls** = Personas × Methods × Instances × Runs

Example: 6 personas × 4 methods × 2 instances × 1 run = **48 calls**

**To reduce costs**:
- Use `llm_model: "gpt-4o"` (cheapest)
- Set `sample_instances: 2`
- Set `runs_per_method: 1`

---

## 4. Creating Persona Files

Persona files define the stakeholders who will evaluate your explanations.

### Persona Structure

```yaml
- name: "Dr. Sarah Jenkins"
  role: "Lead Cardiologist"
  experience_years: 15
  risk_profile: "Extremely risk-averse regarding patient safety"
  decision_style: "Evidence-based, relies on clinical guidelines"
  ai_comfort: "Low to Medium - sees AI as a 'second opinion'"
  
  priorities:
    - "Clinical validity of the risk factors"
    - "Patient safety and early detection"
    - "Understanding the biological mechanism"
  
  mental_model: |
    Heart disease is a complex interplay of physiology and lifestyle.
    Risk factors like chest pain and stress test results are critical.
  
  heuristics:
    - "Chest pain and max heart rate are major indicators"
    - "Abnormal ST depression is a strong warning sign"
  
  explanation_preferences: |
    Needs explanations that use medical terminology and align 
    with clinical guidelines.
```

### Field Guide

| Field | Purpose | Example |
|-------|---------|---------|
| `name` | Persona's name | "Dr. Sarah Jenkins" |
| `role` | Job title | "Lead Cardiologist" |
| `experience_years` | Years in role | 15 |
| `risk_profile` | Risk tolerance | "Extremely risk-averse" |
| `decision_style` | How they decide | "Evidence-based" |
| `ai_comfort` | Trust in AI | "Low to Medium" |
| `priorities` | What matters most | List of 3-5 items |
| `mental_model` | How they think | Paragraph description |
| `heuristics` | Rules of thumb | List of key beliefs |
| `explanation_preferences` | What they need | Paragraph description |

### Example Personas

**Technical Expert**:
- High AI comfort
- Wants comprehensive details
- Prioritizes accuracy over simplicity

**End User**:
- Low AI comfort
- Wants simple, actionable guidance
- Prioritizes clarity and trust

**Regulator**:
- Medium AI comfort
- Wants defensible, auditable explanations
- Prioritizes compliance and fairness

---

## 5. Recommendations: Balancing Criteria

Control how HEXEval balances different criteria:

```yaml
recommendations:
  enabled: true
  dice_sparsity_target: 3    # Target number of feature changes for quality scoring
  
  weights:
    technical_fidelity: 0.3      # Accuracy
    technical_parsimony: 0.2     # Simplicity
    persona_trust: 0.3           # Stakeholder trust
    persona_satisfaction: 0.2    # Stakeholder satisfaction
```

**Weights must sum to 1.0**

**DiCE Sparsity Target Explained**:
- This sets the **ideal** number of feature changes for counterfactuals
- Used to **score** DiCE quality in recommendations
- If average changes = 3 → score = 1.0 (perfect)
- If average changes < 3 → score = 1.0 (even better)
- If average changes > 3 → score decreases (e.g., 6 changes → score = 0.5)
- **Not** the same as `num_counterfactuals` (which controls how many CFs to generate)

**Adjust for your priorities**:
- **Technical focus**: Increase fidelity/parsimony weights
- **User focus**: Increase persona weights
- **Balanced**: Use default (0.3, 0.2, 0.3, 0.2)

---

## Complete Example: Healthcare

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

personas:
  enabled: true
  file: "hexeval/config/personas_healthcare.yaml"
  llm_model: "gpt-4o"
  runs_per_method: 1
  sample_instances: 3
  top_k_features: 5

evaluation:
  sample_size: 100
  random_state: 42
  
  fidelity:
    steps: 20
  
  stability:
    noise_std: 0.05
    repeats: 5
  
  explainers:
    shap:
      enabled: true
      background_size: 100
    
    lime:
      enabled: true
      num_samples: 500
      num_features: 5
      stability_test: true
      stability_subset: 30
    
    anchor:
      enabled: true
      precision_threshold: 0.9
      max_instances: 10
    
    dice:
      enabled: true
      num_counterfactuals: 3
      max_instances: 5
      method: "random"
      sparsity_epsilon: 1e-6

recommendations:
  enabled: true
  dice_sparsity_target: 3
  weights:
    technical_fidelity: 0.3
    technical_parsimony: 0.2
    persona_trust: 0.3
    persona_satisfaction: 0.2
```

---

## Configuration Presets

### Fast Testing (< 2 minutes)

```yaml
evaluation:
  sample_size: 50
personas:
  sample_instances: 2
  runs_per_method: 1
  explainers:
    anchor:
      enabled: false
    dice:
      enabled: false
```

### Balanced Production (5-10 minutes)

```yaml
evaluation:
  sample_size: 150
personas:
  sample_instances: 3
  runs_per_method: 1
```

### Comprehensive Research (20-30 minutes)

```yaml
evaluation:
  sample_size: 300
  fidelity:
    steps: 50
  stability:
    repeats: 10
personas:
  sample_instances: 5
  runs_per_method: 3
```

---

## Troubleshooting

### Evaluation is too slow

**Solutions**:
- Reduce `sample_size` to 50-100
- Disable Anchor and DiCE (slowest methods)
- Reduce `personas.sample_instances` to 2

### AI costs are too high

**Solutions**:
- Use `llm_model: "gpt-4o"` instead of `gpt-4`
- Reduce `sample_instances` to 2
- Set `runs_per_method: 1`
- Use fewer personas

### Results seem inconsistent

**Solutions**:
- Increase `sample_size` to 200+
- Increase `stability.repeats` to 10
- Set `runs_per_method: 2` or higher
- Check your `random_state` is set for reproducibility

---

## Next Steps

1. **Copy an example config** from `hexeval/config/`
2. **Customize the domain** section for your use case
3. **Create or adapt personas** for your stakeholders
4. **Start with fast settings** to test
5. **Scale up** for production evaluation

Need help? Check the other documentation pages for:
- **Prerequisites** - Model and data requirements
- **How It Works** - Understanding the evaluation process
