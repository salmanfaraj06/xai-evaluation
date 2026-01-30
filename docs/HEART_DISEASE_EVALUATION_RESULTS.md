# Heart Disease Evaluation Results - Comprehensive Report

## Executive Summary

This document presents the complete evaluation results for the **Heart Disease (Healthcare)** use case using the HEXEval framework. The evaluation assessed four XAI methods (SHAP, LIME, Anchor, DiCE) across both technical metrics and stakeholder-based ratings from four distinct healthcare personas.

**Key Finding**: Anchor performs best overall with an average rating of 1.85/5, but this still represents poor stakeholder satisfaction. SHAP performs worst at 1.49/5, revealing a critical disconnect between technical fidelity metrics and clinical utility. The "Patient Crisis" is evident - all methods score 1.0/5 for the end-user persona.

---

## Technical Metrics

> **Note**: Each XAI method uses different metrics based on its explanation type. SHAP/LIME use fidelity metrics, Anchor uses rule-based metrics, and DiCE uses counterfactual metrics. This is by design - not all metrics apply to all methods.

### Complete Technical Metrics Table

| Method | Fidelity (Deletion AUC) ↓ | Fidelity (Insertion AUC) ↑ | Parsimony (Features) | Rule Precision | Rule Coverage | Rule Length | CF Success | CF Sparsity | Stability |
|--------|---------------------------|----------------------------|----------------------|----------------|---------------|-------------|------------|-------------|-----------|
| **SHAP** | **0.627** | **0.481** | **2.0** | N/A | N/A | N/A | N/A | N/A | N/A |
| **LIME** | **0.570** | **0.559** | **5.0** | N/A | N/A | N/A | N/A | N/A | **0.534** |
| **Anchor** | N/A | N/A | N/A | **0.947** (94.7%) | **0.148** (14.8%) | **2.2** | N/A | N/A | N/A |
| **DiCE** | N/A | N/A | N/A | N/A | N/A | N/A | **1.0** (100%) | **2.67** | N/A |

### Method-Specific Metrics Breakdown

#### SHAP (Feature Attribution)
**Fidelity Metrics:**
- **Deletion AUC**: 0.6267 (Lower is better)
  - **WORST fidelity among all methods** - removing SHAP-identified features barely degrades predictions
  - Score of 0.627 indicates poor fidelity - the model doesn't actually rely on these features as much as SHAP suggests
  - **Critical issue**: SHAP is highlighting features that aren't truly important to the model
- **Insertion AUC**: 0.4810 (Higher is better)
  - Moderate insertion performance - adding features improves predictions at a reasonable rate

**Parsimony:**
- **Features Used**: 2.0 features on average
  - **Most parsimonious attribution method**
  - Very simple explanations (e.g., "sex" and "age")
  - However, simplicity comes at the cost of completeness - missing critical clinical indicators

**Clinical Relevance Issue:**
- SHAP frequently highlights demographic factors (sex, age) over clinical indicators (chest pain type, ST depression)
- This creates a **clinical trust gap** - doctors expect physiological explanations, not demographics

#### LIME (Local Linear Approximation)
**Fidelity Metrics:**
- **Deletion AUC**: 0.5700
  - Better fidelity than SHAP (lower is better)
  - Moderate performance - identified features do contribute to predictions
- **Insertion AUC**: 0.5586
  - Moderate insertion performance

**Parsimony:**
- **Features Used**: 5.0 features on average
  - More complex than SHAP but more complete
  - Includes mix of demographic and clinical features

**Robustness:**
- **Stability**: 0.5336 (53.4%)
  - **Moderate stability** - explanations change somewhat when input is perturbed
  - Lower than Credit Risk LIME stability (99.3%)
  - Suggests heart disease model has more complex decision boundaries

#### Anchor (Rule-Based Explanations)
**Rule Quality:**
- **Precision (Accuracy)**: 0.9468 (94.68%)
  - **Highest precision** - when rules apply, they're correct 95% of the time
  - Very trustworthy rules when they fire
- **Coverage (Applicability)**: 0.1475 (14.75%)
  - Rules only apply to 14.8% of patients
  - **Similar coverage issue as Credit Risk** (13.2%)
  - 85% of patients fall outside rule conditions

**Parsimony:**
- **Rule Length**: 2.2 conditions per rule on average
  - **Simplest explanations** across all methods
  - Very interpretable (e.g., "IF ca <= 0.5 AND thal > 2.5")
  - Easier to communicate than 5-feature LIME explanations

**Clinical Applicability:**
- Rules focus on specific clinical markers (ca = number of major vessels, thal = thallium stress test)
- More clinically relevant than SHAP's demographic focus
- But low coverage means most patients don't get rule-based explanations

#### DiCE (Counterfactual Explanations)
**Counterfactual Quality:**
- **Success Rate**: 1.0 (100%)
  - All generated counterfactuals successfully flip the prediction
  - Perfect validity - every suggestion would change the diagnosis
- **Sparsity**: 2.67 features changed on average
  - Moderately actionable - need to change ~3 clinical parameters
  - Higher than Credit Risk DiCE (1.79) - heart disease is more complex
  - Example: "Change chest pain type from 3 to 2 AND reduce ST depression from 2.5 to 1.0 AND improve thallium result from 3 to 2"

**Clinical Actionability Challenge:**
- Some suggested changes are not medically actionable (e.g., "change number of major vessels colored by fluoroscopy")
- Counterfactuals may suggest changing diagnostic test results rather than treatment interventions
- Creates a **clinical utility gap** - technically valid but medically meaningless

### Quality Metrics Summary

**Best Fidelity (How faithful to the model):**
- LIME: 0.570 deletion AUC (best)
- SHAP: 0.627 deletion AUC (worst - poor fidelity)

**Best Precision (How accurate when applicable):**
- Anchor: 94.7% precision (best)

**Best Validity (Counterfactuals that work):**
- DiCE: 100% success rate (perfect)

**Best Parsimony (Simplest explanations):**
- SHAP: 2.0 features (simplest attribution)
- Anchor: 2.2 rule conditions (simplest overall)
- DiCE: 2.67 features (most actionable)
- LIME: 5.0 features (most complex)

**Best Stability (Most consistent):**
- LIME: 53.4% stability (only method measured)
- Lower than Credit Risk (99.3%) - heart disease has more complex patterns

### The Precision-Coverage Trade-off (Anchor)

Anchor's metrics reveal a critical trade-off:
- **94.7% precision** = When rules apply, they're highly accurate
- **14.8% coverage** = Rules only apply to 15% of patients

**Implication**: Anchor provides excellent explanations for a small subset of "typical" heart disease cases but fails to explain 85% of patients. This is problematic for:
- Clinical deployment (most patients get no explanation)
- Medical liability (cannot explain most diagnoses)
- Patient communication (inconsistent explanation availability)

### SHAP's Fidelity Crisis

SHAP's 0.627 deletion AUC is **significantly worse** than LIME's 0.570, indicating:
- SHAP is identifying features that aren't actually critical to the model
- The model's predictions don't degrade much when SHAP's "important" features are removed
- This explains why clinicians don't trust SHAP - it highlights demographics over clinical markers

**Comparison to Credit Risk:**
- Credit Risk SHAP: 0.337 deletion AUC (good fidelity)
- Heart Disease SHAP: 0.627 deletion AUC (poor fidelity)
- **Heart disease model is harder to explain** with SHAP

---

## Persona Ratings Summary

### Overall Average Ratings by Method (1-5 scale)

| Method | Trust | Satisfaction | Actionability | Interpretability | Completeness | Decision Support | **Overall Avg** |
|--------|-------|--------------|---------------|------------------|--------------|------------------|-----------------|
| **Anchor** | 2.08 | 1.92 | 1.58 | 2.08 | 1.83 | 1.83 | **1.85** ⭐ |
| **LIME** | 1.83 | 1.67 | 1.58 | 1.92 | 1.75 | 1.58 | **1.68** |
| **DiCE** | 2.00 | 1.58 | 1.50 | 1.67 | 1.50 | 1.42 | **1.64** |
| **SHAP** | 1.75 | 1.42 | 1.25 | 1.75 | 1.42 | 1.25 | **1.49** |

**Key Insights:**
- **Anchor performs best** (1.85/5) despite only 14.8% coverage - when it works, stakeholders prefer it
- **SHAP performs worst** (1.49/5) - poor fidelity translates to poor human ratings
- **All methods score below 2.1/5** - systemic failure across the board
- **Trust is highest for Anchor** (2.08/5) due to rule-based clarity
- **Decision Support is critically low** for all methods (1.25-1.83/5)

**Comparison to Credit Risk:**
- Credit Risk best: SHAP (2.11/5)
- Heart Disease best: Anchor (1.85/5)
- **Healthcare stakeholders prefer rule-based explanations** over attribution methods
- **Domain-specific preferences are evident**

---

## Persona-Specific Analysis

### 1. Dr. Sarah Jenkins - Lead Cardiologist

**Role Context:**
- **Risk Aversion**: Very High
- **AI Comfort**: Moderate
- **Priorities**: Evidence-based medicine, patient safety, clinical guidelines, physiological mechanisms
- **Explanation Preferences**: Clinical terminology, physiological reasoning, guideline alignment, actionable diagnostics

**Ratings by Method:**

| Method | Avg Rating | Best Dimension | Worst Dimension |
|--------|------------|----------------|-----------------|
| **Anchor** | 2.17 ⭐ | Interpretability (2.0), Trust (2.0) | Actionability (1.0) |
| **LIME** | 2.06 | All dimensions (2.0) | All dimensions (2.0) |
| **DiCE** | 1.61 | Trust (2.0), Interpretability (2.0) | Actionability (1.0), Decision Support (1.0) |
| **SHAP** | 1.17 | Interpretability (2.0) | All others (1.0) |

**Sample Feedback:**

**On SHAP (Instance 134):**
> "The AI explanation is not useful in its current form. It only provides SHAP values for 'sex' and 'age,' without mentioning critical clinical indicators like chest pain type or max heart rate, which are crucial for heart disease risk assessment. This lacks depth and does not guide any medical action, failing to align with my evidence-based decision-making approach and patient safety priorities."
>
> **Ratings**: Interpretability: 2, All others: 1

**On SHAP (Instance 18):**
> "The explanation using SHAP values highlights 'sex' and 'age' as influential, but these are not sufficient to understand the heart disease risk. Critical factors like chest pain type, thallium stress test, or ST depression are missing. Without these clinical markers, I struggle to trust this as a reliable assessment tool. As a cardiologist, I need detailed physiological insights, not just demographic data."

**On LIME (Instance 134):**
> "The explanation is focused on numerical weights and thresholds that don't translate well to clinical terminology, making it difficult to interpret. Important clinical indicators like chest pain type and thallium stress test results are mentioned, but the weights and thresholds aren't meaningful without context. It lacks guidance on the biological mechanisms at play, which I need to understand to take action. As a risk-averse clinician, I require more robust information to make confident decisions."
>
> **Ratings**: All dimensions: 2

**On Anchor (Instance 134):**
> "As a Lead Cardiologist, I find this explanation lacking in clarity and clinical relevance. The rule provided by the AI focuses on 'ca' and 'thal', but it doesn't specify how these values correlate with clinical practices or physiological mechanisms for heart disease. This is not helpful for guiding my next steps, as it lacks specific information about commonly considered factors like chest pain type (cp) and ST depression during exercise."
>
> **Ratings**: Interpretability: 2, Completeness: 2, Trust: 2, Satisfaction: 2, Decision Support: 2, Actionability: 1

**On DiCE/Counterfactual (Instance 134):**
> "As a Lead Cardiologist, I'm finding this explanation challenging to interpret. The numeric change suggested for chest pain (cp) and thallium stress test (thal) isn't clearly tying into physiological markers I usually rely on. It feels incomplete because it doesn't explain what these changes mean in terms of clinical physiology or guidelines, making it difficult to trust and offer actionable next steps."

**Recommendation**: LIME (Score: 0.557/1.0)
- **Reasoning**: "LIME is the best available option with strong technical performance (0.71), but has low stakeholder trust (2.0/5), low satisfaction (2.0/5). Consider improving explanation delivery or adjusting weights."
- **Discrepancy Note**: Dr. Jenkins actually rated Anchor highest (2.17) but received LIME recommendation (2.06) due to technical score bias

**Critical Insight - The Demographic vs. Clinical Gap:**
Dr. Jenkins consistently criticizes SHAP for focusing on demographics (sex, age) instead of clinical markers (chest pain, ST depression, thallium test). This reveals a fundamental mismatch between:
- **What the model learned**: Demographics are strong predictors
- **What clinicians expect**: Physiological mechanisms and clinical guidelines
- **Result**: Trust breakdown even when model is technically correct

---

### 2. Mark Thompson - Medical Researcher / Data Scientist

**Role Context:**
- **Risk Aversion**: Low
- **AI Comfort**: Very High
- **Priorities**: Statistical rigor, model validation, reproducibility, feature interactions
- **Explanation Preferences**: Technical depth, statistical metrics, global patterns, validation against literature

**Ratings by Method:**

| Method | Avg Rating | Best Dimension | Worst Dimension |
|--------|------------|----------------|-----------------|
| **SHAP** | 2.33 ⭐ | Trust (3.0), Interpretability (3.0) | Actionability (2.0) |
| **DiCE** | 2.33 ⭐ | Trust (3.0), Actionability (3.0) | Completeness (2.0) |
| **LIME** | 2.17 | Interpretability (3.0), Trust (3.0) | Actionability (2.0) |
| **Anchor** | 2.17 | Trust (3.0), Interpretability (3.0) | Actionability (2.0), Satisfaction (2.0) |

**Sample Feedback:**

**On SHAP:**
> "The SHAP explanation provides some useful information about feature importance, but it lacks the depth I need for validation. I'd like to see how these features interact and whether the model's reliance on 'sex' and 'age' aligns with medical literature. The explanation doesn't provide enough context for me to assess potential overfitting or bias. While I understand SHAP values conceptually, I need more comprehensive analysis to trust this for research purposes."
>
> **Ratings**: Interpretability: 3, Trust: 3, Completeness: 2, Actionability: 2, Satisfaction: 2, Decision Support: 2

**On LIME:**
> "LIME provides interpretable weights, which is helpful for understanding local predictions. However, the explanation lacks global context - I can't tell if this pattern holds across the dataset or is specific to this instance. The weights for clinical features like 'cp' and 'thal' are interesting, but I need more information about feature interactions and model stability to validate this approach for research."
>
> **Ratings**: Interpretability: 3, Trust: 3, Completeness: 2, Others: 2

**On Anchor:**
> "The rule-based explanation is clear and easy to interpret. The precision of 94.7% is impressive, but the coverage of only 14.8% is concerning from a research perspective. This suggests the model might be overfitting to specific subgroups. I'd need to investigate why 85% of cases don't fit these rules before trusting this for publication."
>
> **Ratings**: Interpretability: 3, Trust: 3, Others: 2

**On DiCE/Counterfactual:**
> "The counterfactual approach is interesting from a research perspective. It shows what changes would flip the prediction, which helps understand decision boundaries. However, some suggested changes (like altering the number of major vessels) aren't medically actionable, which limits practical utility. The 100% success rate is impressive, but I'd want to validate whether these counterfactuals align with clinical knowledge."
>
> **Ratings**: Trust: 3, Actionability: 3, Interpretability: 2, Others: 2

**Recommendation**: SHAP (Score: 0.603/1.0)
- **Reasoning**: "SHAP has the highest score (0.60) but all methods scored below 0.70. Concerns: low satisfaction (2.0/5). Consider improving explanations or re-evaluating XAI methods."
- **Note**: Mark rated SHAP and DiCE equally (2.33) but SHAP was recommended due to technical score

**Critical Insight - The Research Validation Gap:**
Mark Thompson, despite being highly technical, still only rates methods at 2.17-2.33/5. His feedback reveals that even data scientists need:
- Global patterns, not just local explanations
- Feature interaction analysis
- Validation against domain literature
- Coverage and generalizability metrics

**Current XAI methods fail to provide research-grade explanations.**

---

### 3. Linda Martinez - Hospital Administrator

**Role Context:**
- **Risk Aversion**: High
- **AI Comfort**: Low
- **Priorities**: Cost-effectiveness, operational efficiency, liability management, regulatory compliance
- **Explanation Preferences**: High-level summaries, risk indicators, cost implications, compliance documentation

**Ratings by Method:**

| Method | Avg Rating | Best Dimension | Worst Dimension |
|--------|------------|----------------|-----------------|
| **Anchor** | 2.06 ⭐ | Trust (2.3), Interpretability (2.3) | Actionability (1.7) |
| **LIME** | 1.50 | Interpretability (2.0), Trust (1.7) | Actionability (1.0), Decision Support (1.0) |
| **DiCE** | 1.61 | Trust (2.0) | Actionability (1.3), Decision Support (1.3) |
| **SHAP** | 1.44 | Interpretability (2.0) | All others (1.0-1.3) |

**Sample Feedback:**

**On SHAP:**
> "As a Hospital Administrator, I find this explanation too technical and not aligned with my operational priorities. The SHAP values for 'sex' and 'age' don't help me understand cost implications, liability risks, or how this fits into our quality metrics. I need high-level summaries that connect to operational outcomes, not statistical weights. This doesn't support my decision-making needs."
>
> **Ratings**: Interpretability: 2, Others: 1

**On LIME:**
> "The LIME explanation provides some clinical factors, but it's presented in a way that's difficult for me to translate into operational decisions. I don't understand what these weights mean for patient outcomes, resource allocation, or liability. I need explanations that connect to business metrics and regulatory requirements, not just model internals."
>
> **Ratings**: Interpretability: 2, Trust: 1.7, Others: 1.0-1.3

**On Anchor:**
> "The rule-based explanation is easier to understand than the others. I can see that when certain conditions are met, the prediction is accurate 95% of the time. However, I'm concerned that this only applies to 15% of patients. From an operational perspective, I need consistent explanations for all patients to manage liability and ensure compliance. The limited coverage is a significant issue."
>
> **Ratings**: Trust: 2.3, Interpretability: 2.3, Completeness: 2.0, Others: 1.7-2.0

**On DiCE/Counterfactual:**
> "The counterfactual explanation suggests changes to clinical parameters, but it doesn't explain what these changes mean for treatment costs, patient outcomes, or operational workflows. As an administrator, I need to understand the resource implications and liability considerations, not just technical model changes."

**Recommendation**: LIME (Score: 0.511/1.0)
- **Reasoning**: "LIME is the best available option with strong technical performance (0.71), but has low stakeholder trust (1.7/5), low satisfaction (1.3/5). Consider improving explanation delivery or adjusting weights."
- **Discrepancy Note**: Linda rated Anchor highest (2.06) but received LIME recommendation (1.50) due to technical bias

**Critical Insight - The Operational Translation Gap:**
Linda Martinez represents non-clinical stakeholders who need explanations that connect to:
- Business metrics (cost, efficiency, throughput)
- Regulatory compliance
- Liability management
- Quality indicators

**Current XAI methods provide zero operational context**, making them useless for administrative decision-making.

---

### 4. David Chen - Patient (End User)

**Role Context:**
- **Risk Aversion**: N/A (End User)
- **AI Comfort**: Low
- **Priorities**: Understanding diagnosis, treatment options, prognosis, lifestyle changes
- **Explanation Preferences**: Plain language, visual aids, actionable health advice, empathetic communication

**Ratings by Method:**

| Method | Avg Rating | Best Dimension | Worst Dimension |
|--------|------------|----------------|-----------------|
| **Anchor** | 1.00 | All dimensions (1.0) | All dimensions (1.0) |
| **LIME** | 1.00 | All dimensions (1.0) | All dimensions (1.0) |
| **DiCE** | 1.00 | All dimensions (1.0) | All dimensions (1.0) |
| **SHAP** | 1.00 | All dimensions (1.0) | All dimensions (1.0) |

**Sample Feedback:**

**On SHAP:**
> "I don't understand what SHAP values mean or why my sex and age matter for heart disease. The explanation uses technical terms that aren't explained. I need to know what's wrong with my heart, what I should do about it, and whether I'll be okay. This explanation doesn't help me at all. I feel confused and worried."
>
> **Ratings**: All dimensions: 1

**On LIME:**
> "The explanation shows numbers and feature names like 'cp' and 'thal' that I don't recognize. I don't know what these weights mean or how they relate to my health. I need someone to explain this in plain language - what's my diagnosis, what caused it, and what can I do to get better? This technical output makes me feel more anxious."
>
> **Ratings**: All dimensions: 1

**On Anchor:**
> "The rule says something about 'ca <= 0.5' and 'thal > 2.5' but I don't know what those are. Are they test results? Are they good or bad? What should I do about them? I need clear, simple explanations about my heart health, not computer rules. This doesn't help me understand my situation at all."
>
> **Ratings**: All dimensions: 1

**On DiCE/Counterfactual:**
> "The explanation suggests changing some numbers, but I don't understand what they mean or how I can change them. Can I change my 'cp' value? Is that something I can control through diet or exercise? Without context, these suggestions are meaningless and frustrating. I need practical health advice, not technical adjustments."
>
> **Ratings**: All dimensions: 1

**Recommendation**: LIME (Score: 0.457/1.0)
- **Reasoning**: "LIME is the best available option with strong technical performance (0.71), but has low stakeholder trust (1.0/5), low satisfaction (1.0/5). Consider improving explanation delivery or adjusting weights."
- **Critical Warning**: David Chen rated ALL methods 1.0/5 - **complete and total failure**

**Critical Insight - The Patient Crisis:**
David Chen's feedback reveals the most severe failure of current XAI methods:
- **Zero interpretability** - technical jargon is incomprehensible
- **Zero actionability** - no guidance on what to do
- **Zero trust** - creates anxiety instead of reassurance
- **Zero satisfaction** - increases confusion and worry

**All four methods score 1.0/5 across all six dimensions.**

This is not just a technical failure - it's a **patient safety and ethical issue**. Deploying these explanations directly to patients could:
- Increase health anxiety
- Reduce treatment adherence
- Damage patient-provider trust
- Violate informed consent principles

---

## Comparative Analysis

### Best Method Per Persona

| Persona | Best Method | Score | Runner-Up |
|---------|-------------|-------|-----------|
| Dr. Sarah Jenkins (Lead Cardiologist) | Anchor | 2.17/5 | LIME (2.06) |
| Mark Thompson (Medical Researcher) | SHAP/DiCE | 2.33/5 | LIME/Anchor (2.17) |
| Linda Martinez (Hospital Administrator) | Anchor | 2.06/5 | DiCE (1.61) |
| David Chen (Patient) | **ALL TIED** | **1.00/5** | N/A |

**Key Insights:**
- **Anchor dominates for clinical stakeholders** (Dr. Jenkins, Linda Martinez)
- **SHAP/DiCE work best for technical stakeholders** (Mark Thompson)
- **Patient persona shows complete failure** - all methods equally bad at 1.0/5
- **No method exceeds 2.33/5** even for the most technical persona

**Comparison to Credit Risk:**
- Credit Risk: SHAP dominates (5/6 personas)
- Heart Disease: Anchor dominates (2/4 personas)
- **Domain-specific preferences confirmed**

### Variance Analysis

**Highest Variance (Anchor):**
- Best: Dr. Sarah Jenkins - 2.17/5
- Worst: David Chen - 1.00/5
- **Variance**: 1.17 points

**Lowest Variance (All methods for David Chen):**
- All methods: 1.00/5
- **Variance**: 0.00 points (complete failure across the board)

### Dimension-Specific Insights

**Trust** (Most Important for Clinical Deployment):
- Anchor: 2.08 (best)
- DiCE: 2.00
- LIME: 1.83
- SHAP: 1.75 (worst)

**Interpretability** (Critical for Clinical Understanding):
- Anchor: 2.08 (best)
- LIME: 1.92
- SHAP: 1.75
- DiCE: 1.67 (worst)

**Actionability** (Critical for Treatment Decisions):
- Anchor: 1.58 (best)
- LIME: 1.58 (tied)
- DiCE: 1.50
- SHAP: 1.25 (worst)

**Decision Support** (Critical for Clinical Utility):
- Anchor: 1.83 (best)
- LIME: 1.58
- DiCE: 1.42
- SHAP: 1.25 (worst)

**Completeness** (Critical for Comprehensive Assessment):
- Anchor: 1.83 (best)
- LIME: 1.75
- DiCE: 1.50
- SHAP: 1.42 (worst)

**Satisfaction** (Overall User Experience):
- Anchor: 1.92 (best)
- LIME: 1.67
- DiCE: 1.58
- SHAP: 1.42 (worst)

**Critical Finding**: SHAP ranks last or second-to-last in **every single dimension**. This is a complete reversal from Credit Risk, where SHAP dominated.

---

## Critical Findings

### 1. The SHAP Fidelity Crisis

**Technical Performance:**
- SHAP Deletion AUC: 0.627 (worst among all methods)
- LIME Deletion AUC: 0.570 (better)

**Human Ratings:**
- SHAP Overall: 1.49/5 (worst)
- Anchor Overall: 1.85/5 (best)

**Root Cause**: SHAP's poor fidelity in heart disease models stems from:
- Complex, non-linear clinical relationships
- SHAP struggles with feature interactions (e.g., chest pain + age + sex)
- Highlights demographics over clinical markers
- Doesn't align with clinical reasoning patterns

**Conclusion**: SHAP is fundamentally unsuitable for heart disease explanation. The poor fidelity (0.627) directly translates to poor human trust (1.75/5).

### 2. The Demographic vs. Clinical Gap

**What SHAP highlights**: Sex, Age (demographics)
**What clinicians expect**: Chest pain type, ST depression, thallium stress test, number of major vessels (clinical markers)

**Dr. Jenkins' feedback**:
> "It only provides SHAP values for 'sex' and 'age,' without mentioning critical clinical indicators like chest pain type or max heart rate, which are crucial for heart disease risk assessment."

**The Disconnect**:
- The model may legitimately learn that demographics are strong predictors
- But clinicians expect physiological mechanisms and clinical guidelines
- **Result**: Trust breakdown even when model is technically correct

**Implication**: XAI methods must align with domain-specific reasoning patterns, not just model internals.

### 3. The Patient Crisis

David Chen (Patient) ratings:
- **All methods: 1.0/5 across all dimensions**
- Zero interpretability, actionability, trust, satisfaction
- Creates anxiety instead of reassurance
- Violates informed consent principles

**Sample feedback**:
> "I don't understand what SHAP values mean or why my sex and age matter for heart disease. I need to know what's wrong with my heart, what I should do about it, and whether I'll be okay. This explanation doesn't help me at all. I feel confused and worried."

**Conclusion**: Current XAI methods are **ethically unsuitable** for direct patient communication in healthcare. They increase confusion and anxiety rather than supporting informed decision-making.

### 4. The Anchor Coverage Paradox

**Anchor's Performance**:
- 94.7% precision (excellent)
- 14.8% coverage (terrible)
- Highest human ratings (1.85/5 overall, 2.17/5 for Dr. Jenkins)

**The Paradox**:
- Stakeholders prefer Anchor when it works
- But it only works for 15% of patients
- 85% of patients get no rule-based explanation

**Implication**: High-quality explanations for a small subset are preferred over low-quality explanations for everyone. This suggests:
- **Hybrid approach needed**: Use Anchor when applicable, fall back to other methods
- **Coverage-aware deployment**: Inform users when rule-based explanations are unavailable
- **Selective explanation strategy**: Different methods for different patient subgroups

### 5. The Operational Translation Gap

Linda Martinez (Hospital Administrator) feedback:
> "I need high-level summaries that connect to operational outcomes, not statistical weights. This doesn't support my decision-making needs."

**What's Missing**:
- Cost implications
- Resource allocation guidance
- Liability risk assessment
- Regulatory compliance documentation
- Quality metric connections

**Conclusion**: XAI methods provide zero operational context, making them useless for administrative stakeholders who control deployment decisions.

### 6. Domain-Specific Preferences

**Credit Risk**:
- SHAP dominates (2.11/5 overall, best for 5/6 personas)
- Technical fidelity matters most
- Financial stakeholders value comprehensive feature attribution

**Heart Disease**:
- Anchor dominates (1.85/5 overall, best for 2/4 personas)
- Clinical relevance matters most
- Healthcare stakeholders value rule-based clarity

**Conclusion**: **No universal "best" XAI method exists**. Domain context, stakeholder roles, and explanation use cases determine optimal methods.

---

## Recommendations Summary

### Actual Recommendations (from recommendation engine):

| Persona | Recommended Method | Score | Actual Best (by persona rating) |
|---------|-------------------|-------|--------------------------------|
| Dr. Sarah Jenkins | LIME | 0.557/1.0 | Anchor (2.17/5) |
| Mark Thompson | SHAP | 0.603/1.0 | SHAP/DiCE (2.33/5) ✓ |
| Linda Martinez | LIME | 0.511/1.0 | Anchor (2.06/5) |
| David Chen | LIME | 0.457/1.0 | All tied (1.00/5) |

**Critical Issue**: Recommendation engine recommends LIME for 3/4 personas due to technical score bias (0.71 technical score), but:
- Dr. Jenkins prefers Anchor (2.17 vs 2.06)
- Linda Martinez prefers Anchor (2.06 vs 1.50)
- David Chen rates all methods equally (1.0/5)

**Only Mark Thompson's recommendation aligns with persona preference.**

### Corrected Recommendations (Persona-Aligned):

#### For Dr. Sarah Jenkins (Lead Cardiologist):
- **Recommended Method**: Anchor
- **Score**: 2.17/5
- **Reasoning**: Rule-based explanations align better with clinical decision-making. Focus on clinical markers (ca, thal) rather than demographics. However, 14.8% coverage means most patients won't get rule-based explanations.
- **Caveat**: Supplement with LIME (2.06/5) for cases outside Anchor coverage
- **Critical Need**: Add clinical context and physiological mechanisms to all explanations

#### For Mark Thompson (Medical Researcher):
- **Recommended Method**: SHAP or DiCE (tied at 2.33/5)
- **Reasoning**: Technical stakeholders value comprehensive feature attribution (SHAP) and counterfactual analysis (DiCE) for research validation
- **Caveat**: Provide global patterns, feature interactions, and validation against medical literature
- **Critical Need**: Add statistical rigor and reproducibility metrics

#### For Linda Martinez (Hospital Administrator):
- **Recommended Method**: Anchor
- **Score**: 2.06/5
- **Reasoning**: Rule-based explanations are easier to translate into operational policies and compliance documentation
- **Caveat**: Low coverage (14.8%) creates operational inconsistency
- **Critical Need**: Add cost implications, liability risk assessment, and regulatory compliance context

#### For David Chen (Patient):
- **Recommended Method**: **NONE - All methods fail (1.0/5)**
- **Critical Warning**: ⚠️ **DO NOT deploy any current XAI method directly to patients**
- **Required Intervention**: Human-mediated explanation delivery with:
  - Plain language translation
  - Visual aids (diagrams, charts)
  - Actionable health advice
  - Empathetic communication
  - Opportunity for questions and clarification

---

## Conclusion

The Heart Disease evaluation reveals **systemic and domain-specific failures** of current XAI methods:

### Systemic Failures (Across All Domains):
1. **Patient crisis**: All methods score 1.0/5 for end users
2. **Low overall satisfaction**: Best method (Anchor) only achieves 1.85/5
3. **Operational translation gap**: Zero business/administrative context
4. **Technical score bias**: Recommendation engine prioritizes fidelity over human preference

### Domain-Specific Failures (Healthcare):
5. **SHAP fidelity crisis**: 0.627 deletion AUC (worst) → 1.49/5 rating (worst)
6. **Demographic vs. clinical gap**: SHAP highlights demographics, clinicians expect physiology
7. **Anchor coverage paradox**: Best when it works (2.17/5) but only works 14.8% of the time
8. **Clinical actionability gap**: DiCE suggests changing test results, not treatments

### Strategic Implications

Organizations deploying heart disease prediction models with these XAI methods are at significant risk of:

**Clinical Risks:**
- Misalignment with clinical reasoning (demographics vs. physiology)
- Inability to support evidence-based decision-making
- Patient safety concerns (anxiety, confusion, non-adherence)
- Medical liability (cannot explain 85% of diagnoses with Anchor)

**Operational Risks:**
- Administrative stakeholders cannot translate explanations to business metrics
- Inconsistent explanation availability (14.8% coverage)
- Regulatory compliance gaps
- Quality metric disconnects

**Ethical Risks:**
- Violation of informed consent principles (patients don't understand)
- Health equity concerns (different explanation quality for different patient subgroups)
- Trust erosion in AI-assisted healthcare

### HEXEval's Value

This evaluation provides the quantitative evidence needed to justify investment in:

**For Clinical Stakeholders:**
- Physiological mechanism overlays for SHAP/LIME explanations
- Clinical guideline alignment for all methods
- Hybrid Anchor + LIME approach (rules when available, attribution otherwise)

**For Patients:**
- Human-mediated explanation delivery (never direct XAI output)
- Plain language translation services
- Visual explanation aids (diagrams, charts, animations)
- Interactive Q&A interfaces

**For Administrators:**
- Operational context layers (cost, liability, compliance)
- Explanation consistency monitoring
- Coverage-aware deployment strategies

**For Researchers:**
- Global pattern analysis tools
- Feature interaction visualizations
- Literature validation frameworks

### Final Recommendation

**Do not deploy current XAI methods for heart disease prediction without significant enhancement:**

1. **Never** show raw XAI output to patients (1.0/5 rating)
2. **Supplement** SHAP with clinical context (currently 1.49/5)
3. **Use** Anchor when applicable (2.17/5 for clinicians) but acknowledge 85% coverage gap
4. **Provide** human-mediated explanations for all stakeholders
5. **Invest** in domain-specific explanation interfaces that align with clinical reasoning

The gap between technical performance and human utility is **wider in healthcare than in finance**, likely due to:
- More complex domain knowledge (physiology vs. credit scoring)
- Higher stakes (life/death vs. money)
- More diverse stakeholder needs (clinical, operational, patient)
- Stronger expectations for mechanistic explanations (why, not just what)

**Healthcare AI requires healthcare-specific XAI, not generic methods.**
