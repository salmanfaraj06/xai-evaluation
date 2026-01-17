# Persona Redesign - Before & After Comparison

**Date:** 2026-01-15  
**Purpose:** Document the methodological improvements to eliminate evaluation bias

---

## Problem Statement

The original personas had three critical flaws that would undermine thesis defense:

1. **Name & Role Bias**: Gender stereotypes (Margaret=conservative woman, David=tech guy)
2. **Metric-Hacking**: Explicitly asking for metrics (David wants "Fidelity") creates circular evaluation
3. **Circular Logic**: Overly simplistic heuristics that pre-determine which XAI method wins

---

## Issue #1: Name & Role Bias

### ❌ BEFORE (Stereotypes)
```
Margaret Chen - Conservative Loan Officer (Asian woman=risk-averse stereotype)
David Rodriguez - Data-Driven Analyst (Latino man=tech role)
Patricia Williams - Risk Manager (woman=compliance role)
Alex Johnson - Anxious/Vulnerable Applicant
```

### ✅ AFTER (Gender-Neutral, Swapped Stereotypes)
```
Jordan Walsh - Policy-Focused Loan Officer (neutral name)
Sam Chen - Model Validation Analyst (swapped: was David)
Taylor Kim - Compliance & Risk Officer (neutral name)
Morgan Patel - Customer Success Manager (neutral name)
Casey Rodriguez - Strategic Planning Director (swapped: was Sarah)
Riley Martinez - Goal-Oriented Applicant (empowered framing, not anxious)
```

**Why This Matters:** Prevents LLM from hal lucinating gender-based behavioral traits based on names alone.

---

## Issue #2: Metric-Hacking Bias

### ❌ BEFORE (Asking for Metrics Directly)

**David Rodriguez priorities:**
- "Fidelity" ← This is a METRIC NAME!
- "Completeness" ← This is a RATING DIMENSION!
- "Consistency with model behavior"

**Margaret Chen priorities:**
- "Actionability" ← This is a RATING DIMENSION!
- "Trust" ← This is a RATING DIMENSION!
- "Clear rules" ← This hints at Anchor (rule-based)

**Problem:** You're telling the LLM "Hi David, rate Fidelity highly!" This defeats the purpose of evaluation.

### ✅ AFTER (Describing Needs, Not Metrics)

**Sam Chen (Analyst) priorities:**
- "Ensuring the model's reasoning aligns with domain knowledge"
- "Understanding all factors that influenced the decision"
- "Detecting potential model errors or unexpected behavior"

**Jordan Walsh (Loan Officer) priorities:**
- "Being able to justify decisions to management and customers"
- "Confidence in the soundness of the recommendation"
- "Clear guidance on what factors drove the decision"

**Why This Works:** The LLM must INFER which explanation method meets these needs. It's not told "rate fidelity" - it has to decide if SHAP/LIME/Anchor/DiCE actually helps with "detecting model errors."

---

## Issue #3: Circular Logic in Heuristics

### ❌ BEFORE (Overly Simplistic)

**Margaret Chen:**
```python
"If CreditScore < 650, lean heavily toward reject."
```

**Problem:** Real loan officers don't just reject based on one number. This is unrealistic and creates artificial preference for simple explanations.

### ✅ AFTER (Nuanced, Realistic)

**Jordan Walsh:**
```python
"Skeptical of low credit scores unless offset by significant tenure or low debt burden."
```

**Why Better:**
- Reflects real-world compensating factors
- Forces explanations to show IF assets/tenure were considered
- SHAP should show "credit_score: -0.3, debt_ratio: +0.15" (offset!)
- Anchor should show "IF score<650 AND debt>40% THEN reject"
- Makes personas MORE discriminating, not less

---

## Issue #4: End-User Empowerment

### ❌ BEFORE (Victim Framing)
```
"Highly anxious about rejection, feels vulnerable and judged by the system"
```

### ✅ AFTER (Empowered Consumer)
```
"Goal-oriented consumer seeking to optimize their financial situation"

Mental Model: "I have financial goals (buying a home, starting a business). 
This decision impacts those goals. I need to know: what factors mattered most, 
were they evaluated fairly, and what concrete steps can I take to improve?"
```

**Why Better:**
- Riley isn't passive/anxious - they're proactive
- Wants actionable steps (rewards DiCE counterfactuals on merit!)
- Checks for fairness (rewards explanations that show relevant factors)
- Realistic: real applicants want to understand AND improve

---

## Verification: No Pre-Determined Winners

**Test: Which persona likes which method?**

### Jordan Walsh (Policy Officer)
- **Could like Anchor**: Maps to policies, clear rules
- **Could like LIME**: Simple top-10 features, easy to explain
- **Unlikely SHAP**: Too many features (24), hard to justify
- **Unlikely DiCE**: Counterfactuals don't help with policy alignment

### Sam Chen (Analyst)
- **Could like SHAP**: Comprehensive, all factors shown
- **Could like LIME**: Fast stability testing
- **Unlikely Anchor**: Lacks detail for model validation
- **Could like DiCE**: Shows what changes flip decision (model behavior test)

### Riley Martinez (End User)
- **Could like DiCE**: "To approve: increase income by $8K" (actionable!)
- **Could like Anchor**: "IF income>$50K THEN approve" (clear!)
- **Unlikely SHAP**: "credit_score: 0.234" (confusing numbers)
- **Could like LIME**: "Top 3 factors: income, debt, score" (simple)

**Key Point:** NO PERSONA IS PRE-WIRED TO ONE METHOD. The LLM must evaluate based on actual explanation quality!

---

## Defense Strategy

**Advisor Question:** "Why didn't you use real behavioral parameters like loss aversion coefficients?"

**Your Answer:**
"I use qualitative descriptions to help the LLM simulate diverse stakeholder perspectives WITHOUT claiming psychometric precision I don't have. The personas describe:

1. **Job responsibilities** (loan officer vs analyst vs applicant)
2. **Information needs** (justify decisions vs audit models vs take action)
3. **Realistic decision-making** (compensating factors, not simple thresholds)

The diversity in ratings (trust 1.0-3.5 across personas/methods) shows this works. Future work could validate with real human studies."

**Advisor Question:** "Isn't Sam Chen rating SHAP highly just circular?"

**Your Answer:**
"Sam's priority isn't 'rate fidelity highly' - it's 'detect model errors.' The LLM must reason: DOES SHAP help detect errors? In practice, Sam might rate SHAP low if it shows 24 features (too noisy to audit). Sam might rate LIME high if 10 features are clearer. The evaluation isn't pre-determined."

---

## Impact on Results

**Expected Changes:**
- ✅ More variance in ratings (no pre-wiring)
- ✅ Surprises (Sam might prefer LIME over SHAP!)
- ✅ Realistic patterns (Riley rates DiCE high for actual actionability)
- ✅ Easier defense (no arbitrary numbers, no circular logic)

**What Stays Same:**
- Overall finding: Technical XAI has low persona scores
- Fidelity-interpretability gap remains
- Contribution: Dual evaluation (technical + human-centered)

---

## Files Changed

1. ✅ `hexeval/evaluation/personas.py` - Complete redesign
2. ⏳ (May need) `persona_evaluator.py` - Update if prompts reference old names

**Commit Message:**
```
refactor: Eliminate evaluation bias in personas

- Gender-neutral names to prevent stereotype hal lucination
- Replace metric names with need descriptions (no "Fidelity" priority)
- Nuanced heuristics with compensating factors (realistic decisions)
- Empowered end-user framing (goal-oriented, not just anxious)

Makes personas methodologically defensible for thesis
```

---

**Approved By:** User (via feedback)  
**Status:** ✅ Ready for Re-Evaluation
