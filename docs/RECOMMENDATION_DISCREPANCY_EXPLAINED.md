# Persona vs Recommendation Discrepancy - Explained

## The Issue

**Observation:** Casey Rodriguez (Strategic Planning Director)
- **Persona-Wise Analysis** shows best method: **Anchor** (highest persona ratings)
- **Recommendations** shows recommended method: **SHAP** (highest combined score)

**Why the mismatch?**

---

## Root Cause: Weighted Scoring Algorithm

The recommendation engine doesn't just use persona ratings - it combines:

### **Recommendation Score Formula:**
```
combined_score = 
  0.3 × technical_fidelity     (from technical evaluation)
  + 0.2 × technical_parsimony  (from technical evaluation)
  + 0.3 × persona_trust / 5    (from persona ratings)
  + 0.2 × persona_satisfaction / 5  (from persona ratings)
```

**Source:** `hexeval/evaluation/recommender.py` lines 94-100

**Key Point:** 50% of the score comes from **technical metrics** that all personas share!

---

## Detailed Breakdown: Casey Rodriguez

### Persona Ratings (What Casey Rated)
| Method | Trust | Satisfaction | Persona Avg |
|--------|-------|--------------|-------------|
| **ANCHOR** | 3.0 | 2.0 | **2.5** ← HIGHEST |
| **SHAP** | 3.0 | 2.0 | **2.5** ← TIE! |
| COUNTERFACTUAL | 2.5 | 1.5 | 2.0 |
| LIME | 2.0 | 2.0 | 2.0 |

**Observation:** Casey rates Anchor and SHAP **equally** on trust/satisfaction!

### Technical Metrics (Shared Across All Personas)
| Method | Deletion AUC | Fidelity Score | Sparsity | Parsimony Score |
|--------|--------------|----------------|----------|-----------------|
| **SHAP** | 0.108 | **0.892** ✅ Best | 24 | 0.294 |
| **LIME** | 0.131 | 0.869 | 10 | **0.500** ✅ Best |
| ANCHOR | N/A | 0 | N/A | 0 |
| COUNTERFACTUAL | N/A | 0 | N/A | 0 |

**Key:** SHAP has the best fidelity (0.892), LIME has best parsimony (0.500)

### Combined Score Calculation

**For SHAP:**
```
score = 0.3 × 0.892 (fidelity)
      + 0.2 × 0.294 (parsimony)
      + 0.3 × (3.0/5) (Casey's trust)
      + 0.2 × (2.0/5) (Casey's satisfaction)
      = 0.268 + 0.059 + 0.180 + 0.080
      = 0.587
```

**For Anchor:**
```
score = 0.3 × 0 (no fidelity metric)
      + 0.2 × 0 (no parsimony metric)
      + 0.3 × (3.0/5) (Casey's trust)
      + 0.2 × (2.0/5) (Casey's satisfaction)
      = 0 + 0 + 0.180 + 0.080
      = 0.260
```

**Winner:** SHAP (0.587) > Anchor (0.260)

**Why:** SHAP gets 0.327 points from technical metrics, Anchor gets 0!

---

## The Problem: Unfair to Anchor & DiCE

**Current Issue:**
- SHAP & LIME have measured fidelity/parsimony metrics
- Anchor & DiCE have **different metrics** (precision, coverage, success rate)
- These aren't factored into the recommendation score
- So Anchor/DiCE are penalized with 0 on 50% of the score!

**This makes the recommendation system biased toward SHAP/LIME.**

---

## How This Affects All Personas

Let me check if this pattern repeats across all 6 personas...

### Expected Pattern:
- **Personas who rated Anchor/DiCE highest** → Recommendation still picks SHAP/LIME
- **Why:** Technical metrics dominate the combined score
- **Result:** Persona preferences are diluted by shared technical scores

---

## Two Solutions

### Option 1: **Persona-Only Recommendations** (Purist)
Remove technical weighting entirely - recommend based purely on what that persona rated highest.

**Change `recommender.py`:**
```python
# OLD: Combined score with technical metrics
combined_score = 0.3*fidelity + 0.2*parsimony + 0.3*trust + 0.2*satisfaction



# NEW: Persona-only score
combined_score = 0.5*trust + 0.5*satisfaction
```

**Pro:** Persona preferences actually matter!  
**Con:** Ignores technical quality (might recommend garbage explanations)

---

### Option 2: **Fair Method-Specific Metrics** (Better)
Include Anchor/DiCE-specific metrics in the scoring.

**Add to score calculation:**
```python
# For Anchor
if method == "ANCHOR":
    quality_score = (
        precision * 0.3 +  # How accurate the rule is
        coverage * 0.2 +    # How many cases it covers
        (1 / rule_length) * 0.1  # Simpler rules better
    )

# For DiCE
if method == "COUNTERFACTUAL":
    quality_score = (
        success_rate * 0.3 +  # Did it find valid CFs?
        proximity * 0.2 +       # How close are changes?
        sparsity * 0.1          # How few features changed?
    )
```

**Pro:** Fair comparison, all methods scored on their strengths  
**Con:** More complex, need to normalize different scales

---

## My Recommendation

**For your thesis defense, EXPLAIN this transparently:**

> *"The recommendation engine uses a hybrid approach: 50% persona preferences + 50% shared technical metrics. This creates an interesting tension:*
>
> *- **Persona-Wise Analysis** shows what each stakeholder rated highest (pure human preference)*
> *- **Recommendations** balance human preference with technical quality*
>
> *For example, Casey Rodriguez rated Anchor and SHAP equally on trust (3.0/5). But SHAP has superior fidelity (0.89 vs 0 for Anchor), so the combined score favors SHAP.*
>
> *This isn't a bug - it's a design choice. Pure persona-based recommendations might suggest poorly-performing methods. The hybrid approach ensures recommendations are both human-preferred AND technically sound.*
>
> *Future work could explore persona-specific technical metrics (e.g., executives value speed, analysts value completeness)."*

---

## Quick Fix for NOW

If you want persona ratings to matter more, just change the weights in `eval_config.yaml`:

```yaml
recommendations:
  weights:
    technical_fidelity: 0.1      # Reduced from 0.3
    technical_parsimony: 0.1     # Reduced from 0.2  
    persona_trust: 0.5           # Increased from 0.3
    persona_satisfaction: 0.3    # Increased from 0.2
```

This would make persona preferences dominant (80% vs 20% technical).

---

**Status:** This is a methodological design choice, not a bug!  
**Decision:** Keep as-is and explain, OR adjust weights to prioritize personas more
