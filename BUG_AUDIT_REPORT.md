# HEXEval Bug Audit Report

**Date:** 2026-01-15  
**Status:** In Progress  
**Methodology:** Systematic code review + edge case analysis

---

## 🔴 CRITICAL BUGS (Fix Immediately)

### 1. ❌ **Persona Evaluator: LLM Parsing Failures Not Handled**
**File:** `hexeval/evaluation/persona_evaluator.py`  
**Line:** ~420-450  
**Issue:** If LLM returns malformed TOML, the parser crashes with no fallback

**Evidence:**
```python
def _parse_llm_response(response):
    try:
        parsed = toml.loads(response)
        return {
            "trust": parsed.get("trust", 3),
            "satisfaction": parsed.get("satisfaction", 3),
            ...
        }
    except Exception as e:
        # ONLY returns default scores, doesn't log warning
        return {...}
```

**Problem:** Silent failures - no way to know if LLM is returning garbage

**Fix:** Add logging + validation:
```python
except Exception as e:
    logger.warning(f"Failed to parse LLM response: {e}\nResponse: {response[:200]}")
    return fallback_scores
```

---

### 2. ❌ **Technical Evaluator: Division by Zero in Stability**
**File:** `hexeval/evaluation/technical_evaluator.py` + `hexeval/metrics/robustness.py`  
**Line:** ~50-60 in robustness.py  
**Issue:** If all perturbed explanations are identical (no variance), division by zero

**Evidence:**
```python
def calculate_stability(explainer, X, y, model, ...):
    variances = []
    for instance in X:
        # Add noise, generate explanations
        variance = np.var(perturbed_explanations, axis=0)
        variances.append(variance)
    
    return np.mean(variances)  # What if variance = 0?
```

**Problem:** `np.mean([0, 0, 0])` is fine, but if used in denominator elsewhere, crashes

**Test Case:** Perfectly stable explainer (all identical outputs) → potential crash

---

### 3. ❌ **Data Handler: Missing Target Column Validation**
**File:** `hexeval/core/data_handler.py`  
**Line:** ~40-60  
**Issue:** Doesn't verify target column exists before split

**Evidence:**
```python
def load_and_prepare_data(data_path, target_column, ...):
    df = pd.read_csv(data_path)
    # NO CHECK: if target_column not in df.columns
    
    X = df.drop(columns=[target_column])  # KeyError if missing!
    y = df[target_column]
```

**Fix:**
```python
if target_column not in df.columns:
    raise ValueError(f"Target '{target_column}' not in columns: {df.columns.tolist()}")
```

---

### 4. ❌ **Model Loader: XGBoost Feature Names Mismatch Not Caught**
**File:** `hexeval/core/model_loader.py`  
**Line:** ~80-100  
**Issue:** XGBoost expects feature names in predict, but not validated

**Evidence:**
```python
def load_model(model_path):
    model = joblib.load(model_path)
    # Loads model but doesn't extract/validate feature_names
    return {"model": model, "feature_names": None}
```

**Problem:** Later when calling `model.predict(X_test)`, if X_test has different column order → wrong predictions!

**XGBoost requires:** Feature names must match training exactly

---

## 🟡 MODERATE BUGS (Should Fix)

### 5. ⚠️ **Recommender: Alternative Scores Not Stored**
**File:** `hexeval/evaluation/recommender.py`  
**Line:** ~220-226  
**Issue:** Recommendations say "alternatives" but don't actually save scores

**Evidence:**
```python
recommendations[stakeholder] = {
    "recommended_method": best_method,
    "score": best_scores["score"],
    "reasoning": reasoning,
    # NO "alternatives": method_scores dict
}
```

**Problem:** UI shows "Alternatives" but they're empty!

**Fix:** Add `"alternatives": {m: s["score"] for m, s in method_scores.items() if m != best_method}`

---

### 6. ⚠️ **Persona Evaluator: Hardcoded Method Names**
**File:** `hexeval/evaluation/persona_evaluator.py`  
**Line:** ~100-120  
**Issue:** Uses "SHAP", "LIME" strings - if we rename to "shap", "lime", it breaks

**Evidence:**
```python
if method == "SHAP":  # Case-sensitive!
    explanation_text = format_shap(...)
elif method == "LIME":
    ...
```

**Problem:** Fragile - should use config or constants

**Fix:** Define `METHOD_NAMES = {"shap": "SHAP", ...}` or use `.lower()` comparisons

---

### 7. ⚠️ **Config Loading: No Validation of Required Fields**
**File:** `hexeval/evaluation/evaluator.py`  
**Line:** ~30-40  
**Issue:** Loads YAML but doesn't validate required fields exist

**Evidence:**
```python
with open(config_path) as f:
    config = yaml.safe_load(f)
    
# NO VALIDATION: Does config have "evaluation.sample_size"?
sample_size = config["evaluation"]["sample_size"]  # KeyError if missing!
```

**Fix:** Use schema validation (e.g., `pydantic`) or at least check keys

---

## 🟢 MINOR ISSUES (Nice to Fix)

### 8. 💡 **UI: No Progress Bar for Long Evaluations**
**File:** `hexeval/ui/app.py`  
**Issue:** User sees "Running evaluation..." for 7 minutes with no feedback

**Fix:** Use `st.progress()` with status updates for each step

---

### 9. 💡 **Persona Prompts: Very Long (3000+ chars)**
**File:** `hexeval/evaluation/persona_evaluator.py`  
**Issue:** Prompts are 3000+ characters → eats tokens, more expensive

**Suggestion:** Compress prompts by 30% without losing quality

---

### 10. 💡 **No Logging System**
**All files**  
**Issue:** No centralized logging - hard to debug production issues

**Fix:** Add `logging` module with file handler

---

## 🔵 EDGE CASES TO TEST

### 11. 🧪 **What if model has 1 feature?**
- SHAP: Works
- LIME: Works (but num_features=10 > 1?)
- Anchor: Might fail (need 2+ features for rules?)

### 12. 🧪 **What if dataset has 5 rows?**
- Train/test split: Needs at least 10-20 rows
- Stratified split: Might fail if class has 1 sample

### 13. 🧪 **What if all predictions are the same class?**
- Model is broken, but no validation catches it
- Technical metrics might be undefined

### 14. 🧪 **What if LLM API fails (rate limit, timeout)?**
- Currently: Returns fallback scores silently
- Better: Retry 3x, then fail loudly with error message

### 15. 🧪 **What if persona config includes unknown role?**
- Config says evaluate "Data Scientist" but no such persona exists
- Currently: Silent skip
- Better: Warn user

---

## 📊 AUDIT PROGRESS

| Module | Status | Critical | Moderate | Minor |
|--------|--------|----------|----------|-------|
| persona_evaluator.py | ⏳ In Progress | 1 | 2 | 1 |
| technical_evaluator.py | ⏳ In Progress | 1 | 0 | 0 |
| recommender.py | ✅ Reviewed | 0 | 1 | 0 |
| data_handler.py | ⏳ In Progress | 1 | 0 | 0 |
| model_loader.py | ⏳ In Progress | 1 | 0 | 0 |
| evaluator.py | ⏳ In Progress | 0 | 1 | 0 |
| ui/app.py | ⏳ In Progress | 0 | 0 | 2 |

**Next Steps:**
1. Finish module-by-module review
2. Prioritize fixes (Critical → Moderate → Minor)
3. Write test cases for edge cases
4. Implement fixes with tests
