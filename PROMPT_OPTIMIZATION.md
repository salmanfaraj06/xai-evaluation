# Prompt Optimization Audit - HEXEval

## Issues Found & Fixed

### 1. ❌ Verbose System Prompt
**Before:** 40+ lines with examples, redundant explanations
**After:** 15 lines, bullet points, no examples
**Savings:** ~60% token reduction

### 2. ❌ JSON Format (verbose)
**Before:** 
```json
{
  "interpretability": 4,
  "completeness": 3,
  ...
}
```
**Token count:** ~80 tokens

**After (TOML):**
```toml
interpretability = 4
completeness = 3
actionability = 3
trust = 4
satisfaction = 3
decision_support = 4
comment = "Clear explanation..."
```
**Token count:** ~50 tokens
**Savings:** 37% reduction

### 3. ❌ Unnecessary Examples in Prompts
**Removed:** Calibration examples (saved ~150 tokens per call)
- Example A/B scenarios
- Verbose rating guidelines
- Redundant dimension descriptions

### 4. ❌ Redundant Information  
**Removed:**
- Instance prediction details (already in explanation)
- Repeated dimension definitions
- Verbose JSON format instructions

### 5. ✅ UI Fix - Missing Dimensions
**Before:** Only showing 3 dimensions (trust, satisfaction, actionability)
**After:** All 6 dimensions displayed:
- trust
- satisfaction  
- actionability
- interpretability
- completeness
- decision_support

## Total Optimization Impact

**Per LLM Call:**
- System prompt: 500 tokens → 200 tokens (-60%)
- User prompt: 150 tokens → 80 tokens (-47%)
- Expected response: 80 tokens → 50 tokens (-37%)

**Total per call:** ~730 tokens → ~330 tokens = **55% reduction**

**For full evaluation (200 calls):**
- Before: ~146,000 tokens
- After: ~66,000 tokens
- **Savings: 80,000 tokens = ~$0.30 per eval (on GPT-4o)**

## New Prompt Structure

### System Prompt (Concise)
```
You are [Name], a [Role].

PROFILE:
• Experience: X years
• Loss Aversion: Xλ
• Risk Tolerance: X
• Trust in AI: X
• Priorities: X, Y, Z

Rate on 6 dimensions (1-5):
[6 one-line descriptions]

Respond in TOML format.
```

### User Prompt (Minimal)
```
Rate this explanation:

Method: SHAP
Explanation: [explanation text]

Provide ratings in TOML format.
```

### Response Format (TOML)
```toml
interpretability = 4
completeness = 3
actionability = 3
trust = 4
satisfaction = 3
decision_support = 4
comment = "..."
```

## Benefits

1. **Cost**: 55% reduction in tokens
2. **Speed**: Faster responses (less to generate)
3. **Quality**: More focused, less confusion
4. **Parsing**: TOML simpler than JSON
5. **UI**: All 6 dimensions now visible

## Next Steps

- Test with optimized prompts
- Verify TOML parsing works
- Compare rating quality (old vs new)
- Consider further optimizations if needed
