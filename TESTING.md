# HEXEval Testing Guide

## Your Two Model-Dataset Pairs

### Pair 1: Credit Risk Model
- **Model**: `xgboost_credit_risk_new.pkl`
- **Dataset**: `credit_risk_dataset.csv`
- **Target**: `loan_status`

### Pair 2: Loan Default Model
- **Model**: `xgboost_loan_default_research_v2.pkl`
- **Dataset**: `loan_default.csv`
- **Target**: (check column name)

---

## Quick Test Commands

### Test Pair 1 (Credit Risk)

```bash
# 1. Validate
python hexeval_cli.py validate \
    xgboost_credit_risk_new.pkl \
    credit_risk_dataset.csv \
    --target loan_status

# 2. Run evaluation (if validation passes)
python -c "
from hexeval import evaluate
results = evaluate(
    model_path='xgboost_credit_risk_new.pkl',
    data_path='credit_risk_dataset.csv',
    target_column='loan_status'
)
print(results['technical_metrics'])
"

# 3. Launch UI
streamlit run hexeval/ui/app.py
# Then upload: xgboost_credit_risk_new.pkl + credit_risk_dataset.csv
```

### Test Pair 2 (Loan Default)

```bash
# 1. Validate
python hexeval_cli.py validate \
    xgboost_loan_default_research_v2.pkl \
    loan_default.csv \
    --target Default

# 2. Run evaluation
python -c "
from hexeval import evaluate
results = evaluate(
    model_path='xgboost_loan_default_research_v2.pkl',
    data_path='loan_default.csv',
    target_column='Default'
)
print(results['technical_metrics'])
"
```

---

## What to Expect

### If Validation Passes ✅
```
✓ Validation passed!
✓ Model can make predictions on data
```

### If Validation Fails ❌
```
✗ Validation failed:
  - Missing required features: [...]
  - Feature mismatch between model training and current data
```

**Solution**: Make sure you're using the correct model-dataset pair!

---

## Full Evaluation Output

When you run `evaluate()`, you'll get:

```python
results = {
    'technical_metrics': DataFrame with columns:
        - method (SHAP, LIME, Anchor, DiCE)
        - fidelity_deletion
        - fidelity_insertion
        - num_important_features
        - rule_accuracy
        - rule_applicability
        - etc.
    
    'persona_ratings': DataFrame (if enabled) with:
        - persona_name
        - explanation_type
        - trust, satisfaction, actionability
        - comment
    
    'recommendations': Dict mapping stakeholder types to best methods
    
    'output_path': Where results were saved
}
```

---

## Tips

1. **Start with validation** - Always validate first to catch issues early
2. **Check feature names** - Model must be trained on same features as dataset
3. **Persona eval is optional** - Technical metrics work standalone
4. **Use Streamlit for easy testing** - No code needed, just upload files

---

## Troubleshooting

**Error: "Missing required features"**
→ Using wrong dataset for that model. Check the pairs above.

**Error: "No module named hexeval"**
→ Run: `pip install -e .`

**Evaluation is slow**
→ Normal! Anchor and DiCE are computationally expensive.
→ Reduce sample sizes in `hexeval/config/eval_config.yaml`

---

## Next Steps After Testing

1. ✅ Verify validation passes
2. ✅ Run technical evaluation
3. ✅ Check results in `outputs/hexeval_results/`
4. ⏳ (Optional) Enable persona evaluation with OpenAI API key
5. 📝 Use insights for your thesis/report
