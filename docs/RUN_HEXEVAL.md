# Running HEXEval

Quick guide to evaluate your model with HEXEval.

## Option 1: Python API (Recommended)

```python
from hexeval import evaluate

results = evaluate(
    model_path="your_model.pkl",
    data_path="your_data.csv",
    target_column="your_target"
)

print(results['technical_metrics'])
print(results['recommendations'])
```

## Option 2: Command Line

```bash
python hexeval_cli.py evaluate your_model.pkl your_data.csv --target your_target
```

## Option 3: Streamlit UI

```bash
streamlit run hexeval/ui/app.py
```

Then upload your files through the web interface.

---

## Example: Credit Risk Model

```python
from hexeval import evaluate

# Using the included example model
results = evaluate(
    model_path="xgboost_loan_default_research_v2.pkl",
    data_path="credit_risk_dataset.csv",
    target_column="loan_status"
)

# View technical metrics
print(results['technical_metrics'])

# View recommendations
for stakeholder, rec in results['recommendations'].items():
    print(f"\n{stakeholder}:")
    print(f"  Recommended: {rec['recommended_method']}")
    print(f"  Reasoning: {rec['reasoning']}")
```

---

## Configuration

Default config: `hexeval/config/eval_config.yaml`

Customize:
- Sample sizes (for faster evaluation)
- Which explainers to run
- Persona evaluation settings

---

## Output

Results saved to: `outputs/hexeval_results/`
- `technical_metrics.csv` - Fidelity, parsimony, etc.
- `persona_ratings.csv` - LLM stakeholder feedback (if enabled)
- `recommendations.json` - Method recommendations per stakeholder

---

## Troubleshooting

**Error: "Model must have predict_proba method"**
- Your model needs probability estimates
- Ensure it's a classifier with `predict_proba()`

**Error: "Missing required features"**
- Model was trained on different features
- Check feature names match between training and evaluation data

**Persona evaluation not running**
- Set `OPENAI_API_KEY` environment variable
- Enable in config: `personas.enabled: true`

---

## Next Steps

- Read `docs/METRICS_GUIDE.md` to interpret results
- Check `examples/quickstart.ipynb` for detailed walkthrough
