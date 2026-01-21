# Testing HEXEval with Persona Evaluation

## Setup

1. **Activate environment**:
```bash
source .venv/bin/activate
```

2. **Set OpenAI API key**:
```bash
export OPENAI_API_KEY="your-key-here"
# OR create .env file:
echo "OPENAI_API_KEY=your-key-here" > .env
```

## Quick Test (Technical + Personas)

```bash
source .venv/bin/activate

python -c "
from hexeval import evaluate

results = evaluate(
    model_path='outputs/models/xgboost_credit_risk_new.pkl',
    data_path='credit_risk_dataset.csv',
    target_column='loan_status'
)

print('\n=== TECHNICAL METRICS ===')
print(results['technical_metrics'])

if results['persona_ratings'] is not None:
    print('\n=== PERSONA RATINGS ===')
    summary = results['persona_ratings'].groupby('explanation_type')[['trust', 'satisfaction']].mean()
    print(summary)
    
    print('\n=== RECOMMENDATIONS ===')
    for stakeholder, rec in list(results['recommendations'].items())[:2]:
        print(f\"{stakeholder}: {rec['recommended_method']}\")
"
```

## Change LLM Model

Edit `hexeval/config/eval_config.yaml`:

```yaml
personas:
  enabled: true
  llm_model: "gpt-4o"  # Options: gpt-4, gpt-4-turbo, gpt-4o, o1-mini, o3-mini
```

**Model recommendations:**
- **gpt-4o**: Fast, cheap, good quality (default)
- **gpt-4-turbo**: Balanced
- **o1-mini / o3-mini**: GPT-5 reasoning models, best quality but slower/more expensive

## Expected Output

```
🚀 Running HEXEval Evaluation...
============================================================
Evaluating SHAP...
✓ SHAP complete
Evaluating LIME...
✓ LIME complete
Evaluating Anchor...
✓ Anchor complete
Evaluating DiCE...
✓ DiCE complete

Running Persona Evaluation (LLM)
Total LLM calls: 200  # 5 personas × 4 methods × 5 instances × 2 runs
100%|████████| 200/200 [02:30<00:00,  1.33it/s]
✓ Persona evaluation complete

✅ Evaluation Complete!

📊 TECHNICAL METRICS:
   method  fidelity_deletion  fidelity_insertion  ...
0    SHAP           0.107                0.249  ...
1    LIME           0.131                0.212  ...
2  Anchor             NaN                  NaN  ...
3    DiCE             NaN                  NaN  ...

📊 PERSONA RATINGS:
                    trust  satisfaction
SHAP                 4.2          4.3
LIME                 3.8          3.9
ANCHOR               4.5          4.4
COUNTERFACTUAL       3.6          3.7

💡 RECOMMENDATIONS:
Risk Manager: Anchor
Data-Driven Analyst: SHAP

💾 Results saved to: outputs/hexeval_results
```

## Cost Estimate

**Per evaluation:**
- 5 personas × 4 methods × 5 instances × 2 runs = **200 LLM calls**

**Approximate costs (as of 2025):**
- **gpt-4o**: ~$0.50 per evaluation
- **gpt-4-turbo**: ~$2.00 per evaluation  
- **o1-mini**: ~$3.00 per evaluation
- **o3-mini**: ~$5.00 per evaluation

## Streamlit UI Test

```bash
source .venv/bin/activate
streamlit run hexeval/ui/app.py
```

Then:
1. Upload model: `outputs/models/xgboost_credit_risk_new.pkl`
2. Upload data: `credit_risk_dataset.csv`
3. Enter target: `loan_status`
4. Enable personas (checkbox)
5. Enter API key (sidebar)
6. Click "Run Evaluation"

## Troubleshooting

**"OPENAI_API_KEY not found"**
→ Set environment variable or create `.env` file

**"Persona evaluation not yet fully implemented"**
→ Old code - make sure you're using the venv: `source .venv/bin/activate`

**LLM call failed**
→ Check API key, model name, and OpenAI quota

**Too expensive**
→ Reduce sample size in config: `sample_instances: 2`, `runs_per_method: 1`
