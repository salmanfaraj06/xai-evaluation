# HEXEval - Holistic Explanation Evaluation Framework

**Evaluate XAI methods for your tabular models using technical metrics + LLM-simulated stakeholder feedback**

## What is HEXEval?

HEXEval helps you answer: **"Which explanation method (SHAP, LIME, Anchor, DiCE) should I use for my stakeholders?"**

Unlike ad-hoc XAI evaluation, HEXEval provides:
- ✅ **Technical rigor**: Fidelity, parsimony, stability metrics
- ✅ **Human-centered**: LLM personas simulate 5 stakeholder types
- ✅ **Actionable recommendations**: "For your conservative loan officers, use Anchor because..."

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from hexeval import evaluate

# Run complete evaluation
results = evaluate(
    model_path="my_model.pkl",      # Your trained model
    data_path="my_data.csv",         # Your dataset
    target_column="target"           # Target variable
)

# View results
print(results['technical_metrics'])
print(results['persona_ratings'])
print(results['recommendations'])
```

**That's it!** HEXEval handles the rest.

---

## User Journey

### Step 1: Prepare Your Model

Train any sklearn or XGBoost model on tabular data:

```python
from sklearn.ensemble import RandomForestClassifier
import joblib

# Train your model (your code)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save it
joblib.dump(model, "my_model.pkl")
```

### Step 2: Run HEXEval

```python
from hexeval import evaluate

results = evaluate(
    model_path="my_model.pkl",
    data_path="my_data.csv",
    target_column="outcome"
)
```

### Step 3: Interpret Results

#### Technical Metrics

```python
print(results['technical_metrics'])
```

| Method | Fidelity (Deletion) | Sparsity | Stability |
|--------|-------------------|----------|-----------|
| SHAP | 0.23 | 12.5 | N/A |
| LIME | 0.31 | 8.2 | 0.89 |
| Anchor | N/A | 3.1 | N/A |
| DiCE | N/A | N/A | N/A |

#### Persona Ratings

```python
print(results['persona_ratings'])
```

| Persona | Method | Trust | Satisfaction | Actionability |
|---------|--------|-------|-------------|--------------|
| Conservative Officer | Anchor | 4.8 | 4.5 | 4.2 |
| Technical Analyst | SHAP | 4.6 | 4.7 | 3.8 |
| ... | ... | ... | ... | ... |

#### Recommendations

```python
print(results['recommendations'])
```

```json
{
  "Conservative Loan Officer": {
    "recommended_method": "Anchor",
    "reasoning": "High stakeholder trust (4.8/5), excellent precision",
    "persona_feedback": "Simple rule-based format preferred..."
  },
  "Data-Driven Analyst": {
    "recommended_method": "SHAP",
    "reasoning": "High satisfaction (4.7/5), excellent fidelity",
    "persona_feedback": "Values comprehensive feature attribution..."
  }
}
```

---

## Configuration

Customize evaluation by editing `hexeval/config/eval_config.yaml`:

```yaml
evaluation:
  sample_size: 150  # Instances to evaluate
  
  explainers:
    shap:
      enabled: true
      background_size: 500
    lime:
      enabled: true
      num_samples: 2000
    anchor:
      enabled: true
      precision_threshold: 0.9
    dice:
      enabled: true
      num_counterfactuals: 3

personas:
  enabled: true
  llm_model: "gpt-4"
  runs_per_method: 2
```

---

## Supported Models

- ✅ scikit-learn (RandomForest, XGBoost, LogisticRegression, etc.)
- ✅ XGBoost
- ✅ Any model with `predict_proba()` method

## Supported Data

- ✅ Tabular data (CSV)
- ✅ Mixed categorical + numeric features
- ✅ Binary classification

---

## UI Dashboard

```bash
streamlit run hexeval/ui/app.py
```

Interactive dashboard with:
- 📤 Model & data upload
- ⚙️ Configuration
- 📊 Visualizations
- 💡 Recommendations

---

## Architecture

```
hexeval/
├── core/              # Model loading, data handling
├── explainers/        # SHAP, LIME, Anchor, DiCE wrappers
├── metrics/           # Fidelity, parsimony, stability metrics
├── evaluation/        # Orchestration & recommendation engine
├── ui/                # Streamlit dashboard
└── config/            # Configuration files
```

---

## Citation

If you use HEXEval in your research, please cite:

```bibtex
@software{hexeval2026,
  title={HEXEval: A Holistic Framework for Standardizing XAI Evaluation},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/hexeval}
}
```

---

## License

MIT License

---

## Contact

Questions? Open an issue or reach out at your.email@example.com
