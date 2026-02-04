# HEXEval: Bridging the Gap Between AI Explanations and Human Trust

[![Test Suite](https://github.com/salmanfaraj06/xai-evaluation/actions/workflows/test.yml/badge.svg)](https://github.com/salmanfaraj06/xai-evaluation/actions/workflows/test.yml)
[![Code Quality](https://github.com/salmanfaraj06/xai-evaluation/actions/workflows/lint.yml/badge.svg)](https://github.com/salmanfaraj06/xai-evaluation/actions/workflows/lint.yml)
[![Live Demo](https://img.shields.io/badge/Demo-Live-brightgreen)](https://hexeval.streamlit.app)

> **"My model is accurate, but why won't anyone use it?"** 
> 
> Technical accuracy doesn't equal human trust. HEXEval is the missing link that helps you evaluate your Explainable AI (XAI) methods not just on **math**, but on **stakeholder needs**.

---

## Why HEXEval?

You have a model. You have explainers like SHAP, LIME, or Anchor. But which one is right for your **Risk Officer**? Which one works for your **End User**? 

HEXEval answers this by running a holistic evaluation:
1.  **Technical Rigor**: We check the math (Fidelity, Stability, Parsimony).
2.  **Human Context**: We use LLM-powered personas to simulate real human stakeholders (e.g., "Skeptical Cardiologist", "Compliance Officer").
3.  **Actionable Advice**: We tell you exactly *which* method to use and *why*.

---

## Quick Start

### 1. The Easy Way: Interactive UI
The best way to use HEXEval is through our visual dashboard.

```bash
# 1. Clone the repo
git clone https://github.com/salmanfaraj06/xai-evaluation.git
cd xai-evaluation

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your OpenAI API Key (for the personas)
echo "OPENAI_API_KEY=your-key-here" > .env

# 4. Launch the App!
streamlit run hexeval/ui/app.py
```
**Then just open your browser to `http://localhost:8501`, upload your dataset, and click "Run Evaluation".**

### 2. The Developer Way: Python API
Want to integrate HEXEval into your CI/CD pipeline or notebook?

```python
from hexeval import evaluate

# One-line evaluation
results = evaluate(
    model_path="models/my_credit_model.pkl",
    data_path="data/loan_applications.csv",
    target_column="default_risk",
    config_path="hexeval/config/eval_config.yaml"
)

# See who wins for your "Conservative Risk Officer"
print(results['recommendations']['Conservative Risk Officer'])
# Output: { "recommended_method": "Anchor", "reasoning": "High trust score (4.8/5)..." }
```

---

## How It Works

HEXEval runs a two-step evaluation process:

| Layer 1: The Math | Layer 2: The Humans |
| :--- | :--- |
| **Fidelity:** Does the explanation actually match the model? | **Trust:** Does the stakeholder believe the explanation? |
| **Stability:** Do small input changes break the explanation? | **Satisfaction:** Is the explanation useful for their job? |
| **Parsimony:** Is the explanation concise? | **Actionability:** Can they make a decision based on it? |

We combine these scores using a weighted algorithm to find the perfect match for each stakeholder type.

---

## Features

*   **Multi-Method Support**: Out-of-the-box support for **SHAP**, **LIME**, **Anchor**, and **DiCE**.
*   **Persona Engine**: highly detailed LLM prompts that simulate distinct personalities (e.g., *Technical Analyst* vs. *Non-Technical Customer*).
*   **Model Agnostic**: Works with any scikit-learn or XGBoost model.
*   **Report Generation**: Exports detailed JSON and CSV reports for your documentation.

---

## Project Structure

*   `hexeval/ui/`: The Streamlit dashboard code.
*   `hexeval/core/`: The engine room (data loading, model wrappers).
*   `hexeval/explainers/`: Adapters for different XAI libraries.
*   `hexeval/evaluation/`: The logic for scoring and recommendations.
*   `hexeval/config/`: YAML files where you define your stakeholders and settings.

---

**Built with ❤️ for Better AI Transparency**
