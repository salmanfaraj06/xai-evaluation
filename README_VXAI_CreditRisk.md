# VXAI-Guided Credit Risk XAI PoC

This repository implements a modular VXAI-aligned pipeline for a loan default classifier. It refactors the original notebook into reusable scripts and modules.

## What’s here
- `config/config_credit.yaml` – data, preprocessing, model, and evaluation settings.
- `src/` – modular code for data loading, preprocessing, model training, explainers (SHAP, LIME, Anchor, DiCE), VXAI metrics, and reporting.
- `scripts/` – CLIs to train the model, run VXAI technical metrics, and generate the human-evaluation template.
- `outputs/` – default location for saved models, metrics, and templates.

## Quickstart
1) Create and activate a virtual env, then install deps  
```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

2) Use the provided pretrained artifact (default: `xgboost_loan_default_research.pkl`)  
No training required. If you still want to retrain, a script is available:
```bash
python scripts/train_model.py --config config/config_credit.yaml
```

3) Run VXAI technical evaluation (SHAP, LIME, Anchor, DiCE) on the pretrained model  
```bash
python scripts/evaluate_vxai.py --config config/config_credit.yaml
```
Outputs: `outputs/vxai_metrics/technical_metrics.csv`

4) Generate human-centred evaluation template  
```bash
python scripts/generate_human_template.py --config config/config_credit.yaml
```
Outputs: `outputs/human_eval_templates/human_eval_template.md`

5) Streamlit UI (inspect plan + run eval + view template)  
```bash
streamlit run streamlit_app.py
```

## Notes
- The data file `loan_default.csv` is expected at the repository root. Update the config path if needed.
- Subset sizes in `config/config_credit.yaml` keep SHAP/LIME/Anchor/DiCE runs lightweight; increase them for more thorough evaluation.
- Metrics follow VXAI mapping in `src/evaluation/vxai_eval_plan.py` (fidelity, parsimony, coverage, robustness).
