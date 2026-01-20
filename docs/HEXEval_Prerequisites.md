# HEXEval Prerequisites & User Guide

This guide details everything you need to successfully evaluate your machine learning models using the HEXEval framework.

## 1. System Requirements

*   **Operating System:** macOS, Linux, or Windows (WSL recommended).
*   **Python Version:** Python 3.8 or higher.
*   **Dependencies:** All required packages are listed in `pyproject.toml` (or `requirements.txt`).
    *   Key libraries: `shap`, `lime`, `anchor-exp`, `dice-ml`, `openai` (optional for Persona mode).

---

## 2. Model Requirements (Critical)

HEXEval is designed to evaluate **black-box tabular classification models**. To function correctly, your model must meet these strict criteria:

### A. Supported Formats
*   **File Type:** Must be saved as a `.pkl` (Pickle) or `.joblib` file.
*   **Libraries:** Scikit-Learn (`sklearn`) or XGBoost (`xgboost`) are natively supported.

### B. The `predict_proba` Requirement
Your model object **must** implement the `predict_proba()` method.
*   **Why?** Most XAI methods (SHAP, LIME, Anchor) rely on probability scores, not just hard class labels, to measure decision boundaries.
*   **Common Pitfall:** If using a Support Vector Machine (SVM), ensure it was trained with `probability=True`.

### C. Pipeline Structure (Recommended)
It is **highly recommended** to wrap your preprocessing (scaling, encoding) and model into a single `sklearn.pipeline.Pipeline`.
*   **Why?** HEXEval passes raw data to validite interpretability. If your model expects pre-processed tensors but you pass raw CSV rows, it will break.
*   **Solution:** Save the entire pipeline:
    ```python
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([
        ('preprocessor', my_preprocessor),
        ('classifier', my_model)
    ])
    joblib.dump(pipeline, "model_pipeline.pkl")
    ```

### D. The "Pro" Format (Dictionary Artifact)
For maximum reliability (especially with correct feature names), save a dictionary containing your components:
```python
artifact = {
    "model": my_model,
    "preprocessor": my_preprocessor,  # Optional
    "feature_names": ["age", "bmi", "glucose"] # Critical for readable plots
}
joblib.dump(artifact, "model_artifact.pkl")
```
HEXEval automatically detects this format and extracts everything.

---

## 3. Data Requirements

### A. Format
*   **File Type:** `.csv` (Comma Separated Values).
*   **Structure:** Standard tabular format where rows are samples and columns are features.

### B. Content
*   **Target Column:** The dataset **must** contain the ground truth/target column (e.g., `target`, `diagnosis`, `churn`). You will specify this column name during execution.
*   **Consistency:** The feature columns in your CSV must match exactly (in name and order) the features your model was trained on.
    *   *Note:* If your model is a Pipeline, your CSV should contain **raw, human-readable data** (e.g., "Male", "Female" strings instead of 0/1), as the pipeline handles the encoding. This produces much better explanations.

---

## 4. Persona (LLM) Requirements

To use the **Persona Evaluator** module (simulating human feedback), you need:

1.  **OpenAI API Key:**
    *   You need a valid API key from [OpenAI Platform](https://platform.openai.com/).
    *   **Cost:** The framework makes multiple calls (Approx. 4 calls * N Personas * N Samples). A typical run might cost $0.50 - $2.00 depending on the model used (`gpt-4` vs `gpt-3.5`).

2.  **Configuration:**
    *   Ensure `enable_personas` is checked in the UI or set to `True` in `eval_config.yaml`.
    *   Ensure the relevant `personas_*.yaml` file is selected for your domain (e.g., Healthcare vs Finance).

---

## 5. Configuration (Optional but Recommended)

While HEXEval works out-of-the-box, customizing `config/eval_config.yaml` greatly improves results:

*   **Domain Context:** defining `domain.name` and reasonable class names (e.g., `Class 0: Healthy`, `Class 1: Disease`) helps the LLM generate accurate qualitative feedback.
*   **Stakeholders:** You can define custom personas in `config/personas_*.yaml` if you want to test specific user types (e.g., "Regulator", "Junior Analyst").
