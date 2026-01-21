# HEXEval Production Data Flow

This document details the exact execution path of the HEXEval framework when running end-to-end.

## 1. Input Stage
**Entry Point:** `hexeval.evaluate()` (called by `app.py` or script)

*   **Inputs:**
    *   `model_path`: Path to pickled model (e.g., `heart_disease_pipeline.pkl`)
    *   `data_path`: Path to tabular CSV (e.g., `heart.csv`)
    *   `target_column`: Name of label column (e.g., `'target'`)
    *   `config_path`: None (defaults to `hexeval/config/eval_config.yaml`)

## 2. Loading & Validation Stage
**Function:** `hexeval.core.load_model(model_path)`
*   **Action:** Loads `.pkl` file using `joblib`.
*   **Pipeline Detection:** Detects if object is `sklearn.pipeline.Pipeline` or raw model.
*   **Wrapper Creation:** Wraps in `ModelWrapper`.
*   **Validation:** Checks for `predict_proba` method. *Fails if missing.*

**Function:** `hexeval.core.load_data(data_path)`
*   **Action:** Reads CSV into Pandas DataFrame.
*   **Splitting:** Splits into Train/Test (80/20) using `random_state=42`.
*   **Return:** Dict `{X_train, X_test, y_train, y_test, feature_names}`.

**Function:** `validate_model_data_compatibility(model, data)`
*   **Action:** Checks feature count match.
*   **Action:** Runs a dummy prediction on one row of `X_test`.
*   **Robustness:** If fails, returns detailed error (prevents cryptic crashes later).

## 3. Data Preprocessing (The "Hidden" Layer)
**Module:** `hexeval.core.data_handler.preprocess_for_model`
*   **Critical Logic:**
    *   If `Pipeline` loaded: Passed raw data (strings/categories preserved). Use pipeline's internal preprocessing.
    *   If `Raw Model` loaded: Attemps `astype(float64)`.
    *   **Fallback:** If float conversion fails (string columns present), falls back to `object` array to allow later encoding. *This prevents crashes on categorical data.*

## 4. Technical Evaluation Stage
**Function:** `hexeval.evaluation.technical_evaluator.run_technical_evaluation`
*   **Config:** Reads `config['explainers']` (SHAP, LIME, Anchor, DiCE).
*   **Sampling:** Subsamples test set (default `N=100`) for speed.

### Explainers:
1.  **SHAP:** Wrapper acts as bridge.
    *   If Pipeline: uses `model.predict_proba` as callable for KernelShap.
    *   If Tree: uses TreeExplainer.
    *   **Metric:** Fidelity (Insertion/Deletion AUC), Parsimony (Sparsity).
2.  **LIME:**
    *   **Class Names:** Pulled from `config.domain` (e.g., "healthy", "heart disease").
    *   **Metric:** Stability (repeated perturbations).
3.  **Anchor:**
    *   **Logic:** Finds "If-Then" rules with `precision > 0.9`.
    *   **Metric:** Coverage, Rule Length.
4.  **DiCE:**
    *   **Logic:** Generates counterfactuals ("change X to flip prediction").
    *   **Metric:** Actionability (Success Rate).

## 5. Persona Evaluation Stage (LLM)
**Function:** `hexeval.evaluation.persona_evaluator.run_persona_evaluation`
*   **Check:** Checks `config['personas']['enabled']`. If False or no API key, skips.
*   **Loading:** Loads personas from `hexeval/config/personas_healthcare.yaml`.
*   **Execution Structure:**
    1.  **System Prompt Construction:**
        *   **Template:** "You are {name}, a {role}. Your context: {context}. The decision is about {prediction_task}..."
        *   **Measured Trace:** The system correctly pulls from `config.domain`.
        *   *Example Trace:* "You are Dr. Sarah Jenkins, a Lead Cardiologist... The AI system helps diagnose patient case..."
    2.  **User Prompt Construction:**
        *   **Header:** "📁 REVIEW - Case #{id}"
        *   **AI Output:** "The system used {method}... Here is what the AI is telling you: {explanation_text}"
        *   **Evaluation Request:** "Rate this explanation... Provide your ratings in TOML format AS {role}."
        *   *Verification:* The prompt correctly uses domain terms ("diagnose", "patient case") instead of generic "credit" terms.
    3.  **LLM Interaction:**
        *   **Model:** `gpt-4o` (configurable).
        *   **Response Format:** TOML code block.
        *   *Trace Finding:* The LLM correctly parses the persona constraints.
            *   *Example Response:* "interpretability = 1... comment = 'Terms like slope change don't make sense to me...'" (This proves the 'Patient' persona was faithfully simulating confusion at technical stats).
    4.  **Error Handling:**
        *   If `OPENAI_API_KEY` is missing in env, system gracefully logs warning and skips without crashing.
    
**No Issues Found:** The prompt construction is robust. Variable injection (`{role}`, `{context}`) works correctly. TOML parsing is reliable.

## 6. Recommendation Stage
**Function:** `hexeval.evaluation.recommender.generate_recommendations`
*   **Input:** Technical Metrics + Persona Ratings.
*   **Logic:**
    *   Calculates weighted score based on `config['recommendations']['weights']`.
    *   Matches stakeholder request (e.g., "I need trust") with method strengths.
*   **Output:** JSON mapping `Stakeholder -> Recommended Method`.

## 7. Output Stage
**Action:** Saves files to `outputs/hexeval_results/`
*   `technical_metrics.csv`
*   `persona_ratings.csv`
*   `recommendations.json`
