# User Guide: Preparing Assets for HEXEval

This guide outlines the necessary file formats and structures required to successfully upload and evaluate your machine learning models on the HEXEval SaaS platform.

To begin an evaluation, you will need to have two distinct assets ready for upload:

1. YOUR **Trained Model Artifact** (e.g., a `.pkl` file)
2. YOUR **Evaluation Dataset** (a `.csv` file)

---

## 1. Model Artifact Requirements

The HEXEval platform is architected to evaluate tabular classification models. To ensure successful ingestion and accurate explanations, your uploaded model file must adhere to the following specifications.

### A. Supported File Formats

The platform accepts trained model objects serialized in the following formats:

* **Pickle** (`.pkl`)
* **Joblib** (`.joblib`)

*Note: The platform currently natively supports Scikit-Learn and XGBoost estimator objects.*

### B. Functional Requirements (`predict_proba`)

The uploaded model object **must** support probabilistic prediction. Internally, the object must implement a functional `predict_proba()` method.

* **Rationale:** HEXEval's interpretability engines (SHAP, LIME, Anchor, DiCE) require access to class probabilities, not just hard class labels, to map decision boundaries accurately.
* **User Action:** Before uploading, verify your model outputs probabilities. For example, if uploading an SVM classifier, ensure it was originally trained with `probability=True`.

### C. Recommended Structure: The "Pipeline" Approach

For the most reliable experience, it is highly recommended to upload a single `sklearn.pipeline.Pipeline` object containing both your data preprocessing steps (e.g., scalers, encoders) and the final estimator.

* **Why this matters on the platform:** This allows you to upload raw, human-readable data (see Section 2). The platform passes the raw data to the pipeline, which handles encoding internally. This results in significantly more interpretable explanations (e.g., seeing "Feature=Urban" instead of "Feature=1").

### D. Best Practice: The Artifact Dictionary

To ensure the platform accurately ingest feature names for visualizations, the optimal upload format is a serialized dictionary containing your components:

```python
# Example structure prior to saving as .pkl
artifact = {
    "model": trained_pipeline_object,
    "feature_names": ["Age", "Income", "Location_Type"] # Critical for readable charts
}

```

*If uploaded in this format, HEXEval will automatically extract the model and apply the correct feature names to all outputs.*

---

## 2. Evaluation Dataset Requirements

You must upload a corresponding dataset that the platform will use to probe the model and generate explanations.

### A. File Format & Structure

* **Format:** Comma Separated Values (`.csv`).
* **Structure:** Standard tabular format where rows represent individual samples and columns represent features.

### B. Content & Integrity Schema

* **Feature Alignment (Critical):** The columns in your CSV must match exactly—in name, order, and count—the features expected by your uploaded model artifact. Mismatches will cause evaluation failure.
* **Target Column:** The dataset should include the ground truth/target column (e.g., "Diagnosis", "Fraud_Label"). You will be asked to identify this column name in the platform UI during setup.
* **Data Type Recommendation:** If you have followed the "Pipeline" recommendation in Section 1.C, your CSV should contain **raw, unencoded data**.
* *Good:* Columns containing strings like "Married", "Single".
* *Bad:* Columns pre-encoded as 0, 1 (unless they are naturally numerical).



---

## 3. Configuring the Evaluation Context

Once your assets are uploaded, you will configure the evaluation context directly in the HEXEval web interface. No local configuration files are required.

### A. Domain Context Selection

To activate the AI-powered **Persona Evaluator**, you must select the appropriate domain context for your model from the dropdown menu (e.g., "Healthcare," "Finance," "Automotive").

* **Rationale:** This selection primes the underlying Large Language Models with domain-specific knowledge and risk frameworks, ensuring that the qualitative feedback provided by simulated stakeholders (e.g., a "Clinical Auditor") is relevant to your use case.