# Project Improvement Plan: Credit Risk XAI Evaluation

## 1. User Experience (UX) & Interface (UI)
The current interface is functional but text-heavy. We can transform it into an interactive dashboard.

### **A. Visualizations**
- **Radar Charts**: Visualize persona ratings (Trust, Satisfaction, etc.) to easily compare XAI methods.
- **Bar Charts**: Compare technical metrics (Fidelity, Stability) across methods.
- **Distribution Plots**: Show the distribution of prediction probabilities and feature importances.

### **B. Interactive "Playground"**
- **Instance Inspector**: Allow users to select a specific test instance (e.g., "Customer #5041") and see:
    - The model's prediction.
    - The actual SHAP force plot.
    - The LIME feature weights.
    - The Anchor rule.
    - The DiCE counterfactuals.
- **"What-If" Analysis**: Let users manually tweak feature values (e.g., increase Income) and see how the prediction and explanations change in real-time.

### **C. Persona Interaction**
- **Chat with Personas**: Implement a chat interface where users can ask the "Executive" or "Data Scientist" persona why they preferred a specific explanation.

## 2. Usability & Workflow
### **A. Configuration Management**
- **In-App Config Editor**: Allow users to adjust parameters (e.g., `threshold`, `num_samples`) directly in the sidebar without editing YAML files.
- **Model Selection**: Dropdown to switch between different trained models (e.g., `xgboost_v1` vs `xgboost_new`).

### **B. Guidance**
- **Tooltips**: Add hover text explaining complex metrics (e.g., "Deletion AUC: Lower is better. Measures how much the model relies on important features.").
- **Guided Tour**: A step-by-step walkthrough for first-time users.

## 3. Scalability & Architecture
### **A. Asynchronous Processing**
- **Background Jobs**: Move the heavy evaluation tasks (which can take minutes) to a background worker (e.g., using `Celery` or Python's `multiprocessing`) so the UI doesn't freeze.
- **Progress Tracking**: Real-time progress bars for each stage of the evaluation.

### **B. Data Handling**
- **Database Integration**: Instead of CSVs, store evaluation results in a lightweight database (SQLite) to handle larger history and enable trend analysis over time.
- **API Layer**: Expose the evaluation logic via a REST API (FastAPI) so it can be integrated into CI/CD pipelines (e.g., "Fail build if Fidelity < 0.8").

## 4. Code Quality & Maintenance
- **Unit Tests**: Add tests for the metric calculations to ensure correctness.
- **Type Hinting**: Ensure full coverage of type hints for better developer experience.
- **Modular Components**: Break the Streamlit app into smaller, reusable component files.

## 5. Immediate Next Steps (Proof of Concept)
We will implement a **Streamlit App v2** that includes:
1.  **Tabs Layout**: Separate "Dashboard", "Detailed Metrics", and "Instance Inspector".
2.  **Visualizations**: Radar charts for human eval and bar charts for technical metrics.
3.  **Interactive Inspector**: A view to see explanations for individual customers.
