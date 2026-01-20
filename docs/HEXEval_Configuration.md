# Configuration Guide

HEXEval is distinct because it **adapts to your domain**. You don't hardcode "Heart Disease" or "Loans" into the python code; you define it here in `eval_config.yaml`.

## 1. Defining Your World (`domain`)
This section tells the LLM Personas what role they are playing. The more specific you are, the better the feedback.

### Example A: Banking
```yaml
domain:
  name: "Consumer Credit Scoring"
  prediction_task: "loan default risk"
  decision_verb: "approve/reject"
  decision_noun: "loan application"
  terms:
    applicant: "borrower"
    risk_factor: "adverse credit history"
```

### Example B: Healthcare
```yaml
domain:
  name: "Oncology Diagnosis"
  prediction_task: "malignancy detection"
  decision_verb: "diagnose"
  decision_noun: "biopsy scan"
  terms:
    applicant: "patient"
    risk_factor: "clinical marker"
```

---

## 2. Defining Your Stakeholders (`personas`)
You can simulated **any** human who reads your model's output.

1.  **Enable Personas:** Set `enabled: true`.
2.  **Point to a File:** Create a simple YAML file (e.g., `my_personas.yaml`) describing the people.

**Example `my_personas.yaml` structure:**
```yaml
- name: "Compliance Officer"
  role: "Regulatory Auditor"
  context: "You are strict. You care about Fair Lending Laws."
  knowledge_level: "High (Legal), Low (Math)"

- name: "Customer Service Rep"
  role: "Front-line Support"
  context: "You need to explain the rejection to an angry customer."
  knowledge_level: "Low"
```

---

## 3. Tuning the Engine (`evaluation`)
Adjust these based on your data size and patience.

*   `sample_size`: How many rows to test?
    *   **50:** Good for quick debugging.
    *   **200+:** Recommended for final reports.
*   `fidelity.steps`: How precise validity checks should be.
    *   Higher (50) = slower, more accurate. Default (20) is usually fine.
*   `stability.noise_std`: How much to "shake" the data to test robustness.
    *   If your data is very noisy/sensitive, lower this (0.01).

---

## 4. Selecting Methods (`explainers`)
Configure the 4 core techniques supported by HEXEval.

### A. Feature Attribution (Importance Scores)
*   **SHAP:** The gold standard for consistency. Slower but reliable.
*   **LIME:** Local approximation. Faster but can be unstable.

### B. Rule-Based (Logic)
*   **Anchor:** Finds "If-Then" rules (e.g., `If Age < 30 AND Income > 50k`). Great for audit trails.
    *   `precision_threshold`: How rigorous the rule must be (0.9 = 90% accuracy).

### C. Counterfactuals (Actionability)
*   **DiCE:** Generates "What-If" scenarios (e.g., *"If you increase Income by $2k, you would be approved"*).
    *   *Note:* Critical for end-user feedback ("What can I do?"), but computationally expensive.
    *   `num_counterfactuals`: How many options to generate (default 3).
