# How HEXEval Works: The Concept

## The Problem: "Good" Math vs. "Useful" Explanations
As an ML practitioner, you often need to explain your black-box model (XGBoost, Random Forest) to humans. You have many tools: SHAP, LIME, Anchors, etc.

But how do you choose?
*   **SHAP** might be mathematically perfect, but can a **Bank Teller** explain it to a customer?
*   **LIME** might be easy to read, but does it **hallucinate** on edge cases?
*   **Anchors** gives clear rules ("If Age < 30..."), but handles complex continuous variables poorly.
*   **DiCE (Counterfactuals)** tells you *what to change* ("Increase income by $5k"), but is computationally slow.

**HEXEval** is a framework that evaluates these XAI methods from two angles simultaneously:
1.  **Objective Reality (The Math):** Is the explanation truthful to the model?
2.  **Subjective Reality (The Human):** Is the explanation useful to the stakeholder?

---

## 🔵 1. The Technical Evaluation (Does it work?)
Before asking humans, we stress-test the explanations against the model itself.

### **Fidelity (Truthfulness)**
We rigorously check if the features the explanation claims are "important" actually drive the model's prediction.
*   *Test:* If we delete the "top features", does the prediction flip?
*   *Goal:* Ensure the explanation isn't lying about what the model actually used.

### **Stability (Robustness)**
We check if the explanation is jittery.
*   *Test:* If we add invisible noise to the input, does the explanation wildly change?
*   *Goal:* You cannot deploy an explanation system that gives different reasons for the same data point on different days.

### **Parsimony (Simplicity)**
We check how "heavy" the cognitive load is.
*   *Test:* How many features/rules are needed to explain the decision?
*   *Goal:* An explanation with 3 key features is better than one with 50 features, even if the 50-feature one is slightly more accurate.

### **Rule Applicability (Coverage)**
*Specific to Anchors.*
*   *Test:* What % of the population does this rule apply to?
*   *Goal:* Avoid hyper-specific rules that only explain one specific patient but fail for everyone else.

### **Actionability (Success Rate)**
*Specific to Counterfactuals (DiCE).*
*   *Test:* If the user follows the advice (e.g., "reduce debt by $500"), does the model actually change its decision?
*   *Goal:* Ensure the advice isn't fake or broken suitable to the model's actual boundary.

---

## 🟢 2. The Persona Evaluation (Is it useful?)
This is where HEXEval is unique. Instead of running a generic survey, we use **LLM Simulations** to interview your specific stakeholders.

You define who matters for your project in the config. For example:
*   **In Finance:** "Loan Officer", "Regulator", "Applicant".
*   **In Healthcare:** "Clinician", "Patient", "Hospital Admin".
*   **In Retail:** "Store Manager", "Marketing Analyst".

**How it works:**
1.  We generate an explanation (e.g., a SHAP plot or LIME weights).
2.  We "show" this to a GPT-4 agent acting as your specific persona (e.g., a strict **Regulator**).
3.  The agent reviews it and scores it on:
    *   **Trust:** "Does this look like a valid decision rationale?"
    *   **Actionability:** "Can I use this to make a decision?"
    *   **Clarity:** "Is this too technical for me?"

*Result:* You might find that **SHAP** fails for your "End Users" (too complex) but wins for your "Data Scientists".

---

## 🔴 3. The Recommendation
Finally, HEXEval aggregates all scores to give you a clear, data-driven recommendation.

It might tell you:
> *"For your **Data Scientists**, use **SHAP** (High fidelity).*
> *But for your **Customers**, use **Counterfactuals** (High actionability, high clarity)."*

This empowers you to deploy the right tool for the right audience, backed by empirical evidence.
