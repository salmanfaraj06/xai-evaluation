# HEXEval - 3-Layered Architecture Description

This document describes the architectural separation of the HEXEval framework, structured into **Presentation**, **Application**, and **Domain** layers.

---

## 1. Architectural Overview

The framework follows a strict hierarchical flow. Requests originate in the **Presentation Layer**, are processed and coordinated by the **Application Layer**, and rely on the core entities and logic defined in the **Domain Layer**.

### [TOP] → Presentation Layer
*Visualizing Interaction & Interface*
- **Primary Interface:** The Streamlit Web Application (`app.py`).
- **Alternative Interface:** The CLI Tool (`scripts/hexeval_cli.py`).
- **Function:** This layer handles everything the user sees. It captures model/data uploads, displays Plotly progress charts, and renders final results. It validates that the user's input (like an API key or a file path) is physically present before starting any logic.

### [MIDDLE] → Application Layer
*Visualizing Orchestration & Workflow*
- **Primary Coordinator:** The Main Evaluator (`evaluator.py`).
- **Recommendation Engine:** The Stakeholder Recommender (`recommender.py`).
- **Function:** This layer behaves like a "project manager." It doesn't know how to calculate SHAP values or Fidelity AUC mathematically; instead, it knows the *sequence* of events. It calls the Domain Layer to load the model, calls the metrics logic to get scores, and finally calls the recommendation logic to summarize findings.

### [BOTTOM] → Domain Layer
*Visualizing Core Entities & Business Rules*
- **Core Entities:** `ModelWrapper` (the model representation) and `Personas` (stakeholder definitions).
- **Business Logic:** The mathematical formulas in `metrics/` (Fidelity, Parsimony, Stability) and the XAI algorithm logic in `explainers/`.
- **Function:** This is the most stable and important part of the code. It defines the "laws" of the framework. For example, the rule that "a model must have a predict_proba method" is a Domain-level requirement enforced here.

---

## 2. Detailed Layer Breakdown

### 📱 Presentation Layer
The system's top layer is in charge of managing user interactions. In HEXEval, this is implemented primarily as a **Streamlit Web Interface**.
- **Data Presentation:** It converts raw CSV/JSON results from the lower layers into interactive Plotly heatmaps and radar charts.
- **User Input Validation:** It ensures that when a user selects a "Healthcare" use case, the corresponding files actually exist on the system.
- **Interface Flexibility:** Because this layer is decoupled, HEXEval could be re-implemented as a Mobile App or a REST API without changing any of the underlying evaluation math.

### ⚙️ Application Layer
The application layer serves as a bridge, sitting between the user-facing UI and the core logic.
- **Data Flow Management:** It takes the configuration file path from the UI and parses it into a structured dictionary that the Domain Layer can understand.
- **Execution Logic:** It controls the "Evaluation Pipeline." It handles the logic of "If Persona Evaluation is enabled, then call the LLM service; otherwise, skip to Technical Evaluation."
- **Services:** It bridges the framework to external services like the OpenAI API, handling the request-response cycle for persona simulations.

### 🧠 Domain Layer
The application's core houses the entities, business logic, and data access rules.
- **System State:** The `ModelWrapper` class maintains the state of the machine learning model being evaluated, abstracting away whether it is an XGBoost or Scikit-Learn model.
- **Business Rules:** This layer contains the fundamental XAI research logic. It defines exactly how to calculate "Deletion AUC" (removing features and Measuring prediction drop).
- **Independence:** Crucially, this layer **never** interacts with the Presentation layer. It does not know if its results are being shown in a browser or saved to a text file; it simply performs the calculations as requested by the Application Layer.

---

## 3. Communication Flow
1. **User Request:** The **Presentation Layer** receives a request to run an evaluation.
2. **Orchestration:** The **Application Layer** receives the inputs, verifies settings, and initiates the pipeline.
3. **Execution:** The **Domain Layer** performs the intensive computation (Metric calculation, XAI generation).
4. **Return:** Results flow back up through the Application Layer to the Presentation Layer for display.
