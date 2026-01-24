# HEXEval: Overview and Operational Methodology

## Executive Summary: The Explainability Challenge

As machine learning models are increasingly deployed in high-stakes domains like finance and healthcare, the need to explain their decisions has become critical. However, the field of Explainable AI (XAI) is fragmented. Data science teams face "choice overload," struggling to select from dozens of techniques (SHAP, LIME, Counterfactuals, Anchors) without a standardized framework to judge their quality.

Furthermore, a critical gap exists between technical validity and practical utility. An explanation that is mathematically precise (high technical fidelity) may still be unintelligible or untrustworthy to a business stakeholder (low human utility).

**HEXEval addresses this gap.** It is the first framework designed to systematically evaluate, compare, and recommend XAI methods by integrating rigorous technical testing with context-aware human simulation.

---

## What is HEXEval?

HEXEval is a model-agnostic, hybrid evaluation framework specifically designed for tabular classification models.

It does not train models. Instead, it acts as an independent auditing layer that sits on top of your already-deployed models. By ingesting your model artifact and a validation dataset, HEXEval runs a battery of tests to answer two fundamental questions:

1. **Is the explanation technically sound?** (Does it accurately reflect the model's behavior without instability?)
2. **Is the explanation fit for purpose?** (Does it meet the specific needs, risk tolerances, and cognitive styles of its intended users?)

By automating the assessment of these two competing dimensions, HEXEval provides organizations with a defensible, data-driven basis for selecting the right interpretability strategy.

---

## The Core Engine: Hybrid Evaluation Methodology

HEXEval’s unique value lies in its dual-track evaluation pipeline, which runs in parallel once assets are ingested.

### Track 1: The Technical Evaluation Engine (Quality & Robustness)

This engine conducts rigorous mathematical probing of the XAI methods to ensure they are objectively reliable. It applies method-appropriate metrics to avoid unfair comparisons:

* **For Attribution Methods (e.g., SHAP, LIME):** We measure **Fidelity** (how accurately feature scores map to model output) and **Stability** (ensuring explanations don’t change drastically with minor input noise).
* **For Rule-Based Methods (e.g., Anchors):** We measure **Precision** (rule accuracy) and **Coverage** (how broad of a population the rule applies to).
* **For Counterfactuals (e.g., DiCE):** We measure **Validity** (success rate in flipping the prediction) and **Sparsity** (cost of change).

All raw metrics are normalized onto a standardized 0–1 scale for subsequent comparison.

### Track 2: The Human-Centered Persona Evaluator (Context & Trust)

While technical metrics are necessary, they are insufficient for adoption. HEXEval utilizes advanced Large Language Models (LLMs) to simulate the qualitative evaluation process of real human stakeholders.

The framework injects domain-specific mental models and constraints into LLM agents to create "Personas" (e.g., a "Conservative Risk Officer" or a "Technical Data Auditor"). These personas review explanations in natural language and rate them across six experiential dimensions:

* **Trust & Reliability**
* **Actionability**
* **Interpretability**
* **Satisfaction**
* **Completeness**
* **Decision Support**

---

## How it Works: The End-to-End Workflow

The HEXEval framework guides users through a four-stage process to transform raw assets into actionable recommendations.

### Stage 1: Ingestion & Configuration

The user uploads their trained model artifact (`.pkl` or `.joblib`) and a representative dataset (`.csv`) via the secure web interface. The user then selects the evaluation context (e.g., "Financial Services," "Clinical Ops"), which primes the system with the appropriate domain knowledge and stakeholder persona definitions.

### Stage 2: Parallel Execution

Upon execution, HEXEval generates explanations for the uploaded data using four industry-standard methods: **SHAP, LIME, Anchor, and DiCE**. These explanations are simultaneously routed to both the Technical Evaluation Engine and the Persona Evaluator for scoring.

### Stage 3: The Multi-Criteria Recommender Engine

This is the framework’s decision-making core. Raw scores from the Technical and Persona tracks flow into a **Context-Aware Weighted Mixer**.

Rather than seeking a universally "best" method, the engine applies Multi-Criteria Decision Analysis (MCDM) logic to balance competing priorities based on the target audience. For example:

* *If the target stakeholder is a Developer,* the engine weights technical Fidelity and Stability higher.
* *If the target stakeholder is a Loan Applicant,* the engine weights Actionability and Interpretability higher.

The engine incorporates a **"Safety Gate"**: if all methods fail minimum technical stability thresholds, the system will recommend *none* rather than promote misleading information.

### Stage 4: Insight Delivery

The framework presents a comprehensive report featuring:

1. **The Primary Recommendation:** The single best method for the selected context, backed by natural language reasoning explaining the trade-offs.
2. **Evaluation Matrices:** Interactive charts showing how each method performed across technical and human dimensions.
3. **Stakeholder Profiles:** Detailed breakdowns of how different personas perceived the same explanations, revealing potential friction points between technical and business teams.