# HEXEval Viva/Demo Script
## 5-7 Minute Presentation Guide

**Purpose:** Demonstrate HEXEval framework to examiners with live UI walkthrough  
**Target Time:** 5-7 minutes  
**Format:** Live demo with prepared results (pre-run to avoid waiting)

---

## Pre-Demo Setup Checklist

**Before the viva:**
- [ ] Pre-run evaluation for "Heart Disease" use case (or Credit Risk)
- [ ] Verify results exist in `outputs/heart_disease/` (or `outputs/credit_risk/`)
- [ ] Have OpenAI API key ready (if doing live persona evaluation)
- [ ] Test UI loads correctly: `streamlit run hexeval/ui/app.py`
- [ ] Have backup screenshots/video if live demo fails
- [ ] Close unnecessary browser tabs/applications

**Recommended:** Use "Heart Disease" use case (smaller dataset, faster to explain)

---

## Script: 5-7 Minute Demo

### [0:00-0:30] Opening Hook & Problem Statement

**What to say:**
> "Good morning/afternoon. Today I'll demonstrate HEXEval, a framework I've developed to solve a critical problem in explainable AI.
>
> **The Problem:** When deploying ML models in healthcare or finance, we need explanations that stakeholders can actually use. But here's the issue: methods like SHAP score excellently on technical metrics—fidelity, stability—yet receive poor ratings from real users. There's a gap between technical excellence and human usability.
>
> **My Research Question:** How can we systematically evaluate XAI methods from both technical rigor and human-centered perspectives to identify the best method for specific stakeholder groups?
>
> **HEXEval's Answer:** A dual evaluation framework that combines technical metrics with LLM-simulated stakeholder personas, enabling data-driven method selection."

**What to show:**
- [Open Streamlit UI] Point to the title: "HEXEval - Holistic Explanation Evaluation"
- [Point to sidebar] "The framework evaluates four methods: SHAP, LIME, Anchor, and DiCE"

**Transition:**
> "Let me show you how it works."

---

### [0:30-1:30] Framework Overview & Configuration

**What to say:**
> "HEXEval has five main components. First, **Configuration**—we select a use case. I've prepared the Heart Disease prediction scenario, which evaluates explanations for a cardiology model.
>
> Notice the framework is domain-agnostic—we can switch to Credit Risk or upload custom models. The configuration includes domain context, stakeholder personas, and evaluation settings.
>
> For this demo, I've pre-run the evaluation to save time, but normally you'd click 'Run Evaluation' here. The framework loads the model, validates compatibility, runs all four XAI methods, and then—this is the key innovation—simulates stakeholder personas using LLMs."

**What to do:**
- [Click Tab 1: "Configuration & Run"]
- [Point to sidebar] "Use Case: Heart Disease (Healthcare)"
- [If results exist] Click "Load Existing Results" button
- [Wait for success message] "Great, results loaded"

**Transition:**
> "Now let's see what the framework evaluated."

---

### [1:30-2:30] Technical Evaluation Results

**What to say:**
> "The first part of HEXEval is **Technical Evaluation**. We measure three key dimensions:
>
> **Fidelity**—does the explanation truthfully reflect the model? We use insertion/deletion AUC. Lower deletion AUC means removing important features actually drops predictions—that's good. Higher insertion AUC means adding important features raises predictions—also good.
>
> **Parsimony**—how simple is the explanation? We count the number of features shown.
>
> **Stability**—does the explanation change wildly with small input changes?
>
> Here are the results for our four methods:"

**What to do:**
- [Click Tab 3: "Results"]
- [Scroll to "Technical Metrics Table"]
- [Point to table] Read out key numbers:

**Script for reading table:**
> "SHAP achieves a deletion AUC of 0.11—excellent fidelity. But notice it shows 24 features—that's overwhelming for users. LIME is similar: good fidelity at 0.13, but still shows 10 features. Anchor achieves 94% rule precision—very high—but only covers 32% of cases. DiCE generates valid counterfactuals 100% of the time, but it's computationally slow.
>
> **So technically, all methods perform well.** But here's where traditional evaluation stops—and where HEXEval's innovation begins."

**Transition:**
> "Let's see what happens when we ask stakeholders to rate these same explanations."

---

### [2:30-4:00] Persona Evaluation & The Gap

**What to say:**
> "The second part of HEXEval is **Persona Evaluation**. Instead of expensive human studies, we use LLM-simulated personas—each representing a real stakeholder type.
>
> For Heart Disease, we have four personas: a Lead Cardiologist, a Medical Researcher, a Hospital Administrator, and a Patient. Each has distinct priorities, risk profiles, and decision-making styles.
>
> We present each persona with explanations from all four methods and ask them to rate on six dimensions: interpretability, completeness, actionability, trust, satisfaction, and decision support."

**What to do:**
- [Scroll to "Persona Ratings Summary"]
- [Point to radar chart or summary table]

**Script for reading persona results:**
> "Here's what we found: **The same explanations that scored excellently on technical metrics receive poor human ratings.**
>
> SHAP: Technical fidelity 0.11—excellent. But average trust from personas? 2.4 out of 5. Actionability? 1.7 out of 5. That's a 2-3 point gap.
>
> LIME: Similar story—good technical scores, but trust at 2.2, actionability at 1.3.
>
> Anchor: 94% precision technically, but trust at 1.9, actionability at 1.3.
>
> **This is the fidelity-interpretability gap**—methods that are mathematically sound are not human-friendly."

**What to show:**
- [Scroll to "Persona-Wise Analysis"]
- [Expand one persona card, e.g., "Dr. Sarah Jenkins - Lead Cardiologist"]

**Script:**
> "Let me show you why. Here's feedback from Dr. Sarah Jenkins, a Lead Cardiologist. She says about SHAP: 'The explanation provides numerical weights without clear clinical context. I need to understand how these translate to patient care decisions.'
>
> This is exactly the problem—technical accuracy doesn't guarantee clinical usability."

**Transition:**
> "But here's the key insight: different personas prefer different methods."

---

### [4:00-5:30] Stakeholder-Specific Recommendations

**What to say:**
> "This brings us to HEXEval's third component: **Stakeholder-Specific Recommendations**.
>
> The framework doesn't just identify that there's a gap—it tells you which method works best for each stakeholder type, combining technical scores with persona preferences."

**What to do:**
- [Click Tab 4: "Recommendations"]
- [Point to recommendation cards]

**Script for reading recommendations:**
> "For the Lead Cardiologist, HEXEval recommends **Anchor**—not because it has the highest technical score, but because it provides rule-based explanations that align with clinical decision-making. The reasoning: 'High stakeholder trust, excellent precision, rule-based format preferred.'
>
> For the Medical Researcher, it recommends **SHAP**—because researchers value comprehensive feature attribution, even if it's technical.
>
> For the Patient, it recommends **DiCE**—because counterfactuals are actionable: 'To reduce risk, reduce your cholesterol by 20 points.'
>
> **This is the core contribution:** One-size-fits-all doesn't work. Different stakeholders need different explanation formats."

**What to show:**
- [Scroll to "Method Comparison Matrix"]
- [Point to heatmap] "You can see the variance—some methods score 4.5 for one persona but 1.5 for another."

**Transition:**
> "Let me show you the framework's architecture."

---

### [5:30-6:30] Architecture & Research Contribution

**What to say:**
> "HEXEval's architecture is three-layered:
>
> **Layer 1: Core Infrastructure**—loads models, handles data, validates compatibility.
>
> **Layer 2: Explainers**—wrappers for SHAP, LIME, Anchor, DiCE.
>
> **Layer 3: Evaluation**—technical metrics and persona simulation.
>
> The framework is domain-agnostic—you configure it via YAML for any binary classification task. No code changes needed.
>
> **Research Contributions:**
> 1. First framework to combine technical and human-centered evaluation systematically.
> 2. Quantifies the fidelity-interpretability gap with concrete evidence.
> 3. Enables stakeholder-specific method selection—not one-size-fits-all.
> 4. LLM-based persona simulation—cost-effective and scalable alternative to human studies."

**What to show:**
- [Click Tab 2: "Use Case Details"] (optional, if time)
- [Or] Reference architecture diagram from documentation

**Transition:**
> "Let me summarize the key findings."

---

### [6:30-7:00] Key Findings & Closing

**What to say:**
> "**Key Findings:**
>
> 1. Technical excellence does not guarantee human interpretability. We see a 2-3 point gap between technical scores and human ratings.
>
> 2. Persona differentiation is real. Different stakeholders prefer different methods—2.5-point variance in ratings.
>
> 3. No method excels universally. Each has strengths for specific personas.
>
> **Impact:** HEXEval enables practitioners to make data-driven decisions about which explanation method to deploy, based on their actual stakeholders—not just technical metrics.
>
> **Future Work:** Validate LLM personas against real humans, extend to multi-class classification, and add natural language explanation generation.
>
> Thank you. I'm happy to answer questions or dive deeper into any component."

**What to do:**
- [Return to Results or Recommendations tab]
- [Be ready to answer questions]

---

## Handling Common Questions

### Q: "Why LLM personas instead of real humans?"

**Answer:**
> "Great question. LLM personas are a cost-effective and scalable alternative—$0.20 per evaluation versus $50-100 per human participant. They're also reproducible, which is critical for research.
>
> However, I acknowledge this is a limitation. In my future work section, I propose a validation study comparing LLM personas to real humans. Preliminary evidence suggests personas capture realistic preferences—we see 2.5-point variance across personas, which matches what we'd expect from real stakeholders.
>
> For now, LLM personas enable rapid iteration and method comparison, which would be prohibitively expensive with real humans."

### Q: "How do you know the personas are realistic?"

**Answer:**
> "The personas are designed based on established HCI research on stakeholder needs in healthcare and finance. Each persona has:
> - Realistic role and experience level
> - Need-based priorities (not metric names)
> - Mental models and heuristics from domain literature
> - Explanation preferences aligned with their decision-making style
>
> The differentiation we see—conservative officers prefer rules, analysts prefer technical detail—matches patterns found in human studies. But you're right—this needs validation, which is why I've proposed a human validation study as future work."

### Q: "What if I want to add a new explainer method?"

**Answer:**
> "The framework is extensible. You'd:
> 1. Create a wrapper class in `hexeval/explainers/` implementing `explain_instance()` and `explain_dataset()`
> 2. Add evaluation logic in `technical_evaluator.py`
> 3. Update the config YAML to enable it
>
> No changes needed to the persona system—it automatically evaluates any method you add. This is documented in the 'Future Extensions' section of the architecture document."

### Q: "What are the limitations?"

**Answer:**
> "Key limitations:
> 1. **Binary classification only**—multi-class and regression not yet supported
> 2. **Tabular data only**—no images or text
> 3. **LLM personas not validated**—need human validation study
> 4. **Single-threaded**—no parallel processing yet
> 5. **Domain-specific**—only validated on healthcare and finance
>
> These are documented in the architecture document, and I've proposed solutions in the future work section."

### Q: "How long does evaluation take?"

**Answer:**
> "For a typical run with 100 instances:
> - Technical evaluation: ~3 minutes (SHAP ~50s, LIME ~100s, Anchor ~20s, DiCE ~15s)
> - Persona evaluation: ~2 minutes (48 LLM API calls)
> - **Total: ~5-7 minutes**
>
> This is much faster than human studies, which can take days or weeks. The framework is designed for rapid iteration."

---

## Backup Plan: If Live Demo Fails

**If UI doesn't load:**
1. Have screenshots ready of each tab
2. Walk through screenshots while narrating
3. Offer to show code/architecture instead

**If results don't load:**
1. Show the code that generates results
2. Explain the data flow
3. Reference the architecture document

**If API key issues:**
1. Show technical evaluation only (no personas)
2. Explain persona evaluation conceptually
3. Show pre-generated persona results from documentation

---

## Practice Tips

1. **Time yourself** — Aim for 6 minutes, leaving 1 minute buffer
2. **Practice transitions** — Smooth handoffs between sections
3. **Know your numbers** — Memorize key metrics (fidelity scores, persona ratings)
4. **Prepare for interruptions** — Examiners may ask questions mid-demo
5. **Have backup visuals** — Screenshots, architecture diagrams, key findings slides

---

## Key Phrases to Remember

- **Opening:** "There's a gap between technical excellence and human usability"
- **Transition:** "But here's where traditional evaluation stops—and where HEXEval's innovation begins"
- **Key Insight:** "Technical excellence does not guarantee human interpretability"
- **Contribution:** "Stakeholder-specific method selection—not one-size-fits-all"
- **Closing:** "HEXEval enables data-driven decisions based on actual stakeholders"

---

## Final Checklist

- [ ] Pre-run evaluation and verify results load
- [ ] Practice script 3+ times
- [ ] Time yourself (target: 6 minutes)
- [ ] Prepare answers to common questions
- [ ] Have backup screenshots ready
- [ ] Test UI loads on presentation machine
- [ ] Have architecture document open as reference
- [ ] Prepare to show code if asked

---

**Good luck with your viva!** 🎓
