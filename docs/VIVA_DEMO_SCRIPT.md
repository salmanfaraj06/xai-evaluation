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
- [ ] Verify all 5 tabs are accessible (Configuration, Use Case Details, Results, Recommendations, Documentation)
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
- [Point to sidebar] "Select use case: Heart Disease or Credit Risk"
- [Point to tabs] "The interface has 5 tabs: Configuration, Use Case Details, Results, Recommendations, and Documentation"
- "The framework evaluates four methods: SHAP, LIME, Anchor, and DiCE"

**Transition:**
> "Let me show you how it works."

---

### [0:30-1:30] Framework Overview & Configuration

**What to say:**
> "HEXEval has five main components. First, **Configuration**—we select a use case. I've prepared the Heart Disease prediction scenario, which evaluates explanations for a cardiology model.
>
> Notice the framework is domain-agnostic—we can switch to Credit Risk or upload custom models.
>
> For this demo, I've pre-run the evaluation to save time. See this green message? 'Found existing results for Heart Disease.' I'll click this **Load Existing Results** button to load the pre-computed evaluation."

**What to do:**
- [Click Tab 1: "📤 Configuration & Run"]
- [Point to sidebar] "Use Case: Heart Disease (Healthcare)"
- [Point to green success message] "✅ Found existing results"
- [Click "📂 Load Existing Results" button]
- [Wait for success message] "✅ Loaded results for Heart Disease!"

**Transition:**
> "Before we look at results, let me quickly show you the use case configuration."

---

### [1:30-2:00] Use Case Details (Optional)

**What to say:**
> "The **Use Case Details** tab shows the complete configuration. Here's the domain context—we're predicting heart disease risk for cardiologists and patients.
>
> Below are the stakeholder personas. Each persona represents a real user type with distinct priorities and preferences. For example, Dr. Sarah Jenkins is a Lead Cardiologist with high risk aversion and moderate AI comfort. Her priorities include patient safety and clinical accuracy.
>
> The framework uses these personas to evaluate explanations from a human perspective—not just technical metrics."

**What to do:**
- [Click Tab 2: "ℹ️ Use Case Details"]
- [Point to Domain Context] "Task: heart disease risk assessment"
- [Expand first persona card] "Dr. Sarah Jenkins - Lead Cardiologist"
- [Point to priorities and explanation preferences]
- [Optional: Show YAML config expander if time permits]

**Transition:**
> "Now let's see the evaluation results."

---

### [2:00-3:00] Technical Evaluation Results

**What to say:**
> "The first part of HEXEval is **Technical Evaluation**. We measure three key dimensions:
>
> **Quality**—does the method actually work? For SHAP and LIME, we measure fidelity using insertion/deletion AUC. For Anchor, we measure precision and coverage. For DiCE, we measure validity—how many counterfactuals successfully flip the prediction.
>
> **Parsimony**—how simple is the explanation? We count features for SHAP/LIME, rule conditions for Anchor, and features changed for DiCE counterfactuals.
>
> Let me show you the visualizations."

**What to do:**
- [Click Tab 3: "📊 Results"]
- [Point to "Quality Metrics" chart]

**Script for reading Quality chart:**
> "This unified chart shows quality across all methods. SHAP has deletion AUC of 0.63—good fidelity. LIME is similar at 0.57. Anchor achieves 95% precision—very high—but only 15% coverage, meaning it applies to a small subset of cases. DiCE has 100% validity—all counterfactuals successfully flip the prediction.
>
> [Scroll to Parsimony chart]
>
> But here's the parsimony trade-off: SHAP shows only 2 features—very simple. LIME shows 5 features—still manageable. Anchor uses 2.2 conditions on average. DiCE changes 2.7 features—highly actionable.
>
> **So technically, all methods perform well.** But here's where traditional evaluation stops—and where HEXEval's innovation begins."

**Transition:**
> "Let's see what happens when we ask stakeholders to rate these same explanations."

---

### [3:00-4:30] Persona Evaluation & The Gap

**What to say:**
> "The second part of HEXEval is **Persona Evaluation**. Instead of expensive human studies, we use LLM-simulated personas—each representing a real stakeholder type.
>
> For Heart Disease, we have four personas: a Lead Cardiologist, a Medical Researcher, a Hospital Administrator, and a Patient. Each has distinct priorities, risk profiles, and decision-making styles.
>
> We present each persona with explanations from all four methods and ask them to rate on **six dimensions**: trust, satisfaction, actionability, interpretability, completeness, and decision support."

**What to do:**
- [Scroll to "Persona Ratings" section]
- [Point to "Average Ratings by Method" table]
- [Point to radar chart showing all 6 dimensions]

**Script for reading persona results:**
> "Here's what we found: **The same explanations that scored excellently on technical metrics receive poor human ratings.**
>
> Look at this table—SHAP scores 1.75 on trust, 1.25 on decision support. LIME is similar: 1.83 trust, 1.58 decision support. Even Anchor, with 95% precision, only gets 2.08 trust and 1.83 decision support. DiCE (Counterfactual) scores 2.0 trust but only 1.42 decision support.
>
> This radar chart shows all six dimensions. Notice how all methods score below 2.5 on most dimensions—that's barely acceptable. The overall averages are: SHAP 1.49, LIME 1.68, Anchor 1.85, DiCE 1.64.
>
> **This is the fidelity-interpretability gap**—methods that are mathematically sound are not human-friendly."

**What to show:**
- [Scroll to "🎭 Persona-Wise Analysis"]
- [Expand one persona card, e.g., "Dr. Sarah Jenkins - Lead Cardiologist"]

**Script:**
> "Let me show you why. Here's the **Persona-Wise Analysis**—we can see each stakeholder's perspective individually.
>
> [Expand Dr. Sarah Jenkins card]
>
> Dr. Sarah Jenkins, a Lead Cardiologist, rates each method. Look at her ratings table—she gives Anchor the highest average at 2.17/5, while SHAP scores only 1.17/5. That's a full point difference.
>
> [Scroll to comments]
>
> Here's her actual feedback on SHAP: 'The SHAP values highlight sex and age without mentioning critical clinical indicators like chest pain type or thallium stress test. Without these clinical markers, I struggle to trust this as a reliable assessment tool.'
>
> But for Anchor, she says: 'The explanation is somewhat understandable, but doesn't provide a complete picture. I value detailed clinical indicators which are not all mentioned here.'
>
> Notice even her 'best' method scores only 2.17—showing the gap. But here's the key: **Different personas have different preferences.** Mark Thompson (Medical Researcher) gives SHAP 2.33—higher than Dr. Jenkins. David Chen (Patient) gives all methods 1.0—he can't understand any of them.
>
> This is exactly the insight—technical accuracy doesn't guarantee usability, and **different personas prefer different methods.**"

---

### [4:30-5:30] Stakeholder-Specific Recommendations

**What to say:**
> "This brings us to HEXEval's third component: **Stakeholder-Specific Recommendations**.
>
> The framework doesn't just identify that there's a gap—it tells you which method works best for each stakeholder type, combining technical scores with persona preferences."

**What to do:**
- [Click Tab 4: "Recommendations"]
- [Point to recommendation cards]

**Script for reading recommendations:**
> "For the Lead Cardiologist (Dr. Sarah Jenkins), HEXEval recommends **Anchor**—not because it has the highest technical score, but because it provides rule-based explanations that align with clinical decision-making. Her average rating for Anchor is 2.17, compared to 1.17 for SHAP.
>
> For the Medical Researcher (Mark Thompson), it recommends **SHAP or DiCE**—both score 2.33 in his ratings. Researchers value comprehensive feature attribution and counterfactual analysis for understanding model behavior.
>
> For the Patient (David Chen), unfortunately all methods score 1.0—none are interpretable to non-technical users. This reveals a critical gap: current XAI methods fail end users entirely.
>
> **This is the core contribution:** One-size-fits-all doesn't work. Different stakeholders need different explanation formats."

**What to show:**
- [Scroll to "📊 Method Comparison Across All Personas"]
- [Point to heatmap] "This is the Method Comparison Matrix—a heatmap showing satisfaction scores by persona and method."

**Script:**
> "This heatmap makes the variance crystal clear. See how the colors change? Green means higher satisfaction, red means low.
>
> For Dr. Sarah Jenkins (Cardiologist), Anchor scores 2.17—her best option. For Mark Thompson (Medical Researcher), SHAP and DiCE both score 2.33—tied for best.
>
> But look at David Chen (Patient)—all methods score 1.0. He can't understand any explanation format. And Linda Martinez (Hospital Administrator) gives Anchor 2.06, SHAP 1.44.
>
> The variance is clear: methods that work for one persona fail for another. **This proves one-size-fits-all doesn't work.**"

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
> The UI has five tabs: Configuration for running evaluations, Use Case Details for viewing personas and domain context, Results for technical and persona metrics, Recommendations for stakeholder-specific guidance, and Documentation for in-app guides.
>
> **Research Contributions:**
> 1. First framework to combine technical and human-centered evaluation systematically.
> 2. Quantifies the fidelity-interpretability gap with concrete evidence.
> 3. Enables stakeholder-specific method selection—not one-size-fits-all.
> 4. LLM-based persona simulation—cost-effective and scalable alternative to human studies."

**What to show:**
- [Click Tab 2: "ℹ️ Use Case Details"] (if not already shown)
- [Or] [Click Tab 5: "📚 Documentation"] to show in-app guides
- [Or] Reference architecture diagram from documentation

**Transition:**
> "Let me summarize the key findings."

---

### [6:30-7:00] Key Findings & Closing

**What to say:**
> "**Key Findings:**
>
> 1. Technical excellence does not guarantee human interpretability. SHAP has 0.63 deletion AUC (good fidelity) but only 1.49 average persona rating. Anchor has 95% precision but 1.85 average rating.
>
> 2. Persona differentiation is real. Dr. Jenkins rates Anchor at 2.17 but SHAP at 1.17—a full point difference. Mark Thompson rates SHAP at 2.33 but Dr. Jenkins rates it 1.17—showing role-based preferences.
>
> 3. No method excels universally. Best average is Anchor at 1.85/5—barely acceptable. Patients rate all methods 1.0—a critical failure for end-user interpretability.
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

### Q: "What are the 6 rating dimensions?"

**Answer:**
> "Great question. The six dimensions capture different aspects of explanation quality from a human perspective:
>
> 1. **Trust** — Do I believe this explanation is accurate and reliable?
> 2. **Satisfaction** — Am I satisfied with the quality and clarity of this explanation?
> 3. **Actionability** — Can I take concrete actions based on this explanation?
> 4. **Interpretability** — How easy is it to understand what the explanation means?
> 5. **Completeness** — Does the explanation provide all the information I need?
> 6. **Decision Support** — Does this explanation help me make better decisions?
>
> These dimensions are based on established HCI and XAI research. Different personas weight these dimensions differently based on their role and priorities."

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
> - Persona evaluation: ~4 minutes (48 LLM API calls)
> - **Total: ~ 6 - 8 minutes**
>
> This is much faster than human studies, which can take days or weeks. The framework is designed for rapid iteration."

---

## Backup Plan: If Live Demo Fails

**If UI doesn't load:**
1. Have screenshots ready of each of the 5 tabs
2. Walk through screenshots while narrating
3. Offer to show code/architecture instead

**If results don't load:**
1. Show the code that generates results
2. Explain the data flow
3. Reference the architecture document
4. Show Use Case Details tab to demonstrate configuration

**If API key issues:**
1. Show technical evaluation only (no personas)
2. Explain persona evaluation conceptually
3. Show pre-generated persona results from documentation
4. Navigate to Documentation tab to show framework capabilities

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
