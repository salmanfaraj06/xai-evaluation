# Human-Centred Evaluation Template (Credit Risk XAI)

This template pairs VXAI technical evaluation with a human-grounded check. Use the personas below to collect Likert-scale ratings for SHAP, LIME, ANCHOR, and counterfactual explanations on a small set of loan cases.

### Persona: Margaret Chen – Conservative Loan Officer
- Experience: 18 years
- Loss aversion: 2.5
- Risk tolerance: Very Low
- Decision speed: Slow (methodical)
- Trust in AI: Low (prefers human oversight)
- Priorities: Actionability, Trust, Clear rules / thresholds
- Mental model: Credit score and payment history are paramount. Any hint of instability (short employment, high debt-to-income) is a red flag. Defaults are catastrophic.
- Heuristics:
  - If CreditScore < 650, lean heavily toward reject.
  - Employment < 12 months is concerning.
  - Multiple recent credit inquiries suggest desperation.
- Explanation preferences: Prefers simple, rule-based explanations (IF-THEN). Distrusts complex statistical methods. Wants clear thresholds and bright-line rules.
- Behavioral signature:
  - favors simplicity: True
  - prefers conservative errors: True
  - values precedent: True
  - skeptical of novelty: True

#### Rating grid

| Construct | Question | 1 (low) – 5 (high) |
| --- | --- | --- |
| Interpretability | How easy was this explanation to understand? | [1] [2] [3] [4] [5] |
| Completeness | Does it cover the important reasons for the model's decision? | [1] [2] [3] [4] [5] |
| Decision support | Did this help you make a better loan decision? | [1] [2] [3] [4] [5] |
| Trust calibration | Does this help you know when to rely on the model? | [1] [2] [3] [4] [5] |

---

### Persona: David Rodriguez – Data-Driven Analyst
- Experience: 5 years
- Loss aversion: 1.5
- Risk tolerance: Moderate
- Decision speed: Moderate (analytical)
- Trust in AI: Medium (trusts validated models)
- Priorities: Fidelity, Completeness, Consistency with model behavior
- Mental model: Statistical models capture patterns humans miss. Focus on performance metrics. Data quality and feature engineering are key.
- Heuristics:
  - Look for feature importance alignment with business logic.
  - Check for overfitting (train vs test performance).
  - Validate explanations against sensitivity analysis.
- Explanation preferences: Prefers comprehensive, quantitative explanations. Values fidelity to the underlying model over simplicity. Wants to see all important features.
- Behavioral signature:
  - favors completeness: True
  - prefers technical rigor: True
  - values consistency: True
  - comfortable with complexity: True

#### Rating grid

| Construct | Question | 1 (low) – 5 (high) |
| --- | --- | --- |
| Interpretability | How easy was this explanation to understand? | [1] [2] [3] [4] [5] |
| Completeness | Does it cover the important reasons for the model's decision? | [1] [2] [3] [4] [5] |
| Decision support | Did this help you make a better loan decision? | [1] [2] [3] [4] [5] |
| Trust calibration | Does this help you know when to rely on the model? | [1] [2] [3] [4] [5] |

---

### Persona: Patricia Williams – Risk Manager
- Experience: 22 years
- Loss aversion: 3.0
- Risk tolerance: Very Low
- Decision speed: Slow (compliance-focused)
- Trust in AI: Very Low (skeptical of automation)
- Priorities: Risk control, Regulatory defensibility, Fair and robust decisions
- Mental model: Portfolio risk trumps individual accuracy. One bad loan can harm metrics. Regulatory compliance is non-negotiable.
- Heuristics:
  - Focus on worst-case scenarios.
  - Require multiple independent signals before approval.
  - Document everything for audit trail.
- Explanation preferences: Needs explanations that satisfy regulators and internal audit. Must explicitly call out risk factors and be defensible.
- Behavioral signature:
  - favors defensibility: True
  - prefers conservative errors: True
  - values documentation: True
  - fears systemic risk: True

#### Rating grid

| Construct | Question | 1 (low) – 5 (high) |
| --- | --- | --- |
| Interpretability | How easy was this explanation to understand? | [1] [2] [3] [4] [5] |
| Completeness | Does it cover the important reasons for the model's decision? | [1] [2] [3] [4] [5] |
| Decision support | Did this help you make a better loan decision? | [1] [2] [3] [4] [5] |
| Trust calibration | Does this help you know when to rely on the model? | [1] [2] [3] [4] [5] |

---

### Persona: James Thompson – Customer Relationship Manager
- Experience: 8 years
- Loss aversion: 1.2
- Risk tolerance: Moderate-High
- Decision speed: Fast (relationship-focused)
- Trust in AI: High (trusts combined human-AI judgment)
- Priorities: Interpretability to non-experts, Simplicity, Communication to borrowers
- Mental model: People are more than numbers. Context matters, and life circumstances change. Relationships have long-term value beyond a single loan.
- Heuristics:
  - Look for positive trajectory (improving credit over time).
  - Consider extenuating circumstances.
  - Balance short-term risk with long-term relationship value.
- Explanation preferences: Needs explanations that can be communicated to customers. Should highlight actionable steps to improve outcomes and avoid heavy jargon.
- Behavioral signature:
  - favors empathy: True
  - prefers actionability: True
  - values communication: True
  - optimistic bias: True

#### Rating grid

| Construct | Question | 1 (low) – 5 (high) |
| --- | --- | --- |
| Interpretability | How easy was this explanation to understand? | [1] [2] [3] [4] [5] |
| Completeness | Does it cover the important reasons for the model's decision? | [1] [2] [3] [4] [5] |
| Decision support | Did this help you make a better loan decision? | [1] [2] [3] [4] [5] |
| Trust calibration | Does this help you know when to rely on the model? | [1] [2] [3] [4] [5] |

---

### Persona: Sarah Martinez – Executive Decision Maker
- Experience: 15 years
- Loss aversion: 1.8
- Risk tolerance: Moderate
- Decision speed: Very Fast (strategic focus)
- Trust in AI: Medium-High (trusts proven systems)
- Priorities: Strategic impact, Alignment with policy, High-level clarity
- Mental model: Needs scalable, efficient decisions. Focus on portfolio-level metrics, not individual loans. Explanations must support business strategy.
- Heuristics:
  - Time is valuable: prioritize high-impact decisions.
  - Delegate details to domain experts.
  - Focus on systemic patterns rather than edge cases.
- Explanation preferences: Needs high-level summaries that align with business objectives and support strategic planning. Details can be delegated.
- Behavioral signature:
  - favors efficiency: True
  - prefers strategic view: True
  - values scalability: True
  - delegates details: True

#### Rating grid

| Construct | Question | 1 (low) – 5 (high) |
| --- | --- | --- |
| Interpretability | How easy was this explanation to understand? | [1] [2] [3] [4] [5] |
| Completeness | Does it cover the important reasons for the model's decision? | [1] [2] [3] [4] [5] |
| Decision support | Did this help you make a better loan decision? | [1] [2] [3] [4] [5] |
| Trust calibration | Does this help you know when to rely on the model? | [1] [2] [3] [4] [5] |

---
