# New End-User Persona: Alex Johnson

## Profile

**Name:** Alex Johnson  
**Role:** Loan Applicant (End User)  
**Experience:** Not a financial professional - just a regular person applying for a loan

## Key Characteristics

### Demographics & Psychology
- **Loss Aversion:** 3.0× (highest) - Loan rejection is personally devastating
- **Trust in AI:** Low - Skeptical of automated systems making life-changing decisions
- **Financial Literacy:** Low - Doesn't understand jargon, formulas, or statistics
- **Anxiety Level:** High when facing rejection

### Mental Model
> "I applied for a loan to improve my life. I don't understand financial formulas. I just want to know: Why was this decision made? Is it fair? What can I do about it?"

### Decision Heuristics
- If I can't understand it, I assume the system is hiding something
- If it gives me specific actions, I trust it more
- Complex numbers make me anxious
- I want to feel respected, not talked down to

### What They Value in Explanations

**Top Priorities:**
1. **Simplicity** - Can I understand this without financial jargon?
2. **Actionability** - What can I do to improve?
3. **Fairness** - Is this decision unbiased?
4. **Transparency** - Why was this the decision?

**Preferences:**
- Plain English, no jargon
- 2-3 main reasons, not 10 features
- Actionable next steps (improve credit score, save more, get co-signer)
- Respect their intelligence but don't assume expertise

**Dislikes:**
- SHAP values, statistical weights
- Complex formulas
- Technical terms without explanation
- Black-box decisions

## Expected Ratings

### SHAP
- **Trust:** 1-2 (too technical, confusing)
- **Interpretability:** 1 (numbers and jargon)
- **Actionability:** 1 (doesn't tell me what to do)
- **Satisfaction:** 1-2 (frustrated)

### LIME
- **Trust:** 1-2 (also technical)
- **Interpretability:** 2 (slightly better with weights)
- **Actionability:** 1 (still no clear actions)

### Anchor (Rules)
- **Trust:** 3-4 (clear IF-THEN makes sense!)
- **Interpretability:** 4-5 (I can understand rules)
- **Actionability:** 3 (shows thresholds I need to meet)
- **Satisfaction:** 4 (finally something I can understand!)

### Counterfactual (DiCE)
- **Trust:** 4-5 (tells me exactly what to change!)
- **Interpretability:** 4 (concrete changes are clear)
- **Actionability:** 5 (perfect - "do THIS to get approved")
- **Satisfaction:** 5 (exactly what I needed!)

## Why This Persona Matters

### Fills Critical Gap
Current personas are all **professionals** evaluating explanations for their jobs.  
Alex represents the **actual customer** who receives the decision.

### Real-World Impact
- Banks must provide explanations to rejected applicants (regulations)
- Customer satisfaction depends on understandable explanations
- Poor explanations lead to complaints, negative reviews, churn

### Different Priorities
**Stakeholders care about:** Accuracy, compliance, efficiency  
**Customers care about:** Understanding, fairness, what to do next

### Expected Insights
Alex will likely rate:
- **Counterfactual highest** - Shows exactly what to change
- **Anchor second** - Simple rules are understandable
- **SHAP/LIME lowest** - Too technical for non-experts

This validates that **different audiences need different explanation types**.

## Implementation Impact

### Minimal Cost Increase
- Old: 5 personas × 4 methods × 2 instances × 1 run = 40 calls
- New: 6 personas × 4 methods × 2 instances × 1 run = 48 calls
- **+20% calls, +$0.03 per evaluation**

### Maximum Value
- Completes the stakeholder coverage
- Adds crucial end-user perspective
- Enables customer-focused recommendations
- Demonstrates real-world applicability

## Sample Expected Comment

**Alex on SHAP:**
> "I have no idea what these SHAP values mean. I'm not a data scientist - I'm just trying to get a loan. This doesn't tell me why I was rejected or what I can do to improve. It feels like the bank is hiding behind numbers instead of giving me a real explanation."

**Alex on Counterfactual:**
> "This is helpful! It tells me exactly what I need to change - increase my income by $5,000 or reduce my debt by $8,000. Now I have a clear path forward instead of just being confused and frustrated."

---

**Decision:** ADD as 6th persona (don't replace) ✅  
**Cost:** Minimal (+$0.03)  
**Value:** Maximum (completes stakeholder landscape)
