# HEXEval Viva - Quick Reference Card

**Print this and keep it next to your screen during the demo**

---

## 🎯 Core Message (Memorize This)

> "Technical excellence does not guarantee human interpretability. HEXEval reveals the fidelity-interpretability gap and enables stakeholder-specific method selection."

---

## ⏱️ Timing Breakdown

| Time | Section | Tab | Key Action |
|------|---------|-----|------------|
| 0:00-0:30 | Hook & Problem | - | Explain the gap |
| 0:30-1:30 | Configuration | Tab 1 | Load existing results |
| 1:30-2:30 | Technical Results | Tab 3 | Show metrics table |
| 2:30-4:00 | Persona Evaluation | Tab 3 | Show persona ratings |
| 4:00-5:30 | Recommendations | Tab 4 | Show stakeholder cards |
| 5:30-6:30 | Architecture | Tab 2 | Quick overview |
| 6:30-7:00 | Findings & Close | - | Summarize |

---

## 📊 Key Numbers to Remember

**Technical Metrics (Heart Disease):**
- SHAP: Deletion 0.11, Insertion 0.25, 24 features
- LIME: Deletion 0.13, Insertion 0.21, 10 features
- Anchor: Precision 94.9%, Coverage 32%
- DiCE: Success rate 100%

**Persona Ratings (Average):**
- SHAP: Trust 2.4/5, Actionability 1.7/5
- LIME: Trust 2.2/5, Actionability 1.3/5
- Anchor: Trust 1.9/5, Actionability 1.3/5
- DiCE: Trust 2.1/5, Actionability 1.6/5

**The Gap:** 2-3 point difference between technical and human scores

---

## 🗣️ Transition Phrases

- **Opening:** "There's a gap between technical excellence and human usability"
- **To Technical:** "Let's see what the framework evaluated"
- **To Personas:** "But here's where traditional evaluation stops"
- **To Recommendations:** "Different personas prefer different methods"
- **Closing:** "HEXEval enables data-driven decisions based on actual stakeholders"

---

## ❓ Quick Answer Bank

**Q: Why LLM personas?**
> "Cost-effective ($0.20 vs $50-100), reproducible, scalable. Acknowledged limitation—validation study proposed."

**Q: Are personas realistic?**
> "Based on HCI research, realistic roles/priorities. Differentiation matches human studies. Needs validation—future work."

**Q: Limitations?**
> "Binary classification only, tabular data, personas not validated, single-threaded. All documented."

**Q: How long?**
> "5-7 minutes total: 3 min technical, 2 min personas. Much faster than human studies."

**Q: Add new explainer?**
> "Create wrapper class, add to evaluator, update config. Personas auto-evaluate any method."

---

## 🚨 Emergency Backup

**If UI fails:**
1. Show screenshots
2. Walk through architecture doc
3. Explain code flow

**If results don't load:**
1. Show code that generates results
2. Explain data flow
3. Reference pre-generated results

**If API issues:**
1. Show technical evaluation only
2. Explain personas conceptually
3. Show pre-generated persona results

---

## ✅ Pre-Demo Checklist

- [ ] Pre-run evaluation
- [ ] Verify results exist
- [ ] Test UI loads
- [ ] Have API key ready
- [ ] Backup screenshots ready
- [ ] Architecture doc open
- [ ] Practice script 3x
- [ ] Time yourself

---

## 💡 Key Points to Emphasize

1. **The Gap:** Technical ≠ Human-friendly
2. **Persona Differentiation:** 2.5-point variance proves stakeholders differ
3. **Stakeholder-Specific:** No one-size-fits-all
4. **Dual Evaluation:** First framework combining both systematically
5. **Domain-Agnostic:** Works for any binary classification task

---

**Remember: You've built something impressive. Be confident!** 🎓
