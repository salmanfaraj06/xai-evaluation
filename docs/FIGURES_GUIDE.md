# Research Paper Figures Guide
## Quick Reference for Using HEXEval Visualizations

---

## 📊 Generated Figures

Run the script to generate all figures:

```bash
python scripts/generate_figures.py
```

**Output:** All figures saved to `figures/` directory in both PDF and PNG formats.

---

## 📋 Figure Checklist for Paper

### **Introduction/Motivation Section:**
- None (text only)

### **Related Work Section:**
- None (text only)

### **Methodology Section:**
- ✅ **Figure 1:** System Architecture (use TikZ/LaTeX)
- ✅ **Figure 2:** Evaluation Pipeline Flow (use TikZ/LaTeX)
- ✅ **Figure 6:** Recommendation Flow Diagram (use TikZ/LaTeX)

### **Results Section:**
- ✅ **Table 1:** Technical Metrics (Heart Disease + Credit Risk)
- ✅ **Table 2:** Persona Ratings Summary
- ✅ **Figure 3:** Fidelity-Interpretability Gap (bar chart)
- ✅ **Table 3:** Persona Differentiation (SHAP ratings)
- ✅ **Figure 4:** Persona Differentiation (radar chart)
- ✅ **Figure 5:** Method-Persona Heatmap
- ✅ **Table 4:** Recommendation Examples

### **Discussion Section:**
- ✅ **Figure 7:** Runtime Breakdown (optional, if discussing performance)

### **Conclusion:**
- None (text only)

---

## 📐 Figure Placement Guidelines

### **Rule of Thumb:**
- **One figure/table per page** (unless small side-by-side)
- **Figure immediately after first mention** in text
- **Caption above table, below figure** (LaTeX default)
- **Reference in text:** "As shown in Figure 3, the fidelity-interpretability gap..."

### **Example Text Integration:**

> "Our evaluation reveals a critical gap between technical excellence and human usability. As shown in Figure 3, methods with high technical fidelity (SHAP: 0.625 deletion AUC) receive poor human ratings (SHAP: 2.4/5 trust). This 2-3 point gap demonstrates the fidelity-interpretability trade-off identified in our research."

---

## 🎨 Figure Quality Checklist

Before including in paper:

- [ ] **Resolution:** 300 DPI minimum (PDF/PNG)
- [ ] **Font Size:** ≥10pt, readable when printed
- [ ] **Colors:** Print-friendly (test grayscale conversion)
- [ ] **Labels:** Clear axis labels, units specified
- [ ] **Legend:** Present and readable
- [ ] **Caption:** Descriptive, explains what figure shows
- [ ] **Consistency:** Same color scheme across figures

---

## 📝 LaTeX Integration

### **Include Generated PDFs:**

```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{figures/fidelity_interpretability_gap.pdf}
\caption{Fidelity-Interpretability Gap: Technical vs Human Ratings}
\label{fig:gap}
\end{figure}
```

### **Reference in Text:**

```latex
As shown in Figure~\ref{fig:gap}, there exists a significant gap...
```

### **Table Example:**

```latex
\begin{table}[h]
\centering
\caption{Technical Evaluation Metrics}
\label{tab:technical}
\input{tables/technical_metrics.tex}  % Or use \begin{tabular}...
\end{table}
```

---

## 🔧 Customization

### **Change Colors:**

Edit `scripts/generate_figures.py`:

```python
# Current colors
colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']

# Change to your preference
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Matplotlib default
```

### **Change Font Sizes:**

```python
# In each plot function, modify:
ax.set_xlabel('...', fontsize=14)  # Increase from 13
ax.set_title('...', fontsize=16)   # Increase from 15
```

### **Change Figure Size:**

```python
fig, ax = plt.subplots(figsize=(12, 8))  # Wider, taller
```

---

## 📊 Data Sources

All figures use data from:

- **Technical Metrics:** `outputs/heart_disease/technical_metrics.csv`
- **Persona Ratings:** `outputs/heart_disease/persona_ratings.csv`
- **Credit Risk:** `outputs/credit_risk/technical_metrics.csv` (for Table 1)

**To update figures with new data:**
1. Re-run HEXEval evaluation
2. Update data arrays in `generate_figures.py`
3. Re-run script

---

## 🎯 Key Messages Per Figure

### **Figure 3 (Gap):**
> "Technical excellence does not guarantee human usability"

### **Figure 4 (Radar):**
> "Different personas value different explanation aspects"

### **Figure 5 (Heatmap):**
> "No single method satisfies all stakeholders"

### **Table 4 (Recommendations):**
> "Stakeholder-specific method selection is necessary"

---

## 📄 Paper Structure Template

```
1. Introduction
   - Problem statement
   - Research question

2. Related Work
   - XAI evaluation frameworks
   - Human-centered evaluation

3. Methodology
   - Figure 1: Architecture
   - Figure 2: Pipeline
   - Figure 6: Recommendations

4. Results
   - Table 1: Technical Metrics
   - Figure 3: Gap Visualization
   - Table 2: Persona Ratings
   - Figure 4: Persona Differentiation
   - Figure 5: Heatmap
   - Table 3: Persona Examples
   - Table 4: Recommendations

5. Discussion
   - Implications
   - Limitations
   - Figure 7: Performance (optional)

6. Conclusion
   - Contributions
   - Future work
```

---

## ✅ Final Checklist Before Submission

- [ ] All figures generated at 300 DPI
- [ ] All tables formatted in LaTeX
- [ ] Figures referenced in text
- [ ] Captions are descriptive
- [ ] Colors are print-friendly
- [ ] Fonts are readable
- [ ] Consistent style across all figures
- [ ] Figures match data in tables
- [ ] Architecture diagrams are clear
- [ ] All figures have labels

---

## 🆘 Troubleshooting

**Issue:** Figures look blurry in PDF
- **Solution:** Ensure 300 DPI, use PDF format (not PNG)

**Issue:** Colors don't print well
- **Solution:** Test grayscale conversion, use high-contrast colors

**Issue:** Text too small
- **Solution:** Increase font sizes in `generate_figures.py`

**Issue:** Figure too large/small
- **Solution:** Adjust `figsize` parameter or use `width` in LaTeX `\includegraphics`

---

**All figures are ready for your research paper!** 🎓
