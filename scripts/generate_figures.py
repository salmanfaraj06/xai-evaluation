"""
Generate all figures for HEXEval research paper.

Usage:
    python scripts/generate_figures.py

Outputs:
    - figures/fidelity_interpretability_gap.pdf
    - figures/persona_differentiation.pdf
    - figures/method_persona_heatmap.pdf
    - figures/runtime_breakdown.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import pandas as pd

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# Create output directory
output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)

# ============================================================================
# Figure 3: Fidelity-Interpretability Gap
# ============================================================================

def plot_fidelity_interpretability_gap():
    """Figure 3: Bar chart comparing technical vs human ratings."""
    
    methods = ['SHAP', 'LIME', 'Anchor', 'DiCE']
    
    # Normalized technical scores (based on deletion AUC, precision, success rate)
    # Heart Disease: SHAP=0.625, LIME=0.570, Anchor=0.951, DiCE=1.0
    # Normalize to 0-1 scale (assuming max is 1.0)
    technical = [0.625, 0.570, 0.951, 1.0]
    
    # Human ratings (trust scores from persona evaluation, normalized to 0-1)
    # SHAP=2.4/5=0.48, LIME=2.2/5=0.44, Anchor=1.9/5=0.38, DiCE=2.1/5=0.42
    human = [0.48, 0.44, 0.38, 0.42]
    
    x = np.arange(len(methods))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, technical, width, label='Technical Score', 
                   color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, human, width, label='Human Rating', 
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add gap annotation
    for i, (t, h) in enumerate(zip(technical, human)):
        gap = t - h
        ax.annotate(f'Gap: {gap:.2f}', 
                   xy=(i, max(t, h)), 
                   xytext=(0, 15), 
                   textcoords='offset points',
                   ha='center', 
                   fontsize=11, 
                   fontweight='bold',
                   color='#34495e',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_ylabel('Normalized Score (0-1)', fontsize=13, fontweight='bold')
    ax.set_xlabel('XAI Method', fontsize=13, fontweight='bold')
    ax.set_title('Fidelity-Interpretability Gap: Technical vs Human Ratings', 
                fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=12)
    ax.legend(fontsize=12, loc='upper left')
    ax.set_ylim([0, 1.2])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fidelity_interpretability_gap.pdf', 
                dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fidelity_interpretability_gap.png', 
                dpi=300, bbox_inches='tight')
    print(f"✓ Generated: {output_dir / 'fidelity_interpretability_gap.pdf'}")
    plt.close()

# ============================================================================
# Figure 4: Persona Differentiation (Radar Chart)
# ============================================================================

def plot_persona_differentiation():
    """Figure 4: Radar chart showing SHAP ratings by persona."""
    
    # Data from persona evaluation (SHAP ratings)
    categories = ['Trust', 'Interpretability', 'Completeness', 
                  'Actionability', 'Satisfaction', 'Decision Support']
    
    personas = {
        'Cardiologist': [1.0, 2.0, 1.0, 1.0, 1.0, 1.0],
        'Researcher': [3.5, 3.8, 3.2, 2.5, 3.0, 3.2],
        'Administrator': [2.0, 2.5, 2.2, 1.5, 2.0, 2.1],
        'Patient': [1.0, 1.5, 1.2, 1.0, 1.0, 1.1]
    }
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    linestyles = ['-', '-', '-', '-']
    linewidths = [2.5, 2.5, 2.5, 2.5]
    
    for i, (persona, values) in enumerate(personas.items()):
        values_plot = values + values[:1]  # Complete the circle
        ax.plot(angles, values_plot, 'o-', 
               linewidth=linewidths[i], 
               label=persona, 
               color=colors[i],
               markersize=8,
               linestyle=linestyles[i])
        ax.fill(angles, values_plot, alpha=0.2, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1', '2', '3', '4', '5'], fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    ax.set_title('Persona Differentiation: SHAP Ratings Across Dimensions', 
                size=16, fontweight='bold', pad=30)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'persona_differentiation.pdf', 
                dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'persona_differentiation.png', 
                dpi=300, bbox_inches='tight')
    print(f"✓ Generated: {output_dir / 'persona_differentiation.pdf'}")
    plt.close()

# ============================================================================
# Figure 5: Method-Persona Heatmap
# ============================================================================

def plot_method_persona_heatmap():
    """Figure 5: Heatmap of satisfaction scores by method-persona."""
    
    # Data: satisfaction scores (1-5 scale)
    data = np.array([
        [1.0, 3.0, 2.0, 1.0],  # SHAP
        [1.0, 2.5, 1.8, 1.0],  # LIME
        [2.0, 2.2, 1.6, 1.5],  # Anchor
        [2.0, 2.0, 1.7, 2.0]   # DiCE
    ])
    
    methods = ['SHAP', 'LIME', 'Anchor', 'DiCE']
    personas = ['Cardiologist', 'Researcher', 'Administrator', 'Patient']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create heatmap
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=1, vmax=5)
    
    # Add text annotations
    for i in range(len(methods)):
        for j in range(len(personas)):
            text_color = 'white' if data[i, j] < 2.5 else 'black'
            text = ax.text(j, i, f'{data[i, j]:.1f}',
                          ha="center", va="center", 
                          color=text_color, 
                          fontweight='bold',
                          fontsize=13)
    
    ax.set_xticks(np.arange(len(personas)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels(personas, fontsize=12, fontweight='bold')
    ax.set_yticklabels(methods, fontsize=12, fontweight='bold')
    ax.set_xlabel('Persona Type', fontsize=13, fontweight='bold')
    ax.set_ylabel('XAI Method', fontsize=13, fontweight='bold')
    ax.set_title('Method-Persona Satisfaction Heatmap', 
                fontsize=15, fontweight='bold', pad=15)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Satisfaction Score (1-5)', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'method_persona_heatmap.pdf', 
                dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'method_persona_heatmap.png', 
                dpi=300, bbox_inches='tight')
    print(f"✓ Generated: {output_dir / 'method_persona_heatmap.pdf'}")
    plt.close()

# ============================================================================
# Figure 7: Runtime Breakdown
# ============================================================================

def plot_runtime_breakdown():
    """Figure 7: Runtime breakdown for evaluation."""
    
    categories = ['SHAP', 'LIME', 'Anchor', 'DiCE', 'Persona\nEvaluation']
    times = [50, 100, 20, 15, 120]  # seconds
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(categories, times, color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{time}s\n({time/60:.1f}min)',
               ha='center', va='bottom',
               fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Time (seconds)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Evaluation Component', fontsize=13, fontweight='bold')
    ax.set_title('HEXEval Runtime Breakdown (Heart Disease, 100 instances)', 
                fontsize=15, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add total time annotation
    total_time = sum(times)
    ax.text(0.98, 0.98, f'Total: {total_time}s ({total_time/60:.1f} min)',
           transform=ax.transAxes,
           fontsize=12, fontweight='bold',
           ha='right', va='top',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'runtime_breakdown.pdf', 
                dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'runtime_breakdown.png', 
                dpi=300, bbox_inches='tight')
    print(f"✓ Generated: {output_dir / 'runtime_breakdown.pdf'}")
    plt.close()

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Generating HEXEval Research Paper Figures")
    print("=" * 60)
    print()
    
    plot_fidelity_interpretability_gap()
    plot_persona_differentiation()
    plot_method_persona_heatmap()
    plot_runtime_breakdown()
    
    print()
    print("=" * 60)
    print("✓ All figures generated successfully!")
    print(f"  Output directory: {output_dir.absolute()}")
    print("=" * 60)
