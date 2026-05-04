import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(1, 1, figsize=(14, 20))
ax.set_xlim(0, 14)
ax.set_ylim(0, 20)
ax.axis('off')

def rbox(ax, x, y, w, h, fc, ec, lw=2, rad=0.12):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad={rad}",
                 facecolor=fc, edgecolor=ec, linewidth=lw))

def arrow(ax, x1, y1, x2, y2, dashed=False):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#111', lw=2.0,
                                linestyle='dashed' if dashed else 'solid'))

# ── Title ──
ax.text(7, 19.5, 'FraudLens – Data Flow Diagram', ha='center', va='center',
        fontsize=18, fontweight='bold', fontstyle='italic', color='#000')

# ── Section: Training pipeline ──
ax.text(7, 18.5, 'Training pipeline', ha='center', va='center',
        fontsize=15, fontweight='bold', fontstyle='italic', color='#111')

# ══════════════════════════════════════════
# TRAINING ROW
# ══════════════════════════════════════════

# Raw CSV
rbox(ax, 0.5, 16.5, 2.6, 1.5, '#4da6ff', '#0066cc')
ax.text(1.8, 17.45, 'Raw CSV', ha='center', va='center',
        fontsize=11, fontweight='bold', color='black')
ax.text(1.8, 16.95, '(Sparkov dataset)', ha='center', va='center',
        fontsize=9, color='black')

arrow(ax, 3.1, 17.25, 4.0, 17.25)

# EDA notebook
rbox(ax, 4.0, 16.5, 2.8, 1.5, '#4da6ff', '#0066cc')
ax.text(5.4, 17.45, 'EDA notebook', ha='center', va='center',
        fontsize=11, fontweight='bold', color='black')
ax.text(5.4, 16.95, '(feature engineering)', ha='center', va='center',
        fontsize=9, color='black')

arrow(ax, 6.8, 17.25, 7.7, 17.25)

# Engineered CSV
rbox(ax, 7.7, 16.5, 2.8, 1.5, '#2b8aff', '#0055cc')
ax.text(9.1, 17.45, 'Engineered CSV', ha='center', va='center',
        fontsize=11, fontweight='bold', color='black')
ax.text(9.1, 16.95, '(19 features)', ha='center', va='center',
        fontsize=9, color='black')

arrow(ax, 10.5, 17.25, 11.2, 17.25)

# Training (3 models) — bold teal
rbox(ax, 11.2, 16.6, 2.3, 1.3, '#00cc99', '#008866', rad=0.25)
ax.text(12.35, 17.4, 'Training', ha='center', va='center',
        fontsize=11, fontweight='bold', color='black')
ax.text(12.35, 16.95, '(3 models)', ha='center', va='center',
        fontsize=9, color='black')

# Arrow ↓ from Training to Saved artefacts
arrow(ax, 12.35, 16.6, 9.8, 15.5)

# Saved artefacts
rbox(ax, 7.8, 14.3, 3.2, 1.2, '#4da6ff', '#0066cc')
ax.text(9.4, 15.05, 'Saved artefacts', ha='center', va='center',
        fontsize=11, fontweight='bold', color='black')
ax.text(9.4, 14.6, '(.joblib + .pt files)', ha='center', va='center',
        fontsize=9, color='black')

# ══════════════════════════════════════════
# SECTION: Live inference pipeline
# ══════════════════════════════════════════
ax.text(3.5, 13.2, 'Live inference pipeline', ha='center', va='center',
        fontsize=15, fontweight='bold', fontstyle='italic', color='#111')

# Dashed arrow from Saved artefacts down to Model manager
arrow(ax, 9.4, 14.3, 9.4, 12.5, dashed=True)

# ══════════════════════════════════════════
# LIVE INFERENCE ROW
# ══════════════════════════════════════════

# User input
rbox(ax, 0.3, 11.0, 2.5, 1.6, '#4da6ff', '#0066cc')
ax.text(1.55, 12.05, 'User input', ha='center', va='center',
        fontsize=11, fontweight='bold', color='black')
ax.text(1.55, 11.5, '(form / stream /\n csv)', ha='center', va='center',
        fontsize=8.5, color='black')

arrow(ax, 2.8, 11.8, 3.7, 11.8)

# Preprocessor
rbox(ax, 3.7, 11.0, 2.8, 1.6, '#4da6ff', '#0066cc')
ax.text(5.1, 12.05, 'Preprocessor', ha='center', va='center',
        fontsize=11, fontweight='bold', color='black')
ax.text(5.1, 11.45, '(velocity +\nscaling)', ha='center', va='center',
        fontsize=8.5, color='black')

arrow(ax, 6.5, 11.8, 7.5, 11.8)

# Model manager
rbox(ax, 7.5, 10.8, 3.0, 1.8, '#2b8aff', '#0055cc')
ax.text(9.0, 12.0, 'Model manager', ha='center', va='center',
        fontsize=11, fontweight='bold', color='black')
ax.text(9.0, 11.35, '(loads active\nmodel)', ha='center', va='center',
        fontsize=8.5, color='black')

arrow(ax, 10.5, 11.8, 11.3, 11.8)

# Inference (score) — bold teal
rbox(ax, 11.3, 11.2, 2.2, 1.2, '#00cc99', '#008866', rad=0.25)
ax.text(12.4, 11.95, 'Inference', ha='center', va='center',
        fontsize=11, fontweight='bold', color='black')
ax.text(12.4, 11.5, '(score)', ha='center', va='center',
        fontsize=9, color='black')

# Arrow ↓ from Inference to Postprocessor
arrow(ax, 12.4, 11.2, 9.5, 9.7)

# Postprocessor
rbox(ax, 7.5, 8.5, 3.2, 1.2, '#2b8aff', '#0055cc')
ax.text(9.1, 9.25, 'Postprocessor', ha='center', va='center',
        fontsize=11, fontweight='bold', color='black')
ax.text(9.1, 8.8, '(SHAP + rules)', ha='center', va='center',
        fontsize=9, color='black')

# Arrow ↓ from Postprocessor to SQLite
arrow(ax, 8.0, 8.5, 5.5, 7.2)

# SQLite database
rbox(ax, 2.2, 5.8, 5.8, 1.4, '#4da6ff', '#0066cc')
ax.text(5.1, 6.7, 'SQLite database', ha='center', va='center',
        fontsize=11, fontweight='bold', color='black')
ax.text(5.1, 6.15, '(persist transaction + score + SHAP)', ha='center', va='center',
        fontsize=9, color='black')

# Arrow ↓ from SQLite to Dashboard
arrow(ax, 5.1, 5.8, 5.1, 4.5)

# Dashboard response
rbox(ax, 2.2, 3.2, 5.8, 1.3, '#4da6ff', '#0066cc')
ax.text(5.1, 4.05, 'Dashboard response', ha='center', va='center',
        fontsize=11, fontweight='bold', color='black')
ax.text(5.1, 3.55, '(score + SHAP + alert badge)', ha='center', va='center',
        fontsize=9, color='black')

plt.tight_layout()
plt.savefig('scripts/figures/fig_3_4_data_flow.png', dpi=300, bbox_inches='tight',
            facecolor='white')
plt.savefig('scripts/figures/fig_3_4_data_flow.pdf', bbox_inches='tight',
            facecolor='white')
print("Saved fig_3_4_data_flow.png and .pdf")
plt.show()
