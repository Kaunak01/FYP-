import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(1, 1, figsize=(16, 17))
ax.set_xlim(0, 16)
ax.set_ylim(0, 17)
ax.axis('off')

def rounded_rect(ax, x, y, w, h, color, edge_color, lw=1.5, radius=0.3):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad={radius}",
                         facecolor=color, edgecolor=edge_color, linewidth=lw)
    ax.add_patch(box)

def arrow_down(ax, x, y1, y2):
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle='->', color='#555555', lw=1.8,
                                linestyle='dashed'))

def arrow_right(ax, x1, x2, y):
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5))

# ── Layer 1: Data pipeline ──
rounded_rect(ax, 0.5, 12.8, 15, 3.8, '#e8f5e9', '#4caf50', lw=2.5, radius=0.4)
ax.text(8, 16.1, 'Layer 1: Data pipeline', ha='center', va='center',
        fontsize=15, fontweight='bold', color='#2e7d32')

# Layer 1 boxes
boxes_l1 = [
    (1.2, 13.6, 'Sparkov dataset', '1.85M transactions'),
    (4.5, 13.6, 'EDA', ''),
    (7.2, 13.6, 'Feature engineering', '19 features incl. 3 velocity'),
    (11.2, 13.6, 'Engineered CSVs', 'Train + test splits'),
]
for (bx, by, title, sub) in boxes_l1:
    w = 3.0 if title != 'EDA' else 2.2
    rounded_rect(ax, bx, by, w, 1.6, '#e3f2fd', '#42a5f5', lw=1.2, radius=0.2)
    ty = by + 1.0 if sub else by + 0.8
    ax.text(bx + w/2, ty, title, ha='center', va='center',
            fontsize=10, fontweight='bold', color='#1565c0')
    if sub:
        ax.text(bx + w/2, by + 0.45, sub, ha='center', va='center',
                fontsize=8.5, fontstyle='italic', color='#555555')

# Layer 1 arrows
arrow_right(ax, 4.2, 4.5, 14.4)
arrow_right(ax, 6.7, 7.2, 14.4)
arrow_right(ax, 10.2, 11.2, 14.4)

# ── Arrow Layer 1 → Layer 2 ──
arrow_down(ax, 8, 12.8, 12.2)

# ── Layer 2: ML pipeline ──
rounded_rect(ax, 0.5, 5.5, 15, 6.6, '#ede7f6', '#5c6bc0', lw=2.5, radius=0.4)
ax.text(8, 11.5, 'Layer 2: ML pipeline', ha='center', va='center',
        fontsize=15, fontweight='bold', color='#283593')

# Model boxes (top row)
model_colors = [
    (1.2, 8.8, 3.5, 'Baseline', 'XGBoost', '#e3f2fd', '#42a5f5', '#1565c0'),
    (5.5, 8.8, 4.0, 'Comparator', 'LSTM + Random Forest', '#fbe9e7', '#e64a19', '#bf360c'),
    (10.5, 8.8, 4.5, 'Proposed', 'AE + BDS + GA + XGBoost', '#fff8e1', '#c8a415', '#795548'),
]
for (bx, by, w, title, sub, fc, ec, tc) in model_colors:
    rounded_rect(ax, bx, by, w, 1.5, fc, ec, lw=1.2, radius=0.2)
    ax.text(bx + w/2, by + 0.95, title, ha='center', va='center',
            fontsize=11, fontweight='bold', color=tc)
    ax.text(bx + w/2, by + 0.4, sub, ha='center', va='center',
            fontsize=9, fontstyle='italic', color='#555555')

# Auxiliary boxes (bottom row)
aux_boxes = [
    (1.2, 6.2, 3.5, 'SHAP explainer'),
    (5.5, 6.2, 4.0, 'GA optimiser'),
    (10.5, 6.2, 4.5, 'PSI drift detector'),
]
for (bx, by, w, title) in aux_boxes:
    rounded_rect(ax, bx, by, w, 1.2, '#eceff1', '#90a4ae', lw=1.0, radius=0.2)
    ax.text(bx + w/2, by + 0.6, title, ha='center', va='center',
            fontsize=10, fontweight='bold', color='#455a64')

ax.text(15.7, 7.0, 'Auxiliary', ha='center', va='center',
        fontsize=9, fontstyle='italic', color='#78909c')

# ── Arrow Layer 2 → Layer 3 ──
arrow_down(ax, 8, 5.5, 5.0)

# ── Layer 3: Flask application ──
rounded_rect(ax, 0.5, 0.5, 15, 3.9, '#f1f8e9', '#689f38', lw=2.5, radius=0.4)
ax.text(8, 4.0, 'Layer 3: Flask application', ha='center', va='center',
        fontsize=15, fontweight='bold', color='#33691e')

# Layer 3 boxes
boxes_l3 = [
    (1.2, 1.2, 3.0, 'SQLite DB', 'Persistence'),
    (4.8, 1.2, 3.0, 'API routes', 'Endpoints'),
    (8.4, 1.2, 3.3, '7-page dashboard', 'Analyst interface'),
    (12.3, 1.2, 3.0, 'PDF reports', 'Compliance'),
]
for (bx, by, w, title, sub) in boxes_l3:
    rounded_rect(ax, bx, by, w, 1.8, '#dcedc8', '#7cb342', lw=1.0, radius=0.2)
    ax.text(bx + w/2, by + 1.1, title, ha='center', va='center',
            fontsize=10, fontweight='bold', color='#33691e')
    ax.text(bx + w/2, by + 0.5, sub, ha='center', va='center',
            fontsize=8.5, fontstyle='italic', color='#555555')

plt.tight_layout()
plt.savefig('scripts/figures/fig_3_1_architecture.png', dpi=300, bbox_inches='tight',
            facecolor='white')
plt.savefig('scripts/figures/fig_3_1_architecture.pdf', bbox_inches='tight',
            facecolor='white')
print("Saved fig_3_1_architecture.png and .pdf")
plt.show()
