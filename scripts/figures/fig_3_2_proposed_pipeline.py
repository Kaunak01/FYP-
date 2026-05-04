import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(1, 1, figsize=(14, 18))
ax.set_xlim(0, 14)
ax.set_ylim(0, 18)
ax.axis('off')

def rounded_rect(ax, x, y, w, h, fc, ec, lw=1.5, radius=0.2):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad={radius}",
                         facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(box)

def arrow_down(ax, x, y1, y2, dashed=False):
    style = 'dashed' if dashed else 'solid'
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5,
                                linestyle=style))

def arrow_diag(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5))

# ── Row 1: Raw transaction ──
rounded_rect(ax, 4.5, 16.2, 5, 1.3, '#f5f5f0', '#bbb', lw=1.2)
ax.text(7, 16.95, 'Raw transaction', ha='center', va='center',
        fontsize=13, fontweight='bold', color='#333')

# Arrows from Raw transaction down to 3 feature boxes
arrow_diag(ax, 5.5, 16.2, 2.5, 14.8)
arrow_down(ax, 7, 16.2, 14.8)
arrow_diag(ax, 8.5, 16.2, 11.5, 14.8)

# ── Row 2: Three feature extraction boxes ──
# Velocity features (green)
rounded_rect(ax, 0.5, 13.5, 4.0, 1.3, '#e0f2f1', '#4db6ac', lw=1.5)
ax.text(2.5, 14.3, 'Velocity features', ha='center', va='center',
        fontsize=11, fontweight='bold', color='#00695c')
ax.text(2.5, 13.85, '1h count, 24h count, amount velocity', ha='center', va='center',
        fontsize=8.5, fontstyle='italic', color='#555')

# BDS scoring (purple/lavender)
rounded_rect(ax, 5.0, 13.5, 4.0, 1.3, '#e8eaf6', '#7986cb', lw=1.5)
ax.text(7, 14.3, 'BDS scoring', ha='center', va='center',
        fontsize=11, fontweight='bold', color='#283593')
ax.text(7, 13.85, '4-dimension GA-optimised', ha='center', va='center',
        fontsize=8.5, fontstyle='italic', color='#555')

# Autoencoder (peach)
rounded_rect(ax, 9.5, 13.5, 4.0, 1.3, '#fbe9e7', '#e57373', lw=1.5)
ax.text(11.5, 14.3, 'Autoencoder', ha='center', va='center',
        fontsize=11, fontweight='bold', color='#bf360c')
ax.text(11.5, 13.85, 'Reconstruction error signal', ha='center', va='center',
        fontsize=8.5, fontstyle='italic', color='#555')

# ── Arrow from BDS down to GA parameter tuning (dashed) ──
arrow_down(ax, 7, 13.5, 12.3, dashed=True)

# ── Row 3: GA parameter tuning ──
rounded_rect(ax, 4.5, 11.0, 5, 1.3, '#f5f5f0', '#bbb', lw=1.2)
ax.text(7, 11.75, 'GA parameter tuning', ha='center', va='center',
        fontsize=12, fontweight='bold', color='#333')
ax.text(7, 11.3, 'F1 fitness function', ha='center', va='center',
        fontsize=9, fontstyle='italic', color='#555')

# ── Arrows from velocity, GA, autoencoder down to Feature fusion ──
arrow_down(ax, 2.5, 13.5, 9.3)
arrow_down(ax, 7, 11.0, 9.3)
arrow_down(ax, 11.5, 13.5, 9.3)

# ── Row 4: Feature fusion ──
rounded_rect(ax, 2.0, 8.0, 10, 1.3, '#fff8e1', '#c8a415', lw=1.5)
ax.text(7, 8.75, 'Feature fusion', ha='center', va='center',
        fontsize=13, fontweight='bold', color='#6d4c00')
ax.text(7, 8.3, '19 engineered + velocity + BDS + AE reconstruction error', ha='center', va='center',
        fontsize=9, fontstyle='italic', color='#555')

# ── Arrow down to XGBoost ──
arrow_down(ax, 7, 8.0, 6.8)

# ── Row 5: XGBoost classifier ──
rounded_rect(ax, 4.0, 5.5, 6, 1.3, '#e3f2fd', '#42a5f5', lw=1.5)
ax.text(7, 6.25, 'XGBoost classifier', ha='center', va='center',
        fontsize=13, fontweight='bold', color='#1565c0')
ax.text(7, 5.8, 'Supervised classification', ha='center', va='center',
        fontsize=9, fontstyle='italic', color='#555')

# ── Arrows from XGBoost down to two outputs ──
arrow_diag(ax, 5.5, 5.5, 3.0, 4.3)
arrow_diag(ax, 8.5, 5.5, 11.0, 4.3)

# ── Row 6: Two output boxes ──
# Fraud probability (green)
rounded_rect(ax, 0.5, 3.0, 5.0, 1.3, '#e8f5e9', '#66bb6a', lw=1.5)
ax.text(3.0, 3.8, 'Fraud probability', ha='center', va='center',
        fontsize=11, fontweight='bold', color='#2e7d32')
ax.text(3.0, 3.3, '3-tier triage (FRAUD/REVIEW/MONITOR)', ha='center', va='center',
        fontsize=8.5, fontstyle='italic', color='#555')

# SHAP attribution (purple)
rounded_rect(ax, 8.5, 3.0, 5.0, 1.3, '#ede7f6', '#7e57c2', lw=1.5)
ax.text(11.0, 3.8, 'SHAP attribution', ha='center', va='center',
        fontsize=11, fontweight='bold', color='#4527a0')
ax.text(11.0, 3.3, 'Per-feature explanation', ha='center', va='center',
        fontsize=8.5, fontstyle='italic', color='#555')

plt.tight_layout()
plt.savefig('scripts/figures/fig_3_2_proposed_pipeline.png', dpi=300, bbox_inches='tight',
            facecolor='white')
plt.savefig('scripts/figures/fig_3_2_proposed_pipeline.pdf', bbox_inches='tight',
            facecolor='white')
print("Saved fig_3_2_proposed_pipeline.png and .pdf")
plt.show()
