import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(1, 1, figsize=(22, 16))
ax.set_xlim(0, 22)
ax.set_ylim(0, 16)
ax.axis('off')

# ── Title ──
ax.text(11, 15.4, 'FraudLens — Entity Relationship Diagram', ha='center', va='center',
        fontsize=20, fontweight='bold', color='#111')

ROW_H = 0.48

def draw_table(ax, x, y, w, title, columns, header_color):
    n = len(columns)
    body_h = ROW_H * n
    total_h = body_h + 0.6

    # Table body
    ax.add_patch(FancyBboxPatch((x, y - body_h), w, body_h,
                 boxstyle="round,pad=0.05", facecolor='#fafafa',
                 edgecolor='#999', linewidth=1.5, zorder=1))

    # Header bar
    ax.add_patch(FancyBboxPatch((x, y), w, 0.6,
                 boxstyle="round,pad=0.05", facecolor=header_color,
                 edgecolor=header_color, linewidth=1.5, zorder=2))
    ax.text(x + w/2, y + 0.3, title, ha='center', va='center',
            fontsize=13, fontweight='bold', color='white', family='monospace', zorder=3)

    for i, (key_type, col_name, col_type) in enumerate(columns):
        cy = y - ROW_H * i - ROW_H / 2 - 0.08

        # Separator
        if i > 0:
            ax.plot([x + 0.08, x + w - 0.08], [cy + ROW_H/2, cy + ROW_H/2],
                    color='#e0e0e0', lw=0.8, zorder=2)

        # PK/FK badge
        if key_type:
            if key_type == 'PK':
                badge_fc = '#1565c0'
            else:
                badge_fc = '#ef6c00'
            ax.text(x + 0.38, cy, key_type, ha='center', va='center',
                    fontsize=9, fontweight='bold', color='white', family='monospace',
                    bbox=dict(boxstyle='round,pad=0.18', facecolor=badge_fc,
                              edgecolor=badge_fc, linewidth=1.2), zorder=5)

        # Column name
        ax.text(x + 0.8, cy, col_name, ha='left', va='center',
                fontsize=10, color='#222', family='monospace', zorder=3)

        # Data type
        ax.text(x + w - 0.2, cy, col_type, ha='right', va='center',
                fontsize=9.5, color='#777', family='monospace', zorder=3)


# ══════════════════════════════════════════
# TABLES
# ══════════════════════════════════════════

# card_history (top-left)
card_cols = [
    ('PK', 'id', 'INTEGER'),
    ('', 'cardholder_id', 'TEXT'),
    ('', 'amount', 'REAL'),
    ('', 'timestamp', 'DATETIME'),
    ('', 'merchant_category', 'TEXT'),
    ('', 'is_fraud', 'INTEGER'),
]
draw_table(ax, 0.3, 14.0, 5.0, 'card_history', card_cols, '#2e7d32')

# transactions (center)
txn_cols = [
    ('PK', 'id', 'INTEGER'),
    ('', 'transaction_id', 'TEXT'),
    ('FK', 'cardholder_id', 'TEXT'),
    ('', 'amount', 'REAL'),
    ('', 'merchant', 'TEXT'),
    ('', 'timestamp', 'DATETIME'),
    ('', 'model_name', 'TEXT'),
    ('', 'fraud_probability', 'REAL'),
    ('', 'risk_level', 'TEXT'),
    ('', 'rule_triggers', 'TEXT'),
    ('', 'features_json', 'TEXT'),
]
draw_table(ax, 7.5, 14.0, 6.5, 'transactions', txn_cols, '#1565c0')

# alerts (top-right)
alert_cols = [
    ('PK', 'id', 'INTEGER'),
    ('FK', 'transaction_id', 'INTEGER'),
    ('', 'alert_type', 'TEXT'),
    ('', 'severity', 'TEXT'),
    ('', 'status', 'TEXT'),
    ('', 'created_at', 'DATETIME'),
]
draw_table(ax, 16.5, 14.0, 5.0, 'alerts', alert_cols, '#bf360c')

# model_metrics (bottom-left)
mm_cols = [
    ('PK', 'id', 'INTEGER'),
    ('', 'model_name', 'TEXT'),
    ('', 'date', 'DATE'),
    ('', 'f1_score', 'REAL'),
    ('', 'precision', 'REAL'),
    ('', 'recall', 'REAL'),
    ('', 'psi_value', 'REAL'),
    ('', 'drift_flag', 'INTEGER'),
]
draw_table(ax, 0.3, 7.0, 5.0, 'model_metrics', mm_cols, '#4527a0')
ax.text(2.8, 2.8, '(standalone — no FK joins)', ha='center', va='center',
        fontsize=9, fontstyle='italic', color='#888')

# feedback (bottom-right)
fb_cols = [
    ('PK', 'id', 'INTEGER'),
    ('FK', 'transaction_id', 'INTEGER'),
    ('', 'actual_label', 'INTEGER'),
    ('', 'analyst_notes', 'TEXT'),
    ('', 'reviewed_at', 'DATETIME'),
]
draw_table(ax, 16.5, 9.5, 5.0, 'feedback', fb_cols, '#6a1b9a')

# ══════════════════════════════════════════
# RELATIONSHIPS
# ══════════════════════════════════════════

# card_history N:1 → transactions
ax.annotate('', xy=(7.5, 12.8), xytext=(5.3, 12.8),
            arrowprops=dict(arrowstyle='->', color='#333', lw=2))
ax.text(6.4, 13.2, 'N : 1', ha='center', va='center',
        fontsize=11, fontstyle='italic', fontweight='bold', color='#444')

# transactions 1:N → alerts
ax.annotate('', xy=(16.5, 12.8), xytext=(14.0, 12.8),
            arrowprops=dict(arrowstyle='->', color='#333', lw=2))
ax.text(15.25, 13.2, '1 : N', ha='center', va='center',
        fontsize=11, fontstyle='italic', fontweight='bold', color='#444')

# transactions 1:1 → feedback
ax.annotate('', xy=(16.5, 9.2), xytext=(14.0, 9.5),
            arrowprops=dict(arrowstyle='->', color='#333', lw=2))
ax.text(15.25, 9.8, '1 : 1', ha='center', va='center',
        fontsize=11, fontstyle='italic', fontweight='bold', color='#444')

plt.tight_layout()
plt.savefig('scripts/figures/fig_3_5_erd.png', dpi=300, bbox_inches='tight',
            facecolor='white')
plt.savefig('scripts/figures/fig_3_5_erd.pdf', bbox_inches='tight',
            facecolor='white')
print("Saved fig_3_5_erd.png and .pdf")
plt.show()
