"""Generate Figure 5.1 — three side-by-side confusion-matrix panels.

Counts taken verbatim from results/verified_metrics.json at threshold 0.50.
Output: fyp_report_assets/figure_5_1_confusion_matrices.png at 300 DPI.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

# (TP, FP, FN, TN) per the brief — locked test-set counts at threshold 0.50
PANELS = [
    ("Baseline\nXGBoost (SMOTE+Tuned)",                  1733, 131, 412, 553443),
    ("Comparator\nLSTM + Random Forest",                  2029, 968, 116, 552606),
    ("Proposed\nAE + BDS + GA + XGBoost",                 1749, 124, 396, 553450),
]

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "fyp_report_assets",
    "figure_5_1_confusion_matrices.png",
)


def main() -> str:
    # Confusion-matrix layout (rows = actual, cols = predicted):
    #   [[TN, FP],
    #    [FN, TP]]
    matrices = []
    for _name, tp, fp, fn, tn in PANELS:
        matrices.append(np.array([[tn, fp], [fn, tp]], dtype=float))

    # Shared LogNorm so all three panels use the same colour mapping.
    # Floor at 1 to avoid log(0).
    vmin = 1.0
    vmax = max(m.max() for m in matrices)
    norm = LogNorm(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    tick_labels = ["Legitimate", "Fraud"]

    for ax, (title, tp, fp, fn, tn), cm in zip(axes, PANELS, matrices):
        # Mask zeros for log-scale display (none here, but defensive)
        cm_for_display = np.where(cm > 0, cm, np.nan)
        im = ax.imshow(cm_for_display, cmap="Blues", norm=norm, aspect="equal")

        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(tick_labels, fontsize=10)
        ax.set_yticklabels(tick_labels, fontsize=10)

        # Annotate every cell with the count. Pick text colour by cell intensity.
        threshold = vmax / 4.0  # rough "dark cell" threshold under log scale
        for (i, j), value in np.ndenumerate(cm):
            text_color = "white" if value > threshold else "#222222"
            ax.text(
                j, i, f"{int(value):,}",
                ha="center", va="center",
                fontsize=12, fontweight="bold",
                color=text_color,
            )

        # Clean academic style: no grid, thin spines
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_edgecolor("#888888")
            spine.set_linewidth(0.6)

    # One shared colourbar across all three panels
    cbar = fig.colorbar(im, ax=axes, shrink=0.7, aspect=20, pad=0.02)
    cbar.set_label("Transaction count (log scale)", fontsize=10)

    fig.patch.set_facecolor("white")
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return OUT


if __name__ == "__main__":
    path = main()
    print(path)
