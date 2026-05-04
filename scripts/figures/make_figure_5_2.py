"""Generate Figure 5.2 — SHAP feature-importance bar chart for the Proposed model.

Reads results/shap_top_features.json and renders a horizontal bar chart of all
19 features ranked by mean |SHAP|, descending. Saved at 300 DPI to
fyp_report_assets/figure_5_2_shap_bar_chart.png.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "results", "shap_top_features.json")
OUT = os.path.join(ROOT, "fyp_report_assets", "figure_5_2_shap_bar_chart.png")


def main() -> str:
    with open(SRC) as f:
        data = json.load(f)

    items = sorted(
        data["top_features"], key=lambda x: x["mean_abs_shap"], reverse=True
    )
    names = [it["feature"] for it in items]
    values = [it["mean_abs_shap"] for it in items]

    # Top of the chart should be the highest-importance feature.
    # matplotlib.barh draws bottom-up, so reverse for plotting.
    names_plot = list(reversed(names))
    values_plot = list(reversed(values))

    fig, ax = plt.subplots(figsize=(8, 10), constrained_layout=True)

    bars = ax.barh(
        names_plot,
        values_plot,
        color="#3a6ea5",
        edgecolor="#1f3d66",
        linewidth=0.6,
        height=0.72,
    )

    # Annotate each bar with its mean |SHAP| value
    xmax = max(values_plot)
    for bar, v in zip(bars, values_plot):
        ax.text(
            v + xmax * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.3f}",
            va="center",
            ha="left",
            fontsize=10,
            color="#222222",
        )

    ax.set_xlabel("Mean |SHAP| value", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_xlim(0, xmax * 1.15)

    # Clean academic style: hide top/right spines, no grid
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_edgecolor("#888888")
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(axis="both", which="both", length=3, color="#888888")
    ax.grid(False)

    fig.patch.set_facecolor("white")
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return OUT


if __name__ == "__main__":
    print(main())
