"""Figure 3: Cluster-bounded reallocation — withdrawals prevented per year vs.
the entry-time window W (months). PJM-scale, 90 seeds.

Single-panel scatter with 95% CI bars at each W ∈ {12, 18, 24, 36, 48, 60, 72}.
Horizontal reference at zero; vertical reference at W=12 (headline).

Reads:  ABM/full_abm/output/cluster_bound_prevention.csv
Writes: paper_figures/fig3_cluster_bound_prevention.{pdf,png}
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREV_CSV = os.path.join(ROOT, "ABM/full_abm/output/cluster_bound_prevention.csv")
OUT_PDF = os.path.join(ROOT, "paper_figures/fig3_cluster_bound_prevention.pdf")
OUT_PNG = os.path.join(ROOT, "paper_figures/fig3_cluster_bound_prevention.png")

CREAM = "#FAF7F2"
INK   = "#1F1A17"
GRID  = "#E6DFD5"
CORAL = "#CC785C"
BROWN = "#8C6F5A"


def main():
    df = pd.read_csv(PREV_CSV).sort_values("W").reset_index(drop=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    fig.patch.set_facecolor(CREAM)
    ax.set_facecolor(CREAM)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # Reference lines
    ax.axhline(0.0, color=INK, linewidth=0.8, alpha=0.55, zorder=1)
    ax.axvline(12.0, color=BROWN, linewidth=0.8, alpha=0.55,
               linestyle=(0, (3, 2)), zorder=1)

    W = df["W"].to_numpy()
    y = df["prev_per_year_pjm"].to_numpy()
    se = df["prev_per_year_pjm_se"].to_numpy()
    yerr = 1.96 * se

    # Connecting line (visual aid)
    ax.plot(W, y, color=CORAL, linewidth=1.2, alpha=0.55,
            zorder=2)

    # CI bars + points
    ax.errorbar(W, y, yerr=yerr, fmt="none", ecolor=CORAL,
                elinewidth=1.4, capsize=4, capthick=1.2, zorder=3)
    ax.plot(W, y, linestyle="none",
            marker="o", markersize=8,
            markerfacecolor=CREAM, markeredgecolor=CORAL,
            markeredgewidth=1.6, zorder=4)

    # Headline label at W=12
    w12 = df.loc[df["W"] == 12].iloc[0]
    ax.annotate(f"+{w12['prev_per_year_pjm']:.1f}/yr  "
                rf"[{w12['prev_per_year_pjm_lo']:+.1f}, {w12['prev_per_year_pjm_hi']:+.1f}]",
                xy=(12, w12["prev_per_year_pjm"]),
                xytext=(20, w12["prev_per_year_pjm"] - 1.5),
                fontsize=9.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.6, alpha=0.7))

    ax.text(12.4, ax.get_ylim()[1] if False else 36, "headline (W=12)",
            fontsize=8.5, color=BROWN, ha="left", va="bottom",
            rotation=0)

    # Axes
    ax.set_xticks([12, 18, 24, 36, 48, 60, 72])
    ax.set_xlim(8, 76)
    ax.set_xlabel("entry-time window W (months)", fontsize=10.5, color=INK)
    ax.set_ylabel("withdrawals prevented per year (PJM-scale)\n"
                  "vs. unbounded reallocation",
                  fontsize=10.5, color=INK)
    y_lo = min(0, (y - yerr).min()) - 3
    y_hi = (y + yerr).max() + 6
    ax.set_ylim(y_lo, y_hi)

    # Legend
    legend_handles = [
        Line2D([0], [0], color=CORAL, lw=1.4, marker="o", markersize=8,
               markerfacecolor=CREAM, markeredgecolor=CORAL,
               markeredgewidth=1.6,
               label="point estimate, 95% CI (n = 90 seeds)"),
        Line2D([0], [0], color=BROWN, lw=0.8, linestyle=(0, (3, 2)),
               label="W=12 (Order 2023 cluster window)"),
        Line2D([0], [0], color=INK, lw=0.8, alpha=0.6,
               label="zero reference"),
    ]
    leg = ax.legend(handles=legend_handles, loc="upper right",
                    frameon=True, framealpha=0.95,
                    edgecolor=GRID, fontsize=9, handlelength=2.6,
                    borderpad=0.6)
    leg.get_frame().set_facecolor(CREAM)

    ax.set_title(
        "Cluster-bound reallocation prevents cascade withdrawals at narrow W",
        fontsize=11.5, color=INK, loc="left", pad=10, weight="bold")

    fig.tight_layout()
    fig.savefig(OUT_PDF, facecolor=CREAM, bbox_inches="tight")
    fig.savefig(OUT_PNG, facecolor=CREAM, dpi=220, bbox_inches="tight")
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
