"""Figure 5: Kernel-scale sensitivity — does the W=12 cluster-bound headline
survive varying `network_distance_scale` s?

Six points (s ∈ {0.1, 0.2, 0.3, 0.5, 1.0, 5.0}) with 95% CI bars; log x-axis
since s spans 50×. Horizontal references at zero and the +30/yr default
headline.

Reads:  ABM/full_abm/output/kernel_sensitivity_summary.csv
Writes: paper_figures/fig5_kernel_sensitivity.{pdf,png}
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KSCALE_CSV = os.path.join(ROOT, "ABM/full_abm/output/kernel_sensitivity_summary.csv")
OUT_PDF = os.path.join(ROOT, "paper_figures/fig5_kernel_sensitivity.pdf")
OUT_PNG = os.path.join(ROOT, "paper_figures/fig5_kernel_sensitivity.png")

CREAM = "#FAF7F2"
INK   = "#1F1A17"
GRID  = "#E6DFD5"
CORAL = "#CC785C"
BROWN = "#8C6F5A"
HEADLINE = 29.85   # W=12 default-α / default-s headline (+30/yr PJM)


def main():
    df = pd.read_csv(KSCALE_CSV).sort_values("scale").reset_index(drop=True)

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

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    fig.patch.set_facecolor(CREAM)
    ax.set_facecolor(CREAM)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # Reference lines
    ax.axhline(0.0, color=INK, linewidth=0.8, alpha=0.55, zorder=1)
    ax.axhline(HEADLINE, color=BROWN, linewidth=0.8, alpha=0.6,
               linestyle=(0, (3, 2)), zorder=1)

    s = df["scale"].to_numpy()
    y = df["prev_per_year_pjm"].to_numpy()
    se = df["prev_per_year_pjm_se"].to_numpy()
    yerr = 1.96 * se

    ax.errorbar(s, y, yerr=yerr, fmt="none", ecolor=CORAL,
                elinewidth=1.4, capsize=5, capthick=1.2, zorder=3)
    ax.plot(s, y, linestyle="none",
            marker="o", markersize=9,
            markerfacecolor=CREAM, markeredgecolor=CORAL,
            markeredgewidth=1.7, zorder=4)

    ax.text(5.0, HEADLINE + 0.6, f"+{HEADLINE:.0f}/yr (default s)",
            fontsize=8.8, color=BROWN, ha="right", va="bottom",
            style="italic")

    for si, yi, yse in zip(s, y, se):
        ax.text(si, yi + 1.96 * yse + 1.5,
                f"+{yi:.1f}", ha="center", va="bottom",
                fontsize=8.8, color=INK)

    ax.set_xscale("log")
    ax.set_xticks(s)
    ax.set_xticklabels([f"{si:g}" for si in s])
    ax.set_xlim(0.07, 7.0)
    ax.set_xlabel(r"kernel decay scale $s$  (unit-square topology)",
                  fontsize=10.5, color=INK)
    ax.set_ylabel("withdrawals prevented per year (PJM-scale)\n"
                  "W=12 vs. unbounded reallocation",
                  fontsize=10.5, color=INK)
    y_lo = min(0, (y - yerr).min()) - 4
    y_hi = (y + yerr).max() + 8
    ax.set_ylim(y_lo, y_hi)

    legend_handles = [
        Line2D([0], [0], color=CORAL, lw=1.4, marker="o", markersize=9,
               markerfacecolor=CREAM, markeredgecolor=CORAL,
               markeredgewidth=1.7,
               label="point estimate, 95% CI (n = 30 seeds)"),
        Line2D([0], [0], color=BROWN, lw=0.8, linestyle=(0, (3, 2)),
               label="default-s headline (+30/yr)"),
    ]
    leg = ax.legend(handles=legend_handles, loc="lower left",
                    frameon=True, framealpha=0.95,
                    edgecolor=GRID, fontsize=9, handlelength=2.4,
                    borderpad=0.6)
    leg.get_frame().set_facecolor(CREAM)

    ax.set_title("Robustness to s (network kernel decay)",
                 fontsize=11.5, color=INK, loc="left", pad=10, weight="bold")

    fig.tight_layout()
    fig.savefig(OUT_PDF, facecolor=CREAM, bbox_inches="tight")
    fig.savefig(OUT_PNG, facecolor=CREAM, dpi=220, bbox_inches="tight")
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
