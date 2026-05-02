"""Figure 6: Combined robustness panel — does the W=12 cluster-bound
prevention headline (and the OFF-reproduces-empirical claim) survive
variation in every parameter governing the network distribution of cost?

2x2 layout:
  (a) α (local share)        — 3 points, n=30 seeds
  (b) s (kernel decay)       — 6 points, n=30 seeds, log-x
  (c) k (network fanout)     — 7 points, n=30 seeds, log-x
  (d) ρ (AR(1) persistence)  — 6 points, n=15 seeds, OFF peak β with
                                empirical reference

Panels (a)-(c) share the y-axis convention (withdrawals prevented per year,
PJM-scale, W=12 vs unbounded) with a horizontal +30/yr default-headline
reference. Panel (d) is on the event-study β scale with the empirical
+0.029 peak as reference.

Reads:
  ABM/full_abm/output/alpha_sensitivity_summary.csv
  ABM/full_abm/output/kernel_sensitivity_summary.csv
  ABM/full_abm/output/fanout_sensitivity_summary.csv
  ABM/full_abm/output/rho_sweep_off_summary.csv

Writes:
  paper_figures/fig6_robustness_panel.{pdf,png}
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALPHA_CSV  = os.path.join(ROOT, "ABM/full_abm/output/alpha_sensitivity_summary.csv")
KSCALE_CSV = os.path.join(ROOT, "ABM/full_abm/output/kernel_sensitivity_summary.csv")
FANOUT_CSV = os.path.join(ROOT, "ABM/full_abm/output/fanout_sensitivity_summary.csv")
RHO_CSV    = os.path.join(ROOT, "ABM/full_abm/output/rho_sweep_off_summary.csv")
OUT_PDF    = os.path.join(ROOT, "paper_figures/fig6_robustness_panel.pdf")
OUT_PNG    = os.path.join(ROOT, "paper_figures/fig6_robustness_panel.png")

CREAM = "#FAF7F2"
INK   = "#1F1A17"
GRID  = "#E6DFD5"
CORAL = "#CC785C"
BROWN = "#8C6F5A"
BLACK = "#1F1A17"

PREV_HEADLINE = 29.85   # +30/yr W=12 default-α/default-s/default-k headline
EMP_PEAK_BETA = 0.0287  # empirical k=7 peak (event_study_coefficients.csv)


def _ci_bars(ax, x, y, se, *, log=False):
    yerr = 1.96 * np.asarray(se)
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor=CORAL,
                elinewidth=1.4, capsize=4, capthick=1.2, zorder=3)
    ax.plot(x, y, linestyle="none", marker="o", markersize=8,
            markerfacecolor=CREAM, markeredgecolor=CORAL,
            markeredgewidth=1.6, zorder=4)


def _label_points(ax, xs, ys, ses, *, fmt="+{:.1f}",
                  color=INK, fontsize=8.2,
                  pad_pixels=8):
    """Place value text above each upper CI cap with a small vertical
    pixel offset. Vertical placement avoids overlap with the CI bar
    (which is vertical) and avoids text-vs-text collisions for close x
    points (which the right-of-point strategy did not solve for log x)."""
    xs = np.asarray(xs); ys = np.asarray(ys); ses = np.asarray(ses)
    upper = ys + 1.96 * ses
    for x, y, ucap in zip(xs, ys, upper):
        x_disp, y_disp = ax.transData.transform((x, ucap))
        x_disp_off, y_disp_off = x_disp, y_disp + pad_pixels
        x_off, y_off = ax.transData.inverted().transform(
            (x_disp_off, y_disp_off))
        ax.text(x_off, y_off, fmt.format(y),
                fontsize=fontsize, color=color,
                ha="center", va="bottom", zorder=6,
                bbox=dict(facecolor=CREAM, edgecolor="none",
                          boxstyle="round,pad=0.12", alpha=0.85))


def panel_alpha(ax):
    df = pd.read_csv(ALPHA_CSV).sort_values("alpha")
    x = df["alpha"].to_numpy()
    y = df["prev_per_year_pjm"].to_numpy()
    se = df["prev_per_year_pjm_se"].to_numpy()

    ax.axhline(0.0, color=INK, linewidth=0.7, alpha=0.55, zorder=1)
    ax.axhline(PREV_HEADLINE, color=BROWN, linewidth=0.8, alpha=0.6,
               linestyle=(0, (3, 2)), zorder=1)

    _ci_bars(ax, x, y, se)
    _label_points(ax, x, y, se)

    ax.set_xticks([0.05, 0.15, 0.30])
    ax.set_xlim(0.005, 0.36)
    ax.set_xlabel(r"local share $\alpha_\text{local}$",
                  fontsize=10, color=INK)
    return y, se


def panel_s(ax):
    df = pd.read_csv(KSCALE_CSV).sort_values("scale")
    x = df["scale"].to_numpy()
    y = df["prev_per_year_pjm"].to_numpy()
    se = df["prev_per_year_pjm_se"].to_numpy()

    ax.axhline(0.0, color=INK, linewidth=0.7, alpha=0.55, zorder=1)
    ax.axhline(PREV_HEADLINE, color=BROWN, linewidth=0.8, alpha=0.6,
               linestyle=(0, (3, 2)), zorder=1)

    _ci_bars(ax, x, y, se)
    _label_points(ax, x, y, se)

    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{si:g}" for si in x])
    ax.set_xlim(0.07, 8.5)
    ax.set_xlabel(r"kernel decay $s$  (unit-square)",
                  fontsize=10, color=INK)
    return y, se


def panel_k(ax):
    df = pd.read_csv(FANOUT_CSV).sort_values("k")
    x = df["k"].to_numpy()
    y = df["prev_per_year_pjm"].to_numpy()
    se = df["prev_per_year_pjm_se"].to_numpy()

    ax.axhline(0.0, color=INK, linewidth=0.7, alpha=0.55, zorder=1)
    ax.axhline(PREV_HEADLINE, color=BROWN, linewidth=0.8, alpha=0.6,
               linestyle=(0, (3, 2)), zorder=1)

    _ci_bars(ax, x, y, se)
    _label_points(ax, x, y, se)

    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(xi)) for xi in x])
    ax.set_xlim(1.5, 280)
    ax.set_xlabel("network fanout $k$  (recipient POIs per fire)",
                  fontsize=10, color=INK)
    return y, se


def panel_rho(ax):
    df = pd.read_csv(RHO_CSV).sort_values("rho")
    x = df["rho"].to_numpy()
    y = df["peak_beta_mean"].to_numpy()
    se = df["peak_beta_se"].to_numpy()

    ax.axhline(0.0, color=INK, linewidth=0.7, alpha=0.55, zorder=1)
    ax.axhline(EMP_PEAK_BETA, color=BLACK, linewidth=0.8, alpha=0.7,
               linestyle=(0, (3, 2)), zorder=1)

    _ci_bars(ax, x, y, se)
    _label_points(ax, x, y, se, fmt="+{:.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{xi:.2f}" for xi in x])
    ax.set_xlim(0.67, 0.98)
    ax.set_xlabel(r"AR(1) persistence $\rho_\text{POI}$",
                  fontsize=10, color=INK)
    return y, se


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
    fig.patch.set_facecolor(CREAM)
    for ax in axes.flat:
        ax.set_facecolor(CREAM)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

    ax_a, ax_s = axes[0]
    ax_k, ax_r = axes[1]

    y_a, se_a = panel_alpha(ax_a)
    y_s, se_s = panel_s(ax_s)
    y_k, se_k = panel_k(ax_k)
    y_r, se_r = panel_rho(ax_r)

    # --- Shared y-axis for prevention panels (a, b, c). Generous top
    # padding so the value labels above the CI caps never collide with
    # adjacent labels or with the legend / headline reference. ---
    all_y = np.concatenate([y_a, y_s, y_k])
    all_se = np.concatenate([se_a, se_s, se_k])
    y_lo = min(0.0, (all_y - 1.96 * all_se).min()) - 4
    y_hi = (all_y + 1.96 * all_se).max() + 13
    for ax in (ax_a, ax_s, ax_k):
        ax.set_ylim(y_lo, y_hi)

    # ρ panel y-axis: leave room above for empirical reference + value labels.
    y_lo_r = min(0.0, (y_r - 1.96 * se_r).min()) - 0.010
    y_hi_r = max(EMP_PEAK_BETA, (y_r + 1.96 * se_r).max()) + 0.022
    ax_r.set_ylim(y_lo_r, y_hi_r)

    # --- y-axis labels (just on left column) ---
    ax_a.set_ylabel("withdrawals prevented per year (PJM-scale)\n"
                    "W=12 vs. unbounded reallocation",
                    fontsize=10, color=INK)
    ax_k.set_ylabel("withdrawals prevented per year (PJM-scale)\n"
                    "W=12 vs. unbounded reallocation",
                    fontsize=10, color=INK)
    ax_r.set_ylabel(r"OFF event-study peak $\beta_{4..8}$"
                    "\n"
                    "(matched-DiD, ABM-OFF regime)",
                    fontsize=10, color=INK)

    # --- subplot titles (sub-letter style) ---
    titles = [
        (ax_a, "(a)  local share α  (n=30)"),
        (ax_s, "(b)  kernel decay s  (n=30)"),
        (ax_k, "(c)  network fanout k  (n=30)"),
        (ax_r, "(d)  AR(1) persistence ρ  (n=15, OFF regime)"),
    ]
    for ax, t in titles:
        ax.set_title(t, fontsize=10.5, color=INK, loc="left",
                     pad=6, weight="bold")

    # Reference-line labels (low-left corner, inside axis but below the data).
    for ax in (ax_a, ax_s, ax_k):
        ax.text(0.018, 0.06, "+30/yr default headline",
                transform=ax.transAxes, fontsize=8, color=BROWN,
                style="italic", ha="left", va="bottom",
                bbox=dict(facecolor=CREAM, edgecolor="none",
                          boxstyle="round,pad=0.18", alpha=0.85))
    ax_r.text(0.018, 0.06,
              rf"empirical $\beta_\mathrm{{peak}}=+{EMP_PEAK_BETA:.3f}$",
              transform=ax_r.transAxes, fontsize=8, color=BLACK,
              style="italic", ha="left", va="bottom",
              bbox=dict(facecolor=CREAM, edgecolor="none",
                        boxstyle="round,pad=0.18", alpha=0.85))

    # --- Single shared legend at the top ---
    legend_handles = [
        Line2D([0], [0], color=CORAL, lw=1.4, marker="o", markersize=8,
               markerfacecolor=CREAM, markeredgecolor=CORAL,
               markeredgewidth=1.6,
               label="point estimate, 95% CI"),
        Line2D([0], [0], color=BROWN, lw=0.8, linestyle=(0, (3, 2)),
               label="+30/yr headline (panels a–c)"),
        Line2D([0], [0], color=BLACK, lw=0.8, linestyle=(0, (3, 2)),
               label="empirical β_peak = +0.029 (panel d)"),
        Line2D([0], [0], color=INK, lw=0.8, alpha=0.55,
               label="zero reference"),
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.005), ncol=4,
               frameon=True, framealpha=0.95,
               edgecolor=GRID, fontsize=9, handlelength=2.4,
               borderpad=0.5).get_frame().set_facecolor(CREAM)

    fig.suptitle(
        "Cluster-bound headline and OFF event-study shape are robust "
        "across network-distribution and persistence parameters",
        x=0.013, y=1.06, ha="left", fontsize=11.5, color=INK, weight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.subplots_adjust(hspace=0.40, wspace=0.28)
    fig.savefig(OUT_PDF, facecolor=CREAM, bbox_inches="tight")
    fig.savefig(OUT_PNG, facecolor=CREAM, dpi=220, bbox_inches="tight")
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
