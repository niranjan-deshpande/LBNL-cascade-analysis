"""Figure 1: Empirical vs. ABM-ON vs. ABM-OFF matched-DiD event-study coefficients.

Reads:
  contagion/output/matched_did/tables/event_study_coefficients.csv  (empirical)
  contagion/output/matched_did/tables/pre_trend_test.csv            (empirical pre-trend F)
  ABM/full_abm/output/cascade_decomposition.csv                     (ABM ON/OFF, 30 seeds)

Writes:
  paper_figures/fig1_event_study.pdf
  paper_figures/fig1_event_study.png
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMP_CSV = os.path.join(ROOT, "contagion/output/matched_did/tables/event_study_coefficients.csv")
PRE_CSV = os.path.join(ROOT, "contagion/output/matched_did/tables/pre_trend_test.csv")
ABM_CSV = os.path.join(ROOT, "ABM/full_abm/output/cascade_decomposition.csv")
OUT_PDF = os.path.join(ROOT, "paper_figures/fig1_event_study.pdf")
OUT_PNG = os.path.join(ROOT, "paper_figures/fig1_event_study.png")

N_PAIRS = 976
N_SEEDS = 30

# Anthropic-ish palette: cream background, warm coral, muted brown, near-black ink.
CREAM   = "#FAF7F2"
INK     = "#1F1A17"
GRID    = "#E6DFD5"
CORAL   = "#CC785C"   # ABM-ON
BROWN   = "#8C6F5A"   # ABM-OFF
BLACK   = "#1F1A17"   # Empirical
RIB_EMP = "#9C9389"
RIB_ON  = "#CC785C"
RIB_OFF = "#8C6F5A"


def load_data():
    emp = pd.read_csv(EMP_CSV)
    abm = pd.read_csv(ABM_CSV)
    pre = pd.read_csv(PRE_CSV)
    return emp, abm, pre


def pre_trend_stat(beta, se, ks):
    """Wald-style joint-zero stat on pre-period (k in {-4,-3}); k=-2,-1 are mechanically zero."""
    mask = np.isin(ks, [-4, -3]) & np.isfinite(se) & (se > 0)
    if mask.sum() == 0:
        return np.nan, np.nan
    z2 = (beta[mask] / se[mask]) ** 2
    chi2 = float(z2.sum())
    df = int(mask.sum())
    # Survival function via scipy if available; else simple normal approx
    try:
        from scipy.stats import chi2 as chi2dist
        p = float(chi2dist.sf(chi2, df))
    except Exception:
        p = np.nan
    return chi2 / df, p  # report as F-style ratio + p


def build_panel(emp, abm):
    # Empirical: drop k=-1 (mechanical zero) for plotting; keep k=-2 even if tiny.
    emp = emp.sort_values("event_time").reset_index(drop=True)
    abm = abm.sort_values("k").reset_index(drop=True)

    # ABM 95% CIs from across-seed SE (honest for the methodological claim).
    abm = abm.assign(
        OFF_lo=abm["OFF_mean"] - 1.96 * abm["OFF_se"],
        OFF_hi=abm["OFF_mean"] + 1.96 * abm["OFF_se"],
        ON_lo =abm["ON_mean"]  - 1.96 * abm["ON_se"],
        ON_hi =abm["ON_mean"]  + 1.96 * abm["ON_se"],
    )
    return emp, abm


def main():
    emp, abm, pre = load_data()
    emp, abm = build_panel(emp, abm)

    ks = emp["event_time"].to_numpy()

    # plotting mask: drop mechanical k=-1; keep k=-2 (essentially zero by construction).
    plot_mask = ks != -1
    ks_p   = ks[plot_mask]
    emp_b  = emp["beta"].to_numpy()[plot_mask]
    emp_lo = emp["ci_lower"].to_numpy()[plot_mask]
    emp_hi = emp["ci_upper"].to_numpy()[plot_mask]

    abm_ks = abm["k"].to_numpy()
    abm_mask = abm_ks != -1
    abm_ks_p = abm_ks[abm_mask]
    on_b   = abm["ON_mean"].to_numpy()[abm_mask]
    on_lo  = abm["ON_lo"].to_numpy()[abm_mask]
    on_hi  = abm["ON_hi"].to_numpy()[abm_mask]
    off_b  = abm["OFF_mean"].to_numpy()[abm_mask]
    off_lo = abm["OFF_lo"].to_numpy()[abm_mask]
    off_hi = abm["OFF_hi"].to_numpy()[abm_mask]

    # ---- pre-trend stats ----
    emp_F = float(pre["f_stat"].iloc[0]); emp_p = float(pre["p_value"].iloc[0])
    on_F,  on_p  = pre_trend_stat(abm["ON_mean"].to_numpy(),
                                   abm["ON_se"].to_numpy(), abm_ks)
    off_F, off_p = pre_trend_stat(abm["OFF_mean"].to_numpy(),
                                   abm["OFF_se"].to_numpy(), abm_ks)

    # ---- figure ----
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

    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    fig.patch.set_facecolor(CREAM)
    ax.set_facecolor(CREAM)

    # gridlines
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # reference lines
    ax.axhline(0.0, color=INK, linewidth=0.8, alpha=0.55, zorder=1)
    ax.axvline(-0.5, color=INK, linewidth=0.8, alpha=0.45,
               linestyle=(0, (2, 2)), zorder=1)

    # ---- ribbons ----
    ax.fill_between(abm_ks_p, off_lo, off_hi, color=RIB_OFF, alpha=0.18,
                    linewidth=0, zorder=2)
    ax.fill_between(abm_ks_p, on_lo,  on_hi,  color=RIB_ON,  alpha=0.20,
                    linewidth=0, zorder=2)
    ax.fill_between(ks_p, emp_lo, emp_hi, color=RIB_EMP, alpha=0.30,
                    linewidth=0, zorder=3)

    # ---- lines ----
    ax.plot(abm_ks_p, off_b, color=BROWN, linewidth=1.6,
            linestyle=(0, (5, 2)), marker="o", markersize=5,
            markerfacecolor=CREAM, markeredgecolor=BROWN, markeredgewidth=1.2,
            zorder=4, label="ABM-OFF")
    ax.plot(abm_ks_p, on_b, color=CORAL, linewidth=1.8,
            linestyle="-", marker="s", markersize=5,
            markerfacecolor=CREAM, markeredgecolor=CORAL, markeredgewidth=1.2,
            zorder=5, label="ABM-ON")
    ax.plot(ks_p, emp_b, color=BLACK, linewidth=1.8,
            linestyle="-", marker="o", markersize=5.5,
            markerfacecolor=BLACK, markeredgecolor=BLACK,
            zorder=6, label="Empirical")

    # ---- annotations: empirical dip & peak ----
    # k=1 dip
    ax.annotate(r"empirical $\beta_{1}=-0.038$",
                xy=(1, emp.loc[emp.event_time == 1, "beta"].iloc[0]),
                xytext=(1.7, -0.075),
                fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.6, alpha=0.7))
    # k=7 peak
    ax.annotate(r"empirical $\beta_{7}=+0.029$",
                xy=(7, emp.loc[emp.event_time == 7, "beta"].iloc[0]),
                xytext=(4.4, 0.062),
                fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.6, alpha=0.7))
    # ABM-OFF dip (magnitude gap)
    off_k1 = float(abm.loc[abm.k == 1, "OFF_mean"].iloc[0])
    ax.annotate(rf"ABM-OFF $\beta_{{1}}={off_k1:+.3f}$",
                xy=(1, off_k1),
                xytext=(2.2, -0.135),
                fontsize=9, color=BROWN,
                arrowprops=dict(arrowstyle="-", color=BROWN, lw=0.6, alpha=0.7))
    # ABM-OFF peak
    off_peak_k = int(abm.loc[abm["OFF_mean"].idxmax(), "k"])
    off_peak_b = float(abm["OFF_mean"].max())
    ax.annotate(rf"ABM-OFF peak $\beta_{{{off_peak_k}}}={off_peak_b:+.3f}$",
                xy=(off_peak_k, off_peak_b),
                xytext=(off_peak_k - 4.5, 0.085),
                fontsize=9, color=BROWN,
                arrowprops=dict(arrowstyle="-", color=BROWN, lw=0.6, alpha=0.7))

    # ---- pre-trend stats box ----
    txt = (
        "Pre-trend joint test ($k\\in\\{-4,-3\\}$)\n"
        f"  Empirical:  F = {emp_F:.2f},  p = {emp_p:.2f}\n"
        f"  ABM-ON:     F = {on_F:.2f},  p = {on_p:.2f}\n"
        f"  ABM-OFF:    F = {off_F:.2f},  p = {off_p:.2f}"
    )
    ax.text(0.012, 0.025, txt, transform=ax.transAxes,
            fontsize=8.5, color=INK, family="monospace",
            verticalalignment="bottom",
            bbox=dict(facecolor=CREAM, edgecolor=GRID, linewidth=0.7,
                      boxstyle="round,pad=0.4"))

    # ---- axes ----
    ax.set_xlabel("event time $k$ (quarters relative to triggering withdrawal)",
                  fontsize=10.5, color=INK)
    ax.set_ylabel(r"matched-DiD coefficient  $\beta_k$"
                  "\n"
                  "(additional withdrawals per POI-quarter)",
                  fontsize=10.5, color=INK)
    ax.set_xticks(np.arange(-4, 9))
    ax.set_xlim(-4.5, 8.5)
    ax.set_ylim(-0.20, 0.10)
    ax.set_yticks(np.arange(-0.20, 0.11, 0.05))

    # custom legend (color + style + marker)
    legend_handles = [
        Line2D([0], [0], color=BLACK, lw=1.8, marker="o", markersize=6,
               markerfacecolor=BLACK, markeredgecolor=BLACK,
               label=f"Empirical (PJM, n = {N_PAIRS} pairs)"),
        Line2D([0], [0], color=CORAL, lw=1.8, marker="s", markersize=5,
               markerfacecolor=CREAM, markeredgecolor=CORAL, markeredgewidth=1.2,
               label=f"ABM-ON  (calibrated, {N_SEEDS} seeds)"),
        Line2D([0], [0], color=BROWN, lw=1.6, marker="o", markersize=5,
               markerfacecolor=CREAM, markeredgecolor=BROWN, markeredgewidth=1.2,
               linestyle=(0, (5, 2)),
               label=f"ABM-OFF (zero-cascade, {N_SEEDS} seeds)"),
        Patch(facecolor=RIB_EMP, alpha=0.30, label="95% CI"),
    ]
    leg = ax.legend(handles=legend_handles, loc="upper left",
                    frameon=True, framealpha=0.95,
                    edgecolor=GRID, fontsize=9, handlelength=2.6,
                    borderpad=0.6)
    leg.get_frame().set_facecolor(CREAM)

    ax.set_title("Matched-DiD event-study: empirical PJM panel vs. ABM with reallocation on/off",
                 fontsize=11.5, color=INK, loc="left", pad=10, weight="bold")

    fig.tight_layout()
    fig.savefig(OUT_PDF, facecolor=CREAM, bbox_inches="tight")
    fig.savefig(OUT_PNG, facecolor=CREAM, dpi=220, bbox_inches="tight")
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
