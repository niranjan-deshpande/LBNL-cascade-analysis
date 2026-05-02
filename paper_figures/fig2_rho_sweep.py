"""Figure 2: ρ-sweep small-multiples — OFF matched-DiD event-study at six AR(1)
persistence levels, with the empirical curve overlaid on each panel.

Visual proof that the OFF-reproduces-empirical-shape result is invariant to
the calibration of `rho_poi`. Six panels (2×3), shared x and y axes.

Reads:
  contagion/output/matched_did/tables/event_study_coefficients.csv
  ABM/full_abm/output/rho_sweep_off_seeds.csv

Writes:
  paper_figures/fig2_rho_sweep.{pdf,png}
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
RHO_CSV = os.path.join(ROOT, "ABM/full_abm/output/rho_sweep_off_seeds.csv")
OUT_PDF = os.path.join(ROOT, "paper_figures/fig2_rho_sweep.pdf")
OUT_PNG = os.path.join(ROOT, "paper_figures/fig2_rho_sweep.png")

# Palette — matches fig1.
CREAM   = "#FAF7F2"
INK     = "#1F1A17"
GRID    = "#E6DFD5"
BROWN   = "#8C6F5A"   # ABM-OFF
BLACK   = "#1F1A17"
RIB_OFF = "#8C6F5A"
RIB_EMP = "#9C9389"

RHOS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
KS_PLOT = list(range(-4, 9))           # -4..8 inclusive
DROP_K = -1                            # drop mechanical zero from plotting


def load_emp() -> pd.DataFrame:
    emp = pd.read_csv(EMP_CSV).sort_values("event_time").reset_index(drop=True)
    return emp[emp["event_time"] != DROP_K].copy()


def load_rho() -> pd.DataFrame:
    df = pd.read_csv(RHO_CSV)
    return df


def compute_rho_panel(df: pd.DataFrame, rho: float) -> dict:
    sub = df[df["rho"] == rho].copy()
    n = len(sub)
    means, ses, lo, hi = [], [], [], []
    for k in KS_PLOT:
        col = f"k{k}"
        x = sub[col].to_numpy()
        m = np.nanmean(x)
        sd = np.nanstd(x, ddof=1) if n > 1 else 0.0
        se = sd / np.sqrt(n) if n > 1 else 0.0
        means.append(m)
        ses.append(se)
        lo.append(m - 1.96 * se)
        hi.append(m + 1.96 * se)
    pre_trend_fail = int((sub["pre_trend_p"] < 0.05).sum())
    return {
        "rho": rho, "n": n,
        "k": np.array(KS_PLOT),
        "mean": np.array(means),
        "se": np.array(ses),
        "lo": np.array(lo),
        "hi": np.array(hi),
        "pre_trend_fail": pre_trend_fail,
        "pre_trend_p_mean": float(sub["pre_trend_p"].mean()),
    }


def main():
    emp = load_emp()
    rho_df = load_rho()
    panels = [compute_rho_panel(rho_df, r) for r in RHOS]

    emp_k    = emp["event_time"].to_numpy()
    emp_b    = emp["beta"].to_numpy()
    emp_lo   = emp["ci_lower"].to_numpy()
    emp_hi   = emp["ci_upper"].to_numpy()

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

    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.0),
                             sharex=True, sharey=True)
    fig.patch.set_facecolor(CREAM)

    for ax, p in zip(axes.flat, panels):
        ax.set_facecolor(CREAM)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.axhline(0.0, color=INK, linewidth=0.7, alpha=0.55, zorder=1)
        ax.axvline(-0.5, color=INK, linewidth=0.7, alpha=0.45,
                   linestyle=(0, (2, 2)), zorder=1)

        # Drop k=-1 mechanical zero from ABM plot too.
        mask = p["k"] != DROP_K
        kk = p["k"][mask]
        mm = p["mean"][mask]
        ll = p["lo"][mask]
        hh = p["hi"][mask]

        # ribbons
        ax.fill_between(kk, ll, hh, color=RIB_OFF, alpha=0.22,
                        linewidth=0, zorder=2)
        ax.fill_between(emp_k, emp_lo, emp_hi, color=RIB_EMP, alpha=0.30,
                        linewidth=0, zorder=3)

        # OFF
        ax.plot(kk, mm, color=BROWN, linewidth=1.5,
                linestyle=(0, (5, 2)), marker="o", markersize=4.2,
                markerfacecolor=CREAM, markeredgecolor=BROWN,
                markeredgewidth=1.1, zorder=4)
        # Empirical
        ax.plot(emp_k, emp_b, color=BLACK, linewidth=1.6,
                linestyle="-", marker="o", markersize=4.2,
                markerfacecolor=BLACK, markeredgecolor=BLACK, zorder=5)

        # title with rho
        ax.set_title(rf"$\rho_{{POI}} = {p['rho']:.2f}$",
                     fontsize=10.5, color=INK, loc="left",
                     pad=4, weight="bold")

        # Per-panel inline summary stats: k1 mean, peak, n seeds
        k1_mean = float(p["mean"][np.where(p["k"] == 1)[0][0]])
        post_mask = (p["k"] >= 4) & (p["k"] <= 8)
        peak_idx = np.argmax(p["mean"][post_mask])
        peak_k = int(p["k"][post_mask][peak_idx])
        peak_b = float(p["mean"][post_mask][peak_idx])
        stat_txt = (rf"$\beta_{{1}}={k1_mean:+.3f}$" "\n"
                    rf"$\beta_{{{peak_k}}}={peak_b:+.3f}$" "\n"
                    rf"$n={p['n']}$")
        ax.text(0.985, 0.030, stat_txt, transform=ax.transAxes,
                fontsize=8.2, color=BROWN, family="DejaVu Sans",
                ha="right", va="bottom",
                bbox=dict(facecolor=CREAM, edgecolor=GRID, linewidth=0.6,
                          boxstyle="round,pad=0.3"))

        # Validity-edge annotation on rho=0.95 panel
        if abs(p["rho"] - 0.95) < 1e-9 and p["pre_trend_fail"] > 0:
            ax.text(0.985, 0.965,
                    f"pre-trend fail: {p['pre_trend_fail']}/{p['n']} seeds",
                    transform=ax.transAxes,
                    fontsize=8.2, color=BROWN, ha="right", va="top",
                    bbox=dict(facecolor=CREAM, edgecolor=BROWN,
                              linewidth=0.7, boxstyle="round,pad=0.3"))

    # Axes labels (shared) — only bottom-row x-labels and left-column y-labels
    for ax in axes[-1, :]:
        ax.set_xlabel("event time $k$ (quarters)",
                      fontsize=10, color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"matched-DiD $\beta_k$",
                      fontsize=10, color=INK)

    # ticks & limits
    axes[0, 0].set_xticks(np.arange(-4, 9, 2))
    axes[0, 0].set_xlim(-4.5, 8.5)
    axes[0, 0].set_ylim(-0.20, 0.10)
    axes[0, 0].set_yticks(np.arange(-0.20, 0.11, 0.05))

    # Legend on figure level
    legend_handles = [
        Line2D([0], [0], color=BLACK, lw=1.6, marker="o", markersize=5,
               markerfacecolor=BLACK, markeredgecolor=BLACK,
               label="Empirical (PJM)"),
        Line2D([0], [0], color=BROWN, lw=1.5, marker="o", markersize=4.5,
               markerfacecolor=CREAM, markeredgecolor=BROWN,
               markeredgewidth=1.1, linestyle=(0, (5, 2)),
               label="ABM-OFF (15 seeds)"),
        Patch(facecolor=RIB_OFF, alpha=0.22, label="ABM 95% CI"),
        Patch(facecolor=RIB_EMP, alpha=0.30, label="Empirical 95% CI"),
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.005), ncol=4,
               frameon=True, framealpha=0.95,
               edgecolor=GRID, fontsize=9, handlelength=2.4,
               borderpad=0.5).get_frame().set_facecolor(CREAM)

    fig.suptitle(
        "OFF event-study shape is invariant to AR(1) persistence",
        x=0.013, y=1.07, ha="left", fontsize=11.5, color=INK, weight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_PDF, facecolor=CREAM, bbox_inches="tight")
    fig.savefig(OUT_PNG, facecolor=CREAM, dpi=220, bbox_inches="tight")
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
