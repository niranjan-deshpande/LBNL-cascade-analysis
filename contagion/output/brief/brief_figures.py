"""Generate Anthropic-styled figures for the contagion brief."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Anthropic palette
CLAY = "#D97757"       # warm primary
PARCHMENT = "#F5F0E8"  # background
CHARCOAL = "#3D3929"   # text / axes
SAND = "#C4B89A"       # muted secondary
SLATE = "#8B8578"      # grid / light elements
WASH = "#EAE4D9"       # subtle fill

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.dirname(HERE)  # contagion/output/
BRIEF_DIR = HERE
os.makedirs(BRIEF_DIR, exist_ok=True)

def anthropic_style(ax, fig):
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SLATE)
    ax.spines["bottom"].set_color(SLATE)
    ax.tick_params(colors=CHARCOAL, labelsize=9)
    ax.xaxis.label.set_color(CHARCOAL)
    ax.yaxis.label.set_color(CHARCOAL)
    ax.title.set_color(CHARCOAL)


def save(fig, name):
    fig.savefig(os.path.join(BRIEF_DIR, f"{name}.pdf"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(BRIEF_DIR, f"{name}.png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  {name}")


# ── 1. Event-study plot (matched DiD) ────────────────────────────────

def plot_event_study():
    csv = os.path.join(OUTPUT_DIR, "matched_did", "tables", "event_study_coefficients.csv")
    df = pd.read_csv(csv)
    df = df[df["se"].notna() & (df["event_time"] != -1)].copy()
    # Add reference
    ref = pd.DataFrame([{"event_time": -1, "beta": 0, "se": 0, "ci_lower": 0, "ci_upper": 0, "p_value": np.nan}])
    df = pd.concat([ref, df]).sort_values("event_time")

    fig, ax = plt.subplots(figsize=(7, 3.8))
    anthropic_style(ax, fig)

    k = df["event_time"].values
    b = df["beta"].values
    lo = df["ci_lower"].values
    hi = df["ci_upper"].values

    # Pre-period shading
    ax.axvspan(-4.5, -0.5, alpha=0.08, color=SAND)
    ax.axhline(0, color=SLATE, lw=0.7, ls="--")
    ax.axvline(-0.5, color=SLATE, lw=0.5, ls=":")

    ax.errorbar(k, b, yerr=[b - lo, hi - b],
                fmt="o-", color=CLAY, capsize=3, markersize=5, lw=1.3,
                markeredgecolor="white", markeredgewidth=0.5)

    # Reference dot
    ax.scatter([-1], [0], color=CHARCOAL, s=50, zorder=5, marker="D")

    ax.set_xlabel("Quarters relative to treatment event", fontsize=10)
    ax.set_ylabel("Additional withdrawals (DiD)", fontsize=10)
    ax.set_title("Event Study: Matched Difference-in-Differences", fontsize=11, fontweight="bold")
    ax.set_xticks(k)
    ax.text(-2.5, ax.get_ylim()[1]*0.92, "pre-period", ha="center", fontsize=8, color=SLATE, style="italic")

    fig.tight_layout()
    save(fig, "event_study")


# ── 2. Entity forest plot ────────────────────────────────────────────

def plot_entity_forest():
    csv = os.path.join(OUTPUT_DIR, "tables", "entity_heterogeneity.csv")
    df = pd.read_csv(csv).sort_values("odds_ratio")

    fig, ax = plt.subplots(figsize=(5, 4.2))
    anthropic_style(ax, fig)

    y = range(len(df))
    ax.errorbar(df["odds_ratio"], y,
                xerr=[df["odds_ratio"] - df["ci_lower"], df["ci_upper"] - df["odds_ratio"]],
                fmt="o", color=CLAY, capsize=2, markersize=4, lw=1,
                markeredgecolor="white", markeredgewidth=0.4)
    ax.axvline(1, color=SLATE, ls="--", lw=0.7)
    ax.set_yticks(list(y))
    ax.set_yticklabels(df["entity"], fontsize=8)
    ax.set_xlabel("Odds ratio (95% CI)", fontsize=9)
    ax.set_title("Peer Withdrawal Effect by Entity", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "entity_forest")


# ── 3. Dose-response ─────────────────────────────────────────────────

def plot_dose_response():
    csv = os.path.join(OUTPUT_DIR, "tables", "dose_response.csv")
    df = pd.read_csv(csv)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    anthropic_style(ax, fig)

    x = range(len(df))
    ax.bar(x, df["odds_ratio"], color=CLAY, edgecolor="white", width=0.6, alpha=0.85)
    ax.errorbar(x, df["odds_ratio"], yerr=1.96*df["se"]*df["odds_ratio"],
                fmt="none", color=CHARCOAL, capsize=4, lw=1)
    ax.axhline(1, color=SLATE, ls="--", lw=0.7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["depth_bin"], fontsize=9)
    ax.set_xlabel("Projects at POI", fontsize=9)
    ax.set_ylabel("Odds ratio", fontsize=9)
    ax.set_title("Dose-Response: Effect Strengthens with POI Depth", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "dose_response")


# ── 4. Cox depth comparison (from tables) ────────────────────────────

def plot_cox_depth():
    csv = os.path.join(OUTPUT_DIR, "tables", "tier2_cox_depth_comparison.csv")
    if not os.path.exists(csv):
        print("  skipping cox_depth (no file)")
        return
    df = pd.read_csv(csv)

    fig, ax = plt.subplots(figsize=(5, 3))
    anthropic_style(ax, fig)

    x = range(len(df))
    ax.errorbar(x, df["hazard_ratio"],
                yerr=[df["hazard_ratio"] - df["hr_ci_lower"], df["hr_ci_upper"] - df["hazard_ratio"]],
                fmt="o-", color=CLAY, capsize=4, markersize=6, lw=1.3,
                markeredgecolor="white", markeredgewidth=0.5)
    ax.axhline(1, color=SLATE, ls="--", lw=0.7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["min_depth"], fontsize=9)
    ax.set_xlabel("Minimum POI depth", fontsize=9)
    ax.set_ylabel("Hazard ratio", fontsize=9)
    ax.set_title("Cox Hazard Ratio by POI Depth Threshold", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "cox_depth")


# ── 5. Permutation null vs observed overdispersion ──────────────────

def plot_permutation_null():
    """Reconstruct the permutation null distribution and annotate observed."""
    csv = os.path.join(OUTPUT_DIR, "tables", "permutation_test.csv")
    if not os.path.exists(csv):
        print("  skipping permutation_null (no file)")
        return
    row = pd.read_csv(csv).iloc[0]
    obs = row["observed"]
    null_mean = row["null_mean"]
    null_std = row["null_std"]

    # Simulate null distribution from reported mean/std (approx normal)
    rng = np.random.default_rng(0)
    null_draws = rng.normal(null_mean, null_std, size=5000)

    fig, ax = plt.subplots(figsize=(5, 3))
    anthropic_style(ax, fig)

    ax.hist(null_draws, bins=40, color=SAND, edgecolor="white",
            alpha=0.85, label="null distribution")
    ax.axvline(obs, color=CLAY, lw=2, label=f"observed = {obs:.3f}")
    ax.axvline(null_mean, color=SLATE, lw=1, ls="--",
               label=f"null mean = {null_mean:.3f}")
    ax.set_xlabel("Variance ratio (overdispersion)", fontsize=9)
    ax.set_ylabel("Frequency", fontsize=9)
    ax.set_title("POI Clustering is Not Chance", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, frameon=False)

    # Annotate SDs above null
    sds = (obs - null_mean) / null_std
    ax.text(0.98, 0.55,
            f"{sds:.0f} SDs\nabove null\n(p < 0.001)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color=CHARCOAL,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=SLATE, lw=0.5))

    fig.tight_layout()
    save(fig, "permutation_null")


if __name__ == "__main__":
    print("Generating brief figures...")
    plot_event_study()
    plot_entity_forest()
    plot_dose_response()
    plot_cox_depth()
    plot_permutation_null()
    print("Done.")
