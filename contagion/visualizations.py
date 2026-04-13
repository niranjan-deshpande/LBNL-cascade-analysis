"""Publication-quality visualizations for contagion analysis."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from config import FIGURES_DIR
import os


# Global style settings
def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

set_style()


def save_fig(fig, name):
    fig.savefig(os.path.join(FIGURES_DIR, f"{name}.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, f"{name}.png"))
    plt.close(fig)
    print(f"  Saved {name}.pdf/.png")


def plot_poi_wd_distribution(poi):
    """Histogram of POI withdrawal rates with binomial expectation overlay."""
    fig, ax = plt.subplots(figsize=(8, 5))

    rates = poi[poi["n_projects"] >= 2]["wd_rate"].dropna()
    ax.hist(rates, bins=30, density=True, alpha=0.7, color="#4878CF", edgecolor="white", label="Observed")

    # Binomial expectation
    p_hat = poi["n_withdrawn"].sum() / poi["n_projects"].sum()
    x = np.linspace(0, 1, 100)
    mean_n = poi[poi["n_projects"] >= 2]["n_projects"].mean()
    # Approximate binomial rate distribution
    binom_std = np.sqrt(p_hat * (1 - p_hat) / mean_n)
    if binom_std > 0:
        binom_pdf = stats.norm.pdf(x, loc=p_hat, scale=binom_std)
        ax.plot(x, binom_pdf, "r--", lw=2, label=f"Binomial expectation (p={p_hat:.2f})")

    ax.set_xlabel("POI Withdrawal Rate")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Withdrawal Rates Across POIs")
    ax.legend()
    save_fig(fig, "01_poi_wd_distribution")


def plot_overdispersion_scatter(poi):
    """Observed vs expected withdrawal count by POI size."""
    fig, ax = plt.subplots(figsize=(8, 6))

    poi_f = poi[poi["n_projects"] >= 2].copy()
    p_hat = poi_f["n_withdrawn"].sum() / poi_f["n_projects"].sum()
    poi_f["expected_wd"] = poi_f["n_projects"] * p_hat

    # Jitter for visibility
    jitter = np.random.RandomState(42).normal(0, 0.15, len(poi_f))
    ax.scatter(poi_f["expected_wd"] + jitter, poi_f["n_withdrawn"] + jitter,
               alpha=0.3, s=15, c="#4878CF")

    # 45-degree line
    max_val = max(poi_f["expected_wd"].max(), poi_f["n_withdrawn"].max())
    ax.plot([0, max_val], [0, max_val], "r--", lw=1.5, label="No overdispersion")

    ax.set_xlabel("Expected Withdrawals (binomial)")
    ax.set_ylabel("Observed Withdrawals")
    ax.set_title("Overdispersion: Observed vs. Expected Withdrawals by POI")
    ax.legend()
    save_fig(fig, "02_overdispersion_scatter")


def plot_entity_summary(ent):
    """Bar chart of withdrawal rates by entity."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ent_sorted = ent.sort_values("wd_rate", ascending=True).tail(15)
    bars = ax.barh(ent_sorted["entity"], ent_sorted["wd_rate"], color="#4878CF", edgecolor="white")

    # Add sample size labels
    for bar, n in zip(bars, ent_sorted["n_projects"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"n={n}", va="center", fontsize=9)

    ax.set_xlabel("Withdrawal Rate")
    ax.set_title("Withdrawal Rate by Entity (Multi-Project POIs)")
    save_fig(fig, "03_entity_summary")


def plot_tier1_coefficients(summary_df):
    """Forest plot of Tier 1 logistic regression odds ratios."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Select key variables (exclude dummies with many levels)
    key_vars = [v for v in summary_df.index if not v.startswith("state_")]
    if len(key_vars) > 20:
        key_vars = key_vars[:20]
    sub = summary_df.loc[key_vars].copy()

    y_pos = range(len(sub))
    ax.errorbar(sub["odds_ratio"], y_pos,
                xerr=[sub["odds_ratio"] - sub["ci_lower"], sub["ci_upper"] - sub["odds_ratio"]],
                fmt="o", color="#4878CF", capsize=3, markersize=5)

    ax.axvline(1, color="gray", linestyle="--", lw=1)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(sub.index, fontsize=9)
    ax.set_xlabel("Odds Ratio (95% CI)")
    ax.set_title("Tier 1: Logistic Regression — Odds of Withdrawal")

    # Highlight peer_wd_rate
    if "peer_wd_rate" in sub.index:
        idx = list(sub.index).index("peer_wd_rate")
        ax.scatter([sub.loc["peer_wd_rate", "odds_ratio"]], [idx],
                   color="red", s=80, zorder=5)

    save_fig(fig, "04_tier1_coefficients")


def plot_tier2_hazard_ratios(cox_summary):
    """Forest plot of Cox model hazard ratios."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Exclude entity dummies for clarity
    key_vars = [v for v in cox_summary.index if not v.startswith("entity_")]
    sub = cox_summary.loc[key_vars].copy()

    y_pos = range(len(sub))
    ax.errorbar(sub["hazard_ratio"], y_pos,
                xerr=[sub["hazard_ratio"] - sub["hr_ci_lower"],
                      sub["hr_ci_upper"] - sub["hazard_ratio"]],
                fmt="o", color="#4878CF", capsize=3, markersize=5)

    ax.axvline(1, color="gray", linestyle="--", lw=1)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(sub.index, fontsize=9)
    ax.set_xlabel("Hazard Ratio (95% CI)")
    ax.set_title("Tier 2: Cox Model — Hazard of Withdrawal")

    if "cumulative_peer_wd" in sub.index:
        idx = list(sub.index).index("cumulative_peer_wd")
        ax.scatter([sub.loc["cumulative_peer_wd", "hazard_ratio"]], [idx],
                   color="red", s=80, zorder=5)

    save_fig(fig, "05_tier2_hazard_ratios")


def plot_tier2_survival_curves(t2_cp):
    """Survival curves stratified by peer withdrawal exposure."""
    from lifelines import KaplanMeierFitter

    fig, ax = plt.subplots(figsize=(8, 5))
    kmf = KaplanMeierFitter()

    # Use the last interval per subject to get final state
    last = t2_cp.sort_values("stop").groupby("q_id").last().reset_index()
    last["max_peer_wd"] = t2_cp.groupby("q_id")["cumulative_peer_wd"].max().values

    groups = {"0 peer WD": last["max_peer_wd"] == 0,
              "1 peer WD": last["max_peer_wd"] == 1,
              "2+ peer WD": last["max_peer_wd"] >= 2}

    colors = ["#4878CF", "#E8A838", "#D65F5F"]
    for (label, mask), color in zip(groups.items(), colors):
        subset = last[mask]
        if len(subset) < 10:
            continue
        duration = subset["stop"] - subset["start"]
        kmf.fit(duration, subset["event"], label=label)
        kmf.plot_survival_function(ax=ax, color=color, ci_show=True)

    ax.set_xlabel("Days in Queue")
    ax.set_ylabel("Survival Probability")
    ax.set_title("Survival by Peer Withdrawal Exposure")
    ax.legend()
    save_fig(fig, "06_tier2_survival_curves")


def plot_permutation_null(perm_results, null_stats):
    """Histogram of permutation null with observed statistic marked."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(null_stats, bins=40, density=True, alpha=0.7, color="#4878CF",
            edgecolor="white", label="Null distribution")
    ax.axvline(perm_results["observed"], color="red", lw=2, linestyle="--",
               label=f"Observed ({perm_results['observed']:.2f})")

    ax.set_xlabel("Overdispersion Statistic (Variance Ratio)")
    ax.set_ylabel("Density")
    ax.set_title(f"Permutation Test (p = {perm_results['p_value']:.4f})")
    ax.legend()
    save_fig(fig, "07_permutation_null")


def plot_entity_forest(eh):
    """Forest plot of contagion coefficient by entity."""
    fig, ax = plt.subplots(figsize=(8, 6))

    if len(eh) == 0:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
        save_fig(fig, "08_entity_forest")
        return

    eh_sorted = eh.sort_values("peer_wd_rate_coef")
    y_pos = range(len(eh_sorted))

    ax.errorbar(eh_sorted["odds_ratio"], y_pos,
                xerr=[eh_sorted["odds_ratio"] - eh_sorted["ci_lower"],
                      eh_sorted["ci_upper"] - eh_sorted["odds_ratio"]],
                fmt="o", color="#4878CF", capsize=3, markersize=6)

    ax.axvline(1, color="gray", linestyle="--", lw=1)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(eh_sorted["entity"], fontsize=10)
    ax.set_xlabel("Odds Ratio (95% CI)")
    ax.set_title("Contagion Effect by Entity")
    save_fig(fig, "08_entity_forest")


def plot_dose_response(dr):
    """Coefficient by POI depth bin."""
    fig, ax = plt.subplots(figsize=(7, 5))

    if len(dr) == 0:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
        save_fig(fig, "09_dose_response")
        return

    x = range(len(dr))
    ax.errorbar(x, dr["odds_ratio"],
                yerr=1.96 * dr["se"] * dr["odds_ratio"],  # delta method approx
                fmt="o-", color="#4878CF", capsize=5, markersize=8)

    ax.axhline(1, color="gray", linestyle="--", lw=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(dr["depth_bin"])
    ax.set_xlabel("POI Depth (Number of Projects)")
    ax.set_ylabel("Odds Ratio for Peer Withdrawal Rate")
    ax.set_title("Dose-Response: Contagion Effect by POI Depth")
    save_fig(fig, "09_dose_response")


def plot_temporal_asymmetry(temporal):
    """Forward vs backward coefficients."""
    fig, ax = plt.subplots(figsize=(6, 5))

    if not temporal:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
        save_fig(fig, "10_temporal_asymmetry")
        return

    labels = ["Forward\n(early→late)", "Backward\n(late→early)"]
    coefs = [temporal["forward_coef"], temporal["backward_coef"]]
    ses = [temporal["forward_se"], temporal["backward_se"]]
    colors = ["#4878CF", "#D65F5F"]

    bars = ax.bar(labels, coefs, yerr=[1.96 * s for s in ses], capsize=5,
                  color=colors, edgecolor="white", width=0.5)
    ax.axhline(0, color="gray", linestyle="--", lw=1)
    ax.set_ylabel("Coefficient")
    ax.set_title("Temporal Asymmetry Test")

    # Add p-values
    for i, (label, p) in enumerate(zip(labels, [temporal["forward_p"], temporal["backward_p"]])):
        ax.text(i, coefs[i] + 1.96 * ses[i] + 0.02, f"p={p:.3f}",
                ha="center", fontsize=10)

    save_fig(fig, "10_temporal_asymmetry")


def plot_shap_summary(shap_values, X):
    """SHAP beeswarm summary plot."""
    import shap as shap_lib

    fig, ax = plt.subplots(figsize=(10, 8))
    shap_lib.summary_plot(shap_values, X, max_display=20, show=False)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    save_fig(plt.gcf(), "11_shap_summary")


def plot_shap_dependence(shap_values, X, feature="peer_wd_count"):
    """SHAP dependence plot for a key feature."""
    import shap as shap_lib

    if feature not in X.columns:
        print(f"  Feature '{feature}' not in X, skipping SHAP dependence plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    feat_idx = list(X.columns).index(feature)
    shap_lib.dependence_plot(feat_idx, shap_values, X, show=False, ax=ax)
    ax.set_title(f"SHAP Dependence: {feature}")
    save_fig(fig, "12_shap_dependence")


def run_visualizations(poi, ent, t1_summary, cox_summary, t2_cp,
                        perm_results, null_stats, eh, dr, temporal,
                        shap_values, X_ml):
    """Generate all visualizations."""
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    plot_poi_wd_distribution(poi)
    plot_overdispersion_scatter(poi)
    plot_entity_summary(ent)
    plot_tier1_coefficients(t1_summary)
    plot_tier2_hazard_ratios(cox_summary)
    plot_tier2_survival_curves(t2_cp)
    plot_permutation_null(perm_results, null_stats)
    plot_entity_forest(eh)
    plot_dose_response(dr)
    plot_temporal_asymmetry(temporal)
    plot_shap_summary(shap_values, X_ml)
    plot_shap_dependence(shap_values, X_ml)

    print("\nAll visualizations saved.")
