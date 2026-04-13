"""Robustness checks: permutation, placebo, temporal asymmetry, dose-response, entity heterogeneity."""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from config import TABLES_DIR, RANDOM_SEED, N_PERMUTATIONS, DEPTH_BINS, DEPTH_LABELS
import os


def permutation_test(t1, n_perms=N_PERMUTATIONS):
    """Shuffle poi_name within entity×q_year_bin. Compare observed overdispersion to null."""
    rng = np.random.RandomState(RANDOM_SEED)

    df = t1.dropna(subset=["entity_poi", "q_year"]).copy()
    df["q_year_bin"] = pd.cut(df["q_year"], bins=5, labels=False)

    # Observed overdispersion statistic
    obs_stat = _overdispersion_stat(df)

    # Permutation distribution
    null_stats = []
    for i in range(n_perms):
        df_perm = df.copy()
        # Shuffle poi_name_clean within entity × q_year_bin
        for _, grp in df_perm.groupby(["entity", "q_year_bin"]):
            idx = grp.index
            shuffled = rng.permutation(df_perm.loc[idx, "poi_name_clean"].values)
            df_perm.loc[idx, "poi_name_clean"] = shuffled
        # Recompute entity_poi
        df_perm["entity_poi"] = df_perm["entity"].astype(str) + "||" + df_perm["poi_name_clean"].astype(str)
        null_stats.append(_overdispersion_stat(df_perm))

        if (i + 1) % 100 == 0:
            print(f"  Permutation {i+1}/{n_perms}")

    null_stats = np.array(null_stats)
    p_value = (null_stats >= obs_stat).mean()

    print(f"\n=== Permutation Test ===")
    print(f"  Observed stat: {obs_stat:.4f}")
    print(f"  Null mean: {null_stats.mean():.4f}, std: {null_stats.std():.4f}")
    print(f"  p-value: {p_value:.4f}")

    results = {
        "observed": obs_stat,
        "null_mean": null_stats.mean(),
        "null_std": null_stats.std(),
        "p_value": p_value,
    }
    pd.DataFrame([results]).to_csv(os.path.join(TABLES_DIR, "permutation_test.csv"), index=False)
    return results, null_stats


def _overdispersion_stat(df):
    """Compute overdispersion statistic (variance ratio of POI withdrawal rates)."""
    poi = df.groupby("entity_poi").agg(
        n=("withdrawn", "count"),
        k=("withdrawn", "sum"),
    )
    poi = poi[poi["n"] >= 2]
    poi["rate"] = poi["k"] / poi["n"]
    p_hat = poi["k"].sum() / poi["n"].sum()
    observed_var = poi["rate"].var()
    expected_var = (p_hat * (1 - p_hat) / poi["n"]).mean()
    if expected_var == 0:
        return 0
    return observed_var / expected_var


def placebo_test(t1):
    """Repeat overdispersion for operational outcomes — should cluster less."""
    df = t1.copy()
    df["operational"] = (df["q_status"] == "operational").astype(int)

    poi = df.groupby("entity_poi").agg(
        n=("q_id", "count"),
        k_wd=("withdrawn", "sum"),
        k_op=("operational", "sum"),
    )
    poi = poi[poi["n"] >= 2]

    # Overdispersion for withdrawals
    poi["rate_wd"] = poi["k_wd"] / poi["n"]
    p_wd = poi["k_wd"].sum() / poi["n"].sum()
    od_wd = poi["rate_wd"].var() / (p_wd * (1 - p_wd) / poi["n"]).mean()

    # Overdispersion for operational
    poi["rate_op"] = poi["k_op"] / poi["n"]
    p_op = poi["k_op"].sum() / poi["n"].sum()
    od_op = poi["rate_op"].var() / (p_op * (1 - p_op) / poi["n"]).mean()

    print(f"\n=== Placebo Test ===")
    print(f"  Withdrawal overdispersion: {od_wd:.4f}")
    print(f"  Operational overdispersion: {od_op:.4f}")
    print(f"  Ratio (wd/op): {od_wd/od_op:.4f}" if od_op > 0 else "  Operational OD = 0")

    results = {"od_withdrawal": od_wd, "od_operational": od_op}
    pd.DataFrame([results]).to_csv(os.path.join(TABLES_DIR, "placebo_test.csv"), index=False)
    return results


def temporal_asymmetry(t1):
    """Granger-style temporal asymmetry test using withdrawal dates.

    Within each POI, split withdrawn projects at the median wd_date into
    "early withdrawers" and "late withdrawers". If contagion is real, the
    early withdrawal rate (relative to POI size) should predict the late
    withdrawal rate (forward direction), but the reverse should be weaker.

    Previous version incorrectly split by q_date (queue entry date), which
    tested cohort correlation rather than temporal ordering of withdrawal
    events, and produced near-symmetric coefficients by construction.
    """
    df = t1.copy()

    # Need wd_date for withdrawn projects to establish temporal ordering
    if "wd_date" not in df.columns:
        print("  wd_date column not available — skipping temporal asymmetry")
        return {}

    results = []
    for epoi, grp in df.groupby("entity_poi"):
        if len(grp) < 3:  # need enough projects for meaningful split
            continue

        # Get withdrawn projects with valid wd_date at this POI
        wd_projects = grp[(grp["withdrawn"] == 1) & grp["wd_date"].notna()]
        if len(wd_projects) < 2:
            continue

        # Split withdrawn projects at median withdrawal date
        median_wd_date = wd_projects["wd_date"].median()
        early_wd_count = (wd_projects["wd_date"] <= median_wd_date).sum()
        late_wd_count = (wd_projects["wd_date"] > median_wd_date).sum()

        if early_wd_count == 0 or late_wd_count == 0:
            continue

        # Normalize by POI size for comparability
        n_total = len(grp)
        results.append({
            "entity_poi": epoi,
            "early_wd_rate": early_wd_count / n_total,
            "late_wd_rate": late_wd_count / n_total,
            "n_total": n_total,
            "n_withdrawn": len(wd_projects),
        })

    asym = pd.DataFrame(results)
    if len(asym) == 0:
        print("  No POIs with 2+ dated withdrawals")
        return {}

    print(f"  POIs with 2+ dated withdrawals: {len(asym)}")

    # Forward: early withdrawal intensity predicts late withdrawal intensity
    X_fwd = sm.add_constant(asym["early_wd_rate"])
    y_fwd = asym["late_wd_rate"]
    mask_fwd = X_fwd.notna().all(axis=1) & y_fwd.notna()
    res_fwd = sm.OLS(y_fwd[mask_fwd], X_fwd[mask_fwd]).fit()

    # Backward: late withdrawal intensity predicts early withdrawal intensity
    X_bwd = sm.add_constant(asym["late_wd_rate"])
    y_bwd = asym["early_wd_rate"]
    mask_bwd = X_bwd.notna().all(axis=1) & y_bwd.notna()
    res_bwd = sm.OLS(y_bwd[mask_bwd], X_bwd[mask_bwd]).fit()

    fwd_coef = res_fwd.params.iloc[1]
    bwd_coef = res_bwd.params.iloc[1]

    print(f"\n=== Temporal Asymmetry (wd_date-based) ===")
    print(f"  Forward (early→late): coef={fwd_coef:.4f}, p={res_fwd.pvalues.iloc[1]:.4f}")
    print(f"  Backward (late→early): coef={bwd_coef:.4f}, p={res_bwd.pvalues.iloc[1]:.4f}")
    diff = fwd_coef - bwd_coef
    print(f"  Asymmetry (fwd - bwd): {diff:.4f}")

    out = {
        "forward_coef": fwd_coef,
        "forward_se": res_fwd.bse.iloc[1],
        "forward_p": res_fwd.pvalues.iloc[1],
        "backward_coef": bwd_coef,
        "backward_se": res_bwd.bse.iloc[1],
        "backward_p": res_bwd.pvalues.iloc[1],
        "asymmetry": diff,
        "n_pois": len(asym),
    }
    pd.DataFrame([out]).to_csv(os.path.join(TABLES_DIR, "temporal_asymmetry.csv"), index=False)
    return out


def dose_response(t1):
    """Run logistic regression within POI depth bins. Coefficient should increase with depth."""
    df = t1.copy()

    results = []
    for (lo, hi), label in zip(DEPTH_BINS, DEPTH_LABELS):
        subset = df[(df["poi_depth"] >= lo) & (df["poi_depth"] <= hi)].copy()
        if len(subset) < 50 or subset["withdrawn"].sum() < 10:
            print(f"  Depth bin {label}: skipping (n={len(subset)}, events={subset['withdrawn'].sum()})")
            continue

        X = sm.add_constant(subset[["peer_wd_rate", "mw1_log", "q_year"]].fillna(0))
        y = subset["withdrawn"]
        try:
            res = sm.Logit(y, X).fit(disp=False)
            coef = res.params["peer_wd_rate"]
            se = res.bse["peer_wd_rate"]
            p = res.pvalues["peer_wd_rate"]
        except Exception as e:
            print(f"  Depth bin {label}: model failed ({e})")
            continue

        results.append({
            "depth_bin": label,
            "n": len(subset),
            "n_events": subset["withdrawn"].sum(),
            "peer_wd_rate_coef": coef,
            "se": se,
            "p_value": p,
            "odds_ratio": np.exp(coef),
        })
        print(f"  Depth {label}: coef={coef:.3f}, OR={np.exp(coef):.3f}, p={p:.4f}, n={len(subset)}")

    dr = pd.DataFrame(results)
    print(f"\n=== Dose-Response ===")
    print(dr.to_string(index=False))
    dr.to_csv(os.path.join(TABLES_DIR, "dose_response.csv"), index=False)
    return dr


def entity_heterogeneity(t1):
    """Run logistic per entity. Extract peer_wd_rate coefficient for forest plot."""
    df = t1.copy()

    results = []
    for entity, grp in df.groupby("entity"):
        if len(grp) < 100 or grp["withdrawn"].sum() < 20:
            continue

        X = sm.add_constant(grp[["peer_wd_rate", "mw1_log", "q_year"]].fillna(0))
        y = grp["withdrawn"]
        try:
            res = sm.Logit(y, X).fit(disp=False)
            coef = res.params["peer_wd_rate"]
            se = res.bse["peer_wd_rate"]
            p = res.pvalues["peer_wd_rate"]
        except Exception:
            continue

        results.append({
            "entity": entity,
            "n": len(grp),
            "n_events": grp["withdrawn"].sum(),
            "peer_wd_rate_coef": coef,
            "se": se,
            "p_value": p,
            "odds_ratio": np.exp(coef),
            "ci_lower": np.exp(coef - 1.96 * se),
            "ci_upper": np.exp(coef + 1.96 * se),
        })

    eh = pd.DataFrame(results).sort_values("peer_wd_rate_coef", ascending=False)
    print(f"\n=== Entity Heterogeneity ===")
    print(eh.to_string(index=False))
    eh.to_csv(os.path.join(TABLES_DIR, "entity_heterogeneity.csv"), index=False)
    return eh


def run_robustness(t1):
    """Run all robustness checks."""
    print("\n" + "=" * 60)
    print("ROBUSTNESS CHECKS")
    print("=" * 60)

    perm_results, null_stats = permutation_test(t1)
    placebo = placebo_test(t1)
    temporal = temporal_asymmetry(t1)
    dr = dose_response(t1)
    eh = entity_heterogeneity(t1)

    return {
        "permutation": (perm_results, null_stats),
        "placebo": placebo,
        "temporal": temporal,
        "dose_response": dr,
        "entity_heterogeneity": eh,
    }
