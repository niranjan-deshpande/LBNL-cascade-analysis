"""Descriptive statistics and overdispersion test."""

import pandas as pd
import numpy as np
from scipy import stats
from config import TABLES_DIR
import os


def poi_summary_stats(t1):
    """Compute POI-level summary statistics."""
    poi = t1.groupby("entity_poi").agg(
        n_projects=("q_id", "count"),
        n_withdrawn=("withdrawn", "sum"),
        wd_rate=("withdrawn", "mean"),
        entity=("entity", "first"),
        type_diversity=("poi_type_diversity", "first"),
    ).reset_index()

    summary = poi[["n_projects", "n_withdrawn", "wd_rate", "type_diversity"]].describe()
    print("\n=== POI-Level Summary ===")
    print(summary.round(3))

    summary.to_csv(os.path.join(TABLES_DIR, "poi_summary_stats.csv"))
    return poi


def overdispersion_test(poi):
    """Test for overdispersion in POI withdrawal rates.

    Under binomial independence, Var(k) = n*p*(1-p).
    If withdrawals cluster at POIs, observed variance > expected.
    """
    # Filter to POIs with enough projects for meaningful test
    poi_f = poi[poi["n_projects"] >= 2].copy()

    # Overall withdrawal rate
    p_hat = poi_f["n_withdrawn"].sum() / poi_f["n_projects"].sum()

    # Expected variance under binomial for each POI
    poi_f["expected_var"] = poi_f["n_projects"] * p_hat * (1 - p_hat)

    # Use chi-squared approach: sum of (observed - expected)^2 / expected
    poi_f["expected_wd"] = poi_f["n_projects"] * p_hat
    chi2_stat = ((poi_f["n_withdrawn"] - poi_f["expected_wd"]) ** 2 / poi_f["expected_var"]).sum()
    df = len(poi_f) - 1
    p_value = 1 - stats.chi2.cdf(chi2_stat, df)

    # Variance ratio
    observed_var = poi_f["wd_rate"].var()
    # Expected variance of rate = p*(1-p)/n, weighted
    weights = poi_f["n_projects"]
    expected_var_rate = (p_hat * (1 - p_hat) / weights).mean()
    variance_ratio = observed_var / expected_var_rate

    results = {
        "overall_wd_rate": p_hat,
        "n_pois": len(poi_f),
        "chi2_statistic": chi2_stat,
        "df": df,
        "p_value": p_value,
        "observed_variance": observed_var,
        "expected_variance": expected_var_rate,
        "variance_ratio": variance_ratio,
    }

    print("\n=== Overdispersion Test ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    pd.DataFrame([results]).to_csv(os.path.join(TABLES_DIR, "overdispersion_test.csv"), index=False)
    return results


def entity_summary(t1):
    """Entity-level summary table."""
    ent = t1.groupby("entity").agg(
        n_projects=("q_id", "count"),
        n_withdrawn=("withdrawn", "sum"),
        wd_rate=("withdrawn", "mean"),
        n_pois=("entity_poi", "nunique"),
        mean_poi_depth=("poi_depth", "mean"),
    ).reset_index()

    ent = ent.sort_values("n_projects", ascending=False)
    print("\n=== Entity Summary ===")
    print(ent.to_string(index=False))

    ent.to_csv(os.path.join(TABLES_DIR, "entity_summary.csv"), index=False)
    return ent


def run_descriptive(t1):
    """Run all descriptive analyses."""
    poi = poi_summary_stats(t1)
    od = overdispersion_test(poi)
    ent = entity_summary(t1)
    return poi, od, ent
