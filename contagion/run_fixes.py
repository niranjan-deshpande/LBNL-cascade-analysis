"""Run all four fixes/improvements and report results."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from config import TABLES_DIR, FIGURES_DIR


def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── Item 4: POI normalization improvement ──────────────────────────
    # Run this FIRST because it affects everything downstream
    print("=" * 70)
    print("ITEM 4: IMPROVED POI NORMALIZATION")
    print("=" * 70)

    from data_prep import load_raw_data, convert_dates, normalize_poi_name

    df_raw = load_raw_data()
    df_raw = convert_dates(df_raw)

    # Compare old vs new normalization
    if "poi_name" in df_raw.columns:
        # Old method: just lowercase + strip
        old_clean = df_raw["poi_name"].astype(str).str.lower().str.strip()
        na_values = {"nan", "na", "n/a", "unknown", "none", ""}
        old_clean = old_clean.where(~old_clean.isin(na_values), other=np.nan)
        old_epoi = df_raw["entity"].astype(str) + "||" + old_clean.astype(str)
        old_epoi = old_epoi.where(old_clean.notna(), other=np.nan)
        n_old_pois = old_epoi.dropna().nunique()

        # New method
        new_clean = normalize_poi_name(df_raw["poi_name"])
        new_epoi = df_raw["entity"].astype(str) + "||" + new_clean.astype(str)
        new_epoi = new_epoi.where(new_clean.notna(), other=np.nan)
        n_new_pois = new_epoi.dropna().nunique()

        pois_merged = n_old_pois - n_new_pois
        print(f"  Old normalization: {n_old_pois} unique entity_poi values")
        print(f"  New normalization: {n_new_pois} unique entity_poi values")
        print(f"  POIs merged: {pois_merged} ({pois_merged/n_old_pois*100:.1f}%)")

        # Show some examples of merges
        old_set = set(old_epoi.dropna().unique())
        new_set = set(new_epoi.dropna().unique())
        lost = old_set - new_set
        if lost:
            print(f"  Example merges (old POIs that no longer exist):")
            for ex in list(lost)[:10]:
                print(f"    {ex}")

    # Now clean with new normalization
    from data_prep import clean_data, build_tier1_sample
    df = clean_data(df_raw.copy())
    t1 = build_tier1_sample(df)

    # Count multi-project POI sample
    print(f"\n  New Tier 1 sample: {len(t1)} projects in {t1['entity_poi'].nunique()} POIs")

    # Sensitivity: re-run Tier 1 logistic
    print("\n  Re-running Tier 1 logistic with improved POI normalization...")
    from tier1_logistic import run_logistic
    t1_model, t1_summary = run_logistic(t1)

    peer_wd_or = t1_summary.loc["peer_wd_rate", "odds_ratio"]
    peer_wd_ci_lo = t1_summary.loc["peer_wd_rate", "ci_lower"]
    peer_wd_ci_hi = t1_summary.loc["peer_wd_rate", "ci_upper"]
    peer_wd_p = t1_summary.loc["peer_wd_rate", "p_value"]
    print(f"\n  peer_wd_rate OR = {peer_wd_or:.2f} [{peer_wd_ci_lo:.2f}, {peer_wd_ci_hi:.2f}], p = {peer_wd_p:.4f}")

    # ── Item 1: Fixed same-day withdrawal bug + re-run Cox ─────────────
    print("\n" + "=" * 70)
    print("ITEM 1: SAME-DAY WITHDRAWAL BUG FIX — COX MODEL RE-RUN")
    print("=" * 70)

    from data_prep import build_tier2_sample
    from tier2_cox import run_cox

    t2_cp = build_tier2_sample(df, lag_days=0)
    print(f"\n  Running Cox model (lag=0, bug fixed)...")
    cox_model, cox_summary = run_cox(t2_cp)

    if "cumulative_peer_wd" in cox_summary.index:
        row = cox_summary.loc["cumulative_peer_wd"]
        print(f"\n  RESULT — cumulative_peer_wd:")
        print(f"    HR = {row['hazard_ratio']:.4f}")
        print(f"    95% CI = [{row['hr_ci_lower']:.4f}, {row['hr_ci_upper']:.4f}]")
        print(f"    p = {row['p']:.4f}")

    # Save updated results
    cox_summary.to_csv(os.path.join(TABLES_DIR, "tier2_cox_results_bugfixed.csv"))

    # ── Item 3: Lagged contagion exposure ──────────────────────────────
    print("\n" + "=" * 70)
    print("ITEM 3: LAGGED CONTAGION EXPOSURE IN COX MODEL")
    print("=" * 70)

    lag_results = []
    for lag_months, lag_days in [(0, 0), (6, 183), (12, 365)]:
        print(f"\n  --- Lag = {lag_months} months ({lag_days} days) ---")
        t2_lag = build_tier2_sample(df, lag_days=lag_days)
        cox_lag, cox_lag_summary = run_cox(t2_lag)

        if "cumulative_peer_wd" in cox_lag_summary.index:
            row = cox_lag_summary.loc["cumulative_peer_wd"]
            lag_results.append({
                "lag_months": lag_months,
                "lag_days": lag_days,
                "hazard_ratio": row["hazard_ratio"],
                "hr_ci_lower": row["hr_ci_lower"],
                "hr_ci_upper": row["hr_ci_upper"],
                "p_value": row["p"],
                "n_intervals": len(t2_lag),
                "n_subjects": t2_lag["q_id"].nunique(),
            })
            print(f"  HR = {row['hazard_ratio']:.4f} [{row['hr_ci_lower']:.4f}, {row['hr_ci_upper']:.4f}], p = {row['p']:.4f}")

        cox_lag_summary.to_csv(os.path.join(TABLES_DIR, f"tier2_cox_lag{lag_months}m.csv"))

    lag_df = pd.DataFrame(lag_results)
    print(f"\n  === Lag Comparison ===")
    print(lag_df.to_string(index=False))
    lag_df.to_csv(os.path.join(TABLES_DIR, "tier2_cox_lag_comparison.csv"), index=False)

    # ── Item 2: Measurement error simulation ───────────────────────────
    print("\n" + "=" * 70)
    print("ITEM 2: MEASUREMENT ERROR ATTENUATION SIMULATION")
    print("=" * 70)

    from simulation_dose_response import run_simulation
    sim_summary, sim_full = run_simulation(n_reps=1000)

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ALL FIXES COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
