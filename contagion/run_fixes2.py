"""Run items 5-8: restricted placebo, deep-POI Cox, PH test, edge-case check."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from config import TABLES_DIR, FIGURES_DIR


def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Load data once
    from data_prep import load_raw_data, convert_dates, clean_data, build_tier1_sample
    df = load_raw_data()
    df = convert_dates(df)
    df = clean_data(df)
    t1 = build_tier1_sample(df)

    # ── Item 5: Restricted placebo test ────────────────────────────────
    print("\n" + "=" * 70)
    print("ITEM 5: RESTRICTED PLACEBO TEST (terminal outcomes only)")
    print("=" * 70)

    run_restricted_placebo(t1)

    # ── Item 8: Edge-case check in counting-process logic ──────────────
    print("\n" + "=" * 70)
    print("ITEM 8: EDGE-CASE CHECK — peer withdrawal at exact stop time")
    print("=" * 70)

    from data_prep import build_tier2_sample
    t2_cp = build_tier2_sample(df, lag_days=0)

    # Check for zero-length intervals
    zero_len = t2_cp[t2_cp["stop"] <= t2_cp["start"]]
    print(f"  Zero-length intervals in output: {len(zero_len)}")

    # Check event attribution: every project with event=1 should have
    # exactly one interval with event=1 (the last one)
    event_counts = t2_cp.groupby("q_id")["event"].sum()
    multi_event = event_counts[event_counts > 1]
    print(f"  Projects with >1 event interval: {len(multi_event)}")
    if len(multi_event) > 0:
        print(f"    WARNING: {multi_event.head().to_dict()}")

    # Verify: for each project, the last interval should carry the event
    last_intervals = t2_cp.sort_values("stop").groupby("q_id").last()
    projects_with_event = t2_cp.groupby("q_id")["event"].max()
    mismatched = (last_intervals["event"] != projects_with_event).sum()
    print(f"  Event on non-final interval: {mismatched}")

    print("  Edge-case check passed." if (len(zero_len) == 0 and len(multi_event) == 0
          and mismatched == 0) else "  Issues detected — see above.")

    # ── Item 6: Cox on deep POIs only ──────────────────────────────────
    print("\n" + "=" * 70)
    print("ITEM 6: COX MODEL ON DEEP POIs (5+ projects)")
    print("=" * 70)

    from tier2_cox import run_cox

    t2_deep = build_tier2_sample(df, lag_days=0, min_poi_depth=5)
    cox_deep, cox_deep_summary = run_cox(t2_deep)

    if "cumulative_peer_wd" in cox_deep_summary.index:
        row = cox_deep_summary.loc["cumulative_peer_wd"]
        print(f"\n  RESULT (depth >= 5) — cumulative_peer_wd:")
        print(f"    HR = {row['hazard_ratio']:.4f}")
        print(f"    95% CI = [{row['hr_ci_lower']:.4f}, {row['hr_ci_upper']:.4f}]")
        print(f"    p = {row['p']:.4f}")
        print(f"    N subjects = {t2_deep['q_id'].nunique()}")
        print(f"    N events = {t2_deep.groupby('q_id')['event'].max().sum()}")
    cox_deep_summary.to_csv(os.path.join(TABLES_DIR, "tier2_cox_deep_pois.csv"))

    # Also try depth >= 3 as intermediate
    print("\n  --- Also trying depth >= 3 ---")
    t2_mid = build_tier2_sample(df, lag_days=0, min_poi_depth=3)
    cox_mid, cox_mid_summary = run_cox(t2_mid)
    if "cumulative_peer_wd" in cox_mid_summary.index:
        row = cox_mid_summary.loc["cumulative_peer_wd"]
        print(f"  RESULT (depth >= 3): HR = {row['hazard_ratio']:.4f} "
              f"[{row['hr_ci_lower']:.4f}, {row['hr_ci_upper']:.4f}], p = {row['p']:.4f}")
    cox_mid_summary.to_csv(os.path.join(TABLES_DIR, "tier2_cox_mid_pois.csv"))

    # Comparison table
    depth_comparison = []
    for label, cp_df, summ in [(">=2", t2_cp, None), (">=3", t2_mid, cox_mid_summary),
                                (">=5", t2_deep, cox_deep_summary)]:
        if label == ">=2":
            # Re-run for >=2 to get summary
            cox2, summ2 = run_cox(t2_cp)
            summ = summ2
        if "cumulative_peer_wd" in summ.index:
            r = summ.loc["cumulative_peer_wd"]
            depth_comparison.append({
                "min_depth": label,
                "n_subjects": cp_df["q_id"].nunique(),
                "n_events": int(cp_df.groupby("q_id")["event"].max().sum()),
                "n_intervals": len(cp_df),
                "hazard_ratio": r["hazard_ratio"],
                "hr_ci_lower": r["hr_ci_lower"],
                "hr_ci_upper": r["hr_ci_upper"],
                "p_value": r["p"],
            })

    dc = pd.DataFrame(depth_comparison)
    print(f"\n  === Cox Model by POI Depth Threshold ===")
    print(dc.to_string(index=False))
    dc.to_csv(os.path.join(TABLES_DIR, "tier2_cox_depth_comparison.csv"), index=False)

    # ── Item 7: Proportional hazards test ──────────────────────────────
    print("\n" + "=" * 70)
    print("ITEM 7: PROPORTIONAL HAZARDS ASSUMPTION TEST")
    print("=" * 70)

    run_ph_test(t2_cp, df)

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ALL ITEMS 5-8 COMPLETE")
    print("=" * 70)


def run_restricted_placebo(t1):
    """Item 5: Placebo test restricted to terminal-outcome projects."""
    # Restrict to terminal outcomes
    terminal = t1[t1["q_status"].isin(["withdrawn", "operational"])].copy()
    print(f"  Terminal-outcome sample: {len(terminal)} projects "
          f"(dropped {len(t1) - len(terminal)} active/suspended)")
    print(f"  Withdrawn: {(terminal['q_status'] == 'withdrawn').sum()}, "
          f"Operational: {(terminal['q_status'] == 'operational').sum()}")

    terminal["operational"] = (terminal["q_status"] == "operational").astype(int)

    # Recompute POI-level stats on terminal sample only
    poi = terminal.groupby("entity_poi").agg(
        n=("q_id", "count"),
        k_wd=("withdrawn", "sum"),
        k_op=("operational", "sum"),
    )
    poi = poi[poi["n"] >= 2]
    print(f"  POIs with 2+ terminal projects: {len(poi)}")

    # Overdispersion for withdrawals
    poi["rate_wd"] = poi["k_wd"] / poi["n"]
    p_wd = poi["k_wd"].sum() / poi["n"].sum()
    expected_var_wd = (p_wd * (1 - p_wd) / poi["n"]).mean()
    od_wd = poi["rate_wd"].var() / expected_var_wd if expected_var_wd > 0 else np.nan

    # Overdispersion for operational
    poi["rate_op"] = poi["k_op"] / poi["n"]
    p_op = poi["k_op"].sum() / poi["n"].sum()
    expected_var_op = (p_op * (1 - p_op) / poi["n"]).mean()
    od_op = poi["rate_op"].var() / expected_var_op if expected_var_op > 0 else np.nan

    ratio = od_wd / od_op if od_op > 0 else np.nan

    print(f"\n  === Restricted Placebo Test (terminal outcomes only) ===")
    print(f"  Withdrawal overdispersion (VR): {od_wd:.4f}")
    print(f"  Operational overdispersion (VR): {od_op:.4f}")
    print(f"  Ratio (wd/op): {ratio:.4f}")

    # Compare with unrestricted
    print(f"\n  For reference, unrestricted values were: VR_wd=1.6726, VR_op=1.5349, ratio=1.0897")

    results = {
        "sample": "terminal_only",
        "n_projects": len(terminal),
        "n_pois": len(poi),
        "od_withdrawal": od_wd,
        "od_operational": od_op,
        "ratio": ratio,
        "p_wd": p_wd,
        "p_op": p_op,
    }
    pd.DataFrame([results]).to_csv(os.path.join(TABLES_DIR, "placebo_test_restricted.csv"), index=False)
    return results


def run_ph_test(t2_cp, df):
    """Item 7: Test proportional hazards assumption via Schoenfeld residuals."""
    from lifelines import CoxTimeVaryingFitter

    # Fit the model
    model_df = t2_cp.copy()
    top_types = model_df["type_clean"].value_counts().head(6).index.tolist()
    model_df["type_group"] = model_df["type_clean"].where(
        model_df["type_clean"].isin(top_types), "Other")
    type_dummies = pd.get_dummies(model_df["type_group"], prefix="type", drop_first=True, dtype=int)
    entity_dummies = pd.get_dummies(model_df["entity"], prefix="entity", drop_first=True, dtype=int)

    fit_df = pd.concat([
        model_df[["q_id", "start", "stop", "event", "cumulative_peer_wd", "mw1_log", "q_year"]],
        type_dummies, entity_dummies,
    ], axis=1).fillna(0)
    fit_df = fit_df[fit_df["stop"] > fit_df["start"]].copy()

    ctv = CoxTimeVaryingFitter(penalizer=0.01)
    ctv.fit(fit_df, id_col="q_id", start_col="start", stop_col="stop", event_col="event")

    # lifelines doesn't directly provide Schoenfeld residuals for time-varying models.
    # Instead, we test the PH assumption by adding a time-interaction term:
    # cumulative_peer_wd * time, where time = midpoint of each interval.
    print("  Testing PH via time-interaction term: cumulative_peer_wd * time")

    fit_df2 = fit_df.copy()
    # Use stop time (not midpoint) — the hazard is evaluated at interval
    # endpoints in counting-process Cox models (Grambsch-Therneau convention)
    time_center = fit_df2["stop"].mean()
    fit_df2["time_centered"] = fit_df2["stop"] - time_center
    fit_df2["peer_wd_x_time"] = fit_df2["cumulative_peer_wd"] * fit_df2["time_centered"]

    ctv2 = CoxTimeVaryingFitter(penalizer=0.01)
    ctv2.fit(fit_df2, id_col="q_id", start_col="start", stop_col="stop", event_col="event")

    print("\n  Model with time-interaction:")
    ctv2.print_summary()

    summ2 = ctv2.summary.copy()
    summ2["hazard_ratio"] = np.exp(summ2["coef"])

    # Check if the interaction term is significant
    if "peer_wd_x_time" in summ2.index:
        interaction = summ2.loc["peer_wd_x_time"]
        print(f"\n  === PH Test Result ===")
        print(f"  peer_wd_x_time coefficient: {interaction['coef']:.6f}")
        print(f"  SE: {interaction['se(coef)']:.6f}")
        print(f"  p-value: {interaction['p']:.4f}")
        if interaction["p"] < 0.05:
            print(f"  CONCLUSION: PH assumption VIOLATED (p < 0.05).")
            print(f"  The contagion effect varies with time.")
            if interaction["coef"] < 0:
                print(f"  Direction: effect WEAKENS over time (consistent with "
                      f"acute shock that fades).")
            else:
                print(f"  Direction: effect STRENGTHENS over time.")
        else:
            print(f"  CONCLUSION: No evidence of PH violation (p = {interaction['p']:.3f}).")
            print(f"  The constant-HR specification appears adequate.")

    # Also check the main effect with the interaction included
    if "cumulative_peer_wd" in summ2.index:
        main = summ2.loc["cumulative_peer_wd"]
        print(f"\n  Main effect (cumulative_peer_wd) with interaction included:")
        print(f"  HR = {main['hazard_ratio']:.4f}, p = {main['p']:.4f}")

    summ2.to_csv(os.path.join(TABLES_DIR, "tier2_cox_ph_test.csv"))

    # Alternative: split by calendar period for a simpler check
    print("\n  --- Alternative: Split-sample by calendar period ---")
    median_time = fit_df["stop"].median()

    for label, mask in [("Early (below median)", fit_df["stop"] <= median_time),
                         ("Late (above median)", fit_df["stop"] > median_time)]:
        subset = fit_df[mask].copy()
        # Need to re-filter to valid subjects
        valid_ids = subset.groupby("q_id").size().index
        subset = subset[subset["q_id"].isin(valid_ids)]
        if len(subset) < 100 or subset["event"].sum() < 50:
            print(f"  {label}: insufficient data (n={len(subset)}, events={subset['event'].sum()})")
            continue
        try:
            ctv_sub = CoxTimeVaryingFitter(penalizer=0.01)
            ctv_sub.fit(subset, id_col="q_id", start_col="start", stop_col="stop", event_col="event")
            s = ctv_sub.summary
            if "cumulative_peer_wd" in s.index:
                r = s.loc["cumulative_peer_wd"]
                hr = np.exp(r["coef"])
                print(f"  {label}: HR = {hr:.4f}, p = {r['p']:.4f}, "
                      f"events = {subset.groupby('q_id')['event'].max().sum():.0f}")
        except Exception as e:
            print(f"  {label}: model failed ({e})")

    return summ2


if __name__ == "__main__":
    main()
