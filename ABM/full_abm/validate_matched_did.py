"""Run the empirical contagion/matched_did.py pipeline on ABM-simulated panels.

This is the most rigorous validation: if the ABM is well-calibrated, running
the *same* matched-DiD estimator used on real LBNL data should produce the
same coefficients on simulated data (or close to it).

Steps:
  1. Simulate N_REPS replications, each with reallocation ON.
  2. Convert each simulated project panel into an LBNL-schema DataFrame.
  3. Call contagion.matched_did's build_poi_quarter_panel ->
     identify_treatment_events -> match_pois -> build_event_study_panel ->
     event_study_regression.
  4. Compare the event-study coefficients to the empirical brief's:
     k=1: -0.038 (p=0.009)
     k=4-8 peak: +0.029 (p=0.004)
     pooled 4-qtr DiD: ~-0.008 (null)
"""

from __future__ import annotations
import os, sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Import ABM bits
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import QueueModel, Params

# Import empirical DiD pipeline
CONTAGION_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "contagion")
sys.path.insert(0, CONTAGION_DIR)
# Bypass matched_did's file-writing side effects by creating its output dirs ahead of time.
for sub in ("tables", "figures"):
    os.makedirs(os.path.join(CONTAGION_DIR, "output", "matched_did", sub), exist_ok=True)
import matched_did as mdid  # noqa: E402

ANCHOR = pd.Timestamp("2000-01-01")  # t=0 maps here


def abm_panel_to_lbnl_schema(panel: pd.DataFrame, horizon_months: int) -> pd.DataFrame:
    """Map ABM project panel to the LBNL-schema DataFrame that matched_did expects.

    Required columns for build_poi_quarter_panel + _compute_poi_static_attrs:
      entity, entity_poi, q_id, q_date, wd_date, on_date, mw1, q_year,
      type_clean, q_status, dominant_type (via _compute_poi_static_attrs)
    """
    df = panel.copy()
    df["entity"] = "PJM"
    df["poi_name_clean"] = "poi_" + df["poi_id"].astype(str)
    df["entity_poi"] = df["entity"] + "||" + df["poi_name_clean"]
    df["q_id"] = df["project_id"].astype(str)
    df["mw1"] = df["mw"]
    df["type_clean"] = "Solar"  # single tech; matched_did matches on dominant type

    df["q_date"] = ANCHOR + pd.to_timedelta(df["t_entry"] * 30.44, unit="D")
    df["q_year"] = df["q_date"].dt.year

    # Operational date: if status==completed, use t_cod; else NaT
    df["on_date"] = pd.NaT
    comp_mask = df["status"] == "completed"
    df.loc[comp_mask, "on_date"] = ANCHOR + pd.to_timedelta(
        df.loc[comp_mask, "t_exit"] * 30.44, unit="D"
    )

    # Withdrawal date
    df["wd_date"] = pd.NaT
    wd_mask = df["status"] == "withdrawn"
    df.loc[wd_mask, "wd_date"] = ANCHOR + pd.to_timedelta(
        df.loc[wd_mask, "t_exit"] * 30.44, unit="D"
    )

    df["q_status"] = df["status"].map({
        "completed": "operational",
        "withdrawn": "withdrawn",
        "active": "active",
    })
    df["withdrawn"] = (df["status"] == "withdrawn").astype(int)

    keep = ["entity", "entity_poi", "poi_name_clean", "q_id", "q_date",
            "wd_date", "on_date", "mw1", "q_year", "type_clean", "q_status",
            "withdrawn"]
    return df[keep]


def run_one_rep(seed: int, params: Params) -> pd.DataFrame:
    p = Params(**{**params.__dict__, "rng_seed": seed})
    model = QueueModel(p).run()
    panel = pd.DataFrame(model.project_panel())
    return abm_panel_to_lbnl_schema(panel, p.horizon_months)


def run_did_on_df(df_sim: pd.DataFrame, label: str):
    print(f"\n── {label} ──")
    panel, df_ = mdid.build_poi_quarter_panel(df_sim)
    events = mdid.identify_treatment_events(panel)
    if len(events) == 0:
        print("  [no treatment events identified]")
        return None
    pairs, unmatched = mdid.match_pois(events, panel)
    if len(pairs) == 0:
        print("  [no matched pairs]")
        return None
    es = mdid.build_event_study_panel(pairs, panel, window=(-4, 8))

    # Two-period DiD
    did_res, _ = mdid.two_period_did(es)

    # Event-study regression
    betas, pre_trend, _ = mdid.event_study_regression(es)
    return {"did": did_res, "betas": betas, "pre_trend": pre_trend,
            "n_pairs": len(pairs), "n_events": len(events)}


def main():
    n_reps = 3
    results = []
    base = Params(n_pois=2000, horizon_months=240, arrivals_per_month=25.0,
                   reallocation_enabled=True)
    for i, seed in enumerate(range(42, 42 + n_reps)):
        print(f"\n=========================  Rep {i+1}/{n_reps}  seed={seed}  =========================")
        df_sim = run_one_rep(seed, base)
        print(f"  simulated projects: {len(df_sim)} (withdrawn={df_sim['withdrawn'].sum()}, "
              f"completed={(df_sim['q_status']=='operational').sum()})")
        r = run_did_on_df(df_sim, f"seed {seed}")
        if r is not None:
            r["seed"] = seed
            results.append(r)

    # Aggregate event-study coefficients across reps
    if results:
        print("\n================= Pooled event-study betas (mean across reps) =================")
        all_betas = []
        for r in results:
            b = pd.DataFrame(r["betas"]).assign(seed=r["seed"])
            all_betas.append(b)
        big = pd.concat(all_betas)
        pooled = big.groupby("event_time").agg(
            mean_beta=("beta", "mean"),
            sd_beta=("beta", "std"),
            mean_p=("p_value", "mean"),
        )
        print(pooled.to_string())

        print("\n================= Pooled two-period DiD =================")
        did_vals = [r["did"]["did_estimate"] for r in results]
        p_vals = [r["did"]["p_value"] for r in results]
        print(f"  DiD across reps: mean={np.mean(did_vals):.4f}  range=[{min(did_vals):.4f}, {max(did_vals):.4f}]")
        print(f"  p-values:         {p_vals}")


if __name__ == "__main__":
    main()
