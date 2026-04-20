"""Sweep alpha_local ∈ {0.10, 0.15, 0.20, 0.25, 0.35} and report matched-DiD
event-study coefficients for each.

Empirical targets (contagion brief):
  k=1:         -0.038 (p=0.009)
  k=4-8 peak:  +0.029 (p=0.004)
  pooled DiD:  ~-0.008 (null)
"""
from __future__ import annotations
import os, sys, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import QueueModel, Params
from validate_matched_did import abm_panel_to_lbnl_schema, run_did_on_df

ALPHAS = [0.10, 0.15, 0.20, 0.25, 0.35]
N_REPS = 2
BASE = dict(n_pois=2000, horizon_months=240, arrivals_per_month=25.0,
            reallocation_enabled=True)


def one_rep(seed: int, alpha: float) -> pd.DataFrame:
    p = Params(**{**BASE, "rng_seed": seed, "alpha_local": alpha})
    m = QueueModel(p).run()
    panel = pd.DataFrame(m.project_panel())
    return abm_panel_to_lbnl_schema(panel, p.horizon_months)


def summarize(results):
    """results: list of dicts from run_did_on_df."""
    all_betas = []
    for i, r in enumerate(results):
        b = pd.DataFrame(r["betas"]).assign(rep=i)
        all_betas.append(b)
    big = pd.concat(all_betas)
    pooled = big.groupby("event_time").agg(
        mean_beta=("beta", "mean"),
        mean_p=("p_value", "mean"),
    )
    did_vals = [r["did"]["did_estimate"] for r in results]
    return pooled, float(np.mean(did_vals))


def main():
    summary_rows = []
    for alpha in ALPHAS:
        print(f"\n########################  alpha_local = {alpha}  ########################")
        t0 = time.time()
        results = []
        for i, seed in enumerate(range(100, 100 + N_REPS)):
            df_sim = one_rep(seed, alpha)
            r = run_did_on_df(df_sim, f"alpha={alpha} seed={seed}")
            if r is not None:
                results.append(r)
        if not results:
            print(f"  [no results for alpha={alpha}]")
            continue
        pooled, did = summarize(results)
        print(f"\n  pooled event-study (alpha={alpha}):")
        print(pooled.to_string().replace("\n", "\n    "))
        print(f"  pooled 2-period DiD: {did:+.4f}")
        # Extract key stats
        betas_series = pooled["mean_beta"]
        k1 = betas_series.get(1, np.nan)
        post_mask = (betas_series.index >= 4) & (betas_series.index <= 8)
        peak_k = betas_series[post_mask].idxmax() if post_mask.any() else np.nan
        peak = betas_series[post_mask].max() if post_mask.any() else np.nan
        summary_rows.append({
            "alpha": alpha,
            "k1": k1,
            "peak_k4_8": peak,
            "peak_at_k": peak_k,
            "pooled_did": did,
            "elapsed_s": time.time() - t0,
        })

    print("\n\n=====================  ALPHA SWEEP SUMMARY  =====================")
    df = pd.DataFrame(summary_rows)
    print(df.to_string(index=False))
    print("\nEmpirical targets:  k=1: -0.038    k=4-8 peak: +0.029    pooled DiD: ~-0.008")
    out_path = os.path.join(HERE, "output", "alpha_sweep.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
