"""Run actual matched-DiD on ABM panels under three regimes to confirm
the out-of-sample prediction from MODEL.md §11.3.

Prediction:
  - ON_no_pool (baseline): matched-DiD peak ~ +0.03-0.06 (matches empirical)
  - ON_with_pool:         matched-DiD peak same as baseline (pool invisible)
  - W=12 cluster-bound:   matched-DiD peak ~30% lower

10 seeds × 3 regimes ≈ 15 min.
"""
from __future__ import annotations
import os, sys, warnings, time
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model import Params
from validate_matched_did import run_one_rep, run_did_on_df

SEEDS = tuple(range(100, 110))  # 10 seeds
BASE = dict(n_pois=2000, horizon_months=240, arrivals_per_month=25.0,
            sigma_poi=2.2e6, alpha_local=0.15)
REGIMES = {
    "ON_no_pool":   dict(reallocation_enabled=True, deposit_pool_enabled=False,
                         cluster_bound_window_months=-1),
    "ON_with_pool": dict(reallocation_enabled=True, deposit_pool_enabled=True,
                         cluster_bound_window_months=-1),
    "W12_both":     dict(reallocation_enabled=True, deposit_pool_enabled=False,
                         cluster_bound_window_months=12),
}


def main():
    t0 = time.time()
    print(f"BASE: {BASE}\nSeeds: {SEEDS}\nRegimes: {list(REGIMES)}", flush=True)
    rows = []
    for name, kw in REGIMES.items():
        for seed in SEEDS:
            ts = time.time()
            p = Params(**{**BASE, **kw, "rng_seed": seed})
            df_sim = run_one_rep(seed, p)
            r = run_did_on_df(df_sim, f"{name} seed={seed}")
            if r is None:
                print(f"  {name} seed={seed}: [skipped, no events/pairs]", flush=True)
                continue
            betas = r["betas"]
            post = betas[betas["event_time"].between(1, 8)]
            peak48 = post[post["event_time"].between(4, 8)]
            peak_row = peak48.iloc[peak48["beta"].argmax()]
            k1_row = post[post["event_time"]==1].iloc[0]
            rows.append({
                "regime": name, "seed": seed,
                "n_pairs": r["n_pairs"], "n_events": r["n_events"],
                "did": r["did"]["did_estimate"], "did_p": r["did"]["p_value"],
                "pre_trend_p": r["pre_trend"].get("p_value", float("nan")),
                "k1_beta": k1_row["beta"], "k1_p": k1_row["p_value"],
                "peak_k": int(peak_row["event_time"]),
                "peak_beta": peak_row["beta"],
                "peak_p": peak_row["p_value"],
            })
            print(f"  {name} seed={seed}: pairs={r['n_pairs']} did={r['did']['did_estimate']:+.4f} "
                  f"k1={k1_row['beta']:+.4f} peak(k={int(peak_row['event_time'])})={peak_row['beta']:+.4f}"
                  f"  pre-p={r['pre_trend'].get('p_value', float('nan')):.3f}  ({time.time()-ts:.1f}s)",
                  flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "output", "matched_did_compare.csv"), index=False)

    n = len(SEEDS)
    print("\n--- Summary (mean ± SE across seeds) ---", flush=True)
    print(f"{'regime':<14} {'did':>18} {'k=1 β':>18} {'peak β':>18} {'pre-p':>8}", flush=True)
    for name in REGIMES:
        sub = df[df["regime"]==name]
        if len(sub) == 0:
            continue
        d_m, d_se = sub["did"].mean(), sub["did"].std(ddof=1)/np.sqrt(len(sub))
        k1_m, k1_se = sub["k1_beta"].mean(), sub["k1_beta"].std(ddof=1)/np.sqrt(len(sub))
        pk_m, pk_se = sub["peak_beta"].mean(), sub["peak_beta"].std(ddof=1)/np.sqrt(len(sub))
        prep = sub["pre_trend_p"].mean()
        print(f"  {name:<14} {d_m:+.4f} ± {d_se:.4f}  {k1_m:+.4f} ± {k1_se:.4f}  "
              f"{pk_m:+.4f} ± {pk_se:.4f}  {prep:.3f}", flush=True)

    # Pairwise prevention reductions
    print("\n--- Prediction test ---", flush=True)
    for comp in ["ON_with_pool", "W12_both"]:
        if comp not in df["regime"].values:
            continue
        a = df[df["regime"]=="ON_no_pool"].set_index("seed")
        b = df[df["regime"]==comp].set_index("seed")
        common = a.index.intersection(b.index)
        peak_diff = (a.loc[common, "peak_beta"] - b.loc[common, "peak_beta"])
        pm = peak_diff.mean(); pse = peak_diff.std(ddof=1)/np.sqrt(len(common))
        print(f"  Δpeak β  ({comp} reduction): {pm:+.4f} ± {pse:.4f}  "
              f"CI [{pm-1.96*pse:+.4f}, {pm+1.96*pse:+.4f}]  (n={len(common)})", flush=True)
    print(f"\nElapsed: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
