"""α_local sensitivity: does the cluster-bound headline hold at other α?

BASE α=0.15 is the default calibration (notes_for_self/notes.md suggests
0.10-0.15 from structural decomposition). This script tests whether the
+30/yr PJM W=12 prevention survives at α=0.05 and α=0.30 — i.e., in
regimes where the local/network split is very different.

3 α values × 3 regimes (OFF, ON_unbounded, W=12 both channels) × 30 seeds.
"""
from __future__ import annotations
import os, sys, warnings, time
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model import QueueModel, Params

SEEDS = tuple(range(100, 130))  # 30 seeds; we're looking for big effects, not fine CIs
BASE = dict(n_pois=2000, horizon_months=240, arrivals_per_month=25.0,
            sigma_poi=2.2e6)
ALPHAS = [0.05, 0.15, 0.30]
REGIME_KWARGS = {
    "OFF":          dict(reallocation_enabled=False),
    "ON_unbounded": dict(reallocation_enabled=True),
    "W12_both":     dict(reallocation_enabled=True, cluster_bound_window_months=12),
}
PJM_PROJECTS_PER_YEAR = 1800


def run_once(alpha, regime_name, regime_kwargs, seed):
    p = Params(**{**BASE, "alpha_local": alpha, **regime_kwargs, "rng_seed": seed})
    m = QueueModel(p).run()
    panel = pd.DataFrame(m.project_panel())
    return {"alpha": alpha, "regime": regime_name, "seed": seed,
            "total_wd": int((panel["status"]=="withdrawn").sum())}


def main():
    t0 = time.time()
    print(f"BASE: {BASE}\nALPHAS: {ALPHAS}\nSeeds: n={len(SEEDS)}", flush=True)
    rows = []
    for alpha in ALPHAS:
        for name, kw in REGIME_KWARGS.items():
            print(f"\n=== alpha={alpha}  {name} ===", flush=True)
            for s in SEEDS:
                ts = time.time()
                r = run_once(alpha, name, kw, s)
                rows.append(r)
                print(f"  seed={s} wd={r['total_wd']} ({time.time()-ts:.1f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "output", "alpha_sensitivity_raw.csv"), index=False)

    print("\n--- Summary: total_wd (mean ± SE) ---", flush=True)
    summary = df.groupby(["alpha", "regime"])["total_wd"].agg(["mean", "std", "count"]).reset_index()
    summary["se"] = summary["std"] / np.sqrt(summary["count"])
    print(summary.to_string(index=False), flush=True)

    print("\n--- W=12 prevention vs ON_unbounded, by α ---", flush=True)
    hy = BASE["horizon_months"]/12
    pjm = PJM_PROJECTS_PER_YEAR / (BASE["arrivals_per_month"]*12)
    n = len(SEEDS)
    out_rows = []
    for alpha in ALPHAS:
        sub = df[df["alpha"]==alpha].pivot(index="seed", columns="regime", values="total_wd")
        prev = sub["ON_unbounded"] - sub["W12_both"]
        pm, pse = prev.mean(), prev.std(ddof=1)/np.sqrt(n)
        pjm_m = pm/hy*pjm; pjm_se = pse/hy*pjm
        full_cascade = sub["ON_unbounded"] - sub["OFF"]
        fc_m = full_cascade.mean()
        print(f"  α={alpha}: prev/run = {pm:+.1f} ± {pse:.1f}  "
              f"prev/yr PJM = {pjm_m:+.2f} ± {pjm_se:.2f} "
              f"CI [{pjm_m-1.96*pjm_se:+.1f}, {pjm_m+1.96*pjm_se:+.1f}]  "
              f"(of full cascade {fc_m:+.0f}/run = {100*pm/fc_m:.0f}% captured)", flush=True)
        out_rows.append({"alpha": alpha, "prev_per_run_mean": pm, "prev_per_run_se": pse,
                         "prev_per_year_pjm": pjm_m, "prev_per_year_pjm_se": pjm_se,
                         "full_cascade_per_run": fc_m,
                         "pct_cascade_captured": 100*pm/fc_m})
    pd.DataFrame(out_rows).to_csv(os.path.join(ROOT, "output", "alpha_sensitivity_summary.csv"), index=False)
    print(f"\nElapsed: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
