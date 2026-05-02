"""network_fanout (k) sensitivity: do the policy counterfactuals survive
when the number of network-share recipient POIs is changed?

`network_fanout` k determines how many *other* POIs each withdrawal's
network-upgrade share (1 - alpha_local)*U gets distributed across. Default
k = 20. Each recipient POI's piece is U_net / k, so k controls the magnitude
of the per-POI cost shock (small k → big concentrated shocks, large k →
dilute shocks across many POIs).

Plausible range: from k=2 (extremely concentrated) to k=200 (10% of POIs per
upgrade). PJM's actual reallocation matrices likely span tens to a few
hundred recipient projects, depending on upgrade scope.

Sweep: k ∈ {2, 5, 10, 20, 50, 100, 200}
       × {OFF, ON_unbounded, W12_both} × 30 seeds (100..129).

Mirrors run_kernel_sensitivity.py. OFF total_wd is invariant to k.

Writes output/fanout_sensitivity_{raw,summary}.csv.
"""
from __future__ import annotations
import os, sys, warnings, time
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model import QueueModel, Params

SEEDS = tuple(range(100, 130))
BASE = dict(n_pois=2000, horizon_months=240, arrivals_per_month=25.0,
            sigma_poi=2.2e6, alpha_local=0.15)
FANOUTS = [2, 5, 10, 20, 50, 100, 200]
REGIME_KWARGS = {
    "OFF":          dict(reallocation_enabled=False),
    "ON_unbounded": dict(reallocation_enabled=True),
    "W12_both":     dict(reallocation_enabled=True, cluster_bound_window_months=12),
}
PJM_PROJECTS_PER_YEAR = 1800


def run_once(k, regime_name, regime_kwargs, seed):
    p = Params(**{**BASE, "network_fanout": k,
                  **regime_kwargs, "rng_seed": seed})
    m = QueueModel(p).run()
    panel = pd.DataFrame(m.project_panel())
    return {"k": k, "regime": regime_name, "seed": seed,
            "total_wd": int((panel["status"] == "withdrawn").sum()),
            "evaporated_U": float(m.evaporated_U)}


def main():
    t0 = time.time()
    print(f"BASE: {BASE}\nFANOUTS: {FANOUTS}\nSeeds: n={len(SEEDS)}", flush=True)
    rows = []
    for k in FANOUTS:
        for name, kw in REGIME_KWARGS.items():
            print(f"\n=== k={k}  {name} ===", flush=True)
            for s in SEEDS:
                ts = time.time()
                r = run_once(k, name, kw, s)
                rows.append(r)
                print(f"  seed={s} wd={r['total_wd']} evap={r['evaporated_U']:.2e} "
                      f"({time.time()-ts:.1f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "output", "fanout_sensitivity_raw.csv"),
              index=False)

    print("\n--- Summary: total_wd (mean ± SE) ---", flush=True)
    summary = df.groupby(["k", "regime"])["total_wd"].agg(
        ["mean", "std", "count"]).reset_index()
    summary["se"] = summary["std"] / np.sqrt(summary["count"])
    print(summary.to_string(index=False), flush=True)

    print("\n--- W=12 prevention vs ON_unbounded, by k ---", flush=True)
    hy = BASE["horizon_months"] / 12
    pjm = PJM_PROJECTS_PER_YEAR / (BASE["arrivals_per_month"] * 12)
    n = len(SEEDS)
    out_rows = []
    for k in FANOUTS:
        sub = df[df["k"] == k].pivot(
            index="seed", columns="regime", values="total_wd")
        prev = sub["ON_unbounded"] - sub["W12_both"]
        pm, pse = prev.mean(), prev.std(ddof=1) / np.sqrt(n)
        pjm_m, pjm_se = pm / hy * pjm, pse / hy * pjm
        full_cascade = sub["ON_unbounded"] - sub["OFF"]
        fc_m = full_cascade.mean()
        fc_se = full_cascade.std(ddof=1) / np.sqrt(n)
        pct = 100 * pm / fc_m if fc_m != 0 else float("nan")
        print(f"  k={k:<4}: cascade ON-OFF = {fc_m:+.1f} ± {fc_se:.1f}  "
              f"prev/run = {pm:+.1f} ± {pse:.1f}  "
              f"prev/yr PJM = {pjm_m:+.2f} ± {pjm_se:.2f} "
              f"CI [{pjm_m-1.96*pjm_se:+.1f}, {pjm_m+1.96*pjm_se:+.1f}]  "
              f"({pct:.0f}% of full cascade captured)", flush=True)
        out_rows.append({
            "k": k,
            "full_cascade_per_run_mean": fc_m, "full_cascade_per_run_se": fc_se,
            "prev_per_run_mean": pm, "prev_per_run_se": pse,
            "prev_per_year_pjm": pjm_m, "prev_per_year_pjm_se": pjm_se,
            "prev_per_year_pjm_lo": pjm_m - 1.96 * pjm_se,
            "prev_per_year_pjm_hi": pjm_m + 1.96 * pjm_se,
            "pct_cascade_captured": pct,
        })
    pd.DataFrame(out_rows).to_csv(
        os.path.join(ROOT, "output", "fanout_sensitivity_summary.csv"),
        index=False)
    print(f"\nElapsed: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
