"""network_distance_scale sensitivity: do the policy counterfactuals survive
when the spatial decay of the network-share kernel is changed?

The exponential kernel W_pq = exp(-||x_p - x_q|| / s) governs how the
network-upgrade share (1 - alpha_local)*U fans out across other POIs. The
default s = 0.3 is a deliberate but loosely calibrated choice; smaller s
concentrates network costs on nearby POIs, larger s diffuses them ~uniformly
over the unit square. If the cluster-bound headline is structural (about the
*entry-time* dimension, not the spatial dimension) it should be roughly flat
across s. If instead it's tightly coupled to a particular spread shape, the
+30/yr PJM W=12 number is fragile.

Sweep: s ∈ {0.1, 0.2, 0.3, 0.5, 1.0, 5.0}  ×  {OFF, ON_unbounded, W12_both}
       × 30 seeds (100..129).

Mirrors run_alpha_sensitivity.py. OFF runs are re-run per s for symmetry,
but should yield identical total_wd across scales at fixed seed (no
reallocation fires → kernel is unused). That invariance is a free sanity
check on the harness.

Writes output/kernel_sensitivity_{raw,summary}.csv.
"""
from __future__ import annotations
import os, sys, warnings, time
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model import QueueModel, Params

SEEDS = tuple(range(100, 130))  # 30 seeds — same as run_alpha_sensitivity.py
BASE = dict(n_pois=2000, horizon_months=240, arrivals_per_month=25.0,
            sigma_poi=2.2e6, alpha_local=0.15)
SCALES = [0.1, 0.2, 0.3, 0.5, 1.0, 5.0]
REGIME_KWARGS = {
    "OFF":          dict(reallocation_enabled=False),
    "ON_unbounded": dict(reallocation_enabled=True),
    "W12_both":     dict(reallocation_enabled=True, cluster_bound_window_months=12),
}
PJM_PROJECTS_PER_YEAR = 1800


def run_once(scale, regime_name, regime_kwargs, seed):
    p = Params(**{**BASE, "network_distance_scale": scale,
                  **regime_kwargs, "rng_seed": seed})
    m = QueueModel(p).run()
    panel = pd.DataFrame(m.project_panel())
    return {"scale": scale, "regime": regime_name, "seed": seed,
            "total_wd": int((panel["status"] == "withdrawn").sum()),
            "evaporated_U": float(m.evaporated_U)}


def main():
    t0 = time.time()
    print(f"BASE: {BASE}\nSCALES: {SCALES}\nSeeds: n={len(SEEDS)}", flush=True)
    rows = []
    for scale in SCALES:
        for name, kw in REGIME_KWARGS.items():
            print(f"\n=== scale={scale}  {name} ===", flush=True)
            for s in SEEDS:
                ts = time.time()
                r = run_once(scale, name, kw, s)
                rows.append(r)
                print(f"  seed={s} wd={r['total_wd']} evap={r['evaporated_U']:.2e} "
                      f"({time.time()-ts:.1f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "output", "kernel_sensitivity_raw.csv"), index=False)

    print("\n--- Summary: total_wd (mean ± SE) ---", flush=True)
    summary = df.groupby(["scale", "regime"])["total_wd"].agg(
        ["mean", "std", "count"]).reset_index()
    summary["se"] = summary["std"] / np.sqrt(summary["count"])
    print(summary.to_string(index=False), flush=True)

    print("\n--- W=12 prevention vs ON_unbounded, by kernel scale ---", flush=True)
    hy = BASE["horizon_months"] / 12
    pjm = PJM_PROJECTS_PER_YEAR / (BASE["arrivals_per_month"] * 12)
    n = len(SEEDS)
    out_rows = []
    for scale in SCALES:
        sub = df[df["scale"] == scale].pivot(
            index="seed", columns="regime", values="total_wd")
        prev = sub["ON_unbounded"] - sub["W12_both"]
        pm, pse = prev.mean(), prev.std(ddof=1) / np.sqrt(n)
        pjm_m, pjm_se = pm / hy * pjm, pse / hy * pjm
        full_cascade = sub["ON_unbounded"] - sub["OFF"]
        fc_m = full_cascade.mean()
        fc_se = full_cascade.std(ddof=1) / np.sqrt(n)
        print(f"  s={scale:<4}: cascade ON-OFF = {fc_m:+.1f} ± {fc_se:.1f}  "
              f"prev/run = {pm:+.1f} ± {pse:.1f}  "
              f"prev/yr PJM = {pjm_m:+.2f} ± {pjm_se:.2f} "
              f"CI [{pjm_m-1.96*pjm_se:+.1f}, {pjm_m+1.96*pjm_se:+.1f}]  "
              f"({100*pm/fc_m:.0f}% of full cascade captured)", flush=True)
        out_rows.append({
            "scale": scale,
            "full_cascade_per_run_mean": fc_m, "full_cascade_per_run_se": fc_se,
            "prev_per_run_mean": pm, "prev_per_run_se": pse,
            "prev_per_year_pjm": pjm_m, "prev_per_year_pjm_se": pjm_se,
            "prev_per_year_pjm_lo": pjm_m - 1.96 * pjm_se,
            "prev_per_year_pjm_hi": pjm_m + 1.96 * pjm_se,
            "pct_cascade_captured": 100 * pm / fc_m,
        })
    pd.DataFrame(out_rows).to_csv(
        os.path.join(ROOT, "output", "kernel_sensitivity_summary.csv"),
        index=False)
    print(f"\nElapsed: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
