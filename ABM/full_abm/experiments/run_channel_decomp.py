"""Channel decomposition: which channel drives cluster-bound prevention?

Compares W=12 applied to: both channels, local only, network only, vs
unbounded and OFF baselines. 5 regimes × 90 seeds (100-189).

Research question: the within-POI matched-DiD headline (+0.029) is what
the empirical brief measures. But the ABM's cluster-bound headline
(+30/yr at W=12) is a system-wide total-withdrawal-count effect.
Decomposition tells us which channel drives it — and whether the
dominant cascade pathway is within-POI local reallocation (α·U) or
across-POI network fanout ((1-α)·U landing at a target POI).

Preliminary 1-seed smoke test: network-only bounding at W=12 recovered
~100% of the full-both-channels effect on total count; local-only
recovered ~42%. If that holds at n=90, the empirical within-POI
matched-DiD target is measuring the smaller channel.
"""
from __future__ import annotations
import os, sys, warnings, time
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model import QueueModel, Params
from validation import full_report

SEEDS = tuple(range(100, 190))
BASE = dict(n_pois=2000, horizon_months=240, arrivals_per_month=25.0,
            sigma_poi=2.2e6, alpha_local=0.15)
REGIMES = {
    "OFF":           dict(reallocation_enabled=False),
    "ON_unbounded":  dict(reallocation_enabled=True),
    "W12_both":      dict(reallocation_enabled=True, cluster_bound_window_months=12),
    "W12_local_only": dict(reallocation_enabled=True,
                           cluster_bound_local_window_months=12,
                           cluster_bound_network_window_months=-1),
    "W12_net_only":   dict(reallocation_enabled=True,
                           cluster_bound_local_window_months=-1,
                           cluster_bound_network_window_months=12),
}
PJM_PROJECTS_PER_YEAR = 1800


def run_once(name, kw, seed):
    p = Params(**{**BASE, **kw, "rng_seed": seed})
    m = QueueModel(p).run()
    panel = pd.DataFrame(m.project_panel())
    rep = full_report(panel, p.horizon_months)
    ev = rep["event_study"].set_index("k_quarter")["diff"]
    return {"regime": name, "seed": seed, "diff": ev,
            "total_wd": int((panel["status"]=="withdrawn").sum()),
            "evaporated_U": float(m.evaporated_U)}


def main():
    t0 = time.time()
    print(f"Config: {BASE}\nSeeds: n={len(SEEDS)}\nRegimes: {list(REGIMES)}", flush=True)
    results = {}
    for name in REGIMES:
        print(f"\n=== {name} ===", flush=True)
        rows = []
        for s in SEEDS:
            ts = time.time()
            rows.append(run_once(name, REGIMES[name], s))
            print(f"  seed={s}  wd={rows[-1]['total_wd']}  ({time.time()-ts:.1f}s)", flush=True)
        results[name] = rows

    # raw CSV
    raw_rows = []
    for name, rows in results.items():
        for r in rows:
            row = {"regime": name, "seed": r["seed"],
                   "total_wd": r["total_wd"], "evaporated_U": r["evaporated_U"]}
            for k, v in r["diff"].items():
                row[f"diff_k{int(k)}"] = v
            raw_rows.append(row)
    pd.DataFrame(raw_rows).to_csv(os.path.join(ROOT, "output", "channel_decomp_rawseeds.csv"), index=False)

    # Counts
    wd = {name: np.array([r["total_wd"] for r in results[name]]) for name in REGIMES}
    n = len(SEEDS)
    hy = BASE["horizon_months"]/12
    pjm = PJM_PROJECTS_PER_YEAR / (BASE["arrivals_per_month"]*12)

    print("\n--- Total withdrawal counts (n=90) ---", flush=True)
    for name in REGIMES:
        a = wd[name]
        print(f"  {name:<18} = {a.mean():.1f} ± {a.std(ddof=1)/np.sqrt(n):.1f}", flush=True)

    print("\n--- Prevention vs ON_unbounded ---", flush=True)
    rows_prev = []
    for name in ["W12_both", "W12_local_only", "W12_net_only"]:
        prevented = wd["ON_unbounded"] - wd[name]
        pm, pse = prevented.mean(), prevented.std(ddof=1)/np.sqrt(n)
        pjm_m = pm/hy*pjm; pjm_se = pse/hy*pjm
        ci_lo = pjm_m - 1.96*pjm_se; ci_hi = pjm_m + 1.96*pjm_se
        print(f"  {name:<18}: prev/run={pm:+.1f} ± {pse:.1f}  "
              f"prev/yr PJM={pjm_m:+.2f} ± {pjm_se:.2f} CI [{ci_lo:+.1f}, {ci_hi:+.1f}]", flush=True)
        rows_prev.append({"regime": name, "prev_per_run_mean": pm, "prev_per_run_se": pse,
                          "prev_per_year_pjm_mean": pjm_m, "prev_per_year_pjm_se": pjm_se,
                          "prev_per_year_pjm_lo": ci_lo, "prev_per_year_pjm_hi": ci_hi})
    pd.DataFrame(rows_prev).to_csv(os.path.join(ROOT, "output", "channel_decomp_prevention.csv"), index=False)

    # Event-study: cascade vs OFF for each regime
    D = {name: pd.DataFrame({r["seed"]: r["diff"] for r in results[name]}) for name in REGIMES}
    print("\n--- Cascade (regime - OFF) event-study peak ---", flush=True)
    cascade = {name: D[name] - D["OFF"] for name in REGIMES if name != "OFF"}
    # argmax on ON_unbounded over k in [4, 8]
    cu = cascade["ON_unbounded"].mean(axis=1)
    peak_k = int(cu.loc[4:8].idxmax())
    print(f"  peak_k = {peak_k}", flush=True)
    rows_cas = []
    for name in ["ON_unbounded", "W12_both", "W12_local_only", "W12_net_only"]:
        c = cascade[name]
        v = c.mean(axis=1).loc[peak_k]
        se = c.std(axis=1, ddof=1).loc[peak_k] / np.sqrt(n)
        print(f"  {name:<18} = {v:+.4f} ± {se:.4f}  CI [{v-1.96*se:+.4f}, {v+1.96*se:+.4f}]", flush=True)
        rows_cas.append({"regime": name, "peak_k": peak_k,
                         "cascade_mean": v, "cascade_se": se,
                         "cascade_lo": v-1.96*se, "cascade_hi": v+1.96*se})
    pd.DataFrame(rows_cas).to_csv(os.path.join(ROOT, "output", "channel_decomp_cascade.csv"), index=False)

    print(f"\nElapsed: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
