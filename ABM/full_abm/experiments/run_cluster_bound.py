"""Cluster-bounded reallocation — 90-seed sweep (primary counterfactual, MODEL.md §11).

9 regimes: OFF + {W=12,18,24,36,48,60,72} + ON_unbounded. For each W, both
local (same-POI) and network (fanned-out) reallocation channels are bounded
to recipients whose t_entry is within W months of the withdrawer's. Maps
onto FERC Order 2023's cluster-study scope reform.

Writes output/cluster_bound_{rawseeds,summary,prevention}.csv.
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

SEEDS = tuple(range(100, 190))           # 90 seeds
BASE = dict(n_pois=2000, horizon_months=240, arrivals_per_month=25.0,
            sigma_poi=2.2e6, alpha_local=0.15)
W_VALUES = [12, 18, 24, 36, 48, 60, 72]
REGIMES = {
    "OFF": dict(reallocation_enabled=False, cluster_bound_window_months=-1),
    **{f"W={w}": dict(reallocation_enabled=True, cluster_bound_window_months=w)
       for w in W_VALUES},
    "ON_unbounded": dict(reallocation_enabled=True, cluster_bound_window_months=-1),
}
PJM_PROJECTS_PER_YEAR = 1800


def run_once(regime_name: str, regime_kwargs: dict, seed: int):
    p = Params(**{**BASE, **regime_kwargs, "rng_seed": seed})
    m = QueueModel(p).run()
    panel = pd.DataFrame(m.project_panel())
    rep = full_report(panel, p.horizon_months)
    ev = rep["event_study"].set_index("k_quarter")["diff"]
    total_wd = int((panel["status"] == "withdrawn").sum())
    return {
        "regime": regime_name,
        "seed": seed,
        "diff": ev,
        "total_wd": total_wd,
        "evaporated_U": float(m.evaporated_U),
    }


def run_regime(regime_name: str):
    print(f"\n=== {regime_name} ===", flush=True)
    rows = []
    for s in SEEDS:
        t0 = time.time()
        r = run_once(regime_name, REGIMES[regime_name], s)
        rows.append(r)
        print(f"  seed={s}  total_wd={r['total_wd']}  "
              f"evap={r['evaporated_U']:.2e}  ({time.time()-t0:.1f}s)",
              flush=True)
    return rows


def diff_matrix(rows):
    return pd.DataFrame({r["seed"]: r["diff"] for r in rows})


def main():
    t0 = time.time()
    print(f"Config: {BASE}", flush=True)
    print(f"Seeds:  n={len(SEEDS)}  {SEEDS[0]}..{SEEDS[-1]}", flush=True)
    print(f"Regimes: {list(REGIMES)}", flush=True)

    results = {name: run_regime(name) for name in REGIMES}

    # Per-seed raw CSV (one row per regime × seed, with k as columns)
    raw_rows = []
    for name, rows in results.items():
        for r in rows:
            row = {
                "regime": name, "seed": r["seed"],
                "total_wd": r["total_wd"],
                "evaporated_U": r["evaporated_U"],
            }
            for k, v in r["diff"].items():
                row[f"diff_k{int(k)}"] = v
            raw_rows.append(row)
    raw_df = pd.DataFrame(raw_rows)
    raw_path = os.path.join(ROOT, "output", "cluster_bound_rawseeds.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f"Saved raw:  {raw_path}", flush=True)

    # Per-seed diff matrices
    D = {name: diff_matrix(results[name]) for name in REGIMES}
    n = len(SEEDS)
    cascade_unbounded = D["ON_unbounded"] - D["OFF"]

    summary_cols = {
        "OFF":          D["OFF"].mean(axis=1),
        "ON_unbounded": D["ON_unbounded"].mean(axis=1),
        "cascade_unbounded":    cascade_unbounded.mean(axis=1),
        "cascade_unbounded_se": cascade_unbounded.std(axis=1, ddof=1) / np.sqrt(n),
    }
    for w in W_VALUES:
        regime = f"W={w}"
        cascade_w = D[regime] - D["OFF"]
        summary_cols[f"{regime}_mean"] = D[regime].mean(axis=1)
        summary_cols[f"cascade_{regime}"]    = cascade_w.mean(axis=1)
        summary_cols[f"cascade_{regime}_se"] = cascade_w.std(axis=1, ddof=1) / np.sqrt(n)
    out = pd.DataFrame(summary_cols)
    out.index.name = "k"
    out_path = os.path.join(ROOT, "output", "cluster_bound_summary.csv")
    out.to_csv(out_path)

    print(f"\n========= Per-k cascade by W (n={n} seeds, mean only) =========", flush=True)
    cascade_cols = ["cascade_unbounded"] + [f"cascade_W={w}" for w in W_VALUES]
    print(out[cascade_cols].round(4).to_string(), flush=True)

    peak_k = int(out.loc[4:8, "cascade_unbounded"].idxmax())
    print(f"\n--- Peak cascade (k={peak_k}, argmax k∈[4,8] on unbounded) ---", flush=True)
    c_u = out.loc[peak_k, "cascade_unbounded"]
    c_u_se = out.loc[peak_k, "cascade_unbounded_se"]
    print(f"  cascade_unbounded = {c_u:+.4f} ± {c_u_se:.4f}  "
          f"(95% CI [{c_u-1.96*c_u_se:+.4f}, {c_u+1.96*c_u_se:+.4f}])",
          flush=True)
    for w in W_VALUES:
        c = out.loc[peak_k, f"cascade_W={w}"]
        c_se = out.loc[peak_k, f"cascade_W={w}_se"]
        print(f"  cascade_W={w:<3}       = {c:+.4f} ± {c_se:.4f}  "
              f"(95% CI [{c-1.96*c_se:+.4f}, {c+1.96*c_se:+.4f}])",
              flush=True)

    wd = {name: np.array([r["total_wd"] for r in results[name]]) for name in REGIMES}
    horizon_years = BASE["horizon_months"] / 12.0
    abm_arrivals_per_year = BASE["arrivals_per_month"] * 12.0
    pjm_scale = PJM_PROJECTS_PER_YEAR / abm_arrivals_per_year

    print("\n--- Total withdrawal counts (mean ± SE across seeds) ---", flush=True)
    print(f"  OFF          = {wd['OFF'].mean():.1f} ± {wd['OFF'].std(ddof=1)/np.sqrt(n):.1f}",
          flush=True)
    print(f"  ON_unbounded = {wd['ON_unbounded'].mean():.1f} ± {wd['ON_unbounded'].std(ddof=1)/np.sqrt(n):.1f}",
          flush=True)
    for w in W_VALUES:
        a = wd[f"W={w}"]
        print(f"  W={w:<3}        = {a.mean():.1f} ± {a.std(ddof=1)/np.sqrt(n):.1f}",
              flush=True)

    print("\n--- Withdrawal prevention per W (vs ON_unbounded) ---", flush=True)
    print(f"{'regime':<8} {'prev/run':>15} {'prev/yr ABM':>14} {'prev/yr PJM (CI)':>28}",
          flush=True)
    rows_prev = []
    for w in W_VALUES:
        prevented = wd["ON_unbounded"] - wd[f"W={w}"]
        pm = prevented.mean()
        pse = prevented.std(ddof=1) / np.sqrt(n)
        pyr_abm = pm / horizon_years
        pyr_abm_se = pse / horizon_years
        pyr_pjm = pyr_abm * pjm_scale
        pyr_pjm_se = pyr_abm_se * pjm_scale
        ci_lo = pyr_pjm - 1.96 * pyr_pjm_se
        ci_hi = pyr_pjm + 1.96 * pyr_pjm_se
        print(f"  W={w:<3}  {pm:+7.1f} ± {pse:4.1f}   {pyr_abm:+5.2f} ± {pyr_abm_se:4.2f}"
              f"   {pyr_pjm:+6.2f} ± {pyr_pjm_se:5.2f}  [{ci_lo:+.1f}, {ci_hi:+.1f}]",
              flush=True)
        rows_prev.append({
            "W": w, "prev_per_run_mean": pm, "prev_per_run_se": pse,
            "prev_per_year_abm": pyr_abm, "prev_per_year_abm_se": pyr_abm_se,
            "prev_per_year_pjm": pyr_pjm, "prev_per_year_pjm_se": pyr_pjm_se,
            "prev_per_year_pjm_lo": ci_lo, "prev_per_year_pjm_hi": ci_hi,
        })
    prev_df = pd.DataFrame(rows_prev)
    prev_path = os.path.join(ROOT, "output", "cluster_bound_prevention.csv")
    prev_df.to_csv(prev_path, index=False)

    print("\n--- Evaporation diagnostic (mean $ across seeds) ---", flush=True)
    for name in REGIMES:
        evap = np.mean([r["evaporated_U"] for r in results[name]])
        print(f"  {name:<14} evaporated_U = {evap:.2e}", flush=True)

    print(f"\nElapsed: {time.time()-t0:.1f}s", flush=True)
    print(f"Saved:   {out_path}", flush=True)
    print(f"Saved:   {prev_path}", flush=True)


if __name__ == "__main__":
    main()
