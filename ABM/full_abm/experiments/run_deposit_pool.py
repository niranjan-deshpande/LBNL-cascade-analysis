"""Deposit-pool counterfactual — 90-seed sweep (secondary counterfactual, MODEL.md §10).

3 regimes (OFF, ON_no_pool, ON_with_pool) on seeds 100-189, matching the
cluster-bound sweep's seed range for apples-to-apples comparison. Pool
absorbs the (1-α)·U network share on withdrawal, with age-ramp forfeiture.

Writes output/deposit_pool_{rawseeds,summary}.csv.
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
    "OFF":          dict(reallocation_enabled=False, deposit_pool_enabled=False),
    "ON_no_pool":   dict(reallocation_enabled=True,  deposit_pool_enabled=False),
    "ON_with_pool": dict(reallocation_enabled=True,  deposit_pool_enabled=True),
}
PJM_PROJECTS_PER_YEAR = 1800


def run_once(regime_name: str, regime_kwargs: dict, seed: int):
    p = Params(**{**BASE, **regime_kwargs, "rng_seed": seed})
    m = QueueModel(p).run()
    panel = pd.DataFrame(m.project_panel())
    rep = full_report(panel, p.horizon_months)
    ev = rep["event_study"].set_index("k_quarter")["diff"]
    total_wd = int((panel["status"] == "withdrawn").sum())
    dep = sum(amt for (_, kind, amt) in m.deposit_pool_log if kind == "deposit")
    absb = sum(amt for (_, kind, amt) in m.deposit_pool_log if kind == "absorb")
    return {"regime": regime_name, "seed": seed, "diff": ev,
            "total_wd": total_wd, "total_deposited": dep, "total_absorbed": absb,
            "pool_residual": m.deposit_pool}


def run_regime(name):
    print(f"\n=== {name} ===", flush=True)
    rows = []
    for s in SEEDS:
        t0 = time.time()
        r = run_once(name, REGIMES[name], s)
        rows.append(r)
        print(f"  seed={s}  wd={r['total_wd']}  dep={r['total_deposited']:.1e}  "
              f"abs={r['total_absorbed']:.1e}  ({time.time()-t0:.1f}s)", flush=True)
    return rows


def main():
    t0 = time.time()
    print(f"Config: {BASE}\nSeeds: n={len(SEEDS)} {SEEDS[0]}..{SEEDS[-1]}\nRegimes: {list(REGIMES)}", flush=True)
    results = {name: run_regime(name) for name in REGIMES}

    raw_rows = []
    for name, rows in results.items():
        for r in rows:
            row = {"regime": name, "seed": r["seed"],
                   "total_wd": r["total_wd"],
                   "total_deposited": r["total_deposited"],
                   "total_absorbed": r["total_absorbed"],
                   "pool_residual": r["pool_residual"]}
            for k, v in r["diff"].items():
                row[f"diff_k{int(k)}"] = v
            raw_rows.append(row)
    raw_df = pd.DataFrame(raw_rows)
    raw_path = os.path.join(ROOT, "output", "deposit_pool_rawseeds.csv")
    raw_df.to_csv(raw_path, index=False)

    D = {name: pd.DataFrame({r["seed"]: r["diff"] for r in results[name]}) for name in REGIMES}
    n = len(SEEDS)
    c_no_pool  = D["ON_no_pool"]   - D["OFF"]
    c_pool     = D["ON_with_pool"] - D["OFF"]
    out = pd.DataFrame({
        "OFF":            D["OFF"].mean(axis=1),
        "ON_no_pool":     D["ON_no_pool"].mean(axis=1),
        "ON_with_pool":   D["ON_with_pool"].mean(axis=1),
        "cascade":        c_no_pool.mean(axis=1),
        "cascade_se":     c_no_pool.std(axis=1, ddof=1) / np.sqrt(n),
        "cascade_pool":   c_pool.mean(axis=1),
        "cascade_pool_se":c_pool.std(axis=1, ddof=1) / np.sqrt(n),
    })
    out["cascade_lo"]      = out["cascade"]      - 1.96 * out["cascade_se"]
    out["cascade_hi"]      = out["cascade"]      + 1.96 * out["cascade_se"]
    out["cascade_pool_lo"] = out["cascade_pool"] - 1.96 * out["cascade_pool_se"]
    out["cascade_pool_hi"] = out["cascade_pool"] + 1.96 * out["cascade_pool_se"]
    out.index.name = "k"
    out_path = os.path.join(ROOT, "output", "deposit_pool_summary.csv")
    out.to_csv(out_path)

    print(f"\n========= Per-k cascade (n={n} seeds) =========", flush=True)
    print(out[["cascade","cascade_lo","cascade_hi","cascade_pool","cascade_pool_lo","cascade_pool_hi"]].round(4).to_string(), flush=True)

    peak_k = int(out.loc[4:8, "cascade"].idxmax())
    print(f"\n--- Peak cascade k={peak_k} ---", flush=True)
    for col_pref in ["cascade","cascade_pool"]:
        v = out.loc[peak_k, col_pref]; se = out.loc[peak_k, col_pref+"_se"]
        print(f"  {col_pref} = {v:+.4f} ± {se:.4f}  CI [{v-1.96*se:+.4f}, {v+1.96*se:+.4f}]", flush=True)

    wd = {name: np.array([r["total_wd"] for r in results[name]]) for name in REGIMES}
    prevented = wd["ON_no_pool"] - wd["ON_with_pool"]
    pm, pse = prevented.mean(), prevented.std(ddof=1)/np.sqrt(n)
    hy = BASE["horizon_months"]/12; pjm = PJM_PROJECTS_PER_YEAR / (BASE["arrivals_per_month"]*12)
    print(f"\n--- Withdrawal counts ---", flush=True)
    for name in REGIMES:
        a = wd[name]
        print(f"  {name:<14} = {a.mean():.1f} ± {a.std(ddof=1)/np.sqrt(n):.1f}", flush=True)
    print(f"\n--- Prevention (ON_no_pool − ON_with_pool) ---", flush=True)
    print(f"  prev/run  = {pm:+.1f} ± {pse:.1f}  CI [{pm-1.96*pse:+.1f}, {pm+1.96*pse:+.1f}]", flush=True)
    print(f"  prev/yr PJM = {pm/hy*pjm:+.2f} ± {pse/hy*pjm:.2f}  CI [{(pm-1.96*pse)/hy*pjm:+.1f}, {(pm+1.96*pse)/hy*pjm:+.1f}]", flush=True)
    print(f"\nElapsed: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
