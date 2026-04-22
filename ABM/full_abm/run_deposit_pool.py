"""Deposit-pool counterfactual.

Three regimes × N seeds:
  OFF           reallocation_enabled=False, deposit_pool_enabled=False
  ON_no_pool    reallocation_enabled=True,  deposit_pool_enabled=False
  ON_with_pool  reallocation_enabled=True,  deposit_pool_enabled=True

Cascade gauge = mini_event_study treated−control `diff` column. Per-k
mean ± SE over seeds; 95% CI = ±1.96·SE. Headline = reduction at the
peak k∈[4,8] with its CI. Peak is argmax over the window on the
point-estimate; the CI is the single-k CI, not a multiple-testing
adjusted bound (peak selection itself is a source of optimism).

Withdrawal prevention is reported directly from panel counts:
  prevented_per_year_abm = (wd_ON_no_pool − wd_ON_with_pool) / horizon_years
Scaled to PJM by (PJM arrivals/yr) / (ABM arrivals/yr) with CI.
"""
from __future__ import annotations
import os, sys, warnings, time
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import QueueModel, Params
from validation import full_report

SEEDS = tuple(range(100, 130))           # 30 seeds
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
    residual = m.deposit_pool
    return {
        "regime": regime_name,
        "seed": seed,
        "diff": ev,
        "total_wd": total_wd,
        "total_deposited": dep,
        "total_absorbed": absb,
        "pool_residual": residual,
    }


def run_regime(regime_name: str):
    print(f"\n=== {regime_name} ===")
    rows = []
    for s in SEEDS:
        t0 = time.time()
        r = run_once(regime_name, REGIMES[regime_name], s)
        rows.append(r)
        print(f"  seed={s}  total_wd={r['total_wd']}  "
              f"dep={r['total_deposited']:.2e}  abs={r['total_absorbed']:.2e}  "
              f"pool={r['pool_residual']:.2e}  ({time.time()-t0:.1f}s)")
    return rows


def diff_matrix(rows):
    """Return DataFrame indexed by k, columns = seeds, values = per-seed diff."""
    return pd.DataFrame({r["seed"]: r["diff"] for r in rows})


def main():
    t0 = time.time()
    print(f"Config: {BASE}")
    print(f"Seeds:  n={len(SEEDS)}  {SEEDS[0]}..{SEEDS[-1]}")
    print(f"Regimes: {list(REGIMES)}")

    results = {name: run_regime(name) for name in REGIMES}

    # Per-seed diff matrices (k × seed)
    D = {name: diff_matrix(results[name]) for name in REGIMES}
    n = len(SEEDS)

    # Cascade per seed = ON_no_pool - OFF; cascade_pool per seed = ON_with_pool - OFF
    cascade_seeds      = D["ON_no_pool"]   - D["OFF"]
    cascade_pool_seeds = D["ON_with_pool"] - D["OFF"]

    out = pd.DataFrame({
        "OFF":               D["OFF"].mean(axis=1),
        "ON_no_pool":        D["ON_no_pool"].mean(axis=1),
        "ON_with_pool":      D["ON_with_pool"].mean(axis=1),
        "cascade":           cascade_seeds.mean(axis=1),
        "cascade_se":        cascade_seeds.std(axis=1, ddof=1) / np.sqrt(n),
        "cascade_pool":      cascade_pool_seeds.mean(axis=1),
        "cascade_pool_se":   cascade_pool_seeds.std(axis=1, ddof=1) / np.sqrt(n),
    })
    out["cascade_lo"]      = out["cascade"]      - 1.96 * out["cascade_se"]
    out["cascade_hi"]      = out["cascade"]      + 1.96 * out["cascade_se"]
    out["cascade_pool_lo"] = out["cascade_pool"] - 1.96 * out["cascade_pool_se"]
    out["cascade_pool_hi"] = out["cascade_pool"] + 1.96 * out["cascade_pool_se"]
    # Reduction per seed, then mean±SE across seeds (propagates noise honestly).
    # Drop seeds where |cascade_seed| is too small to give a stable ratio.
    with np.errstate(divide="ignore", invalid="ignore"):
        red_seeds = 1.0 - cascade_pool_seeds / cascade_seeds.where(cascade_seeds.abs() > 1e-6)
    out["reduction_mean"] = red_seeds.mean(axis=1, skipna=True)
    out["reduction_se"]   = red_seeds.std(axis=1, ddof=1, skipna=True) / np.sqrt(red_seeds.notna().sum(axis=1))
    out.index.name = "k"

    print("\n========= Per-k cascade with 95% CIs (n={} seeds) =========".format(n))
    disp_cols = ["cascade", "cascade_se", "cascade_lo", "cascade_hi",
                 "cascade_pool", "cascade_pool_se", "reduction_mean", "reduction_se"]
    print(out[disp_cols].round(4).to_string())

    # Peak in the 4-8 quarter delayed window (argmax on point estimate)
    peak_k = int(out.loc[4:8, "cascade"].idxmax())
    c_nopool = out.loc[peak_k, "cascade"]
    c_nopool_se = out.loc[peak_k, "cascade_se"]
    c_pool   = out.loc[peak_k, "cascade_pool"]
    c_pool_se = out.loc[peak_k, "cascade_pool_se"]
    red_mean = out.loc[peak_k, "reduction_mean"]
    red_se   = out.loc[peak_k, "reduction_se"]

    print(f"\n--- Peak cascade (k = {peak_k}, picked by argmax k∈[4,8]) ---")
    print(f"  cascade_no_pool = {c_nopool:+.4f} ± {c_nopool_se:.4f}  "
          f"(95% CI [{c_nopool-1.96*c_nopool_se:+.4f}, {c_nopool+1.96*c_nopool_se:+.4f}])")
    print(f"  cascade_pool    = {c_pool:+.4f} ± {c_pool_se:.4f}  "
          f"(95% CI [{c_pool-1.96*c_pool_se:+.4f}, {c_pool+1.96*c_pool_se:+.4f}])")
    print(f"  reduction       = {red_mean:.3f} ± {red_se:.3f}   "
          f"(95% CI [{red_mean-1.96*red_se:+.3f}, {red_mean+1.96*red_se:+.3f}])")

    # Direct withdrawal-count comparison (per-seed diff, then mean ± SE)
    wd_no_pool   = np.array([r["total_wd"] for r in results["ON_no_pool"]])
    wd_with_pool = np.array([r["total_wd"] for r in results["ON_with_pool"]])
    wd_off       = np.array([r["total_wd"] for r in results["OFF"]])
    prevented_per_run = wd_no_pool - wd_with_pool
    prev_mean = prevented_per_run.mean()
    prev_se   = prevented_per_run.std(ddof=1) / np.sqrt(n)

    horizon_years = BASE["horizon_months"] / 12.0
    abm_arrivals_per_year = BASE["arrivals_per_month"] * 12.0
    pjm_scale = PJM_PROJECTS_PER_YEAR / abm_arrivals_per_year

    prev_per_year_abm = prev_mean / horizon_years
    prev_per_year_abm_se = prev_se / horizon_years
    prev_per_year_pjm = prev_per_year_abm * pjm_scale
    prev_per_year_pjm_se = prev_per_year_abm_se * pjm_scale

    print("\n--- Direct withdrawal-count prevention (no peak-k weighting) ---")
    print(f"  mean total_wd  OFF          = {wd_off.mean():.1f} ± {wd_off.std(ddof=1)/np.sqrt(n):.1f}")
    print(f"  mean total_wd  ON_no_pool   = {wd_no_pool.mean():.1f} ± {wd_no_pool.std(ddof=1)/np.sqrt(n):.1f}")
    print(f"  mean total_wd  ON_with_pool = {wd_with_pool.mean():.1f} ± {wd_with_pool.std(ddof=1)/np.sqrt(n):.1f}")
    print(f"  prevented_per_run = {prev_mean:+.1f} ± {prev_se:.1f}  "
          f"(95% CI [{prev_mean-1.96*prev_se:+.1f}, {prev_mean+1.96*prev_se:+.1f}])")
    print(f"  prevented/yr (ABM scale, {abm_arrivals_per_year:.0f} arrivals/yr) "
          f"= {prev_per_year_abm:+.2f} ± {prev_per_year_abm_se:.2f}")
    print(f"  prevented/yr (PJM scale, ×{pjm_scale:.2f}) "
          f"= {prev_per_year_pjm:+.2f} ± {prev_per_year_pjm_se:.2f}  "
          f"(95% CI [{prev_per_year_pjm-1.96*prev_per_year_pjm_se:+.1f}, "
          f"{prev_per_year_pjm+1.96*prev_per_year_pjm_se:+.1f}])")

    print("\n--- Pool diagnostics (ON_with_pool, mean over seeds) ---")
    dep_m = np.mean([r["total_deposited"] for r in results["ON_with_pool"]])
    abs_m = np.mean([r["total_absorbed"]  for r in results["ON_with_pool"]])
    res_m = np.mean([r["pool_residual"]   for r in results["ON_with_pool"]])
    print(f"  total_deposited = {dep_m:.3e}")
    print(f"  total_absorbed  = {abs_m:.3e}")
    print(f"  pool_residual   = {res_m:.3e}   ({100*res_m/dep_m:.2f}% of deposited)")
    if abs_m > dep_m + 1.0:
        print("  WARNING: absorbed > deposited — accounting bug.")

    out_path = os.path.join(HERE, "output", "deposit_pool_counterfactual.csv")
    out.to_csv(out_path)
    print(f"\nElapsed: {time.time()-t0:.1f}s")
    print(f"Saved:   {out_path}")


if __name__ == "__main__":
    main()
