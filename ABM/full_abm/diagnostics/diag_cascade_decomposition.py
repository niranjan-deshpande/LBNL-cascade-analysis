"""Cascade decomposition: split matched-DiD event-study coefficients into
cascade (ON − OFF) and non-cascade (OFF alone) components.

Step 1 showed σ_poi doesn't move OFF_k1 — the 3× k-coefficient magnitude gap
isn't the η channel. This diagnostic isolates what IS driving it. OFF β_k is
the non-cascade component (selection + matching bias when the empirical
estimator is run on simulated panels); ON β_k − OFF β_k is the cascade
component (mechanical reallocation effect).

Decision rule:
  (ON − OFF) at peak ≈ +0.03  →  cascade calibrated correctly, gap is selection.
  (ON − OFF) at peak ≈ +0.06-0.09 →  cascade is genuinely too strong; tune α/φ.
"""
from __future__ import annotations
import os, sys, warnings, time
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model import QueueModel, Params
from validate_matched_did import abm_panel_to_lbnl_schema, run_did_on_df

SEEDS = tuple(range(100, 130))          # 30 seeds for defensible SEs
BASE = dict(n_pois=2000, horizon_months=240, arrivals_per_month=25.0,
            sigma_poi=2.2e6, alpha_local=0.15)


def beta_vector(reallocation_enabled: bool, seed: int):
    p = Params(**{**BASE, "rng_seed": seed,
                  "reallocation_enabled": reallocation_enabled})
    m = QueueModel(p).run()
    panel = pd.DataFrame(m.project_panel())
    df_sim = abm_panel_to_lbnl_schema(panel, p.horizon_months)
    label = f"{'ON ' if reallocation_enabled else 'OFF'} seed={seed}"
    r = run_did_on_df(df_sim, label)
    if r is None:
        return None
    return pd.DataFrame(r["betas"]).set_index("event_time")["beta"]


def run_regime(reallocation_enabled: bool) -> pd.DataFrame:
    cols = {}
    for s in SEEDS:
        b = beta_vector(reallocation_enabled, s)
        if b is not None:
            cols[s] = b
    return pd.DataFrame(cols)


def main():
    t0 = time.time()
    print(f"Config: {BASE}")
    print(f"Seeds:  n={len(SEEDS)}  {SEEDS[0]}..{SEEDS[-1]}\n")

    print("=== OFF regime ===")
    off_raw = run_regime(False)
    print("\n=== ON regime ===")
    on_raw = run_regime(True)

    # Align on the intersection of seeds that returned valid betas in both regimes.
    common = [s for s in off_raw.columns if s in on_raw.columns]
    off = off_raw[common]
    on  = on_raw[common]
    n = len(common)
    cascade_seeds = on - off   # k × seed

    out = pd.DataFrame({
        "OFF_mean":   off.mean(axis=1),
        "OFF_sd":     off.std(axis=1, ddof=1),
        "OFF_se":     off.std(axis=1, ddof=1) / np.sqrt(n),
        "ON_mean":    on.mean(axis=1),
        "ON_sd":      on.std(axis=1, ddof=1),
        "ON_se":      on.std(axis=1, ddof=1) / np.sqrt(n),
        "cascade":    cascade_seeds.mean(axis=1),
        "cascade_se": cascade_seeds.std(axis=1, ddof=1) / np.sqrt(n),
    })
    out["cascade_lo"] = out["cascade"] - 1.96 * out["cascade_se"]
    out["cascade_hi"] = out["cascade"] + 1.96 * out["cascade_se"]
    out["ci_excludes_zero"] = (out["cascade_lo"] > 0) | (out["cascade_hi"] < 0)
    out.index.name = "k"

    print(f"\n========= Cascade decomposition ({n} seeds, ±1.96·SE) =========")
    disp = out[["OFF_mean", "OFF_se", "ON_mean", "ON_se",
                "cascade", "cascade_se", "cascade_lo", "cascade_hi",
                "ci_excludes_zero"]]
    print(disp.round(4).to_string())

    # Empirical-target comparison (no peak-picking on noisy data — report full window).
    print("\n--- Cascade (ON−OFF) at each k with 95% CI ---")
    for k in sorted(out.index):
        row = out.loc[k]
        star = " *" if row["ci_excludes_zero"] else ""
        print(f"  k={k:>3}  cascade={row['cascade']:+.4f} ± {row['cascade_se']:.4f}   "
              f"CI [{row['cascade_lo']:+.4f}, {row['cascade_hi']:+.4f}]{star}")

    print("\n--- Comparison to empirical targets (NOT bootstrap SEs) ---")
    print(f"  k=1 empirical: -0.038   ABM k=1 cascade: {out.loc[1, 'cascade']:+.4f} ± {out.loc[1, 'cascade_se']:.4f}")
    peak_k = int(out.loc[4:8, "cascade"].idxmax())
    print(f"  k=5 empirical: +0.029   ABM k={peak_k} cascade (argmax k∈[4,8]): "
          f"{out.loc[peak_k, 'cascade']:+.4f} ± {out.loc[peak_k, 'cascade_se']:.4f}")
    print("  Note: peak picked by argmax is upward-biased; the per-k CIs above are the honest summary.")

    out_path = os.path.join(ROOT, "output", "cascade_decomposition.csv")
    out.to_csv(out_path)
    print(f"\nElapsed: {time.time()-t0:.1f}s")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
