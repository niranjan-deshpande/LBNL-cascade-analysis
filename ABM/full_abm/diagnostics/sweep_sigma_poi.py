"""Step 1: σ_poi sweep to calibrate the shared-conditions channel.

Hypothesis: σ_poi (AR(1) shared shock) is ~3× too large. The OFF-regime
diagnostic (diag_off_k1) confirmed that OFF k=1 = −0.108 vs empirical
−0.038, and α doesn't affect OFF dynamics. Sweep σ_poi downward to fix
OFF k=1 first (pure η channel, no cascade contamination); if VR drops
below target, compensate with σ_between (sweep_sigma_between.py). α stays
at 0.15 unless the peak is still too high after fixing η.

Metrics tracked (5 reps per σ_poi):
  OFF k=1        target ≈ −0.038  (pure η channel)
  ON  k=1        target ≈ −0.038
  ON  peak k=4-8 target ≈ +0.029
  ON  DiD        target ≈ −0.008  (near null)
  OFF VR         target ≈ 1.6
  ON  completion target ≈ 0.20
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
from validate_matched_did import abm_panel_to_lbnl_schema, run_did_on_df

SIGMAS = [0.8e6, 1.0e6, 1.2e6, 1.5e6, 2.2e6]
SEEDS = tuple(range(100, 105))
BASE = dict(n_pois=2000, horizon_months=240, arrivals_per_month=25.0,
            alpha_local=0.15)


def simulate(reallocation_enabled: bool, seed: int, sigma_poi: float):
    p = Params(**{**BASE, "rng_seed": seed, "sigma_poi": sigma_poi,
                  "reallocation_enabled": reallocation_enabled})
    m = QueueModel(p).run()
    panel = pd.DataFrame(m.project_panel())
    return panel, p


def did_result(panel, p, label):
    df_sim = abm_panel_to_lbnl_schema(panel, p.horizon_months)
    return run_did_on_df(df_sim, label)


def main():
    rows = []
    for sp in SIGMAS:
        t0 = time.time()
        print(f"\n================  sigma_poi = {sp:.2e}  ================")
        off_k1, on_k1, on_peak, on_did, off_vr, on_compl = [], [], [], [], [], []
        for seed in SEEDS:
            # OFF regime → OFF k=1, OFF VR
            panel_off, p_off = simulate(False, seed, sp)
            rep_off = full_report(panel_off, p_off.horizon_months)
            off_vr.append(rep_off["variance_ratio"]["vr"])
            r_off = did_result(panel_off, p_off, f"OFF sp={sp:.1e} seed={seed}")
            if r_off is not None:
                betas_off = pd.DataFrame(r_off["betas"]).set_index("event_time")["beta"]
                off_k1.append(betas_off.get(1, np.nan))
            # ON regime → ON k=1, ON peak, ON DiD, ON completion
            panel_on, p_on = simulate(True, seed, sp)
            rep_on = full_report(panel_on, p_on.horizon_months)
            on_compl.append(rep_on["completion_rate"])
            r_on = did_result(panel_on, p_on, f"ON  sp={sp:.1e} seed={seed}")
            if r_on is not None:
                betas_on = pd.DataFrame(r_on["betas"]).set_index("event_time")["beta"]
                on_k1.append(betas_on.get(1, np.nan))
                peak_window = betas_on.loc[(betas_on.index >= 4) & (betas_on.index <= 8)]
                on_peak.append(peak_window.max())
                on_did.append(r_on["did"]["did_estimate"])

        rows.append({
            "sigma_poi": sp,
            "OFF_k1": np.nanmean(off_k1),
            "ON_k1": np.nanmean(on_k1),
            "ON_peak_k4_8": np.nanmean(on_peak),
            "ON_pooled_DiD": np.nanmean(on_did),
            "OFF_VR": np.nanmean(off_vr),
            "ON_completion": np.nanmean(on_compl),
            "elapsed_s": time.time() - t0,
        })
        # Incremental save so we can inspect partial results while the sweep runs
        pd.DataFrame(rows).to_csv(
            os.path.join(ROOT, "output", "sigma_poi_sweep.csv"), index=False
        )

    df = pd.DataFrame(rows)
    print("\n=======================  SIGMA_POI SWEEP  =======================")
    print(df.to_string(index=False))
    print("\nTargets:  OFF_k1 ≈ -0.038   ON_k1 ≈ -0.038   ON_peak ≈ +0.029")
    print("          ON_pooled_DiD ≈ -0.008   OFF_VR ≈ 1.6   ON_completion ≈ 0.20")
    out = os.path.join(ROOT, "output", "sigma_poi_sweep.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
