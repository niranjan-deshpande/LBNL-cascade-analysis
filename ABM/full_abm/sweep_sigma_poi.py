"""Sweep sigma_poi at alpha_local=0.15.

For each sigma_poi, run ON and OFF regimes and report:
  - OFF k=1 (pure eta channel — should approach -0.038)
  - ON  k=1
  - ON  peak over k=4-8 (should also shrink as eta stops inflating it)
  - OFF variance ratio (cross-sectional — constraint; target ~1.6)
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
from validate_matched_did import abm_panel_to_lbnl_schema, run_did_on_df

SIGMAS = [1.0e6, 1.5e6, 2.2e6]
SEEDS = (100, 101)
BASE = dict(n_pois=2000, horizon_months=240, arrivals_per_month=25.0,
            alpha_local=0.15)


def simulate(alpha_on: bool, seed: int, sigma_poi: float):
    p = Params(**{**BASE, "rng_seed": seed, "sigma_poi": sigma_poi,
                  "reallocation_enabled": alpha_on})
    m = QueueModel(p).run()
    panel = pd.DataFrame(m.project_panel())
    return panel, p


def did_for(panel, p, label):
    df_sim = abm_panel_to_lbnl_schema(panel, p.horizon_months)
    r = run_did_on_df(df_sim, label)
    if r is None:
        return None
    return pd.DataFrame(r["betas"]).set_index("event_time")["beta"]


def main():
    rows = []
    for sp in SIGMAS:
        t0 = time.time()
        print(f"\n================  sigma_poi = {sp:.2e}  ================")
        off_k1_list, on_k1_list, on_peak_list, off_vr_list = [], [], [], []
        for seed in SEEDS:
            # OFF
            panel_off, p_off = simulate(False, seed, sp)
            rep_off = full_report(panel_off, p_off.horizon_months)
            off_vr_list.append(rep_off["variance_ratio"]["vr"])
            betas_off = did_for(panel_off, p_off, f"OFF sp={sp:.1e} seed={seed}")
            if betas_off is not None:
                off_k1_list.append(betas_off.get(1, np.nan))
            # ON
            panel_on, p_on = simulate(True, seed, sp)
            betas_on = did_for(panel_on, p_on, f"ON  sp={sp:.1e} seed={seed}")
            if betas_on is not None:
                on_k1_list.append(betas_on.get(1, np.nan))
                peak_range = betas_on.loc[(betas_on.index >= 4) & (betas_on.index <= 8)]
                on_peak_list.append(peak_range.max())

        rows.append({
            "sigma_poi": sp,
            "OFF_k1": np.nanmean(off_k1_list),
            "ON_k1": np.nanmean(on_k1_list),
            "ON_peak_k4_8": np.nanmean(on_peak_list),
            "OFF_VR": np.nanmean(off_vr_list),
            "elapsed_s": time.time() - t0,
        })
    df = pd.DataFrame(rows)
    print("\n=======================  SIGMA_POI SWEEP  =======================")
    print(df.to_string(index=False))
    print("\nTargets:  OFF_k1 ≈ -0.038   ON_k1 ≈ -0.038   ON_peak ≈ +0.029   OFF_VR ≈ 1.6")
    out = os.path.join(HERE, "output", "sigma_poi_sweep.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
