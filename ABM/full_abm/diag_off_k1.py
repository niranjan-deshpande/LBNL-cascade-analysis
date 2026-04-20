"""Diagnostic: OFF-regime k=1 at alpha=0.15 vs alpha=0.35.

Reallocation is disabled, so only the eta (AR(1)) channel should drive the
treated-vs-control gap. If both alphas give the same OFF k=1, the eta channel
is truly independent of alpha and the proposed sigma_poi fix is well-targeted.
"""
from __future__ import annotations
import os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import QueueModel, Params
from validate_matched_did import abm_panel_to_lbnl_schema, run_did_on_df

BASE = dict(n_pois=2000, horizon_months=240, arrivals_per_month=25.0,
            reallocation_enabled=False)
SEEDS = (100, 101)


def run_cell(alpha: float):
    print(f"\n########## OFF regime,  alpha_local={alpha} ##########")
    betas_by_seed = []
    for s in SEEDS:
        p = Params(**{**BASE, "rng_seed": s, "alpha_local": alpha})
        m = QueueModel(p).run()
        panel = pd.DataFrame(m.project_panel())
        df_sim = abm_panel_to_lbnl_schema(panel, p.horizon_months)
        r = run_did_on_df(df_sim, f"OFF alpha={alpha} seed={s}")
        if r is None:
            continue
        b = pd.DataFrame(r["betas"])
        betas_by_seed.append(b.assign(seed=s))
    big = pd.concat(betas_by_seed)
    pooled = big.groupby("event_time")["beta"].mean()
    return pooled


if __name__ == "__main__":
    p015 = run_cell(0.15)
    p035 = run_cell(0.35)
    print("\n================  OFF-regime event-study betas (mean over seeds)  ================")
    out = pd.DataFrame({"alpha=0.15": p015, "alpha=0.35": p035})
    out["diff"] = out["alpha=0.35"] - out["alpha=0.15"]
    print(out.to_string())
    print(f"\nk=1 at alpha=0.15: {p015.get(1, float('nan')):+.4f}")
    print(f"k=1 at alpha=0.35: {p035.get(1, float('nan')):+.4f}")
    print(f"delta:             {p035.get(1, float('nan')) - p015.get(1, float('nan')):+.4f}")
