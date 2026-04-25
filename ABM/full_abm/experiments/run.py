"""Multi-seed validation sweep for the multi-POI ABM.

Runs the model under reallocation ON and OFF across N seeds, aggregates the
key validation statistics, and prints a comparison table.
"""

from __future__ import annotations
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")  # silence Mesa FutureWarnings during sweeps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model import QueueModel, Params
from validation import full_report

OUT = os.path.join(ROOT, "output")
os.makedirs(OUT, exist_ok=True)


def _one(params, seed):
    p = Params(**{**params.__dict__, "rng_seed": seed})
    m = QueueModel(p).run()
    panel = pd.DataFrame(m.project_panel())
    rep = full_report(panel, p.horizon_months)
    return panel, rep


def _agg(reports):
    comp = np.mean([r["completion_rate"] for r in reports])
    vr = np.mean([r["variance_ratio"]["vr"] for r in reports])
    or_vals = [r["peer_effect"].get("or", np.nan) for r in reports]
    or_vals = [x for x in or_vals if np.isfinite(x)]
    or_ = np.mean(or_vals) if or_vals else float("nan")
    # Event study: average diff by k across replications
    es = pd.concat([r["event_study"].assign(rep=i) for i, r in enumerate(reports)])
    es_agg = es.groupby("k_quarter").agg(
        mean_diff=("diff", "mean"),
        sd_diff=("diff", "std"),
    )
    # Dose response: pool across reps, take mean wd_rate by bin
    dr = pd.concat([r["dose_response"].assign(rep=i) for i, r in enumerate(reports)])
    dr_agg = dr.groupby("depth_bin", observed=True).agg(mean_wd_rate=("mean_wd_rate", "mean"))
    return {
        "completion_rate": comp,
        "variance_ratio": vr,
        "peer_or": or_,
        "event_study": es_agg,
        "dose_response": dr_agg,
    }


def sweep(label, base_params, seeds=tuple(range(42, 52))):
    t0 = time.time()
    reports = []
    n_projects = []
    for s in seeds:
        _, rep = _one(base_params, s)
        reports.append(rep)
        n_projects.append(int(sum(rep.get("variance_ratio", {}).get("n_pois", 0) > 0 for _ in [0]) or 0))
    out = _agg(reports)
    print(f"\n=== {label}  ({len(seeds)} seeds, {time.time() - t0:.1f}s) ===")
    print(f"  completion_rate = {out['completion_rate']:.3f}   (target ~0.20)")
    print(f"  variance_ratio  = {out['variance_ratio']:.3f}   (target ~1.6)")
    print(f"  peer OR         = {out['peer_or']:.2f}")
    print("  dose response (mean wd_rate by depth bin):")
    print(out["dose_response"].to_string().replace("\n", "\n    "))
    print("  event study (mean treated-control diff):")
    print(out["event_study"].to_string().replace("\n", "\n    "))
    return out


def main():
    base = Params(n_pois=2000, horizon_months=240, arrivals_per_month=25.0,
                   reallocation_enabled=True)
    on_ = sweep("realloc_ON", base)
    off_ = sweep("realloc_OFF", Params(**{**base.__dict__, "reallocation_enabled": False}))

    # Diff ON - OFF in the event study: isolates the cascade channel
    print("\n=== cascade channel (ON event study minus OFF event study) ===")
    es_diff = (on_["event_study"]["mean_diff"] - off_["event_study"]["mean_diff"]).rename("cascade_diff")
    print(es_diff.to_string())


if __name__ == "__main__":
    main()
