"""Run the toy ABM over many replications and dump sanity plots / CSVs.

Two comparisons:
  1. Reallocation ON  vs. OFF  (same seed per replication): does the cascade channel show up?
  2. Completion rate under baseline: should calibrate near the ~27% empirical rate.

Outputs go to ABM/toy_one_poi/output/.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from calibrate import sample_mw, sample_duration_months, sample_upgrade_dollars_per_kw
from model import Params, draw_projects, simulate

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)


def run_replications(n_reps: int = 500, seed: int = 42, params: Params | None = None,
                      reallocation: bool = True):
    params = params or Params()
    params.reallocation_enabled = reallocation
    master = np.random.default_rng(seed)
    rows = []
    wd_path_sum = np.zeros(params.horizon_months)
    for r in range(n_reps):
        rng = np.random.default_rng(master.integers(0, 2**31 - 1))
        mw_draws = sample_mw(rng, params.n_projects)
        X = sample_upgrade_dollars_per_kw(rng) * 1000.0 * mw_draws.sum()  # $/kW * kW

        # monkey-patch the MW draws into a sampler that just returns them once
        def _mw(r, n, _x=mw_draws):
            return _x[:n]
        projects = draw_projects(params, rng, _mw, sample_duration_months, params.horizon_months)
        res = simulate(projects, X_total=X, params=params, rng=rng)
        rows.append({
            "rep": r,
            "reallocation": reallocation,
            "n_completed": res.n_completed,
            "n_withdrawn": res.n_withdrawn,
            "n_active_at_end": (res.status == "active").sum(),
            "X_total": X,
            "total_mw": float(mw_draws.sum()),
            "first_wd_month": int(res.t_exit[res.status == "withdrawn"].min()) if res.n_withdrawn else -1,
        })
        wd_path_sum += res.withdrawal_months
    df = pd.DataFrame(rows)
    wd_path_mean = wd_path_sum / n_reps
    return df, wd_path_mean


def conditional_cascade(df_on: pd.DataFrame, df_off: pd.DataFrame, params: Params):
    """Among reps with >=1 early withdrawal (first_wd_month <= 24), compare avg subsequent
    withdrawal count in months [36, 60] under reallocation ON vs OFF."""
    # We can't compare month-by-month paths here without storing them per-rep,
    # so use the summary: avg #withdrawn overall among reps with an early trigger.
    def slice_(df):
        trig = df[(df["first_wd_month"] >= 0) & (df["first_wd_month"] <= 24)]
        return trig["n_withdrawn"].mean(), len(trig)
    m_on, n_on = slice_(df_on)
    m_off, n_off = slice_(df_off)
    return {"mean_wd_on": m_on, "n_on": n_on, "mean_wd_off": m_off, "n_off": n_off}


def main():
    params = Params()
    df_on, path_on = run_replications(n_reps=500, seed=42, params=Params(), reallocation=True)
    df_off, path_off = run_replications(n_reps=500, seed=42, params=Params(), reallocation=False)

    df_on.to_csv(os.path.join(OUT, "reps_realloc_on.csv"), index=False)
    df_off.to_csv(os.path.join(OUT, "reps_realloc_off.csv"), index=False)

    # Summary
    def _summary(df, label):
        n = len(df)
        total = n * params.n_projects
        return {
            "label": label,
            "reps": n,
            "mean_completion_rate": df["n_completed"].sum() / total,
            "mean_withdrawn_rate": df["n_withdrawn"].sum() / total,
            "mean_active_at_end_rate": df["n_active_at_end"].sum() / total,
        }
    summary = pd.DataFrame([_summary(df_on, "realloc_on"), _summary(df_off, "realloc_off")])
    summary.to_csv(os.path.join(OUT, "summary.csv"), index=False)

    cond = conditional_cascade(df_on, df_off, params)
    with open(os.path.join(OUT, "cascade_check.txt"), "w") as f:
        f.write(str(cond) + "\n")

    # Plot: monthly withdrawal rate paths (ON vs OFF)
    fig, ax = plt.subplots(figsize=(8, 4))
    months = np.arange(params.horizon_months)
    ax.plot(months, path_off, label="Reallocation OFF", lw=1.5)
    ax.plot(months, path_on, label="Reallocation ON", lw=1.5)
    ax.set_xlabel("Month")
    ax.set_ylabel("Avg withdrawals / month (per 6-project POI)")
    ax.set_title("Toy one-POI ABM: withdrawal paths, reallocation ON vs OFF")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "withdrawal_paths.png"), dpi=140)

    print(summary.to_string(index=False))
    print("cascade check:", cond)
    print(f"outputs written to {OUT}")


if __name__ == "__main__":
    main()
