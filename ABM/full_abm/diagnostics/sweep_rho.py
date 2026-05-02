"""rho_poi sweep on OFF regime — does the AR(1) persistence calibration drive
the OFF-reproduces-empirical-shape result, or is the shape robust?

The OFF regime (no reallocation cascade) currently reproduces most of the
empirical event-study shape (k=1 dip, k=4-8 hump). rho_poi = 0.85 was set as
a default and never swept — sigma_poi was. This script sweeps rho across
[0.70, 0.95] holding sigma_poi at 2.2e6 (the calibration default), and
measures how each rho's OFF event-study betas compare to the empirical ones.

Metric: per-k betas and an RMSE shape distance against the canonical
empirical event-study coefficients (from contagion/output/matched_did/tables/
event_study_coefficients.csv at k=1..8).
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

RHOS = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
SEEDS = tuple(range(105, 115))
BASE = dict(n_pois=2000, horizon_months=240, arrivals_per_month=25.0,
            sigma_poi=2.2e6, alpha_local=0.15, reallocation_enabled=False)

# Canonical empirical betas at k=1..8 (from contagion/output/matched_did/
# tables/event_study_coefficients.csv after the 2026-04-25 bug fix).
EMP_BETAS = {
    1: -0.0379, 2: -0.0041, 3: +0.0113, 4: +0.0246,
    5: +0.0215, 6: +0.0092, 7: +0.0287, 8: +0.0113,
}
EMP_K1 = EMP_BETAS[1]
EMP_PEAK_K = 7
EMP_PEAK_BETA = EMP_BETAS[EMP_PEAK_K]


def simulate_off(seed: int, rho: float):
    p = Params(**{**BASE, "rng_seed": seed, "rho_poi": rho})
    m = QueueModel(p).run()
    panel = pd.DataFrame(m.project_panel())
    return panel, p


def main():
    t_start = time.time()
    seeds_path = os.path.join(ROOT, "output", "rho_sweep_off_seeds.csv")
    existing_df = pd.read_csv(seeds_path) if os.path.exists(seeds_path) else pd.DataFrame()
    print(f"Existing seeds rows on disk: {len(existing_df)}", flush=True)
    new_rows = []
    for rho in RHOS:
        t0 = time.time()
        print(f"\n================  rho = {rho:.2f}  ================", flush=True)
        for seed in SEEDS:
            if not existing_df.empty and (
                (existing_df["rho"] == rho) & (existing_df["seed"] == seed)
            ).any():
                print(f"  seed={seed}: already in CSV, skipping", flush=True)
                continue
            panel, p = simulate_off(seed, rho)
            df_sim = abm_panel_to_lbnl_schema(panel, p.horizon_months)
            r = run_did_on_df(df_sim, f"OFF rho={rho:.2f} seed={seed}")
            if r is None:
                continue
            betas = pd.DataFrame(r["betas"]).set_index("event_time")["beta"]
            row = {
                "rho": rho, "seed": seed,
                "n_pairs": r["n_pairs"],
                "did": r["did"]["did_estimate"],
                "pre_trend_p": r["pre_trend"].get("p_value", float("nan")),
            }
            for k in range(-4, 9):
                row[f"k{k}"] = float(betas.get(k, np.nan))
            post = betas.loc[(betas.index >= 4) & (betas.index <= 8)]
            row["peak_beta"] = float(post.max())
            row["peak_k"] = int(post.idxmax())
            new_rows.append(row)
            print(f"  seed={seed}: pairs={r['n_pairs']} k1={row['k1']:+.4f} "
                  f"peak(k={row['peak_k']})={row['peak_beta']:+.4f} pre-p={row['pre_trend_p']:.3f}",
                  flush=True)
            # Incremental save so partial progress survives interruption.
            combined = pd.concat([existing_df, pd.DataFrame(new_rows)], ignore_index=True)
            combined.to_csv(seeds_path, index=False)
        print(f"  rho={rho:.2f} elapsed: {time.time()-t0:.1f}s", flush=True)

    seeds_df = pd.concat([existing_df, pd.DataFrame(new_rows)], ignore_index=True)
    seeds_df.to_csv(seeds_path, index=False)
    print(f"\nTotal seeds rows after merge: {len(seeds_df)}", flush=True)

    # Summary: per-rho mean ± SE of k1, peak_beta, peak_k, pre_trend_p, plus shape RMSE.
    summary_rows = []
    for rho in RHOS:
        sub = seeds_df[seeds_df["rho"] == rho]
        if len(sub) == 0:
            continue
        n = len(sub)
        emp_vec = np.array([EMP_BETAS[k] for k in range(1, 9)])
        sim_means = np.array([sub[f"k{k}"].mean() for k in range(1, 9)])
        rmse = float(np.sqrt(((sim_means - emp_vec) ** 2).mean()))
        summary_rows.append({
            "rho": rho,
            "n_seeds": n,
            "k1_mean": sub["k1"].mean(),
            "k1_se": sub["k1"].std(ddof=1) / np.sqrt(n),
            "peak_beta_mean": sub["peak_beta"].mean(),
            "peak_beta_se": sub["peak_beta"].std(ddof=1) / np.sqrt(n),
            "peak_k_mode": int(sub["peak_k"].mode().iloc[0]),
            "did_mean": sub["did"].mean(),
            "pre_trend_p_mean": sub["pre_trend_p"].mean(),
            "shape_rmse_vs_emp": rmse,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(ROOT, "output", "rho_sweep_off_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n========================  RHO SWEEP — OFF  ========================", flush=True)
    print(summary_df.to_string(index=False), flush=True)
    print(f"\nEmpirical reference: k1 = {EMP_K1:+.4f}, "
          f"peak (k={EMP_PEAK_K}) = {EMP_PEAK_BETA:+.4f}", flush=True)
    print(f"Saved: {summary_path}", flush=True)
    print(f"Total elapsed: {time.time()-t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
