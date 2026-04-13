"""Simulation: does measurement error attenuation explain the dose-response gradient?

Generate synthetic data under a pure shared-confounder model (zero contagion):
each POI gets a latent quality alpha_g, projects withdraw as Bernoulli(logit^-1(alpha_g)).
Then run the dose-response analysis and compare the gradient to real data.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
from config import RANDOM_SEED, DEPTH_BINS, DEPTH_LABELS, TABLES_DIR
import os


def simulate_one(rng, poi_depths, alpha_sd=1.0, base_rate_logit=0.0):
    """Generate one synthetic dataset under pure confounding (no contagion).

    Args:
        poi_depths: array of POI sizes (from real data)
        alpha_sd: SD of POI-level latent quality (controls overdispersion)
        base_rate_logit: mean of alpha distribution on logit scale
    """
    records = []
    for n_g in poi_depths:
        alpha_g = rng.normal(base_rate_logit, alpha_sd)
        p_g = expit(alpha_g)
        withdrawals = rng.binomial(1, p_g, size=n_g)

        # Compute leave-one-out peer withdrawal rate
        total_wd = withdrawals.sum()
        for i in range(n_g):
            peer_wd = total_wd - withdrawals[i]
            peer_rate = peer_wd / (n_g - 1)
            records.append({
                "withdrawn": withdrawals[i],
                "peer_wd_rate": peer_rate,
                "poi_depth": n_g,
            })

    return pd.DataFrame(records)


def run_dose_response_on_synthetic(df):
    """Run logistic regression by depth bin, return coefficients."""
    results = {}
    for (lo, hi), label in zip(DEPTH_BINS, DEPTH_LABELS):
        subset = df[(df["poi_depth"] >= lo) & (df["poi_depth"] <= hi)]
        if len(subset) < 50 or subset["withdrawn"].sum() < 10:
            results[label] = np.nan
            continue
        X = sm.add_constant(subset[["peer_wd_rate"]])
        y = subset["withdrawn"].astype(float)
        try:
            res = sm.Logit(y, X).fit(disp=False, maxiter=50)
            results[label] = res.params["peer_wd_rate"]
        except Exception:
            results[label] = np.nan
    return results


def run_simulation(n_reps=1000):
    """Run the full simulation study."""
    rng = np.random.RandomState(RANDOM_SEED)

    # Use real POI depth distribution from data
    # We'll load the real data to get POI depths
    from data_prep import load_raw_data, convert_dates, clean_data, build_tier1_sample
    df = load_raw_data()
    df = convert_dates(df)
    df = clean_data(df)
    t1 = build_tier1_sample(df)

    poi_depths = t1.groupby("entity_poi")["poi_depth"].first().values
    print(f"Using {len(poi_depths)} POIs from real data")
    print(f"Depth distribution: min={poi_depths.min()}, median={np.median(poi_depths):.0f}, "
          f"max={poi_depths.max()}, mean={poi_depths.mean():.1f}")

    # Calibrate alpha_sd to match observed overdispersion (~1.67 variance ratio)
    # and base_rate to match observed withdrawal rate (~0.54)
    base_rate_logit = np.log(0.544 / (1 - 0.544))  # ~0.18
    alpha_sd = 0.85  # calibrated to roughly match VR ~1.67

    # Get real-data coefficients for comparison
    from robustness import dose_response
    real_dr = dose_response(t1)
    real_coefs = dict(zip(real_dr["depth_bin"], real_dr["peer_wd_rate_coef"]))

    print(f"\nReal-data coefficients: {real_coefs}")
    print(f"\nRunning {n_reps} simulation replications...")

    sim_results = []
    for i in range(n_reps):
        syn = simulate_one(rng, poi_depths, alpha_sd=alpha_sd, base_rate_logit=base_rate_logit)
        coefs = run_dose_response_on_synthetic(syn)
        sim_results.append(coefs)
        if (i + 1) % 100 == 0:
            print(f"  Replication {i+1}/{n_reps}")

    sim_df = pd.DataFrame(sim_results)

    # Summary
    print("\n=== Simulation Results (Pure Confounding, No Contagion) ===")
    print(f"{'Depth Bin':>10}  {'Sim Mean':>10}  {'Sim SD':>10}  {'Real Coef':>10}  {'Real > Sim%':>12}")
    summary_rows = []
    for label in DEPTH_LABELS:
        sim_vals = sim_df[label].dropna()
        if len(sim_vals) == 0:
            continue
        real_val = real_coefs.get(label, np.nan)
        pct_above = (sim_vals >= real_val).mean() * 100 if not np.isnan(real_val) else np.nan
        print(f"{label:>10}  {sim_vals.mean():>10.3f}  {sim_vals.std():>10.3f}  "
              f"{real_val:>10.3f}  {pct_above:>11.1f}%")
        summary_rows.append({
            "depth_bin": label,
            "sim_mean": sim_vals.mean(),
            "sim_sd": sim_vals.std(),
            "sim_median": sim_vals.median(),
            "real_coef": real_val,
            "pct_sim_exceeds_real": pct_above,
        })

    # Check gradient: does the simulated coefficient increase with depth?
    sim_gradient = sim_df[DEPTH_LABELS].mean()
    real_gradient = [real_coefs.get(l, np.nan) for l in DEPTH_LABELS]
    print(f"\nSimulated gradient (mean coefs): {sim_gradient.values}")
    print(f"Real gradient:                   {real_gradient}")

    # Ratio of real gradient slope to simulated gradient slope
    sim_slope = sim_gradient.iloc[-1] - sim_gradient.iloc[0]
    real_slope = real_gradient[-1] - real_gradient[0]
    if sim_slope != 0:
        print(f"\nGradient slope — real: {real_slope:.3f}, sim: {sim_slope:.3f}, "
              f"ratio: {real_slope/sim_slope:.2f}x")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(TABLES_DIR, "simulation_dose_response.csv"), index=False)

    # Also save the full sim distribution for plotting
    sim_df.to_csv(os.path.join(TABLES_DIR, "simulation_dose_response_full.csv"), index=False)

    return summary_df, sim_df


if __name__ == "__main__":
    run_simulation()
