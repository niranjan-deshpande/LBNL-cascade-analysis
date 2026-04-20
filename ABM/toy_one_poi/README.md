# Toy One-POI ABM

Proof-of-concept for the full ABM. Single POI, 6 projects, monthly time step.

## Model

NPV-style decision rule: project `i` stays at month `t` iff
`U_it < H_it + eps_it + eta_t`, where `U_it` is the allocated upgrade cost
(share × total POI upgrade $), `H_it` is headroom ramping up from a floor to
full value at COD, `eps_it` is idiosyncratic monthly noise, `eta_t` is a
shared POI-level monthly shock.

Headroom is hierarchical: `mu_POI ~ N(mu0, sigma_between^2)` is drawn once per
replication; then per-project `H_i ~ N(mu_POI, sigma_within^2)` with
`sigma_within << sigma_between` (dominant variance across POIs).

Reallocation: a withdrawal at `t` schedules a redistribution of the
withdrawer's share across still-active peers (pro-rata by MW) at `t + tau`,
with `tau ~ U[12, 18]` months — matching the DiD's delayed-positive window.

Counterfactual comparator (`reallocation_enabled=False`): withdrawer's share
evaporates; peers are unaffected. Isolates the cascade channel.

## Calibration

Bootstrapped from PJM rows of the LBNL queue data:
- `mw`: nameplate MW per project (pre-filtered to (1, 2000) MW).
- `dur_months`: queue→COD duration from completed projects.
- `$/kW` upgrade cost: log-uniform between 71 (avg for completions) and 563
  (avg for withdrawals), per the brief.

Headroom scale (`mu_H_mean`, `sigma_between`) picked so the baseline
completion rate hits ~27% (matching LBNL empirical). Current defaults land
at 26.8% over 500 reps.

## Files

- `calibrate.py` — loads LBNL Excel once, caches to `_calib_cache.pkl`,
  exposes MW / duration samplers.
- `model.py` — `Project`, `Params`, `simulate`, `draw_projects`.
- `run_toy.py` — runs 500 reps under reallocation ON vs OFF, writes CSVs
  and `output/withdrawal_paths.png`.

## Run

```
python calibrate.py    # one-time: build the cache
python run_toy.py      # run replications and produce outputs
```

## Current results (seed=42, 500 reps)

| regime         | mean completion | mean withdrawn |
|----------------|-----------------|----------------|
| realloc ON     | 26.8%           | 73.2%          |
| realloc OFF    | 49.6%           | 50.4%          |

Conditional on an early withdrawal (`first_wd_month <= 24`): ON averages
5.06 withdrawals per 6-project POI, OFF averages 3.48 — i.e. the cascade
channel adds ~1.6 extra withdrawals per triggered POI in this toy.

## Known deferred pieces

- Underfunded-upgrade deposit pool counterfactual.
- Per-developer headroom heterogeneity (beyond hierarchical POI/project).
- Formal fit to the DiD event-time profile (requires matched-pair simulation,
  not per-POI summary).
- Time-varying `C_i` / `E[R_i]` is currently folded into a simple linear
  ramp on `H_i`; a richer discounting treatment is deferred to the full ABM.
