# LBNL Interconnection Queue — Withdrawal Cascades

Empirical test for withdrawal cascades in U.S. electricity interconnection queues: when a project withdraws from a point of interconnection (POI), shared network-upgrade costs reallocate to remaining projects — does that trigger further withdrawals?

The analysis uses the full LBNL queue (36,441 projects across 53 entities, through 2024) and proceeds in layers of increasing causal rigor, followed by an agent-based model (ABM) to reproduce the empirical patterns and run counterfactual policy experiments.

## Data

Source file: `lbnl_ix_queue_data_file_thru2024_v2.xlsx`, sheet `03. Complete Queue Data`.

Key fields used: `q_id`, `q_status`, `q_date`, `wd_date`, `poi_name`, `entity`, `type_clean`, `mw1`, `developer`.

## Repository layout

```
contagion/
├── config.py                        # paths, constants, entity lists
├── data_prep.py                     # loading, cleaning, POI normalization
├── descriptive.py                   # overdispersion test, POI/entity summaries
├── tier1_logistic.py                # cross-sectional logistic regression
├── tier2_cox.py                     # Cox time-varying covariate model
├── tier3_ml.py                      # gradient boosting + SHAP
├── robustness.py                    # permutation, placebo, temporal, dose-response
├── simulation_dose_response.py      # calibrated simulation under zero contagion
├── matched_did.py                   # matched difference-in-differences
├── visualizations.py                # main-pipeline figures
├── run_all.py                       # main pipeline
├── run_fixes.py, run_fixes2.py      # follow-up fixes and extensions
├── briefs/                          # 2-page brief + full report + DiD writeup
└── output/                          # figures, tables, DiD artifacts
ABM/
├── notes.md                         # model design notes & PJM cost-allocation mechanics
├── toy_one_poi/                     # proof-of-concept: single POI, 6 projects
│   ├── model.py                     # Project, Params, simulate
│   ├── calibrate.py                 # bootstrap samplers from LBNL data
│   ├── run_toy.py                   # 500-rep ON vs OFF sweep
│   └── output/                      # CSVs + withdrawal path figure
└── full_abm/                        # PJM-calibrated multi-POI Mesa model
    ├── model.py                     # Project (Mesa Agent), POI, QueueModel
    ├── calibrate.py                 # PJM-only bootstrap samplers (cached)
    ├── validation.py                # completion rate, VR, dose response, mini event study
    ├── run.py                       # multi-seed ON vs OFF sweep
    ├── sweep_alpha.py               # sensitivity over alpha_local parameter
    ├── sweep_sigma_poi.py           # sensitivity over sigma_poi parameter
    ├── validate_matched_did.py      # full matched-DiD pipeline on simulated panels
    ├── diag_off_k1.py               # diagnostic for k=1 dip under reallocation OFF
    ├── diag_cascade_decomposition.py# ON−OFF cascade decomposition (DiD identification)
    ├── run_deposit_pool.py          # deposit-pool counterfactual (3-regime × 5-seed)
    ├── MODEL.md                     # full model specification
    └── output/                      # panel CSVs, sweep results, logs
docs/
├── contagion_plan.md                # design & rationale
└── insights_from_data.md            # initial exploration
```

## Running the pipeline

**Empirical contagion analysis:**
```bash
python -m contagion.run_all
python -m contagion.run_fixes
python -m contagion.run_fixes2
python -m contagion.matched_did
```

**Toy ABM (single POI):**
```bash
cd ABM/toy_one_poi
python calibrate.py   # one-time: build the cache
python run_toy.py     # 500 reps, ON vs OFF
```

**Full ABM (PJM-calibrated):**
```bash
cd ABM/full_abm
python calibrate.py                    # one-time: build the cache
python run.py                          # multi-seed ON vs OFF sweep
python validate_matched_did.py         # matched-DiD pipeline on simulated panels
python diag_cascade_decomposition.py   # ON−OFF cascade decomposition
python run_deposit_pool.py             # deposit-pool counterfactual
python sweep_alpha.py                  # alpha_local sensitivity
python sweep_sigma_poi.py              # sigma_poi sensitivity
```

Outputs land in `contagion/output/` and `ABM/*/output/`.

## Headline results

### Empirical analysis

- **Overdispersion:** variance ratio 1.67 across POIs, 33 SDs above the permutation null.
- **Logistic:** OR of 13.2 on peer-withdrawal rate; positive in all 19 testable entities.
- **Cox (time-varying):** HR 1.032 per peer withdrawal (p = 0.002); strengthens at deeper POIs.
- **Matched DiD:** 951 matched pairs, parallel trends pass; short-run null, delayed positive effects at 4–8 quarters (peak 0.042, p = 0.002) — consistent with formal cost reallocation through restudies. See identification caveat below.
- **Simulation:** zero-contagion DGP cannot reproduce observed Cox/logistic coefficients (0/1000 replications).

See `contagion/briefs/contagion_brief.pdf` for the 2-page summary.

#### DiD identification caveat

Running the empirical matched-DiD estimator on ABM panels with a *known, mechanically operative* cascade (`ABM/full_abm/diag_cascade_decomposition.py`, 30 seeds) shows that the OFF regime alone (no reallocation) reproduces most of the event-study shape. The ABM demonstrates that matched-DiD estimates in this DGP setting are sensitive to residual selection on permanent POI heterogeneity; the empirical +0.029 at peak should be interpreted as an **upper bound on the causal cascade effect**, not a direct measurement of it. See `ABM/notes_for_self/forward_path_4-20.md` for the decomposition and `ABM/full_abm/MODEL.md` §7c and §10 for how the ABM works around this (`mini_event_study` ON − OFF has no fixed effects, so it preserves the cascade contrast).

### Agent-based model

The full ABM (Mesa, PJM-calibrated) reproduces the two-channel empirical picture:

- **Shared-conditions channel:** persistent AR(1) POI-level shock ($\rho = 0.85$) replicates the overdispersion variance ratio (ABM: 1.67; target: ~1.6) and the excess withdrawals visible immediately after a first withdrawal even without reallocation.
- **Cascade channel:** reallocation ON vs OFF shows a flat pre-period, sharp onset at k=4–5 (matching the 12–18 month restudy lag), peak at k=5–6, and clean decay — reproducing the empirical event-study *shape*. The full matched-DiD pipeline run on simulated panels passes parallel-trends tests and recovers a near-null two-period DiD.
- **DiD identification issue:** the matched-DiD estimator applied to ABM panels — where the cascade is mechanically present by construction — leaves an ON − OFF peak that is small and mostly swallowed by fixed effects. The ABM thus demonstrates that matched-DiD estimates in this DGP setting are sensitive to residual selection on permanent POI heterogeneity; the empirical +0.029 should be read as an upper bound on the causal cascade, not a direct measurement. MODEL.md §7c's "3× magnitude gap" reframes as this identification artifact rather than a calibration miscalibration. `mini_event_study` ON − OFF (no fixed effects) is the working cascade gauge.

**Deposit-pool counterfactual** (`ABM/full_abm/run_deposit_pool.py`, 30 seeds): adding PJM's RD1→RD4 security-deposit pool — forfeited deposits absorb the network share of underfunded upgrade costs before cascade — reduces total withdrawals by **60.4 ± 20.0 per 240-month run** (95% CI [+21, +100]). Scaled to PJM (6× ABM arrival rate): **18 ± 6 withdrawals/year prevented** (95% CI [+6, +30]). The per-k proportional-reduction ratio is uninterpretable at this sample size (noisy denominator), so the count-based CI is the headline. Per-k cascade signal (ON − OFF) is detectable at k ∈ {1, 4, 6} and at-noise elsewhere. See MODEL.md §10.

See `ABM/full_abm/MODEL.md` for the full model specification and validation table.

## TODO

- [x] Deposit-pool counterfactual (`ABM/full_abm/run_deposit_pool.py`, MODEL.md §10).
- [ ] Additional policy counterfactuals: DFAX-threshold changes, cluster-bounded vs. unbounded reallocation, regional cost-allocation shifts.
- [ ] Redesign the empirical identification strategy to exploit network topology and PJM's actual DFAX reallocation matrices (the current matched-DiD's confound with persistent POI economics is structural, not fixable with parameter tuning).
