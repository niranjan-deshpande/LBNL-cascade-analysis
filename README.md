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
├── notes_for_self/                  # design notes, forward-path writeups, session synthesis
├── toy_one_poi/                     # proof-of-concept: single POI, 6 projects
│   ├── model.py                     # Project, Params, simulate
│   ├── calibrate.py                 # bootstrap samplers from LBNL data
│   ├── run_toy.py                   # 500-rep ON vs OFF sweep
│   └── output/                      # CSVs + withdrawal path figure
└── full_abm/                        # PJM-calibrated multi-POI Mesa model
    ├── MODEL.md                     # full model specification
    ├── model.py                     # Project (Mesa Agent), POI, QueueModel
    ├── validation.py                # completion rate, VR, dose response, mini event study
    ├── validate_matched_did.py      # library: runs contagion/matched_did.py on ABM panels
    ├── calibrate.py                 # PJM-only bootstrap samplers (cached)
    ├── experiments/
    │   ├── run.py                   # multi-seed ON vs OFF sweep (entry point)
    │   ├── run_cluster_bound.py     # cluster-bounded reallocation, 90-seed W sweep (primary, §11)
    │   ├── run_deposit_pool.py      # deposit-pool counterfactual, 90 seeds (secondary, §10)
    │   ├── run_channel_decomp.py    # local vs network channel decomposition (§11.4)
    │   ├── run_alpha_sensitivity.py # α_local robustness (§11.5)
    │   └── run_matched_did_compare.py # matched-DiD on ABM panels under 3 regimes (§11.3)
    ├── diagnostics/
    │   ├── diag_cascade_decomposition.py # matched-DiD ON vs OFF decomposition (§7c caveat)
    │   ├── diag_off_k1.py           # diagnostic for k=1 dip under reallocation OFF
    │   ├── sweep_alpha.py           # sensitivity over alpha_local
    │   └── sweep_sigma_poi.py       # sensitivity over sigma_poi
    └── output/                      # panel CSVs, sweep results, logs
docs/
├── contagion_plan.md                # design & rationale
└── insights_from_data.md            # initial exploration
```

## Running the pipeline

**Empirical contagion analysis:**
```bash
cd contagion
python run_all.py          # descriptive, logit, Cox, ML
python run_fixes.py        # robustness follow-ups
python run_fixes2.py       # additional extensions
python matched_did.py      # matched DiD event study
```

**Toy ABM (single POI):**
```bash
cd ABM/toy_one_poi
python calibrate.py         # one-time: build the cache
python run_toy.py           # 500 reps, ON vs OFF
```

**Full ABM (PJM-calibrated).** Run everything from `ABM/full_abm/`:
```bash
cd ABM/full_abm
python calibrate.py                                  # one-time: build the cache

# Primary counterfactual and extensions
python experiments/run.py                            # multi-seed ON vs OFF sweep
python experiments/run_cluster_bound.py              # cluster-bounded reallocation, 90-seed W sweep (§11)
python experiments/run_deposit_pool.py               # deposit-pool counterfactual, 90 seeds (§10)
python experiments/run_channel_decomp.py             # local vs network channel decomposition (§11.4)
python experiments/run_alpha_sensitivity.py          # α_local robustness (§11.5)
python experiments/run_matched_did_compare.py        # matched-DiD on ABM panels under 3 regimes (§11.3)

# Diagnostics (identification caveat, parameter sweeps)
python diagnostics/diag_cascade_decomposition.py     # matched-DiD ON vs OFF decomposition (§7c)
python diagnostics/diag_off_k1.py                    # k=1 dip under reallocation OFF
python diagnostics/sweep_alpha.py                    # alpha_local sensitivity
python diagnostics/sweep_sigma_poi.py                # sigma_poi sensitivity
```

All `ABM/full_abm/*` outputs land in `ABM/full_abm/output/`; `contagion/*` outputs land in `contagion/output/`.

## Headline results

### Empirical analysis

- **Overdispersion:** variance ratio 1.67 across POIs, 33 SDs above the permutation null.
- **Logistic:** OR of 13.2 on peer-withdrawal rate; positive in all 19 testable entities.
- **Cox (time-varying):** HR 1.032 per peer withdrawal (p = 0.002); strengthens at deeper POIs.
- **Matched DiD:** 951 matched pairs, parallel trends pass; short-run null, delayed positive effects at 4–8 quarters (peak 0.042, p = 0.002) — consistent with formal cost reallocation through restudies. See identification caveat below.
- **Simulation:** zero-contagion DGP cannot reproduce observed Cox/logistic coefficients (0/1000 replications).

See `contagion/briefs/contagion_brief.pdf` for the 2-page summary.

#### DiD identification caveat

Running the empirical matched-DiD estimator on ABM panels with a *known, mechanically operative* cascade (`ABM/full_abm/diagnostics/diag_cascade_decomposition.py`, 30 seeds) shows that the OFF regime alone (no reallocation) reproduces most of the event-study shape. The ABM demonstrates that matched-DiD estimates in this DGP setting are sensitive to residual selection on permanent POI heterogeneity; the empirical +0.029 at peak should be interpreted as an **upper bound on the causal cascade effect**, not a direct measurement of it. See `ABM/notes_for_self/forward_path_4-20.md` for the decomposition and `ABM/full_abm/MODEL.md` §7c and §10 for how the ABM works around this (`mini_event_study` ON − OFF has no fixed effects, so it preserves the cascade contrast).

### Agent-based model

The full ABM (Mesa, PJM-calibrated) reproduces the two-channel empirical picture:

- **Shared-conditions channel:** persistent AR(1) POI-level shock ($\rho = 0.85$) replicates the overdispersion variance ratio (ABM: 1.67; target: ~1.6) and the excess withdrawals visible immediately after a first withdrawal even without reallocation.
- **Cascade channel:** reallocation ON vs OFF shows a flat pre-period, sharp onset at k=4–5 (matching the 12–18 month restudy lag), peak at k=5–6, and clean decay — reproducing the empirical event-study *shape*. The full matched-DiD pipeline run on simulated panels passes parallel-trends tests and recovers a near-null two-period DiD.
- **DiD identification issue:** the matched-DiD estimator applied to ABM panels — where the cascade is mechanically present by construction — leaves an ON − OFF peak that is small and mostly swallowed by fixed effects. The ABM thus demonstrates that matched-DiD estimates in this DGP setting are sensitive to residual selection on permanent POI heterogeneity; the empirical +0.029 should be read as an upper bound on the causal cascade, not a direct measurement. MODEL.md §7c's "3× magnitude gap" reframes as this identification artifact rather than a calibration miscalibration. `mini_event_study` ON − OFF (no fixed effects) is the working cascade gauge.

**Cluster-bounded reallocation — primary counterfactual** (`ABM/full_abm/experiments/run_cluster_bound.py`, 90 seeds, MODEL.md §11). Restricts reallocation (both local and network channels) to recipients whose `t_entry` is within `W` months of the withdrawer's — maps onto FERC Order 2023's cluster-study scope reform. 9-regime sweep. Tight windows prevent a statistically meaningful share of withdrawals: **W=12 prevents +29.9 ± 3.2 /yr at PJM scale, 95% CI [+23.6, +36.0]**; W=18 prevents +10.5 ± 3.4 /yr, CI [+3.9, +17.1]; W=24 prevents +6.5 ± 3.2 /yr, CI [+0.4, +12.7]. W ≥ 36 is indistinguishable from unbounded at this sample size. Within-POI integrated cascade (the quantity matched-DiD measures) drops by ~32% at W=12. Channel decomposition (`run_channel_decomp.py`, §11.4) shows network fanout drives ~75% of total-count prevention while the local share drives the within-POI event-study signal. α-robustness (`run_alpha_sensitivity.py`, §11.5): headline holds across α ∈ {0.05, 0.15, 0.30}.

**Deposit-pool counterfactual — secondary** (`ABM/full_abm/experiments/run_deposit_pool.py`, 90 seeds, MODEL.md §10). Pool absorbs the `(1−α)·U` network share before cascade. Prevents **+15.1 ± 3.4 /yr at PJM scale, CI [+8.4, +21.8]** in total counts — but within-POI integrated cascade is **essentially zero** (+0.0000 ± 0.0076). Mechanism: pool reduces short-run cascade (k=1 by 35%) but offsets it with increased medium-run cascade (k=5 by 163%). Delays, doesn't extinguish. Because matched-DiD integrates within-POI, the pool's prevention is invisible to that estimator, while cluster-bound's is detectable. Confirmed by running the actual `contagion/matched_did.py` pipeline on ABM panels under both regimes (`run_matched_did_compare.py`, §11.3).

See `ABM/full_abm/MODEL.md` for the full model specification, validation table, and counterfactual sections.

## TODO

- [x] Deposit-pool counterfactual (MODEL.md §10).
- [x] Cluster-bounded reallocation counterfactual (MODEL.md §11) — primary, supersedes deposit pool.
- [x] Channel decomposition and α-sensitivity (MODEL.md §11.4, §11.5).
- [x] Validate on the actual empirical matched-DiD estimator (MODEL.md §11.3).
- [ ] Additional policy counterfactuals: DFAX-threshold changes, regional cost-allocation shifts, mixed reforms (pool + cluster-bound).
- [ ] Network-topology sensitivity (`network_fanout` sweep): §11.4 attributes 75% of total-count prevention to the network channel, which may be topology-dependent.
- [ ] Redesign the empirical identification strategy to exploit network topology and PJM's actual DFAX reallocation matrices (the current matched-DiD's confound with persistent POI economics is structural, not fixable with parameter tuning).
