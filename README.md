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
    ├── validation.py                # completion rate, VR, dose response, event study
    ├── run.py                       # multi-seed ON vs OFF sweep
    ├── sweep_alpha.py               # sensitivity over alpha_local parameter
    ├── sweep_sigma_poi.py           # sensitivity over sigma_poi parameter
    ├── validate_matched_did.py      # full matched-DiD pipeline on simulated panels
    ├── diag_off_k1.py               # diagnostic for k=1 dip under reallocation OFF
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
python calibrate.py             # one-time: build the cache
python run.py                   # multi-seed ON vs OFF sweep
python validate_matched_did.py  # matched-DiD pipeline on simulated panels
python sweep_alpha.py           # alpha_local sensitivity
python sweep_sigma_poi.py       # sigma_poi sensitivity
```

Outputs land in `contagion/output/` and `ABM/*/output/`.

## Headline results

### Empirical analysis

- **Overdispersion:** variance ratio 1.67 across POIs, 33 SDs above the permutation null.
- **Logistic:** OR of 13.2 on peer-withdrawal rate; positive in all 19 testable entities.
- **Cox (time-varying):** HR 1.032 per peer withdrawal (p = 0.002); strengthens at deeper POIs.
- **Matched DiD:** 951 matched pairs, parallel trends pass; short-run null, delayed positive effects at 4–8 quarters (peak 0.042, p = 0.002) — consistent with formal cost reallocation through restudies.
- **Simulation:** zero-contagion DGP cannot reproduce observed Cox/logistic coefficients (0/1000 replications).

See `contagion/briefs/contagion_brief.pdf` for the 2-page summary.

### Agent-based model

The full ABM (Mesa, PJM-calibrated) reproduces the two-channel empirical picture:

- **Shared-conditions channel:** persistent AR(1) POI-level shock ($\rho = 0.85$) replicates the overdispersion variance ratio (ABM: 1.67; target: ~1.6) and the excess withdrawals visible immediately after a first withdrawal even without reallocation.
- **Cascade channel:** reallocation ON vs OFF shows a flat pre-period, sharp onset at k=4–5 (matching the 12–18 month restudy lag), peak at k=5–6, and clean decay — reproducing the matched-DiD event-study *shape* from the empirical analysis. The full matched-DiD pipeline run on simulated panels passes parallel-trends tests (F ~ 0.1–1.1) and recovers a near-null two-period DiD (ABM: −0.020; empirical: −0.008).
- **Known calibration gap:** event-study coefficient *magnitudes* are ~3× the empirical values. Current parameters represent a local sweet spot (DiD near null, pre-trends clean, shape correct); closing the magnitude gap is the main remaining calibration task.

See `ABM/full_abm/MODEL.md` for the full model specification and validation table.

## TODO

- [ ] Test policy counterfactuals using the ABM (deposit-pool mechanism, DFAX-threshold changes, cluster-bounded vs. unbounded reallocation, regional cost-allocation shifts).
