# LBNL Interconnection Queue — Withdrawal Cascades

Empirical test for withdrawal cascades in U.S. electricity interconnection queues: when a project withdraws from a point of interconnection (POI), shared network-upgrade costs reallocate to remaining projects — does that trigger further withdrawals?

The analysis uses the full LBNL queue (36,441 projects across 53 entities, through 2024) and proceeds in layers of increasing causal rigor.

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
docs/
├── contagion_plan.md                # design & rationale
└── insights_from_data.md            # initial exploration
```

## Running the pipeline

```bash
python -m contagion.run_all
python -m contagion.run_fixes
python -m contagion.run_fixes2
python -m contagion.matched_did
```

Outputs land in `contagion/output/` (figures as PDF + PNG, tables as CSV).

## Headline results

- **Overdispersion:** variance ratio 1.67 across POIs, 33 SDs above the permutation null.
- **Logistic:** OR of 13.2 on peer-withdrawal rate; positive in all 19 testable entities.
- **Cox (time-varying):** HR 1.032 per peer withdrawal (p = 0.002); strengthens at deeper POIs.
- **Matched DiD:** 951 matched pairs, parallel trends pass; short-run null, delayed positive effects at 4–8 quarters (peak 0.042, p = 0.002) — consistent with formal cost reallocation through restudies.
- **Simulation:** zero-contagion DGP cannot reproduce observed Cox/logistic coefficients (0/1000 replications).

See `contagion/briefs/contagion_brief.pdf` for the 2-page summary and `contagion/briefs/contagion_report.md` for the full report.
