# LBNL Interconnection Queue Analysis — Project Context

## Project Goal
Test whether withdrawal cascades exist in U.S. electricity interconnection queues. When one project withdraws from a point of interconnection (POI), shared network upgrade costs are reallocated to remaining projects — does this trigger further withdrawals? The analysis uses the full queue (all 36,441 projects, not just post-ISA) and proceeds in layers of increasing causal rigor: overdispersion tests, logistic regression, Cox time-varying models, gradient boosting + SHAP, robustness checks, and a matched difference-in-differences design.

The original plan was a post-ISA survival model, but `ia_date` sparsity made that infeasible. The contagion analysis sidesteps this by using `q_status`, `wd_date`, `q_date`, and `poi_name`, which have far better coverage.

## Repository Structure

```
LBNL/
├── CLAUDE.md                     # this file
├── lbnl_ix_queue_data_file_thru2024_v2.xlsx  # source data (14 MB)
├── docs/                         # planning & reference documents
│   ├── contagion_plan.md         # analysis design and rationale
│   └── insights_from_data.md     # initial data exploration findings
├── contagion/                    # analysis code (Python)
│   ├── config.py                 # paths, constants, entity lists
│   ├── data_prep.py              # loading, cleaning, POI normalization, sample construction
│   ├── descriptive.py            # overdispersion test, POI/entity summaries
│   ├── tier1_logistic.py         # logistic regression (cross-sectional)
│   ├── tier2_cox.py              # Cox time-varying covariate model
│   ├── tier3_ml.py               # gradient boosting + SHAP
│   ├── robustness.py             # permutation, placebo, temporal asymmetry, dose-response, entity heterogeneity
│   ├── simulation_dose_response.py  # calibrated simulation under zero contagion
│   ├── matched_did.py            # matched difference-in-differences
│   ├── visualizations.py         # publication figures for main pipeline
│   ├── run_all.py                # main pipeline (steps 1–7)
│   ├── run_fixes.py              # bug fixes: same-day withdrawal, lagged exposure, simulation
│   ├── run_fixes2.py             # restricted placebo, deep-POI Cox, PH test
│   ├── briefs/                   # write-ups and reports
│   │   ├── contagion_brief.tex/.pdf  # 2-page summary of all contagion results
│   │   ├── contagion_report.md       # full detailed report (all tiers + robustness)
│   │   └── matched_did_results.md    # detailed matched DiD report
│   └── output/
│       ├── brief/                # Anthropic-styled figures for the 2-page brief
│       │   └── brief_figures.py  # generates those figures
│       ├── figures/              # full pipeline figures (12 plots, PDF + PNG)
│       ├── tables/               # full pipeline CSVs
│       └── matched_did/          # DiD-specific figures & tables
```

## Data

**Source:** `lbnl_ix_queue_data_file_thru2024_v2.xlsx`, sheet `03. Complete Queue Data`. 36,441 interconnection queue requests across 53 entities, 31 columns. The code reads directly from this Excel file.

### Key Fields
- `q_id`: Queue ID. Not globally unique — use `entity + q_id` as composite key.
- `q_status`: `active`, `withdrawn`, `suspended`, `operational`
- `q_date`: Queue entry date (YYYY-MM-DD format)
- `wd_date`, `on_date`, `ia_date`, `prop_date`: **Excel serial numbers** (convert via `datetime(1899, 12, 30) + timedelta(days=value)`)
- `poi_name`: Point of interconnection. Very messy — `data_prep.py` normalizes abbreviations, whitespace, and creates `entity_poi = entity || poi_name_clean` as the grouping key.
- `entity`: Transmission provider. Tier 2 entities (good `wd_date` coverage): PJM, ERCOT, CAISO, SOCO, ISO-NE, PSCo.
- `type_clean`: Resource type (Solar, Wind, Gas, Battery, etc.)
- `mw1`: Capacity in MW. Has some negatives.
- `q_year`: Year from `q_date`. 56 entries = 1970 (anomaly), 101 = NA.
- `developer`: Project developer name. Used in matched DiD sensitivity analysis.

### Data Constraints
- **`ia_date` is mostly empty** (2,358/36,441). The contagion analysis does not use it.
- **`wd_date` coverage varies by entity.** PJM (97%), ERCOT (100%), CAISO (83%), SOCO (96%). SPP, NYISO, PacifiCorp have zero.
- **Date format:** All date columns including `q_date` are Excel serial numbers (int64/float64) in the raw Excel file.
- **POI names are unstandardized.** Measurement error likely attenuates contagion estimates.

## Key Results (Current State)

### Correlation (very strong)
- Overdispersion variance ratio: 1.67 (chi-squared p < 10^-15). Permutation test: 33 SDs above null.
- Logistic OR for peer withdrawal rate: **13.2** (p < 0.001). Positive in all 19 testable entities.
- Gradient boosting AUC: 0.87. Peer withdrawal count is top SHAP feature by 1.5x.
- Dose-response: monotonic increase with POI depth (OR 10.6 at depth 2, 88.3 at depth 10+).

### Temporal evidence (significant, modest)
- Cox HR per peer withdrawal: **1.032** (p = 0.002). Strengthens at deeper POIs (1.052 at depth 5+).
- Effect vanishes at 6- and 12-month lags.
- Simulation under zero contagion cannot reproduce observed coefficients (0/1000 replications).

### Causal evidence (matched DiD)
- 951 matched pairs, 97% match rate. Parallel trends pass (F = 1.00, p = 0.39).
- Short-run DiD is null (-0.028, p = 0.27).
- **Delayed positive effects at k=4–8 quarters** (peak 0.042, p = 0.002). Survives developer and batch restrictions.
- The 1–2 year delay is consistent with formal cost reallocation through restudies.

### Robustness checks
- **Placebo test:** Unrestricted sample shows 9% excess withdrawal clustering over operational outcomes (1.67 vs 1.53). Restricted (terminal-only) shows equal overdispersion, but this is mechanical — withdrawals and successes are complements in that sample, so their variance ratios are identical by construction.
- **Temporal asymmetry:** Inconclusive. Forward and backward coefficients are similar (~0.42 vs ~0.39).
- **Entity heterogeneity:** Positive in all 19 entities. Nationwide pattern.

### Interpretation
The cascade mechanism is empirically supported by multiple independent tests but its magnitude is modest relative to overall POI-level clustering. The cross-sectional OR of ~13 captures both shared confounders and peer effects; the Cox HR of ~1.03 isolates the temporal component; the matched DiD localizes it to a 4–8 quarter lag. For the planned agent-based model, combine heterogeneous POI-level cost shocks (dominant channel) with a modest cascade multiplier on a 12–18 month delay.
