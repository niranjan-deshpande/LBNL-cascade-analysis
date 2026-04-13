# Withdrawal Cascade / Contagion Analysis Plan

## Research Question

When one project at a point of interconnection withdraws, does it increase the probability that other projects sharing the same POI or network upgrade cluster also withdraw?

## Why This Is Novel

- Johnston et al. describe the cascade mechanism (withdrawal → cost reallocation → more withdrawals) but model it theoretically, not empirically.
- LBNL reports describe the phenomenon qualitatively but don't model spatial/cluster correlation.
- This analysis would directly measure peer withdrawal effects at the POI/cluster level.

## Why This Works With the Data

The original survival model was bottlenecked by `ia_date` sparsity (only 2,358/36,441 rows). This analysis **does not need `ia_date`** — it uses `q_status`, `wd_date`, `q_date`, `poi_name`, `cluster`, and `entity`, which have far better coverage.

| Constraint | Original survival model | Cascade analysis |
|---|---|---|
| Needs `ia_date` | Yes (only 2,358 have it) | No |
| Sample size | 1,796 (149 withdrawal events) | 13,714+ projects in multi-project POIs |
| Entity coverage | ERCOT-dominated (59%) | All entities with `wd_date` usable |
| POI high cardinality | Liability | Asset (defines the clusters) |
| Cost data overlap | None | Not needed |
| PJM usable | No (3 `ia_date` values) | Yes — best entity in the dataset |

## Data Inventory

### Primary Sample

Entities with strong `wd_date` coverage for the time-varying analysis:

| Entity | Total projects | Withdrawn | Has `wd_date` | Coverage | Multi-project POIs | Withdrawals in multi-POIs (dated) |
|---|---|---|---|---|---|---|
| PJM | 8,152 | 4,885 | 4,734 | 97% | 1,331 | 1,774 |
| ERCOT | 3,282 | 955 | 955 | 100% | 323 | 141 |
| CAISO | 2,837 | 1,971 | 1,629 | 83% | 412 | 554 |
| SOCO | 1,158 | 863 | 829 | 96% | 214 | 476 |
| ISO-NE | 1,281 | 663 | 649 | 98% | 115 | 142 |
| PSCo | 375 | 290 | 281 | 97% | 62 | 194 |
| **Total** | **17,085** | **9,627** | **9,077** | | **2,457** | **3,281** |

Entities with zero `wd_date` (SPP, PacifiCorp, NYISO, Duke, BPA, IP, TVA, APS, etc.) can still be used for the logistic (non-temporal) version.

### Grouping Variables

1. **`entity + poi_name`** — geographic co-location at a substation/interconnection point.
   - 4,787 multi-project groups across all entities (13,714 projects)
   - 3,364 of these have at least 1 withdrawal
   - 1,518 have mixed outcomes (both withdrawals and operational/active)
   - Median group size: 2; mean: 2.9; max: 35

2. **`cluster`** — interconnection study cohort (entity-specific).
   - 242 multi-project clusters (9,465 projects)
   - Closer to the actual cost-sharing mechanism that drives cascades
   - Only available for 9,505/36,441 projects (CAISO, MISO, NYISO, BPA, PacifiCorp mainly)
   - Much larger groups (median: 9.5, max: 558)

### Preliminary Signal

Observed variance in POI-level withdrawal rates is **1.68x** what independence would predict. Withdrawals cluster within POIs more than chance alone. This is necessary (though not sufficient) for the contagion story.

## Analysis Plan

### Tier 1: Logistic Regression (Simplest)

**Question:** Does the withdrawal rate at a POI predict whether individual projects also withdraw?

- **Dependent variable:** `withdrawn` (binary: `q_status == 'withdrawn'`)
- **Key regressor:** Peer withdrawal count or rate at the same POI (excluding project i)
- **Controls:** `type_clean`, `mw1`, `entity`, `q_year`, `state`
- **Sample:** All projects in multi-project POIs (~13,714), all entities
- **No date requirements** beyond `q_date` for controls
- **Concern:** Simultaneity — peer withdrawal rate is endogenous. This is descriptive, not causal.

### Tier 2: Cox Model with Time-Varying Covariate

**Question:** Does withdrawal of project i at POI k at time t increase the hazard of withdrawal for remaining projects at POI k?

- **Time variable:** Days since queue entry (`q_date`) to event (withdrawal via `wd_date`, or COD via `on_date`, or right-censored)
- **Key covariate:** Cumulative peer withdrawals at same POI up to time t (time-varying)
- **Stratification:** Entity (allows baseline hazard to differ)
- **Controls:** `type_clean`, `mw1`, `q_year`
- **Sample:** Projects in multi-project POIs within entities that report `wd_date` (~3,281 dated withdrawals across 2,457 POI groups)
- **Strengths:** Temporal ordering addresses reverse causality; time-varying covariate captures the cascade dynamic
- **Concern:** Common shocks (projects at same POI face same transmission constraint). Partially addressed by controls and robustness checks.

### Tier 3: ML Feature Importance

**Question:** How much predictive power does "peer withdrawal history at POI" add relative to project-level characteristics?

- **Model:** Random forest or gradient boosting classifier
- **Target:** Withdrawal (binary)
- **Features:** Project-level (type, capacity, entity, year, state) + POI-level (peer withdrawal count, POI queue depth, POI congestion)
- **Metric:** Feature importance (permutation or SHAP) for the peer withdrawal feature
- **Purpose:** Complement the econometric analysis; show magnitude of contagion signal vs. other factors

### Robustness / Identification Checks

These address the key threat: **contagion vs. common shocks**.

1. **Temporal asymmetry test:** Does withdrawal of earlier-queued projects predict withdrawal of later-queued ones, but not vice versa? (Use `q_date` ordering within POIs.)
2. **Permutation test:** Randomly shuffle POI assignments across projects within entity×year cells. Re-compute within-POI withdrawal clustering. If the observed clustering is significantly higher than permuted, there's excess co-movement beyond entity×year common shocks.
3. **Placebo test:** Do *operational* outcomes also cluster within POIs at the same rate? If clustering is specific to withdrawals (not just "POIs have correlated outcomes of any kind"), that supports the cost-reallocation mechanism.
4. **Dose-response:** Does the cascade effect strengthen with POI queue depth? (More projects sharing upgrades → stronger contagion.)
5. **Entity heterogeneity:** Run the analysis separately by entity. Cost allocation rules differ across ISOs — cascade effects should be stronger where cost reallocation is more direct.

## Policy Relevance

- Directly supports ACP's interest in post-GIA cost reallocation dynamics.
- If withdrawal cascades are empirically real and quantifiable, that's a concrete input for reforming cost allocation rules.
- Entity-level heterogeneity results could identify which ISO cost allocation designs are most cascade-prone.

## Implementation Order

1. [ ] Data preparation: clean POI names, build `entity_poi` groups, convert dates, construct panel
2. [ ] Descriptive statistics: withdrawal clustering by POI, overdispersion tests, visualizations
3. [ ] Tier 1: Logistic regression with peer withdrawal rate
4. [ ] Tier 2: Cox model with time-varying peer withdrawal covariate
5. [ ] Robustness checks (permutation test, temporal asymmetry, placebo)
6. [ ] Tier 3: ML feature importance (if time permits)
7. [ ] Write-up and visualizations
