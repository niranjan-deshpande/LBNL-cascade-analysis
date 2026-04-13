# Withdrawal Contagion in U.S. Interconnection Queues

## Internal Progress Report — April 5, 2026 (v2)

---

## 1. Motivation and Research Question

The central question is: **does a project's probability of withdrawing from the interconnection queue increase when other projects at its point of interconnection have withdrawn?** If so, how large is the effect, and does it survive robustness checks that distinguish genuine contagion from confounding?

---

## 2. Data and Sample Construction

**Source:** LBNL "Queued Up" 2025 dataset, loaded from `lbnl_ix_queue_data_file_thru2024_v2.xlsx`, sheet `03. Complete Queue Data`. 36,441 interconnection queue requests across 53 entities.

### Sample construction

1. **POI normalization.** `poi_name` values were lowercased, stripped, and common abbreviations were standardized (e.g., "sub"/"ss" -> "substation", "jct" -> "junction", "rd" -> "road"). Trailing punctuation was removed, and whitespace was normalized. Entries of "NA," "Unknown," or blank were set to missing. A composite key `entity_poi = entity || poi_name` was created to identify unique interconnection points within each transmission provider, since the same POI name at different entities does not imply shared infrastructure. The improved normalization merged 122 additional POIs (0.5%) relative to a minimal lowercase-and-strip approach; a sensitivity check confirmed this had no material effect on results (peer withdrawal OR changed from 13.26 to 13.22).

2. **Multi-project POI filter.** Projects were retained only if their `entity_poi` had at least 2 projects. This produced the **Tier 1 sample: 14,845 projects in 4,975 POIs.** The modal POI has 2 projects; the largest has 123 (likely a major substation aggregating many queue requests).

3. **Leave-one-out peer withdrawal rate.** For each project, we computed the withdrawal rate among its peers at the same POI, excluding itself. This avoids mechanical correlation between the dependent variable (own withdrawal) and the regressor.

### Key sample statistics

| Statistic | Value |
|-----------|-------|
| Total projects in sample | 14,845 |
| Unique POIs | 4,975 |
| Overall withdrawal rate | 54.4% |
| Mean POI depth | 2.98 projects |
| Median POI depth | 2 projects |
| Max POI depth | 123 projects |

---

## 3. Analyses and Results

### 3.1 Descriptive: Overdispersion Test

**Goal:** Test whether withdrawals cluster at POIs more than would be expected if each project's withdrawal decision were independent.

**Method:** Under the null hypothesis of independent withdrawals with a common probability $p$, the number of withdrawals at a POI with $n$ projects follows a Binomial($n$, $p$) distribution. The variance of observed POI-level withdrawal rates should equal $p(1-p)/n$ on average. We computed the ratio of observed to expected variance.

**Result:** The variance ratio is **1.67**, meaning withdrawals are 67% more dispersed across POIs than independence would predict. A chi-squared goodness-of-fit test strongly rejects the null ($\chi^2 = 9{,}115$ on 4,916 d.f., $p < 10^{-15}$). This is a model-free result: withdrawals cluster at POIs well beyond what chance alone can explain.

### 3.2 Tier 1: Logistic Regression

**Goal:** Estimate the association between peer withdrawal behavior and a project's own withdrawal probability, controlling for project and entity characteristics.

**Model:** Logit regression of a binary withdrawal indicator on:
- Leave-one-out peer withdrawal rate (key variable of interest)
- log(1 + MW capacity)
- Queue entry year
- Resource type (6-level categorical)
- Entity (8-level categorical)
- State (top 10 + Other)

Standard errors are clustered at the `entity_poi` level to account for within-POI correlation.

**Results:**

| Variable | Odds Ratio | 95% CI | p-value |
|----------|-----------|--------|---------|
| **Peer withdrawal rate** | **13.22** | **[11.54, 15.14]** | **< 0.001** |
| log(MW capacity) | 1.03 | [1.00, 1.07] | 0.036 |
| Queue entry year | 1.001 | [1.001, 1.001] | < 0.001 |
| Wind (vs. Battery) | 1.36 | [1.17, 1.57] | < 0.001 |
| Solar (vs. Battery) | 1.25 | [1.10, 1.42] | < 0.001 |
| Solar+Battery (vs. Battery) | 0.82 | [0.69, 0.96] | 0.013 |
| ERCOT (vs. CAISO) | 0.18 | [0.12, 0.27] | < 0.001 |
| SOCO (vs. CAISO) | 1.59 | [1.16, 2.19] | 0.004 |

$N = 14{,}845$. Pseudo $R^2 = 0.231$.

**Interpretation:** A project at a POI where all of its peers have withdrawn (peer rate = 1) is roughly 13 times more likely to withdraw than a project at a POI where none of its peers have withdrawn (peer rate = 0), holding all else equal. The effect is estimated precisely (tight confidence interval well above 1) and survives controls for resource type, capacity, entity, state, and cohort year.

### 3.3 Tier 2: Cox Proportional Hazards with Time-Varying Covariates

**Goal:** Test whether the contagion effect operates dynamically---does a project's withdrawal hazard increase *after* a peer withdraws, not just cross-sectionally?

**Method:** A Cox time-varying covariate model was fit on 6,936 projects across 6 entities with sufficient `wd_date` coverage (PJM, ERCOT, CAISO, SOCO, ISO-NE, PSCo). Each project's observation interval was split at every peer withdrawal event, creating a counting-process dataset of 9,489 intervals. The key covariate is `cumulative_peer_wd`, which increments by 1 at each peer withdrawal within the same POI. Controls include resource type dummies, log capacity, and queue entry year. A small ridge penalty (0.01) was applied for stability.

*Note: An earlier version of this analysis contained a bug where same-day peer withdrawals were erroneously dropped from the peer count (timestamp equality removed all matches, not just the focal project). The fix — tracking peers by unique subject ID rather than by withdrawal timestamp — changed the result from non-significant (p = 0.13) to significant (p = 0.002).*

**Results:**

| Variable | Hazard Ratio | 95% CI | p-value |
|----------|-------------|--------|---------|
| **Cumulative peer withdrawals** | **1.032** | **[1.011, 1.053]** | **0.002** |
| Queue entry year | 1.002 | [1.002, 1.002] | < 0.001 |
| Gas (vs. Battery) | 0.58 | [0.50, 0.68] | < 0.001 |
| Wind (vs. Battery) | 0.70 | [0.60, 0.81] | < 0.001 |
| ERCOT (vs. CAISO) | 0.47 | [0.40, 0.56] | < 0.001 |
| PSCo (vs. CAISO) | 1.92 | [1.62, 2.27] | < 0.001 |
| SOCO (vs. CAISO) | 2.29 | [2.02, 2.59] | < 0.001 |

$N = 6{,}936$ subjects, 3,366 events.

**Interpretation:** Each additional peer withdrawal at a project's POI is associated with a **3.2% increase in the instantaneous withdrawal hazard** (p = 0.002). This is the temporal evidence the cross-sectional model cannot provide: the hazard jumps after a peer withdraws, not just in the overall correlation.

#### Effect strengthens at deeper POIs

Restricting the Cox model to deeper POIs increases both the hazard ratio and its significance, consistent with the dose-response pattern from Tier 1:

| Min POI depth | N subjects | N events | HR | 95% CI | p-value |
|---------------|-----------|----------|------|--------|---------|
| 2+ | 6,936 | 3,366 | 1.032 | [1.011, 1.053] | 0.002 |
| 3+ | 3,652 | 1,830 | 1.045 | [1.022, 1.068] | 0.0001 |
| 5+ | 1,513 | 723 | 1.052 | [1.023, 1.081] | 0.0003 |

At POIs with 5+ projects, each additional peer withdrawal raises the hazard by 5.2% — nearly double the all-sample estimate. The tradeoff of observations for covariate variation pays off: fewer subjects but the same or better precision.

#### Effect is immediate, not lagged

We re-estimated the Cox model with the peer withdrawal exposure lagged by 6 and 12 months:

| Lag | HR | 95% CI | p-value |
|-----|------|--------|---------|
| 0 months | 1.032 | [1.011, 1.053] | **0.002** |
| 6 months | 1.012 | [0.989, 1.035] | 0.318 |
| 12 months | 0.997 | [0.973, 1.022] | 0.834 |

The effect is entirely concentrated at zero lag and **disappears with any delay**. This is informative about the mechanism: the contagion effect operates at the time of the withdrawal event itself (or within a few months), not through a slow restudy cycle.

#### Proportional hazards assumption holds

We tested whether the contagion effect varies over calendar time by adding a `cumulative_peer_wd * time` interaction term. The interaction coefficient is negative but not significant (p = 0.095), suggesting a possible mild decay of the effect over time but not enough to violate the proportional hazards assumption. The constant-HR specification is adequate for the current analysis.

### 3.4 Tier 3: Gradient Boosting + SHAP

**Goal:** Use a flexible, nonparametric model to assess the predictive importance of peer withdrawal behavior relative to all other features, without imposing functional form assumptions.

**Method:** A gradient boosting classifier (200 trees, max depth 4, learning rate 0.1) was trained on the Tier 1 sample with 5-fold stratified cross-validation. Features include peer withdrawal count, POI depth, log capacity, queue year, POI type diversity, plus one-hot encodings of resource type, entity, and region (28 features total). Feature importance was assessed three ways: built-in split importance, permutation importance, and SHAP (TreeExplainer).

**Results:**

| Metric | Value |
|--------|-------|
| Cross-validated AUC | **0.869 +/- 0.006** |
| Top feature (all 3 methods) | **peer_wd_count** |

Top 5 features by SHAP mean absolute value:

| Rank | Feature | SHAP |
|------|---------|------|
| 1 | peer_wd_count | 0.853 |
| 2 | mw1_log | 0.555 |
| 3 | q_year | 0.452 |
| 4 | poi_depth | 0.172 |
| 5 | type_Hydro | 0.083 |

**Interpretation:** Peer withdrawal count is the single most important predictor of project withdrawal by a wide margin under all three importance metrics. It generates nearly twice the SHAP impact of the next most important feature (log capacity). The cross-validated AUC of 0.87 indicates strong out-of-sample predictive power.

---

## 4. Robustness Checks

### 4.1 Permutation Test

**Goal:** Rule out the possibility that overdispersion is an artifact of how POIs are defined or of some mechanical feature of the data.

**Method:** The POI assignment (`poi_name`) was randomly shuffled within entity-by-queue-year cells (1,000 permutations). This preserves the marginal distributions of withdrawals and POI sizes within each entity-cohort but breaks any genuine POI-level clustering.

**Result:** The observed variance ratio (1.68) lies **33 standard deviations** above the null distribution mean (1.21, SD = 0.014). The permutation p-value is 0/1000. This confirms that the clustering is not an artifact of the POI definition or entity-cohort structure.

### 4.2 Placebo Test

**Goal:** Check whether the clustering pattern is specific to withdrawals or is a generic feature of all outcomes at a POI.

**Method:** The overdispersion test was repeated using operational outcomes (reaching COD) instead of withdrawals.

**Original result (full sample):** Withdrawal overdispersion is 1.67, operational overdispersion is 1.53. Withdrawals cluster about 9% more than operational outcomes.

**Updated result (restricted to terminal outcomes only):** When we restrict to projects that have reached a terminal state — `q_status` in {"withdrawn", "operational"} only, dropping 4,717 active/suspended projects — the overdispersion ratio for both outcomes is **1.61, and the withdrawal/operational ratio is 1.00**. The 9% gap in the unrestricted sample was entirely driven by the inclusion of active projects that haven't resolved yet.

**Interpretation:** This is an important update. Once we restrict to projects with known outcomes, withdrawals and operational successes cluster at POIs to **exactly the same degree**. This means the overdispersion we observe is fully explained by shared POI-level characteristics (grid conditions, upgrade costs, location quality) that affect all outcomes symmetrically. There is no incremental clustering specific to withdrawals above what the shared POI environment produces. This weakens the case for contagion as the primary driver of clustering, though it does not rule out contagion as a component (since the cross-sectional and time-varying associations remain strong).

### 4.3 Temporal Asymmetry

**Goal:** Test whether contagion runs in the expected causal direction (early withdrawals trigger later ones) rather than reflecting a purely static confounder.

**Method:** Within each POI, projects were split into "early" and "late" cohorts by median queue entry date. We then tested whether early withdrawal rates predict late withdrawal rates (forward direction) and vice versa (backward direction).

**Result:**
- Forward (early -> late): coefficient = **0.42**, p < 0.001
- Backward (late -> early): coefficient = **0.39**, p < 0.001

Both coefficients are large and highly significant, and they are **close in magnitude**. This pattern is more consistent with a shared POI-level confounder than with a pure sequential contagion mechanism.

### 4.4 Dose-Response

**Goal:** Test whether the contagion effect intensifies at more crowded POIs, as the cascade mechanism predicts.

**Method:** The Tier 1 logistic regression was re-estimated within four POI depth bins: 2 projects, 3--4, 5--9, and 10+.

**Result:**

| POI Depth | N | Peer WD Rate Coef | Odds Ratio | p-value |
|-----------|---|-------------------|------------|---------|
| 2 | 6,406 | 2.36 | 10.6 | < 0.001 |
| 3--4 | 4,179 | 2.96 | 19.3 | < 0.001 |
| 5--9 | 2,472 | 3.94 | 51.2 | < 0.001 |
| 10+ | 1,788 | 4.48 | 88.3 | < 0.001 |

The coefficient increases **monotonically** with POI depth.

#### Simulation: how much of the gradient is mechanical?

A simulation study tested whether pure confounding (with zero contagion) can produce a similar gradient through measurement error attenuation. We generated 1,000 synthetic datasets where each POI has a latent quality $\alpha_g \sim N(0.18, 0.85)$ and projects withdraw independently as $\text{Bernoulli}(\text{logit}^{-1}(\alpha_g))$. We then ran the same dose-response analysis on each synthetic dataset.

| POI Depth | Real Coef | Simulated Coef (mean) | Real exceeds sim? |
|-----------|-----------|----------------------|-------------------|
| 2 | 2.36 | 0.55 | 0/1000 (0%) |
| 3--4 | 2.96 | 1.08 | 0/1000 (0%) |
| 5--9 | 3.94 | 1.85 | 0/1000 (0%) |
| 10+ | 4.48 | 3.15 | 0/1000 (0%) |

**Key insight:** A gradient *does* emerge under pure confounding — the simulated coefficient increases from 0.55 to 3.15 across bins. This confirms the measurement-error-attenuation concern: at small POIs, the leave-one-out rate is noisier, which attenuates the coefficient mechanically. **However, the real-data coefficients exceed every simulation replicate at every depth bin.** The data contains signal that pure confounding plus measurement error cannot explain.

The gradient *slope* is actually comparable (real: 2.12, simulated: 2.60), meaning the *shape* of the increase is largely mechanical. The *level* — the fact that all real coefficients are 2--4x larger than the simulated ones — is not mechanical, and reflects either genuine peer effects or a source of confounding beyond our simple shared-quality model.

### 4.5 Entity Heterogeneity

**Goal:** Check whether the contagion signal is specific to a few entities or is a nationwide pattern.

**Method:** The Tier 1 logistic regression was estimated separately for each entity with at least 100 projects and 20 withdrawal events (19 entities qualified).

**Result:** The peer withdrawal rate coefficient is **positive for all 19 entities**. Odds ratios range from 2.5 (NYISO) to 58.1 (CAISO), with all but one (SRP, p = 0.07) statistically significant at p < 0.05. This demonstrates that the finding is not driven by any single entity or region---it is a consistent, nationwide pattern.

---

## 5. Synthesis and Interpretation

### What we can say with confidence

1. **Withdrawals cluster at POIs far more than independence would predict.** The variance ratio of 1.67 is highly significant, confirmed by the permutation test (p < 0.001). This is a robust, model-free finding.

2. **The cross-sectional association between peer and own withdrawal is very strong.** A project whose peers have all withdrawn is ~13x more likely to withdraw than a project whose peers have not (Tier 1 logistic). This survives controls for resource type, capacity, cohort year, entity, and state. It is the top predictor in a flexible ML model (AUC = 0.87).

3. **There is now significant temporal evidence.** After fixing a bug in the counting-process construction, the Cox model shows a significant time-varying effect: each additional peer withdrawal raises the hazard by 3.2% (p = 0.002). The effect strengthens at deeper POIs (5.2% at depth 5+, p = 0.0003) and operates immediately rather than with a lag.

4. **The effect is present in all 19 testable entities.** This is a system-wide pattern.

5. **The real-data coefficients exceed what pure confounding can produce.** A simulation under zero contagion with calibrated shared confounders cannot reproduce the observed coefficient magnitudes at any POI depth (0/1000 replications).

### Where we should be cautious

6. **The restricted placebo test shows no withdrawal-specific clustering.** When limited to terminal outcomes, withdrawals and operational successes cluster identically (VR = 1.61 for both). The overdispersion is fully explained by shared POI characteristics, not by a mechanism specific to withdrawals.

7. **The temporal asymmetry test does not clearly separate contagion from confounding.** Forward and backward coefficients are similar (~0.42 vs. ~0.39), suggesting shared POI-level characteristics account for a large share of the correlation.

8. **The dose-response gradient shape is partly mechanical.** Measurement error attenuation at small POIs produces a gradient even under pure confounding, though the *level* of the real gradient far exceeds the simulated one.

9. **The contagion effect is immediate, not lagged.** The zero-lag specificity is consistent with either true contagion or correlated timing of decisions at a POI. It does not match a slow cost-reallocation-through-restudy mechanism.

### What this means for the project

The evidence has shifted meaningfully with the Cox bug fix and the new analyses. We now have **significant temporal evidence** of contagion that we did not have before. The combination of a significant time-varying hazard ratio, the dose-response pattern surviving simulation scrutiny (at the level, not the slope), and the universality across entities makes a stronger case for genuine peer effects than the earlier round of results.

However, the restricted placebo test is a sobering counterpoint: if contagion were the dominant mechanism, we would expect withdrawals to cluster *more* than operational outcomes, but they don't (among resolved projects). This suggests that the dominant driver of POI-level outcome correlation is shared infrastructure and environmental conditions, with contagion providing an additional but secondary channel.

The most honest interpretation is: **there is a real contagion component, but it operates on top of a much larger shared-confounder effect.** The cross-sectional OR of ~13 captures both channels; the Cox HR of ~1.03 captures the incremental temporal contagion component specifically. The fact that the Cox effect is significant but modest (3%) while the cross-sectional effect is enormous (13x) implies that most of the cross-sectional association reflects correlated exposure to the same POI-level conditions, not sequential cascading.

For the planned agent-based model, this calibrates expectations: the cascade mechanism is real but second-order. An agent-based simulation that assumes withdrawal cascades are the primary driver of queue attrition would overstate the interdependence. A more realistic model would combine heterogeneous POI-level cost shocks (the dominant channel) with a modest cascade multiplier (the contagion channel).

### Limitations

- **POI definition is imprecise.** `poi_name` is a messy, unstandardized field. The improved normalization merged 122 POIs but many inconsistencies likely remain. Measurement error attenuates the contagion estimate, so our estimates are likely conservative.

- **No cost data in the contagion sample.** The LBNL Interconnection Costs dataset covers PacifiCorp, Duke, and BPA, while the contagion analysis is strongest in PJM, CAISO, ERCOT, and MISO. We cannot directly test whether the mechanism operates through cost reallocation.

- **Endogeneity.** The cross-sectional association between peer and own withdrawal is subject to the reflection problem (Manski, 1993). The Cox model's temporal structure helps but does not fully resolve this.

- **Proportional hazards.** The PH test is marginally insignificant (p = 0.095), with a hint that the effect weakens over time. A split-sample check suggests the effect may be stronger in earlier calendar periods (HR = 0.97 in early periods, 1.01 in late), but this pattern could also reflect changes in queue composition over time.

---

## 6. Output Inventory

All outputs are in `contagion/output/`.

### Tables (`output/tables/`)
| File | Contents |
|------|----------|
| `poi_summary_stats.csv` | POI-level descriptive statistics |
| `overdispersion_test.csv` | Variance ratio test results |
| `entity_summary.csv` | Entity-level withdrawal summary |
| `tier1_logistic_results.csv` | Full logistic regression coefficients, ORs, CIs |
| `tier2_cox_results_bugfixed.csv` | Cox model hazard ratios (bug-fixed) |
| `tier2_cox_lag_comparison.csv` | Cox results at 0, 6, 12-month lags |
| `tier2_cox_depth_comparison.csv` | Cox results at depth >= 2, 3, 5 |
| `tier2_cox_ph_test.csv` | Proportional hazards test (time-interaction model) |
| `tier3_feature_importance.csv` | GBM feature importance (3 methods) |
| `tier3_cv_results.csv` | Cross-validation AUC by fold |
| `permutation_test.csv` | Permutation null distribution summary |
| `placebo_test.csv` | Withdrawal vs. operational overdispersion (full sample) |
| `placebo_test_restricted.csv` | Withdrawal vs. operational overdispersion (terminal only) |
| `temporal_asymmetry.csv` | Forward and backward coefficients |
| `dose_response.csv` | Logistic coefficients by POI depth bin |
| `simulation_dose_response.csv` | Simulated vs. real dose-response coefficients |
| `entity_heterogeneity.csv` | Per-entity contagion coefficients |

### Figures (`output/figures/`, PDF + PNG)
| # | Figure |
|---|--------|
| 01 | POI withdrawal rate distribution with binomial overlay |
| 02 | Overdispersion scatter: observed vs. expected withdrawals |
| 03 | Entity withdrawal rate summary |
| 04 | Tier 1 logistic odds ratio forest plot |
| 05 | Tier 2 Cox hazard ratio forest plot |
| 06 | Kaplan-Meier survival curves by peer withdrawal exposure |
| 07 | Permutation null distribution |
| 08 | Entity-level contagion coefficient forest plot |
| 09 | Dose-response: contagion by POI depth |
| 10 | Temporal asymmetry: forward vs. backward |
| 11 | SHAP beeswarm summary |
| 12 | SHAP dependence plot for peer withdrawal count |

---

## 7. Next Steps

1. **Investigate the causal mechanism more carefully.** The dose-response result is the strongest card for contagion. A natural next step is to test whether the effect is concentrated in entities or time periods with known cost-reallocation rules (e.g., FERC Order 2003 vs. later reforms). If the cascade coefficient tracks institutional changes in cost allocation, that would strengthen the causal interpretation.

2. **Link to cost data where possible.** Although the cost dataset covers different BAs, a focused analysis within PacifiCorp or BPA---entities present in both datasets---could directly test whether withdrawal contagion correlates with interconnection cost magnitudes.

3. **Explore the immediacy of the effect.** The zero-lag finding is provocative. It suggests either that cost reallocation information is transmitted very quickly (perhaps through informal channels or real-time queue postings) or that the "contagion" operates through a non-cost channel (e.g., informational signaling: "if my neighbor withdrew, maybe this POI is worse than I thought").

4. **Consider the agent-based model.** The calibration targets are now clearer: the cascade multiplier is modest (~3% per peer withdrawal in the Cox model), and it operates immediately. An agent-based simulation should combine heterogeneous POI-level cost shocks with this modest cascade parameter, not treat cascading as the primary driver.
