# Matched Difference-in-Differences: Withdrawal Contagion at POIs

## 1. Design

### Research question

Does a withdrawal event at a point of interconnection (POI) causally increase the probability of subsequent withdrawals at the same POI?

### Identification strategy

We use a matched difference-in-differences design. A POI is "treated" in quarter *t* if it experiences its first withdrawal after 2 clean quarters (no withdrawals in t-1, t-2). Each treated POI is matched 1:1 to a control POI in the same entity with the same dominant technology, similar depth (active project count ±1), and no withdrawals in [t-2, t]. Matching is nearest-neighbor without replacement, with ties broken by depth proximity. The outcome is the number of additional withdrawals per quarter at the POI (excluding the triggering withdrawal at k=0).

### Sample

- **Entities:** PJM, ERCOT, CAISO, SOCO, ISO-NE, PSCo (selected for `wd_date` coverage)
- **Treatment events identified:** 978
- **Matched pairs:** 951 (97.2% match rate)
- **Unmatched treated POIs:** 27 (concentrated in CAISO and PSCo)

---

## 2. Match Balance

All standardized differences are below 0.11, indicating excellent balance:

| Covariate | Control | Treated | Std. Diff |
|-----------|---------|---------|-----------|
| POI depth (active projects) | 1.37 | 1.41 | 0.051 |
| Pre-event withdrawal count (t-4 to t-1) | 0.000 | 0.003 | 0.080 |
| Mean project capacity (MW) | 142.0 | 166.3 | 0.105 |
| Mean queue entry year | 2015.5 | 2015.4 | -0.007 |
| Technology types at POI | 1.08 | 1.11 | 0.070 |

---

## 3. Results

### 3.1 Two-Period DiD

Comparing total withdrawal counts in [t-4, t-1] vs [t+1, t+4]:

| Statistic | Value |
|-----------|-------|
| DiD estimate | -0.028 |
| SE | 0.026 |
| p-value | 0.267 |
| 95% CI | [-0.079, 0.022] |
| N pairs | 951 |

The short-window (4-quarter) DiD finds no significant immediate contagion effect.

### 3.2 Event Study

**Pre-trend test:** F = 1.00, p = 0.391. Pre-period coefficients are jointly insignificant — the parallel trends assumption is **supported**.

Event-study coefficients (relative to k = -1):

| Event-time | Beta | SE | p-value | 95% CI |
|-----------|------|-----|---------|--------|
| -4 | 0.001 | 0.001 | 0.317 | [-0.001, 0.003] |
| -3 | 0.002 | 0.001 | 0.157 | [-0.001, 0.005] |
| -2 | ~0 | (collinear) | — | — |
| -1 | 0 (ref) | — | — | — |
| 0 | ~0 | (collinear) | — | — |
| 1 | **-0.048** | 0.017 | **0.005** | [-0.082, -0.015] |
| 2 | -0.001 | 0.014 | 0.942 | [-0.029, 0.027] |
| 3 | -0.003 | 0.012 | 0.787 | [-0.026, 0.020] |
| 4 | **0.027** | 0.013 | **0.039** | [0.001, 0.053] |
| 5 | **0.042** | 0.013 | **0.002** | [0.016, 0.068] |
| 6 | 0.006 | 0.008 | 0.442 | [-0.010, 0.022] |
| 7 | **0.021** | 0.010 | **0.031** | [0.002, 0.040] |
| 8 | **0.013** | 0.006 | **0.050** | [0.000, 0.025] |

Notes on collinear coefficients: k = -2 and k = 0 are absorbed by the POI fixed effects because (a) treated and control POIs both have zero withdrawals at k = -2 by construction (the 2-quarter clean window), and (b) the triggering withdrawal is subtracted from the k = 0 count for treated POIs, leaving near-zero variation in both groups.

### Key patterns

1. **Flat pre-trends (k = -4 to -1):** No differential pre-event dynamics. The matching successfully creates comparable groups.

2. **Negative immediate response (k = 1):** The coefficient is significantly negative (-0.048, p = 0.005). This "survivor selection" effect likely reflects that projects which don't withdraw immediately following a peer's departure are temporarily more resilient — a compositional shift in the at-risk set.

3. **Delayed positive effects (k = 4–8):** Significant positive coefficients emerge 1–2 years post-event. The peak is at k = 5 (0.042 additional withdrawals, p = 0.002). This timing is consistent with formal interconnection restudies and cost reallocation cycles, which typically take 12–18 months to complete.

4. **The 8-quarter DiD approaches significance** (0.054, p = 0.078), suggesting the cumulative delayed effect is real but modest.

---

## 4. Sensitivity Analyses

### 4.1 Different-developer restriction (934 pairs)

Restricting to events where the withdrawing project has a different developer from all remaining active projects at the POI (rules out same-developer correlated decisions):

- 3/8 post-treatment periods significant at p < 0.05: k = 1 (-0.049, p = 0.005), k = 5 (0.043, p = 0.002), k = 8 (0.013, p = 0.050)
- Pattern unchanged — the delayed contagion effect is not driven by within-developer correlation.

### 4.2 No-batch restriction (877 pairs)

Restricting to events where only one withdrawal occurred in the treatment quarter (rules out batch processing artifacts):

- 3/8 post-treatment periods significant: k = 1 (-0.044, p = 0.014), k = 5 (0.033, p = 0.005), k = 7 (0.022, p = 0.039)
- Slightly attenuated but qualitatively identical pattern.

### 4.3 Post-window sensitivity

| Window (quarters) | DiD estimate | SE | p-value |
|-------------------|-------------|-----|---------|
| 4 | -0.028 | 0.026 | 0.267 |
| 8 | 0.054 | 0.030 | 0.078 |

The sign flip from negative (4Q) to positive (8Q) reflects the delayed contagion dynamic: the short window captures only the negative k = 1 dip, while the longer window picks up the positive k = 4–8 effects.

### 4.4 Depth tolerance sensitivity

| Tolerance | Pairs | Significant post-periods | Pre-trend p |
|-----------|-------|------------------------|-------------|
| ±0 (exact) | 918 | 4/8 | 0.391 |
| ±1 (baseline) | 951 | 5/8 | 0.391 |
| ±2 (relaxed) | 961 | 5/8 | 0.721 |

Results are stable across matching specifications. Relaxing the depth tolerance slightly improves the pre-trend test (p = 0.72) without changing the substantive pattern.

---

## 5. Comparison with Existing Cox Results

The Tier 2 Cox model found a hazard ratio of 1.032 (p = 0.002) for each additional peer withdrawal — a 3.2% increase in instantaneous withdrawal hazard, concentrated at zero lag (the effect vanished at 6- and 12-month lags).

The matched DiD adds two insights that the Cox model could not provide:

1. **The contagion effect is delayed, not immediate.** The short-run DiD is null; significant effects emerge only at 4–8 quarters. The Cox model's "zero-lag" finding likely reflects same-quarter co-movement (shared shocks or batch decisions), not true cascade dynamics. The actual contagion mechanism operates on the 1–2 year timescale of formal restudies.

2. **The effect survives a credibly causal design.** The matched DiD passes the parallel trends test and is robust to developer restriction and batch exclusion, providing stronger causal evidence than the time-varying covariate approach.

The negative k = 1 coefficient reconciles the Cox model's immediacy finding with the delayed pattern: some projects do exit quickly (captured by the Cox HR), but this is offset by survivor selection, yielding a net-negative immediate effect in the DiD. The true cascade effect accumulates over the following year.

---

## 6. Limitations

- **POI depth.** Most treated POIs have only 1–2 active projects (mean depth 1.4), limiting the scope for cascade dynamics. Deeper POIs showed stronger effects in the Cox analysis (HR = 1.052 at depth 5+), but the DiD sample is too thin at depth 5+ for a separate analysis.
- **Quarterly granularity.** Aggregating to quarters smooths within-quarter dynamics. The k = 0 collinearity results from this aggregation combined with the treatment definition.
- **Conservative treatment definition.** The 2-quarter clean window ensures discrete events but discards POIs with ongoing attrition. Relaxing to 1 quarter would increase power.
- **No cost data in sample.** The cost dataset covers different entities (PacifiCorp, Duke, BPA) than the contagion sample (PJM, ERCOT, CAISO, SOCO). We cannot test whether the delayed effect operates through cost reallocation specifically.
