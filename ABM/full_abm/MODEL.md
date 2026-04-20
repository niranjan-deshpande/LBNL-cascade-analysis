# Multi-POI PJM Interconnection-Queue ABM — Specification

This document specifies the agent-based model in `ABM/full_abm/` for the PJM
old-serial-regime interconnection queue. It targets replication of the
empirical patterns established in `contagion/briefs/contagion_brief.tex`.

## 1. Purpose & scope

- **Scope**: PJM, old serial regime (unbounded reallocation within a POI, no
  deposit pool).
- **Purpose**: reproduce the empirical two-channel picture of withdrawal
  clustering — dominant shared POI conditions, modest delayed cascade from
  formal cost reallocation — and provide a platform for counterfactual
  policy experiments (deposit pool, DFAX-threshold changes, regional cost
  allocation shifts).

## 2. Entities

### Project (Mesa `Agent`)
State:
- `mw` — nameplate capacity ($MW$), bootstrap-sampled from the PJM LBNL panel.
- `t_entry` — queue-entry month.
- `t_cod` — scheduled commercial-operation month. Duration $t_\text{cod} - t_\text{entry}$
  bootstrap-sampled from completed PJM projects (q_date → on_date), with a floor of 24 months.
- `H_base` — baseline headroom (dollars), drawn as $H_i \sim \mathcal{N}(\mu_{\text{POI}}, \sigma_\text{within}^2)$.
- `U` — cumulative allocated upgrade cost (dollars). Initial: $U_i^{(0)} = c_p \cdot \text{mw}_i \cdot 1000$.
- `status ∈ {active, withdrawn, completed}`.
- `t_exit` — month of terminal transition (or `None`).

### POI (plain dataclass container; not a Mesa Agent)
State:
- `poi_id`
- `c_per_kw` — initial cost-intensity ($/kW), drawn once per POI: $c_p \sim \text{LogUniform}(20, 150)$.
- `mu_poi` — POI-level headroom mean, drawn once: $\mu_p \sim \mathcal{N}(\mu_0, \sigma_\text{between}^2)$.
- `hotness` — arrival-weight, drawn once: $h_p \sim \text{Gamma}(1.2, 1.0)$ then normalized.
- `eta` — AR(1) shared shock state, evolves each month.
- `pending` — list of `(t_fire, withdrawer)` pending reallocation events.

### QueueModel (Mesa `Model`)
Owns all POIs and Projects. Holds `Params`, an independent `numpy.Generator`
(`rng`) for reproducibility, the event log, and a Mesa `DataCollector`.

## 3. Time and step order

Monthly discrete time. Horizon default $T = 180$ months (15 years).
Each call to `model.step()` executes (at time $t$):

1. **POI shock update** (AR(1))
   $$\eta_{p,t} = \rho \cdot \eta_{p,t-1} + \sqrt{1 - \rho^2}\cdot \xi_{p,t},\qquad \xi_{p,t}\sim\mathcal{N}(0, \sigma_\text{POI}^2)$$
   Default $\rho = 0.85$ → ~7-month half-life. This is the **shared conditions** channel.

2. **Fire reallocations** for events with `t_fire == t`. See §5.

3. **Poisson arrivals**
   - $N_t \sim \text{Poisson}(\lambda)$, default $\lambda = 10$ projects/month.
   - For each arrival, target POI sampled $\propto h_p$; MW and COD-duration
     bootstrapped from the PJM panel; new Project created with
     $U_0 = c_p \cdot \text{mw} \cdot 1000$ and
     $H_i \sim \mathcal{N}(\mu_p, \sigma_\text{within}^2)$.

4. **Completions**: every active project with $t \ge t_\text{cod}$ transitions to `completed`.
   Its $U$ is frozen (not reallocated).

5. **Decision rule** (simultaneous across active projects):
   $$\text{stay}_{i,t} = \mathbf{1}\left[\; U_{i,t} \;<\; H_i \cdot r(t, t_\text{entry}, t_\text{cod}) + \varepsilon_{i,t} + \eta_{p(i),t}\;\right]$$
   with
   $$r(t, a, b) = r_\text{floor} + (1 - r_\text{floor})\cdot\min\{1, \max\{0, (t-a)/(b-a)\}\},\qquad r_\text{floor}=0.55,$$
   and $\varepsilon_{i,t} \sim \mathcal{N}(0, \sigma_\varepsilon^2)$.

6. **Withdrawals**: every project where `stay==0` transitions to `withdrawn`. For each such
   project, schedule a reallocation event at $t + \tau$ with $\tau \sim \text{Uniform}\{12, \dots, 18\}$
   months (matching the empirical DiD delayed window).

7. **Data collection**.

## 4. Hierarchical headroom

Most variance is **across** POIs (dominant driver of the empirical
overdispersion VR ≈ 1.6 and the strict-control upper-bound pattern). Modest
variance **within** POIs captures project-specific economics:

$$\mu_p \sim \mathcal{N}(\mu_0, \sigma_\text{between}^2),\qquad H_i \mid \mu_p \sim \mathcal{N}(\mu_p, \sigma_\text{within}^2)$$

with $\sigma_\text{within} \ll \sigma_\text{between}$.

## 5. Reallocation (PJM DFAX-style, with local/network decomposition)

Empirically, only ~10–15% of shared interconnection costs are POI-local
(Attachment Facilities / TOIF); the remaining ~85–90% are network upgrades
whose costs are driven by power-flow contributions and should in principle
spread across projects at *other* POIs. We model this with a fixed split
parameter $\alpha = $ `alpha_local` (default $0.15$).

**Topology.** At construction, each POI is assigned 2D coordinates
$(x_p, y_p) \sim \text{Uniform}([0,1]^2)$. A dense weight matrix
$W_{pq} = \exp(-\|x_p - x_q\| / s)$ (with $s = $ `network_distance_scale`,
default $0.3$; diagonal zeroed, rows normalized) defines distance-biased
sampling probabilities for network-share recipients.

When a reallocation event for withdrawer $j$ at POI $p$ fires at time $t$,
split $U_j = U_j^\text{loc} + U_j^\text{net}$ with $U_j^\text{loc} = \alpha U_j$:

**Local share $U_j^\text{loc}$** (DFAX, unchanged logic):
1. Let $A$ = currently active projects at $p$; $\text{share}_q = \text{mw}_q / (\text{mw}_j + \sum_{q \in A} \text{mw}_q)$.
2. Eligible set $E = \{q \in A : \text{share}_q > \theta\}$, default $\theta = 0.03$.
3. If $E \neq \emptyset$: $U_q \mathrel{{+}{=}} U_j^\text{loc} \cdot \mathrm{mw}_q / \sum_{q' \in E} \mathrm{mw}_{q'}$. Else the local share evaporates.

**Network share $U_j^\text{net}$** (new):
1. Sample $k = $ `network_fanout` other POIs without replacement using row $p$ of $W$ (default $k = 20$).
2. Split $U_j^\text{net}$ equally across the $k$ sampled POIs.
3. At each recipient POI $q$, distribute its piece pro-rata by MW across all active projects at $q$ (no DFAX threshold — the share is already small per project after the fanout). If $q$ has no active projects, that piece evaporates.

The 12–18 month lag matches the empirical DiD's delayed-positive window; the
DFAX filter on the local share protects small co-POI projects from the
cascade. The within-POI matched DiD identifies only the local channel (plus
any induced spillover when a nearby POI happens to be in both the network
fanout and the control pool), so the calibrated $\alpha$ directly controls
per-peer cost shock magnitude — the dominant lever for the event-study peak
height.

## 6. Parameters

| Parameter | Default | Calibration rationale |
|---|---|---|
| `n_pois` | 400 | Small-medium scale for iteration |
| `horizon_months` | 180 | 15 years; covers full project lifecycles |
| `arrivals_per_month` | 10 | Matches LBNL PJM ~12.7/month at the entity level, scaled to 400 POIs |
| `mu_H_center` | $9.5 \times 10^6$ | Tuned so ON completion rate ≈ 0.18 (target PJM 0.197) |
| `sigma_between` | $3.5 \times 10^6$ | Tuned so VR ≈ 1.6 under OFF (structural POI heterogeneity only) |
| `sigma_within` | $2.0 \times 10^6$ | Modest within-POI heterogeneity |
| `sigma_eps` | $2.2 \times 10^6$ | Idiosyncratic monthly noise; large enough that reallocation isn't knife-edge |
| `sigma_poi` | $2.2 \times 10^6$ | Unconditional SD of $\eta_{p,t}$; innovation variance scaled by $\sqrt{1-\rho^2}$ |
| `rho_poi` | 0.85 | ~7-month half-life; matches the empirical temporal decay pattern |
| `ramp_floor` | 0.55 | $H_i$ ramps from $0.55H$ at entry to $H$ at COD |
| `dfax_threshold` | 0.03 | PJM rule (`ABM/notes.md` §2b) |
| `alpha_local` | 0.15 | POI-local fraction of shared cost (Attachment Facilities / TOIF share); 1 − α is network-upgrade share routed to other POIs |
| `network_fanout` | 20 | Other POIs sharing each withdrawal's network cost |
| `network_distance_scale` | 0.3 | exp(−d/s) bias on unit-square topology |
| `lag_low, lag_high` | 12, 18 | Matches DiD peak at 4–8 quarters |
| `dollars_per_kw` | LogUniform(20, 150) | **Initial** cost allocations; lower than the brief's 71–563/kW headline since those are *final* costs including reallocations |

## 7. Validation (baseline, 10 seeds)

Targets taken from `contagion/briefs/contagion_brief.tex` (PJM-specific where
available, national otherwise).

### 7a. PJM-scale run (2000 POIs, 240 months, ~6000 projects/rep, 37s / 10 seeds)

| Metric | ABM (ON) | ABM (OFF) | PJM target | Read |
|---|---|---|---|---|
| Completion rate (terminal) | 0.221 | 0.262 | 0.197 | ✓ within [0.17, 0.23] |
| POI variance ratio | 1.67 | 1.51 | ~1.6 | ✓ on target |
| Dose response (wd_rate by depth 2 → 10+) | 0.70 → 0.78 | 0.69 → 0.71 flat | monotonic rise | ✓ |
| Cascade channel (ON − OFF), k=1 | +0.7 pp | — | ~null / slight negative in DiD | ✓ |
| Cascade channel, k=2–4 | all within [−0.6, +0.6] pp | — | pre-reallocation null | ✓ |
| Cascade channel, **k=5** | **+6.3 pp** | — | matched DiD peak +2.9 pp | shape + timing ✓; mag ~2× |
| Cascade channel, k=6 | +4.9 pp | — | — | ✓ |
| Cascade channel, k=7 | +2.6 pp | — | — | ✓ |
| Cascade channel, k=8 | +0.1 pp | — | decay | ✓ |

The persistent ~2× overshoot on the cascade peak magnitude is reproducible
at both 400-POI and 2000-POI scales, so it is a structural feature rather
than sampling noise. The likely driver is that our event study uses a
random non-treated POI as the control, rather than the propensity-matched
twin used by `contagion/matched_did.py`. Running the full matched-DiD
pipeline on simulated panels is the natural next validation.

### 7b. Small-scale run (400 POIs, 180 months, ~1750 projects, 1.6s / 10 seeds)

Same qualitative pattern: completion 0.175, VR 1.94, cascade channel
flat at k=1–4, peak +6.1 to +6.5 pp at k=5–6, decay to ~0 at k=8.
Small-scale SDs are ~2× the large-scale SDs, as expected.

### 7c. Full matched-DiD pipeline on simulated panels (3 reps × 2000 POIs × 240 months)

The most rigorous validation: run the *exact* `contagion/matched_did.py`
estimator used on real LBNL data against ABM-generated LBNL-schema panels.
Driver: `validate_matched_did.py`.

**Event-study coefficients (pooled across 3 reps; quarterly outcome = POI
withdrawals per quarter):**

| k | ABM (this run) | Empirical (brief) | Match? |
|---|---|---|---|
| −4 | +0.015 | ~0 | within SE |
| −3 | +0.001 | ~0 | ✓ |
| Pre-trend F | 0.10–1.06 (p > 0.37) | F = 0.47 (p = 0.71) | ✓ passes |
| **k = 1** | **−0.109** (p ≈ 0.007 pooled) | **−0.038** (p = 0.009) | ✓ sign + significance; 3× empirical magnitude |
| k = 2 | −0.010 | ~null | ✓ |
| k = 3 | +0.031 | ~null | slight overshoot |
| k = 4 | +0.084 | ~+0.02 | shape ✓; 4× |
| **k = 5 (peak)** | **+0.097** (p ≈ 5e-5) | **+0.029** (p = 0.004) | ✓ shape; 3× magnitude |
| k = 6 | +0.077 | +0.025 | ✓ shape; 3× |
| k = 7 | +0.035 | ~+0.02 | ✓ |
| k = 8 | +0.051 | — | — |
| **Two-period DiD** (4Q pre vs 4Q post) | **−0.020** | **−0.008 (null)** | **✓ near-null** |
| Matching rate | 99.2–99.7% | 95.6% | ✓ |
| Treated events per rep | ~370 | 976 (pooled across national Tier-2) | proportional |

**Interpretation.** The ABM reproduces the empirical event-study *shape*
nearly exactly:

1. Statistically significant **negative dip at k=1** (the brief's central k=1
   finding of −0.038, p=0.009 is replicated with sign and significance).
2. **Null at k=2–3** (pre-reallocation quiet period).
3. **Sharp onset at k=4**, **peak at k=5**, decay through k=8 — matching the
   12–18 month reallocation lag convolved with quarterly binning.
4. **Pre-trends pass** in every replication (F ~ 0.1–1.1, well above the DiD's
   parallel-trends threshold).
5. **Two-period DiD of −0.020 is near-null** (empirical: −0.008), reproducing
   one of the brief's most-quoted headline numbers. The k=1 dip and the k=5–7
   positives offset in the 4-quarter pooled window.

**Known calibration gap.** Event-study coefficient *magnitudes* are ~3× the
empirical values (peak +0.097 vs empirical +0.029). Tuning further requires
trading off other targets — raising `sigma_eps` to 6e6 pushes the peak down
to +0.086 but swings the two-period DiD to −0.077 and pre-trend F to 2.5.
Current parameters are a local sweet spot: DiD near null, pre-trends clean,
shape pristine, magnitudes 3× too hot.

Closing this magnitude gap is the main calibration task remaining. Likely
levers: finer POI-level heterogeneity in `c_per_kw`, realistic attrition of
reallocation dollars (fraction absorbed by the system analogous to the new
regime's pool mechanism), or adding correlated intra-POI shocks to reduce
reallocation-induced synchronization.

### Interpretation

- The **shared-conditions channel** is visible in both regimes: treated POIs
  (those that saw a first withdrawal) show excess withdrawals immediately at
  k=1 even with reallocation OFF, driven by the persistent AR(1) $\eta_{p,t}$.
  This is the `+5.8 pp` at k=1 under OFF — empirically it shows up as the
  strict-control upper-bound (+6–8 pp) in the brief.
- The **cascade channel** only appears under ON, delayed to k=5–6 (15–18
  months; the lag distribution convolved with the quarterly binning). The
  ON − OFF contrast isolates this channel cleanly.
- Overall cascade magnitude is ~2× the matched-DiD peak (+6 pp vs +2.9 pp).
  One likely contributor: our matched-null for the event study is a random
  non-treated POI, not a propensity-matched twin as in the brief's DiD.
  Expect the magnitude to shrink when we later run the full `contagion/`
  pipeline on simulated panels.

## 8. File layout

```
ABM/full_abm/
├── calibrate.py   — PJM-only LBNL bootstrap samplers (cached pickle)
├── model.py       — Project (Mesa Agent), POI, QueueModel (Mesa Model), Params
├── validation.py  — completion rate, variance ratio, dose response, mini event study
├── run.py         — multi-seed sweep, ON vs OFF comparison
└── output/        — panel CSVs
```

## 9. Known limitations & deferred work

- **Cascade magnitude ~2× target**: likely shrinks with a propensity-matched
  control, to be confirmed by running the `contagion/matched_did.py` pipeline
  on simulated panels.
- **Peer-effect OR is compressed** (~4 vs national ~10) because the overall
  withdrawal rate is very high (matching PJM's 80%+ withdrawal rate); at
  high base rates, the logistic OR mechanically compresses.
- **Deposit-pool counterfactual** is not yet implemented; the cycle-regime
  reform is the next modeling step.
- **Project-entry covariates**: tech type, developer, state are not yet
  tracked. Needed for entity-level stratification and developer-level
  heterogeneity tests.
- **Re-running full `contagion/` pipeline on simulated output** is deferred
  until the Mesa model is stabilized; that's the most rigorous validation.
- **Cost formulation**: the initial $c_p$ is a POI-level scalar; in reality
  different upgrades (direct-connection, network upgrades) have different
  cascade susceptibility. A future version could split $U$ into shares
  against different facility types.
