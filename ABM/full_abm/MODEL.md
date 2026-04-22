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
- `U_initial` — frozen copy of $U_i^{(0)}$ captured at construction; used only by the deposit-pool forfeiture rule in §10 (never mutated).
- `status ∈ {active, withdrawn, completed}`.
- `t_exit` — month of terminal transition (or `None`).

### POI (plain dataclass container; not a Mesa Agent)
State:
- `poi_id`
- `c_per_kw` — initial cost-intensity ($/kW), drawn once per POI: $c_p \sim \text{LogUniform}(20, 150)$.
- `mu_poi` — POI-level headroom mean, drawn once: $\mu_p \sim \mathcal{N}(\mu_0, \sigma_\text{between}^2)$.
- `hotness` — arrival-weight, drawn once: $h_p \sim \text{Gamma}(1.2, 1.0)$ then normalized.
- `x`, `y` — unit-square topology coordinates used for distance-biased network-fanout sampling (§5).
- `eta` — AR(1) shared shock state, evolves each month.
- `projects` — list of all `Project`s ever assigned to this POI (regardless of status).
- `pending` — list of `(t_fire, withdrawer)` pending reallocation events.

### QueueModel (Mesa `Model`)
Owns all POIs and Projects. Holds:
- `params`, plus `t` (current month).
- `rng` — primary `numpy.Generator` seeded from `params.rng_seed`; drives POI-shock draws, arrivals, MW/duration bootstraps, stay-rule noise, reallocation lag sampling, and local-share target selection.
- `fanout_rng` — dedicated `numpy.Generator` seeded `params.rng_seed + 7919`, used *only* for selecting network-fanout target POIs. Kept separate so that ON_no_pool and ON_with_pool runs with the same `rng_seed` draw the same network targets and differ only in the pool-absorption branch.
- `_network_weights` — precomputed $(n \times n)$ row-stochastic distance-weight matrix for network-fanout sampling.
- `_poi_weights` — normalized Gamma hotness vector used as arrival probabilities.
- `event_log` — list of `(t, event_type, project_id, poi_id)` tuples.
- `deposit_pool` (float, initialized 0) and `deposit_pool_log` (list of `(t, action, amount)`) — always present; only mutated when `params.deposit_pool_enabled` is True (see §10).
- `datacollector` — Mesa `DataCollector` recording per-step active / cumulative-withdrawn / cumulative-completed counts.

## 3. Time and step order

Monthly discrete time. Horizon default $T = 180$ months (15 years).
Each call to `model.step()` executes (at time $t$):

1. **POI shock update** (AR(1))
   $$\eta_{p,t} = \rho \cdot \eta_{p,t-1} + \sqrt{1 - \rho^2}\cdot \xi_{p,t},\qquad \xi_{p,t}\sim\mathcal{N}(0, \sigma_\text{POI}^2)$$
   Default $\rho = 0.85$ gives impulse-response half-life $-\ln 2 / \ln \rho \approx 4.3$ months (e-folding time $1/(1-\rho) \approx 6.7$ months). This is the **shared conditions** channel. The innovation SD is computed in code as `sigma_poi * sqrt(1 - rho^2)` so that the stationary SD of $\eta$ is exactly `sigma_poi`.

2. **Fire reallocations** for events with `t_fire == t`. See §5. If `deposit_pool_enabled`, pool absorption against the network share of the reallocating cost happens inside this step, before co-POI and cross-POI distribution.

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
   and $\varepsilon_{i,t} \sim \mathcal{N}(0, \sigma_\varepsilon^2)$. If $t_\text{cod} \le t_\text{entry}$ (degenerate duration), the ramp returns $1$.

6. **Withdrawals**: every project where `stay==0` transitions to `withdrawn`. For each such
   project:
   - If `deposit_pool_enabled`, forfeit $(\phi_0 + (1-\phi_0) f) \cdot U_i^{(0)}$ into the pool (see §10.1).
   - If `reallocation_enabled`, schedule a reallocation event at $t + \tau$ with $\tau \sim \text{Uniform}\{12, \dots, 18\}$ months (matching the empirical DiD delayed window). Events scheduled beyond `horizon_months` are dropped.

7. **Data collection**.

Steps 5 and 6 are implemented together in `_decisions_and_withdrawals`: decisions are evaluated simultaneously against the current shocks, then all projects with `stay==0` transition atomically.

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
split $U_j = U_j^\text{loc} + U_j^\text{net}$ with $U_j^\text{loc} = \alpha U_j$.
If `deposit_pool_enabled` (§10), the pool absorbs against $U_j^\text{net}$
first, possibly reducing it to zero; only the residual $U_j^\text{net}$
continues through the network-fanout step below. The local share
$U_j^\text{loc}$ is never absorbed. If `pj.U <= 0` on entry (should not
happen under normal operation), the event is silently dropped. When
`reallocation_enabled` is False, no events are ever scheduled (see §3
step 6), so this step is a no-op; due-date events in `pending` are still
popped off each step, but no cost is redistributed.

**Local share $U_j^\text{loc}$** (DFAX, unchanged logic):
1. Let $A$ = currently active projects at $p$; $\text{share}_q = \text{mw}_q / (\text{mw}_j + \sum_{q \in A} \text{mw}_q)$.
2. Eligible set $E = \{q \in A : \text{share}_q > \theta\}$, default $\theta = 0.03$.
3. If $E \neq \emptyset$: $U_q \mathrel{{+}{=}} U_j^\text{loc} \cdot \mathrm{mw}_q / \sum_{q' \in E} \mathrm{mw}_{q'}$. Else the local share evaporates.

**Network share $U_j^\text{net}$** (new):
1. Sample $k = \min(\text{`network\_fanout`}, n_\text{POI} - 1)$ other POIs without replacement using row $p$ of $W$ (default $k = 20$). Target POIs are drawn from `fanout_rng`, not the primary `rng`, so that runs at the same `rng_seed` with different reallocation / pool settings share network-target draws (see §2 QueueModel).
2. Split $U_j^\text{net}$ equally across the $k$ sampled POIs: each gets $U_j^\text{net} / k$.
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
| `n_pois` | 400 | Small-medium scale for iteration; PJM-scale sweeps set this to 2000. |
| `horizon_months` | 180 | 15 years; covers full project lifecycles. PJM-scale sweeps use 240. |
| `burn_in_months` | 36 | Ignored when computing calibration/validation metrics. |
| `arrivals_per_month` | 10.0 | PJM-scale sweeps use 25.0. Matches LBNL PJM ~12.7/month at entity level, scaled to 400 POIs. |
| `hotness_shape`, `hotness_scale` | 1.2, 1.0 | Gamma shape/scale for POI arrival-weight draw; heavier tail → deeper POIs. |
| `mu_H_center` | $13.0 \times 10^6$ | Tuned against the current cost scale so ON completion rate lands near the PJM 0.197 target. |
| `sigma_between` | $3.5 \times 10^6$ | Tuned so VR ≈ 1.6 under OFF (structural POI heterogeneity only). |
| `sigma_within` | $2.0 \times 10^6$ | Modest within-POI heterogeneity. |
| `sigma_eps` | $4.0 \times 10^6$ | Idiosyncratic monthly noise; large enough that reallocation isn't knife-edge. |
| `sigma_poi` | $2.2 \times 10^6$ | Unconditional SD of $\eta_{p,t}$; innovation variance scaled by $\sqrt{1-\rho^2}$. |
| `rho_poi` | 0.85 | AR(1) persistence; half-life ≈ 4.3 months. |
| `ramp_floor` | 0.55 | $H_i$ ramps from $0.55H$ at entry to $H$ at COD. |
| `reallocation_enabled` | `True` | Turn off to run the OFF regime (no scheduling at withdrawal). |
| `dfax_threshold` | 0.03 | PJM rule (`ABM/notes_for_self/notes.md` §2b). |
| `lag_low, lag_high` | 12, 18 | Inclusive; Matches DiD peak at 4–8 quarters. |
| `alpha_local` | 0.15 | POI-local fraction of shared cost (Attachment Facilities / TOIF share); 1 − α is network-upgrade share routed to other POIs. |
| `network_fanout` | 20 | Other POIs sharing each withdrawal's network cost. |
| `network_distance_scale` | 0.3 | exp(−d/s) bias on unit-square topology. |
| `poi_topology_seed` | `None` | If `None`, topology draws use `rng_seed`; set explicitly to decouple topology from run RNG across sweeps. |
| `dollars_per_kw_low, _high` | 20, 150 | **Initial** cost allocations (LogUniform). Lower than the brief's 71–563/kW headline since those are *final* costs including reallocations. |
| `deposit_pool_enabled` | `False` | Counterfactual switch; see §10. |
| `deposit_floor` | 0.10 | $\phi_0$ in the deposit-forfeiture ramp; see §10. |
| `rng_seed` | 42 | Seeds `rng`; `fanout_rng` is seeded from `rng_seed + 7919`. |

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

> **Caveat (added after §10).** The matched-DiD event-study coefficients
> below are not a clean estimate of the reallocation cascade.
> `diag_cascade_decomposition.py` (30 seeds) shows the OFF regime alone
> reproduces most of the event-study shape — OFF k=1 = −0.111, OFF peak
> (k=6) = +0.038 — and cascade (ON − OFF) CIs exclude zero only at k=5
> (+0.014 ± 0.005) and k=8 (+0.010 ± 0.004); at all other k's the CI
> includes zero. The ABM demonstrates that matched-DiD estimates in this
> DGP setting are sensitive to residual selection on permanent POI
> heterogeneity; the empirical +0.029 should be interpreted as an
> **upper bound on the causal cascade effect**, not a direct measurement
> of it. §10 uses the `mini_event_study` ON − OFF contrast as the cleaner
> cascade gauge. See `notes_for_self/forward_path_4-20.md` for the full
> decomposition.

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

**Reframed: "3× too hot" is an identification artifact, not a calibration
gap.** `diag_cascade_decomposition.py` decomposes the ABM's matched-DiD
event study into OFF (no reallocation) and ON − OFF (the cascade channel).
The OFF regime alone reproduces most of the event-study shape — POI/time
fixed effects plus propensity matching soak up the cascade alongside the
selection-on-$\mu_\text{POI}$ signal that matched-DiD was meant to remove.
The ABM demonstrates that matched-DiD estimates in this DGP are sensitive
to residual selection on permanent POI heterogeneity, so the empirical
+0.029 peak should be read as an upper bound on the causal cascade. The
$\sigma_\text{POI}$ sweep (`sweep_sigma_poi.py`) confirms that varying
μ-POI heterogeneity does not collapse the OFF shape, consistent with the
confound being structural rather than parameter-sensitive.

See `notes_for_self/forward_path_4-20.md` for the full decomposition.

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
├── MODEL.md                           — this document
├── calibrate.py                       — PJM-only LBNL bootstrap samplers (cached pickle)
├── model.py                           — Params, POI, Project (Mesa Agent), QueueModel (Mesa Model)
├── validation.py                      — completion rate, variance ratio, dose response, mini_event_study, full_report
├── run.py                             — multi-seed ON vs OFF sweep
├── validate_matched_did.py            — runs contagion/matched_did.py on ABM-generated LBNL-schema panels (§7c)
├── diag_off_k1.py                     — diagnostic: decomposes the OFF k=1 dip
├── diag_cascade_decomposition.py      — 30-seed matched-DiD ON vs OFF decomposition (§7c caveat)
├── sweep_alpha.py                     — sensitivity sweep over alpha_local
├── sweep_sigma_poi.py                 — sensitivity sweep over sigma_poi
├── run_deposit_pool.py                — 3-regime × 30-seed deposit-pool counterfactual (§10)
└── output/                            — panel CSVs, sweep results, logs
```

## 9. Known limitations & deferred work

- **Cascade magnitude ~2× target**: likely shrinks with a propensity-matched
  control, to be confirmed by running the `contagion/matched_did.py` pipeline
  on simulated panels.
- **Peer-effect OR is compressed** (~4 vs national ~10) because the overall
  withdrawal rate is very high (matching PJM's 80%+ withdrawal rate); at
  high base rates, the logistic OR mechanically compresses.
- **Deposit-pool counterfactual** — implemented; see §10.
- **Project-entry covariates**: tech type, developer, state are not yet
  tracked. Needed for entity-level stratification and developer-level
  heterogeneity tests.
- **Re-running full `contagion/` pipeline on simulated output** is deferred
  until the Mesa model is stabilized; that's the most rigorous validation.
- **Cost formulation**: the initial $c_p$ is a POI-level scalar; in reality
  different upgrades (direct-connection, network upgrades) have different
  cascade susceptibility. A future version could split $U$ into shares
  against different facility types.

## 10. Deposit-pool counterfactual

PJM's new cycle regime posts a schedule of increasing security deposits
(RD1 = \$4K/MW at application, RD2 = 10% of allocated cost at DP1, RD3 = 50%
at DP2, RD4 = 100% at DP3; see `notes_for_self/notes.md` §3a–3b). Forfeited
deposits fund a system-wide pool that absorbs underfunded upgrade costs
*before* they cascade onto remaining projects. This section adds a minimal
version of that mechanism to the ABM and measures its effect on the
cascade channel.

### 10.1 Mechanism

The ABM has no study-phase structure, so the RD schedule is approximated
by a linear age ramp.

**Forfeiture** (at withdrawal, §3 step 6). When project $i$ withdraws at
time $t$, let
$f = \min\{1, \max\{0, (t - t_\text{entry}) / (t_\text{cod} - t_\text{entry})\}\}$
(with $f = 1$ if $t_\text{cod} \le t_\text{entry}$). The forfeited amount
is
$$\text{forfeit}_i = \bigl(\phi_0 + (1 - \phi_0) \cdot f\bigr) \cdot U_i^{(0)},$$
pegged to the project's **original** allocation $U_i^{(0)}$ (captured at
entry as `U_initial`), not its running $U_i$, since PJM security deposits
are posted against each project's own allocated cost and not against costs
inherited via reallocation. The amount is added to `deposit_pool` and
logged as `("deposit", forfeit)` in `deposit_pool_log`.

**Absorption** (at reallocation firing, §3 step 2). When a scheduled
reallocation for withdrawer $j$ fires at POI $p$, the standard local /
network split is computed first:
$$U_j^\text{loc} = \alpha \, U_j, \qquad U_j^\text{net} = (1-\alpha) \, U_j.$$
Then, if the pool is non-empty and $U_j^\text{net} > 0$, the pool absorbs
against the network share only:
$$\text{absorbed} = \min(\text{pool}, U_j^\text{net}), \qquad U_j^\text{net} \mathrel{-}= \text{absorbed}, \qquad \text{pool} \mathrel{-}= \text{absorbed}.$$
The residual $U_j^\text{net}$ (possibly zero) then follows the distance-
biased fanout in §5; the local share $U_j^\text{loc}$ is untouched and
falls on co-POI peers as usual. The pool is monotone-cumulative (no
reset). Absorptions are logged as `("absorb", amount)` in
`deposit_pool_log`; `pj.U` itself is not mutated.

**Why the network share only.** The local share is Attachment-Facilities /
TOIF-style cost that is largely same-POI (and frequently same-developer)
and is not the economic target of PJM's deposit pool — the pool is
designed to cover system-wide network-upgrade shortfalls. Operationally
this also keeps the `mini_event_study` co-POI contrast measuring the
*unabsorbed* local cascade, so any pool effect visible there comes
through the mechanism of fewer network-driven withdrawals propagating
into local cascades downstream (not from the pool eating the local
channel directly).

### 10.2 Parameters

| Parameter | Default | Notes |
|---|---|---|
| `deposit_pool_enabled` | `False` | When `False`, pool state is initialized but never mutated; no behavioral change. |
| `deposit_floor` | 0.10 | $\phi_0$ at $f = 0$. Calibrated to RD1 ($4K/MW ≈ 8% of $U$ for a typical project); rounded up as a defensible minimum. |

### 10.3 Results

3-regime × 30-seed sweep (`run_deposit_pool.py`) at the §7c configuration
(2000 POIs, 240 months, $\sigma_\text{POI} = 2.2 \times 10^6$,
$\alpha = 0.15$, seeds 100–129). Cascade gauge = `mini_event_study`
treated − control `diff`; per-seed cascade = ON_no_pool − OFF. Mean ± SE
across seeds, 95% CI = ±1.96·SE.

| k | cascade | 95% CI | cascade_pool | 95% CI |
|---|---|---|---|---|
| 1 | +0.020 | **[+0.006, +0.034]** | +0.011 | [−0.001, +0.023] |
| 2 | +0.002 | [−0.011, +0.015] | +0.002 | [−0.011, +0.014] |
| 3 | +0.003 | [−0.008, +0.014] | +0.005 | [−0.004, +0.013] |
| 4 | +0.012 | **[+0.002, +0.023]** | +0.007 | [−0.003, +0.017] |
| 5 | +0.003 | [−0.003, +0.010] | +0.001 | [−0.007, +0.009] |
| 6 | +0.005 | **[+0.001, +0.009]** | +0.004 | [+0.000, +0.009] |
| 7 | +0.004 | [−0.002, +0.010] | +0.009 | [+0.004, +0.013] |
| 8 | +0.004 | [−0.002, +0.009] | +0.002 | [−0.003, +0.007] |

Cascade is detectable (CI excludes zero) at $k \in \{1, 4, 6\}$; elsewhere
it sits inside noise. The peak-k argmax over $[4,8]$ lands at $k=4$ with
$+0.012 \pm 0.005$, but peak-picking is upward-biased and the per-k CIs
above are the honest summary.

**Proportional reduction is uninterpretable at this sample size.** At the
argmax $k=4$, reduction $= 1 - \text{cascade\_pool}/\text{cascade} =
-0.92 \pm 1.00$ (95% CI $[-2.87, +1.04]$). The ratio's denominator is at
noise ($+0.012 \pm 0.005$) and the per-seed ratio has a very heavy tail;
larger-magnitude point estimates do not change this. Reporting a single
"% reduction" headline is not defensible here.

**Direct withdrawal-count prevention** (primary policy headline). Per-seed
$\text{prevented} = \text{wd}_\text{ON\_no\_pool} - \text{wd}_\text{ON\_with\_pool}$,
averaged across 30 seeds:

| Regime | mean total_wd ± SE |
|---|---|
| OFF | 4407.2 ± 14.3 |
| ON_no_pool | 4762.7 ± 15.1 |
| ON_with_pool | 4702.4 ± 16.6 |

$\text{prevented/run (240-mo horizon)} = +60.4 \pm 20.0$ (95% CI
$[+21.2, +99.5]$). Scaling to PJM's 1,800 arrivals/year (ABM has
$25 \times 12 = 300$ arrivals/year, so scale factor 6.0):

$$\text{prevented/yr at PJM scale} = 18.1 \pm 6.0 \quad (95\%\,\text{CI}\,[+6.4, +29.9]).$$

This CI **excludes zero**: the pool produces a measurable reduction in
total withdrawals. The count-based CI is well-behaved because it does not
divide by a noisy denominator; it is the headline to quote.

**Pool accounting** (30-seed mean): total deposited 4.86e9, total absorbed
4.83e9, end-of-horizon residual 2.4e7 (0.5% of deposited). The pool is
sized so that essentially every absorption opportunity finds enough
balance; $\text{absorbed} \leq \text{deposited}$ holds in every run.

### 10.4 Caveats

- **`mini_event_study` ON − OFF only sees the co-POI cascade.** At
  $\alpha = 0.15$, 85% of each withdrawer's $U$ is distributed across 20
  other POIs; `mini_event_study` only observes the co-POI peers. The
  direct withdrawal-count headline (18 ± 6 /yr at PJM scale) captures the
  full effect — network-fanout withdrawals do show up in total counts —
  but the per-k cascade CIs above do not.
- **`deposit_floor = 0.10` approximates RD1.** A withdrawal at $f = 0$
  forfeits 10% of the original $U_i^{(0)}$; at $f = 1$ it forfeits 100%.
  The linear ramp approximates the RD1→RD4 trajectory but does not
  reproduce the discrete DP milestones. Raising the floor models the
  later RD phases; we have not swept this.
- **BOTE scales arrivals, not withdrawals.** The 18 ± 6 /yr figure scales
  by (PJM arrivals/yr) / (ABM arrivals/yr) = 1800 / 300 = 6.0. If the
  true PJM arrival rate is different, scale accordingly.
- **Matched-DiD is not used here.** §7c's matched-DiD ON − OFF is
  confounded by fixed-effects absorption of both the cascade and
  persistent POI heterogeneity; `mini_event_study` (no fixed effects) is
  the cleaner cascade gauge.
- **Bug-fix history (2026-04).** Three bugs were fixed before the final
  30-seed run: (a) absorption now targets $(1-\alpha) U_j$ rather than
  full $U_j$, so local-share cascade onto co-POI peers is not
  over-reduced; (b) forfeiture is pegged to original $U_i^{(0)}$
  (`U_initial`) rather than running $U$ (which had been inflated by prior
  reallocations); (c) a dedicated `fanout_rng` (seeded
  `rng_seed + 7919`) drives network-target selection so that ON_no_pool
  and ON_with_pool draw the same targets for any fire event they share,
  removing RNG-trajectory divergence as a source of between-regime noise.
