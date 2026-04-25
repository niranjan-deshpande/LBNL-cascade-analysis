# Multi-POI PJM Interconnection-Queue ABM — Specification

Here, we specify the agent-based model in `ABM/full_abm/` for the PJM
old-serial-regime interconnection queue. It targets replication of the
empirical patterns established in `contagion/briefs/contagion_brief.tex`.
We then apply a policy counterfactual---underfunded-pool---and simulate results.

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
- `U` — cumulative allocated upgrade cost (dollars). Initial: $U_i^{(0)} = c_p \cdot \text{mw}_i \cdot 1000$. The multiplaction by 1000 is for conversion from kW to mW
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
> `diagnostics/diag_cascade_decomposition.py` (30 seeds) shows the OFF regime alone
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
gap.** `diagnostics/diag_cascade_decomposition.py` decomposes the ABM's matched-DiD
event study into OFF (no reallocation) and ON − OFF (the cascade channel).
The OFF regime alone reproduces most of the event-study shape — POI/time
fixed effects plus propensity matching soak up the cascade alongside the
selection-on-$\mu_\text{POI}$ signal that matched-DiD was meant to remove.
The ABM demonstrates that matched-DiD estimates in this DGP are sensitive
to residual selection on permanent POI heterogeneity, so the empirical
+0.029 peak should be read as an upper bound on the causal cascade. The
$\sigma_\text{POI}$ sweep (`diagnostics/sweep_sigma_poi.py`) confirms that varying
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
├── model.py                           — Params, POI, Project (Mesa Agent), QueueModel (Mesa Model)
├── validation.py                      — completion rate, variance ratio, dose response, mini_event_study, full_report
├── validate_matched_did.py            — runs contagion/matched_did.py on ABM-generated LBNL-schema panels (§7c)
├── calibrate.py                       — PJM-only LBNL bootstrap samplers (cached pickle)
├── experiments/
│   ├── run.py                         — multi-seed ON vs OFF sweep (entry point)
│   ├── run_cluster_bound.py           — cluster-bounded reallocation, 90-seed W sweep (§11, primary)
│   ├── run_deposit_pool.py            — deposit-pool counterfactual, 90 seeds (§10, secondary)
│   ├── run_channel_decomp.py          — local vs network channel decomposition at W=12 (§11.4)
│   ├── run_alpha_sensitivity.py       — α_local robustness for the W=12 headline (§11.5)
│   └── run_matched_did_compare.py     — runs contagion/matched_did.py on ABM panels under 3 regimes (§11.3)
├── diagnostics/
│   ├── diag_cascade_decomposition.py  — matched-DiD ON vs OFF decomposition (§7c caveat)
│   ├── diag_off_k1.py                 — decomposition of the OFF k=1 dip
│   ├── sweep_alpha.py                 — sensitivity sweep over alpha_local
│   └── sweep_sigma_poi.py             — sensitivity sweep over sigma_poi
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

**30-seed version** retained below for comparability. A **90-seed
re-run** (`experiments/run_deposit_pool.py`, seeds 100-189) at the same BASE
config gave: OFF = 4419.3 ± 7.8, ON_no_pool = 4752.5 ± 8.6, ON_with_pool
= 4702.2 ± 9.5. Prevention = **+15.1 ± 3.4 /yr PJM, 95% CI
[+8.4, +21.8]** — within the 30-seed CI but meaningfully tighter. The
OFF and ON_no_pool means are identical (to 1 decimal) with the
cluster-bound run at the same seeds, confirming DGP consistency.
**Crucially**, at n=90 the per-k cascade at peak k=7 is +0.0090 ± 0.0017
for ON_no_pool vs +0.0093 ± 0.0014 for ON_with_pool — essentially
identical. See §11.3 for the comparative discussion: the pool is
effective on total withdrawal counts but invisible to the within-POI
mini_event_study signal, because the withdrawals it prevents are at
*other* POIs (network-fanout targets), not at the original withdrawer's
POI.

3-regime × 30-seed sweep (original version of `experiments/run_deposit_pool.py`) at the §7c configuration
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

## 11. Cluster-bounded reallocation counterfactual

FERC Order 2023 replaced PJM's old serial/rolling interconnection process
with a cluster-study regime: projects enter in discrete study cycles and
cost reallocation is confined to cycle-mates rather than spreading
throughout the whole active queue. This section adds a minimal proxy for
that reform to the ABM and measures its effect on the cascade channel.

**Why this is the primary counterfactual.** The empirical brief's
matched-DiD headline is a **within-POI** quantity: does a withdrawal at
POI $p$ predict additional withdrawals at POI $p$ over the following
4–8 quarters? The deposit-pool counterfactual in §10 only neutralizes the
inter-POI network-fanout share $(1-\alpha)U_j$, so it targets a channel
one step removed from what the matched-DiD picks up. Cluster bounding, by
contrast, constrains *both* reallocation channels and maps cleanly onto
Order 2023's structural reform language.

### 11.1 Mechanism

Add a new parameter:

| Parameter | Default | Notes |
|---|---|---|
| `cluster_bound_window_months` | `-1` | Sentinel for disabled (current unbounded behavior). When $\ge 0$, reallocation recipients are filtered by entry-time proximity. |

When `cluster_bound_window_months = W >= 0`, the reallocation firing in
§5 is modified so that *both* local and network shares only reach
recipient projects whose entry time is within $W$ months of the
withdrawer's:

- **Local share** ($\alpha U_j$, same POI): the candidate set
  `poi.active_projects()` is first filtered to
  $\{q : |q.t_\text{entry} - j.t_\text{entry}| \le W\}$ before the DFAX
  threshold is applied. If the filtered set is empty, $U_j^\text{loc}$
  evaporates.
- **Network share** ((1 − α) U_j, distance-biased fanout): for each of
  the `network_fanout` target POIs, the recipient set is filtered by the
  same window. If any target POI has zero in-window active projects, its
  per-POI share $U_j^\text{net}/k$ evaporates for that target.

Evaporation is accumulated in `QueueModel.evaporated_U` (dollars).
Tracking is unconditional so totals are comparable across regimes; under
the unbounded regime this field captures the baseline evaporation from
DFAX-threshold filtering and empty POIs.

$W \to \infty$ (in practice $W \ge$ horizon) recovers the unbounded
reallocation. $W = 0$ effectively disables reallocation since co-entry
peers at monthly granularity are rare.

### 11.2 Results

9-regime × 90-seed sweep (`experiments/run_cluster_bound.py`, 3119s elapsed) at
the §7c configuration (2000 POIs, 240 months, $\sigma_\text{POI} = 2.2
\times 10^6$, $\alpha = 0.15$, seeds 100–189). Cascade gauge =
`mini_event_study` treated − control `diff`; per-seed cascade $=$ regime
$-$ OFF. Mean ± SE across seeds, 95% CI = ±1.96·SE. An earlier 30-seed
sweep (seeds 100–129) was consistent but underpowered in the middle
of the gradient; the 90-seed numbers below supersede it.

**Total withdrawal counts** (mean ± SE across 90 seeds, 240-month run):

| Regime | total_wd | vs ON_unbounded |
|---|---|---|
| OFF            | 4419.3 ± 7.8  | — |
| W=12           | 4653.0 ± 7.5  | **+99.5 ± 10.5** |
| W=18           | 4717.3 ± 9.5  | **+35.1 ± 11.2** |
| W=24           | 4730.7 ± 8.8  | **+21.8 ± 10.5** |
| W=36           | 4736.8 ± 8.2  | +15.7 ± 10.4 |
| W=48           | 4745.7 ± 8.8  | +6.8 ± 8.3 |
| W=60           | 4748.7 ± 8.2  | +3.8 ± 7.6 |
| W=72           | 4743.7 ± 7.1  | +8.8 ± 5.4 |
| ON_unbounded   | 4752.5 ± 8.6  | — |

**Withdrawal prevention (primary policy headline).** Per-seed
$\text{prevented} = \text{wd}_\text{ON\_unbounded} - \text{wd}_{W}$,
averaged across 90 seeds, scaled to PJM by (1800 / 300) = 6.0:

| W (months) | prevented/run | prevented/yr PJM | 95% CI |
|---|---|---|---|
| **12** | **+99.5 ± 10.5** | **+29.9 ± 3.2** | **[+23.6, +36.0]** |
| **18** | **+35.1 ± 11.2** | **+10.5 ± 3.4** | **[+3.9, +17.1]** |
| **24** | **+21.8 ± 10.5** | **+6.5 ± 3.2**   | **[+0.4, +12.7]** |
| 36 | +15.7 ± 10.4 | +4.7 ± 3.1 | [−1.4, +10.8] |
| 48 | +6.8 ± 8.3   | +2.0 ± 2.5 | [−2.9, +6.9]  |
| 60 | +3.8 ± 7.6   | +1.1 ± 2.3 | [−3.4, +5.6]  |
| 72 | +8.8 ± 5.4   | +2.7 ± 1.6 | [−0.5, +5.8]  |

**Three regimes now have CIs that exclude zero**: $W = 12$, $18$, $24$.
The prevention gradient is clean and monotone in the tight end: $+30$
→ $+10.5$ → $+6.5$ /yr as the window widens from 12 to 24 months. Above
$W \ge 36$, all point estimates fall within the 1$\sigma$ band of zero.

**Interpretation.** The cascade channel in this DGP is concentrated
among close-entry peers — projects within roughly a 2-year entry window
of the withdrawer. A 1-year cluster (W=12) captures $+30$ /yr PJM-scale
prevention; loosening to a 2-year cluster (W=24) drops that by $4\times$
to $+6.5$ /yr; widening beyond that buys essentially nothing. For
context, fully disabling reallocation (OFF) reduces withdrawals by
$+333$ /run = $+100$ /yr PJM-scale, so cluster bounding at $W=12$
mitigates roughly 30% of the full cascade channel. This is materially
smaller than the "turn it all off" counterfactual but meaningfully
larger than any realistic widening of the cluster window.

This is a reasonably direct statement on FERC Order 2023: **short
study cycles do real work; longer cycles recover most of the
unbounded-reallocation pathology.** A 12-month cycle prevents roughly
3× more withdrawals than a 24-month cycle (CI-clean), and a 3-year
cycle (W=36) is statistically indistinguishable from no reform at
all at n = 90.

**Peak-k cascade (argmax k∈[4,8] on unbounded at n=90 now at k=7):**

| Regime | cascade at k=7 | 95% CI |
|---|---|---|
| ON_unbounded | +0.0090 ± 0.0017 | **[+0.0056, +0.0123]** |
| W=12  | +0.0058 ± 0.0016 | **[+0.0026, +0.0091]** |
| W=18  | +0.0054 ± 0.0017 | **[+0.0021, +0.0088]** |
| W=24  | +0.0058 ± 0.0017 | **[+0.0025, +0.0091]** |
| W=36  | +0.0080 ± 0.0018 | **[+0.0044, +0.0116]** |
| W=48  | +0.0096 ± 0.0021 | **[+0.0055, +0.0137]** |
| W=60  | +0.0091 ± 0.0017 | **[+0.0058, +0.0125]** |
| W=72  | +0.0078 ± 0.0018 | **[+0.0042, +0.0113]** |

All event-study CIs now exclude zero at n = 90. The point estimates
order roughly as expected: tight-cluster regimes (W = 12–24) sit around
$+0.005$; loose regimes (W ≥ 36) sit around $+0.008\text{--}0.010$;
unbounded at $+0.009$. The gradient on the event-study is visible but
shallower than on total counts, which is consistent with the count
metric picking up the cumulative network-fanout effect over the full
run while `mini_event_study` only sees the co-POI contrast.

**Evaporation diagnostic.** Under all bounded regimes, total
evaporated $U$ sits around $3.58\text{–}3.64 \times 10^{10}$ — very
close to the ON_unbounded baseline ($3.59 \times 10^{10}$). The bound
doesn't change how much $U$ evaporates so much as where non-evaporated
$U$ lands (more concentrated on cluster-mates when tight).

**Note on peak-k shift.** The 30-seed sweep identified k=4 as the
argmax; at 90 seeds the argmax shifts to k=7 with all k's now
CI-clean. This is a better-sampled view of the same underlying shape
and does not change any qualitative conclusion — the count-based
headline is the defensible summary regardless of peak selection.

### 11.3 Side-by-side with the deposit pool (why cluster-bound is primary)

Both counterfactuals were re-run at n=90 on seeds 100-189 with identical
BASE config. OFF and ON_unbounded/ON_no_pool match to one decimal across
runs. Comparing at the same sample size:

| Metric | Deposit pool (§10) | Cluster-bound W=12 (§11.2) |
|---|---|---|
| Total-count prevention /yr PJM | **+15.1 ± 3.4** CI [+8.4, +21.8] | **+29.9 ± 3.2** CI [+23.6, +36.0] |
| Peak-k cascade, unbounded regime  | +0.0090 ± 0.0017 | +0.0090 ± 0.0017 |
| Peak-k cascade, treated regime    | +0.0093 ± 0.0014 | +0.0058 ± 0.0016 |
| Reduction in peak-k cascade       | **~0** (CIs overlap completely) | **~35%** (CIs separate) |
| Integrated within-POI cascade (Σ k=1..8) | **+0.0435 → +0.0435 (0.0% ± 17%, CI centered at 0)** | +0.0435 → +0.0298 (**+32% ± 15%, CI [+0.2%, +61%]**) |

Two sharp observations:

1. **Cluster-bound prevents roughly 2× as many withdrawals as the pool**
   on total counts (+30 /yr vs +15 /yr at PJM scale) at the same sample
   size and DGP.
2. **Only cluster-bound visibly shifts the within-POI mini_event_study
   cascade.** The pool's per-k peak at k=7 is indistinguishable from
   unbounded; cluster-bound's is ~35% lower.

Observation 2 is the important one. `mini_event_study` measures the same
quantity the empirical matched-DiD measures: within-POI cascade. If the
pool were a good analogue for what the empirical paper headlines
preventing, it would show up there. It does not. The pool's
total-withdrawal prevention is happening *elsewhere in the network* —
at POIs that are downstream fanout targets, not at the treated
withdrawer's POI. The matched-DiD research design (treated POI vs
matched control POI) is blind to this channel by construction.

Cluster-bound, which bounds both local α·U and network-fanout (1−α)·U
shares, *does* show up in mini_event_study because its mechanism
constrains exactly the within-POI pathway that the empirical design
measures.

This is the operational reason cluster-bound supersedes the pool as the
primary policy counterfactual going forward. If the empirical brief is
to claim "reform X prevents Y /yr of the cascade I measured," the
counterfactual needs to act on the same channel the empirical design
identifies. Cluster-bound does; the pool does not.

**Per-k decomposition makes this even sharper.** Looking at the full
per-k cascade profile of the pool (cascade = ON_no_pool − OFF;
cascade_pool = ON_with_pool − OFF):

| k | cascade | cascade_pool | pool reduction |
|---|---|---|---|
| 1 | +0.0130 | +0.0084 | **+35%** |
| 2 | +0.0045 | +0.0040 | +11% |
| 3 | +0.0015 | +0.0032 | −114% |
| 4 | +0.0053 | +0.0046 | +14% |
| 5 | +0.0018 | +0.0049 | **−163%** |
| 6 | +0.0063 | +0.0077 | **−22%** |
| 7 | +0.0090 | +0.0093 | −3% |
| 8 | +0.0021 | +0.0015 | +27% |

The pool **does** reduce the short-run cascade (k=1 by 35%) — but that
reduction is offset by an *increase* in medium-run cascade (k=5, 6).
Economically: the pool absorbs the short-run shock at withdrawal, which
buys time; but cost eventually migrates onto the network (pool depletes,
or subsequent withdrawers' balances run low) and the cascade propagates
at a longer lag. The pool delays the cascade; it does not extinguish it.

Cluster-bound W=12 by contrast shows positive reductions across most k
(28%, 38%, 74%, 5%, 35%, 46%) with no large negative offset, producing
the 32% integrated reduction. This is a real structural change to the
DGP, not a timing shift.

**Formal test** (per-seed paired differences, n=90):

- Pool: integrated cascade reduction = $+0.0000 \pm 0.0076$, 95% CI
  $[-0.0148, +0.0149]$. Literally zero point estimate.
- Cluster-bound (W=12, both channels): integrated cascade reduction =
  $+0.0137 \pm 0.0065$, 95% CI $[+0.0010, +0.0265]$. Excludes zero,
  though the lower bound is close.

**Confirmation: running the actual matched-DiD pipeline on ABM panels.**
The mini_event_study result above is a proxy for what the full empirical
matched-DiD estimator would find. To verify, `experiments/run_matched_did_compare.py`
runs `contagion/matched_did.py`'s full pipeline (build panel → identify
events → match POIs → event-study with POI + event-time FEs, clustered
SEs) on ABM-simulated panels from each regime. 10 seeds × 3 regimes
(14 min at ~30s/seed for the matched-DiD pipeline):

| Regime | DiD | k=1 β | peak β (k∈[4,8]) | pre-trend p |
|---|---|---|---|---|
| ON_no_pool      | −0.059 ± 0.013 | −0.118 ± 0.012 | **+0.060 ± 0.003** | 0.68 |
| ON_with_pool    | −0.064 ± 0.018 | −0.125 ± 0.010 | **+0.061 ± 0.005** | 0.53 |
| W=12 both channels | −0.062 ± 0.007 | −0.106 ± 0.011 | **+0.050 ± 0.004** | 0.52 |

**Peak-β reduction vs baseline (ON_no_pool), paired on seed, n=10:**

- **Pool:** $-0.0005 \pm 0.0070$, 95% CI $[-0.0143, +0.0133]$ — **zero**.
  Matched-DiD cannot distinguish the pool regime from the baseline
  unbounded regime.
- **Cluster-bound W=12:** $+0.0104 \pm 0.0043$, 95% CI $[+0.0020,
  +0.0187]$ — CI **excludes zero**. ~17% reduction in the matched-DiD
  peak estimate.

This is the strongest form of the prediction: the *actual empirical
estimator*, applied to simulated panels under each counterfactual,
confirms that cluster-bound reduces the matched-DiD-measured cascade
while the pool does not. Pre-trend F-tests pass under all three
regimes ($p > 0.5$), so the parallel-trends assumption holds under the
reform counterfactuals as well.

Note: the ABM's matched-DiD peak (+0.060 at ON_no_pool) is about 2×
the empirical brief's +0.029. This is consistent with the §7c
identification caveat (matched-DiD in this DGP setting is sensitive
to μ_POI selection, so peak estimates are inflated). The ~17%
reduction from cluster-bound is still the quantity of interest — it
says "a reform that changes the cascade DGP by ~32% (per §11.3
integrated-cascade measurement) produces a ~17% visible shift in
matched-DiD estimates," which bounds the detection power of the
empirical design.

### 11.4 Channel decomposition

5-regime × 90-seed sweep (`experiments/run_channel_decomp.py`, 1720s) at the same
BASE as §11.2. Bounds the local α·U and network (1−α)·U channels
independently via two new Params (`cluster_bound_local_window_months`,
`cluster_bound_network_window_months`) to attribute the W=12 prevention
to its source.

**Total-count prevention vs ON_unbounded** (n=90):

| Regime | prev/run | prev/yr PJM | 95% CI | % of combined |
|---|---|---|---|---|
| W=12 both channels | +99.5 ± 10.5 | +29.85 ± 3.16 | **[+23.6, +36.0]** | 100% |
| W=12 network-only  | +74.2 ± 10.1 | +22.25 ± 3.02 | **[+16.3, +28.2]** | 75% |
| W=12 local-only    | +15.4 ± 8.9  | +4.63 ± 2.68  | [−0.6, +9.9]      | 15% |

**The network-fanout channel is the dominant pathway for total-count
prevention.** Bounding the network channel alone recovers 75% of the
combined effect; bounding the local channel alone has an effect whose
CI touches zero. This is consistent with the structural split at
α=0.15: 85% of each withdrawer's $U$ enters the network-fanout
pipeline.

**Peak-k within-POI cascade** (k=7 argmax, n=90):

| Regime | cascade at k=7 | 95% CI |
|---|---|---|
| ON_unbounded        | +0.0090 ± 0.0017 | [+0.0056, +0.0123] |
| W=12 both channels  | +0.0058 ± 0.0016 | [+0.0026, +0.0091] |
| W=12 local-only     | **+0.0057 ± 0.0016** | [+0.0026, +0.0088] |
| W=12 network-only   | +0.0067 ± 0.0016 | [+0.0035, +0.0099] |

**The within-POI mini_event_study signal is driven almost entirely by
the local channel.** Bounding the local share alone (W=12 local-only)
produces essentially the same peak-k reduction as bounding both
(+0.0057 vs +0.0058). Bounding the network share alone has only a
modest effect on peak-k (+0.0067 vs +0.0090 baseline, 26% reduction).

**Key insight — the two channels map to different empirical targets:**

| Channel | Drives | Observable via |
|---|---|---|
| Network fanout (1−α)·U | **Total-count** prevention | Whole-queue panel counts |
| Local α·U              | **Within-POI** mini_event_study | Matched-DiD within-POI design |

So matched-DiD, as currently implemented in the empirical brief, sees
the *local* cascade channel (~15% of total-count prevention) while
being blind to the *network* channel (~75%). Empirical claims about
"the cascade we measured" should be interpreted as a claim about the
local, within-POI DFAX reallocation channel specifically — not about
the full system-wide cascade.

This also explains why the deposit pool (§10, which targets the
network channel) doesn't show up in mini_event_study (§11.3): the
pool's mechanism acts on the channel matched-DiD is blind to.
Cluster-bound's prevention is ~25% local-attributable (W=12_local_only
as a fraction of W=12_both in peak-k terms), which is exactly why it
*is* visible in mini_event_study.

### 11.5 α_local sensitivity

3 α × 3 regime × 30 seeds (`experiments/run_alpha_sensitivity.py`, 114s). Tests
whether the W=12 headline is robust to the local/network split
calibration.

| α | full cascade (OFF → ON_unbounded) | W=12 prevention | prev/yr PJM | 95% CI | % captured |
|---|---|---|---|---|---|
| 0.05 | +343 /run | +116.2 ± 20.6 | +34.9 ± 6.2 | **[+22.8, +47.0]** | 34% |
| 0.15 | +356 /run | +110.6 ± 21.8 | +33.2 ± 6.6 | **[+20.3, +46.0]** | 31% |
| 0.30 | +328 /run | +78.6 ± 15.3  | +23.6 ± 4.6 | **[+14.6, +32.6]** | 24% |

All three CIs exclude zero. The W=12 prevention is robust across the
α range: point estimates span +24 to +35 /yr PJM, consistent with a
true effect in the "roughly 30% of the cascade channel prevented" band.

The monotone pattern (α=0.05 > α=0.15 > α=0.30) is consistent with
§11.4: a lower α means more of $U$ enters the network-fanout pipeline,
where cluster bounding does most of its work. At α=0.30 only 70% of
$U$ enters the network — less for cluster bounding to intercept — so
total prevention drops to +23.6 /yr.

**Conclusion**: the +30 /yr W=12 headline is not an artifact of the
α=0.15 calibration. It persists across the plausible α range
(0.10-0.15 from structural decomposition, with +/-2x as sensitivity
anchors).

### 11.6 Caveats

- **Same matched-DiD identification caveat as §7c applies.** The ABM's
  matched-DiD estimates are contaminated by selection on permanent POI
  heterogeneity; `mini_event_study` (no fixed effects) is the correct
  ABM-internal cascade gauge for cluster-bound comparisons.
- **Evaporation is total, not bound-attributable.** The
  `evaporated_U` field counts every dollar that failed to place
  (empty-POI, sub-DFAX, and filter-exclusion sources combined). To
  isolate the bound-attributable component, subtract the unbounded
  regime's evaporation from the bounded regime's.
- **W = 0 is a stress test, not a realistic reform.** Real cluster
  cycles are 12–24 months, not zero. W = 0 is included only to verify
  the filter actually bites at the tight end.
- **Entry-time granularity.** The ABM tracks entry in months; PJM
  cluster cycles are years. This means a W of 24 months in the ABM
  corresponds roughly to a 2-year cycle in PJM — reasonable but not
  exact.
