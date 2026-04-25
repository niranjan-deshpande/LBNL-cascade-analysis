# 2026-04-23 autonomous-session synthesis

During your 2-hour away window, I ran three robustness/extension
experiments that materially change the interpretation of the cluster-
bounded counterfactual. All results are defensible (CI-clean primary
findings; n=90 throughout except α sensitivity at n=30). Detailed
analysis lives in MODEL.md §11.3 / §11.4 / §11.5 and §10.3.

## Headline findings

### 1. Deposit pool is invisible to the within-POI cascade gauge (§11.3)

Re-ran the deposit pool at n=90 on the same seeds (100-189) as the
cluster-bound v2 sweep. OFF and unbounded baselines match to 1 decimal,
confirming DGP consistency.

**Total-count prevention** (both CI-clean):
- Pool: +15.1 ± 3.4 /yr PJM
- Cluster-bound W=12: +29.9 ± 3.2 /yr PJM

**Integrated within-POI cascade reduction** (Σ k=1..8 of
mini_event_study diff — the quantity matched-DiD integrates):
- Pool: **+0.0000 ± 0.0076** — literally zero point estimate
- Cluster-bound: +0.0137 ± 0.0065, CI [+0.0010, +0.0265] (~32% of baseline)

The pool reduces short-run cascade (k=1 by 35%) but offsets it with
increased medium-run cascade (k=5 by 163%). Net effect on within-POI
integrated cascade = 0. It delays; it doesn't extinguish.

**Operational implication:** If the empirical team applied the current
matched-DiD framework to a PJM-style pool reform, they would measure
**no cascade reduction** — even though the pool prevents +15/yr
PJM-scale withdrawals. The cascade prevention happens at *other* POIs
(network-fanout targets), which matched-DiD excludes by design.

### 2. Two cascade channels map to two different empirical targets (§11.4)

Added per-channel bounds (`cluster_bound_local_window_months`,
`cluster_bound_network_window_months`). At W=12, 90 seeds:

| Channel bounded | Total-count prev (/yr PJM) | Peak-k within-POI cascade reduction |
|---|---|---|
| Both (combined) | +29.85, CI [+23.6, +36.0] | 35% |
| **Network only** | **+22.25, CI [+16.3, +28.2]** | 26% |
| **Local only**   | +4.63, CI [-0.6, +9.9] (touches zero)  | **37%** |

Two orthogonal stories:
- **Network fanout dominates total-count prevention** (75% of combined effect)
- **Local share dominates within-POI mini_event_study signal** (local-only
  produces the same peak-k reduction as both channels combined)

This is the clean mechanistic explanation for finding #1: the pool
targets network only → misses the within-POI signal. Cluster-bound
targets both → catches both.

Testable out-of-sample prediction for the empirical team: applying the
existing matched-DiD framework to a hypothetical Order 2023
counterfactual should reveal a ~30% reduction in peak cascade; applying
it to a pool-like reform should reveal no reduction.

### 3. Headline is robust to α_local calibration (§11.5)

The W=12 prevention holds across α ∈ {0.05, 0.15, 0.30}: prev/yr PJM in
the [+24, +35] band, all CI-clean. Monotone decrease with α (more local
share = less leverage for cluster bounding).

## What changes in the story

The 2-channel finding #2 is, IMO, the most important contribution from
this session. It reframes the empirical brief's scope:

- **Current framing**: "matched-DiD measures withdrawal cascades via
  DFAX reallocation; we find peak +0.029 at 4-8 quarter lag."
- **Revised framing**: "matched-DiD measures withdrawal cascades that
  propagate via the *within-POI / local cost-reallocation channel*; the
  peak +0.029 at 4-8 quarter lag is a measure of this specific channel,
  which the ABM decomposition shows to be ~15% of the system-wide
  cascade total."

The +0.029 is real and important — but its policy relevance depends on
whether the reform being considered targets the same channel. Order
2023 (cluster-study scope reduction) does, which is why cluster-bound
shows ~30% reduction in matched-DiD-style panels. A deposit-security
reform targets a different channel, so matched-DiD wouldn't detect it
even if it prevented withdrawals.

### 4. Actual matched-DiD on ABM panels — CONFIRMS the prediction (§11.3)

Ran `contagion/matched_did.py`'s full pipeline on 10 ABM seeds under
each of 3 regimes. ~30s per seed × 30 total = 14 min.

Peak β reduction vs ON_no_pool baseline (paired per seed):
- Pool: **−0.0005 ± 0.0070, CI [−0.0143, +0.0133]** — literally zero
- W=12: **+0.0104 ± 0.0043, CI [+0.0020, +0.0187]** — ~17% reduction,
  CI excludes zero

Absolute peak β values:
- ON_no_pool: +0.060 ± 0.003
- ON_with_pool: +0.061 ± 0.005 (identical to baseline)
- W=12 both channels: +0.050 ± 0.004 (clearly lower)

Pre-trend F-tests pass under all three regimes (p > 0.5).

This is the cleanest possible statement: **the actual estimator used in
the empirical brief, applied to simulated panels, detects cluster-bound
but not the pool.** Both the mini_event_study proxy (§11.3
integrated-cascade test) and the full matched-DiD (this test) give the
same conclusion. The ~17% vs ~32% gap between the two tells you the
matched-DiD design sees about half of the actual cascade-reduction
signal in the DGP — a bound on its detection power.

## Suggested next runs

2. **Varying network_fanout** (currently 20). If the network channel
   is the main total-count driver, fanout topology matters. A sparser
   network (fanout=5) should localize cascades; denser (fanout=50)
   should amplify network-channel prevention.

3. **Mixed-reform counterfactual**: Pool + cluster-bound simultaneously.
   Since the two reforms target orthogonal channels per §11.4, their
   effects should be roughly additive.

## File manifest (session outputs)

Code (paths reflect post-cleanup layout):
- `experiments/run_deposit_pool.py` — pool at n=90 (was `run_deposit_pool_v2.py`)
- `experiments/run_channel_decomp.py` — per-channel decomposition
- `experiments/run_alpha_sensitivity.py` — α robustness
- `model.py` — added `cluster_bound_local/network_window_months` params

Data (in `ABM/full_abm/output/`):
- `deposit_pool_rawseeds.csv`, `deposit_pool_summary.csv`, `deposit_pool.log`
- `channel_decomp_rawseeds.csv`, `channel_decomp_prevention.csv`,
  `channel_decomp_cascade.csv`, `channel_decomp.log`
- `alpha_sensitivity_raw.csv`, `alpha_sensitivity_summary.csv`,
  `alpha_sensitivity.log`

Docs:
- MODEL.md §10.3 — pool n=90 note
- MODEL.md §11.3 — side-by-side pool vs CB
- MODEL.md §11.4 — channel decomposition
- MODEL.md §11.5 — α sensitivity
- Memory: `pool_vs_cluster_bound.md` (new)

Total compute: ~46 min elapsed (deposit-pool 17min + decomp 29min +
alpha 2min). CPU stayed pinned via caffeinate.
