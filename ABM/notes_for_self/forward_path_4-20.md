## Issue:
As per findings from `diag_cascade_decomposition.py`, most of the matched DiD effects come from different $\mu_{POI}$, not from the actual selection effect. This confounds our original empirical analysis, since we're just picking up on un-matched characteristics. 
The ABM's OFF regime alone produces most of the signal (OFF k=1 = -0.114, OFF peak = +0.044). Reallocation adds only
~1 pp marginal, scattered across k. Thus: The cascade channel (ON−OFF) at peak is +0.009. The mini_event_study sees +6.3 pp. The matched-DiD sees +0.9 pp. The cascade is real in the model but the matched-DiD almost completely absorbs it into its POI/time controls. The entire event-study shape — the k=1 dip, the k=4–8 hump — is reproduced by OFF alone, meaning it's driven by selection on μ_POI and η persistence, not by cascade.

   This isn't just a model validation issue. It reflects back onto the brief's empirical results.

The brief's matched-DiD reports +0.029 at peak and calls this "the central causal finding" for cascade. But your ABM just demonstrated that the matched-DiD applied to a DGP with a known, mechanically operative cascade produces a near-zero cascade coefficient — because the estimator can't separate cascade from the persistent shared conditions that selected the treated POI in the first place.

This means the empirical +0.029 is probably also a mix of cascade + residual selection that the matching doesn't absorb. The true causal cascade in the data could be smaller than +0.029, possibly much smaller. Your ABM didn't fail to reproduce the cascade — it showed you that the estimator you used to measure it has a structural identification problem.

This gives us extreme interpretability issues: one the one hand, our estimate is a *lower bound*, since ~85% of the network-cascade effect goes onto projects that aren't within a given POI; on the other hand, our estimate is an *upper bound* in a different respect, since most of the empirical DiD results come from fitting to different pre-existing $\mu_{POI}$ across projects.

It seems like to make meaningful progress here, I'd need to basically completely redesign the empirical section to take into account the network's topology, and the cost reallocation matrices (from digging through PJM dockets, etc). This is basically unfeasible for me right now. 

One potential fix: Don't report the deposit-pool effect in pp. Report it as a proportional reduction in the cascade multiplier. If ON−OFF at peak is +0.009 without the pool and +0.003 with the pool, the headline is "the deposit pool eliminates ~67% of the cascade channel," not "the deposit pool reduces withdrawals by 0.006 pp." Then do a back-of-envelope scaling calculation: PJM processes ~1,800 projects/year, baseline withdrawal rate ~75%, cascade adds X% on top — the deposit pool prevents N withdrawals per year. That's a concrete, interpretable number even if the per-project effect is small.

## Update (2026-04-21): softened framing + 30-seed results

Red-team pass surfaced three problems with the above: (1) matched-DiD was run at 5 seeds so ON−OFF ≈ +0.009 had no defensible SE; (2) three code bugs inflated pool effects (absorption against full U, forfeiture against inflated U, RNG divergence between regimes); (3) the BOTE composed fractional-excess withdrawals with peak-k reduction, mixing units.

**30-seed cascade decomposition numbers:**
- OFF k=1 = −0.111 ± 0.005 (reproduces empirical −0.038 roughly 3× over, driven by selection)
- OFF peak at k=6 = +0.038 ± 0.003 (close to empirical +0.029 at k=5)
- Cascade (ON − OFF) CI excludes zero only at k=5 (+0.014 ± 0.005) and k=8 (+0.010 ± 0.004)
- At k=1 cascade = −0.012 ± 0.008 (CI includes zero) — the empirical k=1 dip is NOT driven by reallocation cascade
- At k=4 cascade = +0.004 ± 0.006 (CI includes zero)
- argmax peak over k∈[4,8] = k=5 at +0.014 ± 0.005 (don't over-read; peak-picking is upward-biased)

**Framing to use going forward (user's exact phrasing):** "the ABM demonstrates that matched-DiD estimates in this DGP setting are sensitive to residual selection on permanent POI heterogeneity; the empirical +0.029 should be interpreted as an upper bound on the causal cascade effect." This is defensible without committing to a specific ON−OFF point estimate (matching-quality differences between the ABM and empirical estimators could shift it).

**Deposit-pool numbers to use** (30 seeds, bugs fixed):
- Per-k cascade CI excludes zero at k ∈ {1, 4, 6}; at-noise elsewhere.
- Per-k proportional-reduction ratio is **uninterpretable** at this n (argmax-k reduction = −0.92 ± 1.00, CI [−2.87, +1.04] — noisy denominator plus heavy ratio tail). Don't revive the "54% reduction" headline from the 5-seed run.
- **Direct withdrawal-count prevention is the headline.** prevented_per_run = 60.4 ± 20.0 (CI [+21, +100]). PJM-scaled by arrivals ratio (×6.0): 18 ± 6 /yr (CI [+6, +30]). CI excludes zero.
- The 18/yr figure is ~3.5× smaller than the original 65/yr BOTE. Both the bug fixes and the unit-correct scaling matter.
