"""Multi-POI interconnection-queue ABM (PJM, old serial regime).

See MODEL.md for the full specification. This module contains the Mesa model
and agent classes.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import numpy as np
import mesa

from calibrate import (
    sample_mw, sample_duration_months, sample_dollars_per_kw
)


# ---------- Parameter bundle ---------------------------------------------

@dataclass
class Params:
    # Size / horizon
    n_pois: int = 400
    horizon_months: int = 180          # 15 years
    burn_in_months: int = 36           # ignore for calibration/validation metrics
    rng_seed: int = 42

    # Arrivals
    arrivals_per_month: float = 10.0   # total new projects/month across all POIs
    hotness_shape: float = 1.2         # Gamma shape for POI hotness (heavier tail → deeper POIs)
    hotness_scale: float = 1.0

    # Headroom hierarchy (dollars). Scales are tuned against the lower c_p range
    # below; the $71–563/kW from the brief are *final* allocated costs.
    mu_H_center: float = 13.0e6        # grand mean of mu_POI
    sigma_between: float = 3.5e6       # POI-to-POI spread in mu_POI (dominant variance)
    sigma_within: float = 2.0e6        # project-within-POI spread

    # Shocks (dollars). Raising sigma_eps shrinks the cascade bite relative to noise.
    sigma_eps: float = 4.0e6           # idiosyncratic per-project monthly shock
    sigma_poi: float = 2.2e6           # POI-level per-month innovation
    rho_poi: float = 0.85              # AR(1) persistence of POI shock (~7-month half-life)

    # Headroom ramp over project life
    ramp_floor: float = 0.55           # H_i(t_entry) = ramp_floor * H_i

    # Reallocation (PJM DFAX-style)
    reallocation_enabled: bool = True
    dfax_threshold: float = 0.03        # 3% DFAX: peers below this MW-share-at-POI don't get reallocated to
    lag_low: int = 12
    lag_high: int = 18                 # inclusive

    # Local/network cost decomposition. On withdrawal, alpha_local * U stays at
    # the POI (Attachment Facilities / TOIF proxy) and follows the DFAX rule;
    # the remaining (1 - alpha_local) * U is the network-upgrade share and
    # distributes across a distance-biased subset of *other* POIs.
    # Empirical structural decomposition suggests alpha_local ~ 0.10-0.15.
    alpha_local: float = 0.15
    network_fanout: int = 20            # number of other POIs that share the network portion
    network_distance_scale: float = 0.3 # exp(-d/scale) bias on a unit-square topology
    poi_topology_seed: int | None = None  # if set, decouples topology RNG from run RNG

    # Initial cost scale ($/kW) — drawn per POI, log-uniform between these bounds.
    # Range is lower than the brief's 71–563/kW headline because those are *final*
    # allocated costs (including accumulated reallocations). Initial direct allocations
    # are smaller; reallocations during the run drive the effective cost up.
    dollars_per_kw_low: float = 20.0
    dollars_per_kw_high: float = 150.0


# ---------- POI container (not a Mesa Agent — it doesn't act) -------------

@dataclass
class POI:
    poi_id: int
    c_per_kw: float
    mu_poi: float
    hotness: float
    x: float = 0.0                                # topology coord (unit square)
    y: float = 0.0
    eta: float = 0.0                              # AR(1) state
    projects: List["Project"] = field(default_factory=list)
    pending: List[tuple] = field(default_factory=list)  # (t_fire, withdrawer_project)

    def active_projects(self):
        return [p for p in self.projects if p.status == "active"]


# ---------- Project agent -------------------------------------------------

class Project(mesa.Agent):
    def __init__(self, model, poi: POI, mw: float, t_entry: int, t_cod: int, H_base: float):
        super().__init__(model)
        self.poi = poi
        self.mw = float(mw)
        self.t_entry = int(t_entry)
        self.t_cod = int(t_cod)
        self.H_base = float(H_base)
        self.U = poi.c_per_kw * self.mw * 1000.0   # initial allocated $ (mw*1000 = kW)
        self.status = "active"
        self.t_exit: Optional[int] = None
        poi.projects.append(self)

    def ramp(self, t):
        floor = self.model.params.ramp_floor
        if self.t_cod <= self.t_entry:
            return 1.0
        frac = (t - self.t_entry) / (self.t_cod - self.t_entry)
        frac = max(0.0, min(1.0, frac))
        return floor + (1.0 - floor) * frac


# ---------- Model ---------------------------------------------------------

class QueueModel(mesa.Model):
    def __init__(self, params: Params | None = None):
        super().__init__(seed=(params.rng_seed if params else 42))
        self.params = params or Params()
        self.rng = np.random.default_rng(self.params.rng_seed)
        self.t = 0

        # Build POIs
        hot = self.rng.gamma(self.params.hotness_shape, self.params.hotness_scale,
                              size=self.params.n_pois)
        hot = hot / hot.sum()  # probability weights for arrivals
        c_vals = sample_dollars_per_kw(self.rng, self.params.n_pois,
                                        self.params.dollars_per_kw_low,
                                        self.params.dollars_per_kw_high)
        mu_vals = self.rng.normal(self.params.mu_H_center, self.params.sigma_between,
                                   size=self.params.n_pois)
        # Topology: unit-square coords. Separate RNG so topology is stable
        # across calibration sweeps that vary rng_seed, unless explicitly tied.
        topo_seed = (self.params.poi_topology_seed
                     if self.params.poi_topology_seed is not None
                     else self.params.rng_seed)
        topo_rng = np.random.default_rng(topo_seed)
        coords = topo_rng.uniform(0.0, 1.0, size=(self.params.n_pois, 2))
        self.pois: List[POI] = [
            POI(poi_id=i, c_per_kw=float(c_vals[i]), mu_poi=float(mu_vals[i]),
                hotness=float(hot[i]),
                x=float(coords[i, 0]), y=float(coords[i, 1]))
            for i in range(self.params.n_pois)
        ]
        self._poi_weights = hot

        # Precompute distance-biased network reallocation weights (row i: prob of
        # sending network share from POI i to each other POI j). Diagonal zeroed.
        n = self.params.n_pois
        dx = coords[:, 0][:, None] - coords[:, 0][None, :]
        dy = coords[:, 1][:, None] - coords[:, 1][None, :]
        dist = np.sqrt(dx * dx + dy * dy)
        scale = max(1e-6, self.params.network_distance_scale)
        W = np.exp(-dist / scale)
        np.fill_diagonal(W, 0.0)
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        self._network_weights = W / row_sums  # (n, n) each row sums to 1

        # Running logs
        self.event_log = []   # tuples (t, event_type, project_id, poi_id)
        self.poi_shock_log = []  # optional diagnostics

        # DataCollector for per-step counts
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "t": "t",
                "n_active": lambda m: sum(1 for a in m.agents if a.status == "active"),
                "n_withdrawn_cumulative": lambda m: sum(1 for a in m.agents if a.status == "withdrawn"),
                "n_completed_cumulative": lambda m: sum(1 for a in m.agents if a.status == "completed"),
            }
        )

    # ---- step helpers ------------------------------------------------

    def _update_poi_shocks(self):
        rho = self.params.rho_poi
        sd = self.params.sigma_poi * math.sqrt(max(0.0, 1 - rho * rho))
        for p in self.pois:
            p.eta = rho * p.eta + self.rng.normal(0.0, sd)

    def _fire_reallocations(self):
        """Local/network split reallocation.

        On a fired withdrawal, decompose U_j into:
          - local share    = alpha_local * U_j  → DFAX pro-rata among co-POI peers
          - network share  = (1 - alpha_local) * U_j → distance-biased distribution
            across network_fanout *other* POIs; within each recipient POI,
            pro-rata by MW over active projects (no DFAX threshold).
        If there are no eligible recipients (local or network), that share
        evaporates (absorbed by the TO / load / unmodeled projects).
        """
        thr = self.params.dfax_threshold
        alpha = float(self.params.alpha_local)
        fanout = int(self.params.network_fanout)
        n_pois = self.params.n_pois
        for poi in self.pois:
            fire_now = [(tf, pj) for (tf, pj) in poi.pending if tf == self.t]
            if not fire_now:
                continue
            poi.pending = [(tf, pj) for (tf, pj) in poi.pending if tf > self.t]
            if not self.params.reallocation_enabled:
                continue
            for _, pj in fire_now:
                if pj.U <= 0:
                    continue
                U_local = alpha * pj.U
                U_net = (1.0 - alpha) * pj.U

                # ---- Local share: DFAX pro-rata among co-POI peers ----
                if U_local > 0:
                    active = poi.active_projects()
                    if active:
                        denom = pj.mw + sum(q.mw for q in active)
                        eligible = [q for q in active if (q.mw / denom) > thr]
                        if eligible:
                            elig_mw = sum(q.mw for q in eligible)
                            for q in eligible:
                                q.U += U_local * (q.mw / elig_mw)

                # ---- Network share: distance-biased fanout to other POIs ----
                if U_net > 0 and fanout > 0 and n_pois > 1:
                    probs = self._network_weights[poi.poi_id]
                    k = min(fanout, n_pois - 1)
                    targets = self.rng.choice(n_pois, size=k, replace=False, p=probs)
                    per_poi = U_net / k
                    for j in targets:
                        tgt = self.pois[int(j)]
                        t_active = tgt.active_projects()
                        if not t_active:
                            continue  # evaporates
                        mw_sum = sum(q.mw for q in t_active)
                        if mw_sum <= 0:
                            continue
                        for q in t_active:
                            q.U += per_poi * (q.mw / mw_sum)

                self.event_log.append((self.t, "realloc_fired", pj.unique_id, poi.poi_id))

    def _poisson_arrivals(self):
        lam = self.params.arrivals_per_month
        n_new = int(self.rng.poisson(lam))
        if n_new == 0:
            return
        target_pois = self.rng.choice(self.params.n_pois, size=n_new,
                                       p=self._poi_weights, replace=True)
        mws = sample_mw(self.rng, n_new)
        durs = sample_duration_months(self.rng, n_new)
        for i in range(n_new):
            poi = self.pois[int(target_pois[i])]
            mw = float(mws[i])
            t_cod = int(self.t + max(24, min(int(round(durs[i])),
                                              self.params.horizon_months * 2)))
            # H hierarchical draw (per-project within POI)
            H_base = float(self.rng.normal(poi.mu_poi, self.params.sigma_within))
            Project(self, poi, mw=mw, t_entry=self.t, t_cod=t_cod, H_base=H_base)

    def _completions(self):
        for a in list(self.agents):
            if a.status == "active" and self.t >= a.t_cod:
                a.status = "completed"
                a.t_exit = self.t
                self.event_log.append((self.t, "completed", a.unique_id, a.poi.poi_id))

    def _decisions_and_withdrawals(self):
        """Simultaneous decision: all active agents evaluate stay-rule with current shocks."""
        newly = []
        sigma_eps = self.params.sigma_eps
        for a in list(self.agents):
            if a.status != "active":
                continue
            H_it = a.H_base * a.ramp(self.t)
            eps = self.rng.normal(0.0, sigma_eps)
            eta = a.poi.eta
            if not (a.U < H_it + eps + eta):
                newly.append(a)
        for a in newly:
            a.status = "withdrawn"
            a.t_exit = self.t
            self.event_log.append((self.t, "withdrawn", a.unique_id, a.poi.poi_id))
            # Schedule reallocation (or drop if not enabled)
            if self.params.reallocation_enabled:
                lag = int(self.rng.integers(self.params.lag_low, self.params.lag_high + 1))
                if self.t + lag < self.params.horizon_months:
                    a.poi.pending.append((self.t + lag, a))

    # ---- public step -------------------------------------------------

    def step(self):
        self._update_poi_shocks()
        self._fire_reallocations()
        self._poisson_arrivals()
        self._completions()
        self._decisions_and_withdrawals()
        self.datacollector.collect(self)
        self.t += 1

    def run(self):
        for _ in range(self.params.horizon_months):
            self.step()
        return self

    # ---- extraction helpers -----------------------------------------

    def project_panel(self) -> "list[dict]":
        """Return one row per project with summary fields."""
        out = []
        for a in self.agents:
            out.append({
                "project_id": a.unique_id,
                "poi_id": a.poi.poi_id,
                "mw": a.mw,
                "t_entry": a.t_entry,
                "t_cod": a.t_cod,
                "t_exit": a.t_exit,
                "status": a.status,
                "H_base": a.H_base,
                "U_final": a.U,
                "poi_mu": a.poi.mu_poi,
                "poi_c_per_kw": a.poi.c_per_kw,
            })
        return out
