"""Toy one-POI ABM. Monthly time-step, 6 projects, DFAX-proxy by MW share.

Decision rule (per active project i, each month t):
    stay iff U_it < H_it + eps_it + eta_t
where
    U_it = s_i(t) * X_remaining(t)        # allocated upgrade cost ($)
    H_it = H_i * ramp(t, t_entry, T_COD)  # headroom, rising as project matures
    eps_it ~ N(0, sigma_eps^2)            # idiosyncratic shock (monthly)
    eta_t  ~ N(0, sigma_poi^2)            # shared POI-level shock (monthly)

Headroom is drawn hierarchically:
    mu_POI ~ N(mu0, sigma_between^2)      # one draw per POI per replication
    H_i    ~ N(mu_POI, sigma_within^2)    # per project, small within-POI spread

Reallocation (serial, unbounded-within-POI):
    When project j withdraws at time t, schedule a reallocation event at
    t + tau (tau ~ U[12,18] months). At that event, the share s_j is removed
    and renormalized across still-active peers (pro-rata by MW).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class Project:
    idx: int
    mw: float
    t_entry: int           # month index of queue entry (always 0 in toy)
    t_cod: int             # scheduled COD month
    H: float               # headroom (dollars, baseline)
    status: str = "active" # active | withdrawn | completed
    t_exit: Optional[int] = None
    share: float = 0.0     # current share of X (pro-rata by MW among active)


@dataclass
class Params:
    n_projects: int = 6
    horizon_months: int = 120
    dollars_per_kw_low: float = 71.0
    dollars_per_kw_high: float = 563.0
    mu_H_mean: float = 40e6          # POI mean headroom center ($) — calibration knob
    sigma_between: float = 25e6      # cross-POI spread in mu_POI (dominant)
    sigma_within: float = 5e6        # within-POI spread in H_i (small)
    sigma_eps: float = 4e6           # idiosyncratic monthly shock (sd)
    sigma_poi: float = 6e6           # shared POI monthly shock (sd)
    ramp_floor: float = 0.6          # H(t_entry) = ramp_floor * H_i; H(t_COD) = H_i
    lag_low: int = 12
    lag_high: int = 18               # inclusive upper bound for uniform lag
    reallocation_enabled: bool = True


@dataclass
class ReallocEvent:
    t_fire: int
    withdrawer_idx: int


@dataclass
class RunResult:
    status: np.ndarray            # shape (N,), final status strings
    t_exit: np.ndarray            # shape (N,), exit month (or -1 if no exit)
    withdrawal_months: np.ndarray # shape (H,), monthly withdrawal counts
    completion_months: np.ndarray # shape (H,), monthly completion counts
    U_trace: np.ndarray           # shape (N, H), allocated upgrade $ over time
    n_withdrawn: int
    n_completed: int


def _ramp(t, t_entry, t_cod, floor):
    if t_cod <= t_entry:
        return 1.0
    frac = min(1.0, max(0.0, (t - t_entry) / (t_cod - t_entry)))
    return floor + (1.0 - floor) * frac


def simulate(projects_init, X_total, params: Params, rng: np.random.Generator) -> RunResult:
    """Run one replication. `projects_init` is a list of partially-specified Projects
    (with idx, mw, t_entry, t_cod, H set). Returns a RunResult."""
    projects: List[Project] = [Project(**{k: getattr(p, k) for k in ("idx","mw","t_entry","t_cod","H")})
                                for p in projects_init]
    N = len(projects)
    T = params.horizon_months

    def _redistribute_from(withdrawer_idx: int):
        """Pro-rata (by MW) redistribute withdrawer's share to still-active peers."""
        wp = projects[withdrawer_idx]
        active = [p for p in projects if p.status == "active"]
        tot_mw = sum(p.mw for p in active)
        if tot_mw <= 0 or wp.share <= 0:
            wp.share = 0.0
            return
        for p in active:
            p.share += wp.share * (p.mw / tot_mw)
        wp.share = 0.0

    tot0 = sum(p.mw for p in projects)
    for p in projects:
        p.share = p.mw / tot0 if tot0 > 0 else 0.0

    # X_remaining is fixed for the toy — withdrawals don't reduce total upgrade scope;
    # they only redistribute shares across remaining active projects. This is the
    # "unbounded serial DFAX" assumption.
    X_remaining = X_total

    pending: List[ReallocEvent] = []
    wd_months = np.zeros(T, dtype=int)
    comp_months = np.zeros(T, dtype=int)
    U_trace = np.zeros((N, T))

    for t in range(T):
        # 1) Fire scheduled reallocations (disperse withdrawer's share to active peers).
        fired = [ev for ev in pending if ev.t_fire == t]
        if params.reallocation_enabled:
            for ev in fired:
                _redistribute_from(ev.withdrawer_idx)
        pending = [ev for ev in pending if ev.t_fire > t]

        # 2) Shared POI-level shock this month.
        eta = rng.normal(0.0, params.sigma_poi)

        # 3) Completions (reached COD while active). Their share stays frozen on them
        # (they're built — cost is committed); it is not reallocated to peers.
        for p in projects:
            if p.status == "active" and t >= p.t_cod:
                p.status = "completed"
                p.t_exit = t
                comp_months[t] += 1

        # 4) Decision rule for each active project.
        new_withdrawals = []
        for p in projects:
            if p.status != "active":
                U_trace[p.idx, t] = 0.0
                continue
            U_it = p.share * X_remaining
            U_trace[p.idx, t] = U_it
            H_it = p.H * _ramp(t, p.t_entry, p.t_cod, params.ramp_floor)
            eps = rng.normal(0.0, params.sigma_eps)
            if not (U_it < H_it + eps + eta):
                new_withdrawals.append(p)

        for p in new_withdrawals:
            p.status = "withdrawn"
            p.t_exit = t
            wd_months[t] += 1
            if params.reallocation_enabled:
                lag = int(rng.integers(params.lag_low, params.lag_high + 1))
                if t + lag < T:
                    pending.append(ReallocEvent(t_fire=t + lag, withdrawer_idx=p.idx))
                else:
                    # Fire within-horizon reallocations only; beyond horizon, just drop share.
                    p.share = 0.0
            else:
                p.share = 0.0  # "no cascade" counterfactual: share evaporates, peers unaffected.

    status = np.array([p.status for p in projects])
    t_exit = np.array([(p.t_exit if p.t_exit is not None else -1) for p in projects])
    return RunResult(
        status=status,
        t_exit=t_exit,
        withdrawal_months=wd_months,
        completion_months=comp_months,
        U_trace=U_trace,
        n_withdrawn=int((status == "withdrawn").sum()),
        n_completed=int((status == "completed").sum()),
    )


def draw_projects(params: Params, rng: np.random.Generator,
                  mw_sampler, duration_sampler, horizon: int) -> List[Project]:
    """Draw N projects + their headrooms using the hierarchical scheme.
    Returns a list of Projects with idx/mw/t_entry/t_cod/H set (status/share fresh)."""
    mws = mw_sampler(rng, params.n_projects).astype(float)
    durs = duration_sampler(rng, params.n_projects).astype(float)

    mu_poi = rng.normal(params.mu_H_mean, params.sigma_between)
    H_draws = rng.normal(mu_poi, params.sigma_within, size=params.n_projects)

    out = []
    for i in range(params.n_projects):
        t_cod = int(np.clip(np.round(durs[i]), 24, horizon - 1))
        out.append(Project(idx=i, mw=float(mws[i]), t_entry=0, t_cod=t_cod, H=float(H_draws[i])))
    return out
