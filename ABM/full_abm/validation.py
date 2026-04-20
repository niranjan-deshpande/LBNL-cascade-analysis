"""Summary-stat validation metrics for the multi-POI ABM.

Metrics (intended to be compared against PJM empirical targets from the
contagion brief):

  - completion_rate: fraction of terminal projects that completed (target ~0.20 for PJM).
  - withdrawal_variance_ratio: variance-ratio overdispersion statistic on
    POI-level withdrawal rates (target ~1.6).
  - dose_response: withdrawal rate by POI-depth bin (expect monotonic increase,
    or equivalently an increasing logistic OR on peer-wd-rate).
  - peer_wd_logistic_or: OR from a cross-sectional logistic of project
    withdrawal on peer withdrawal rate (target ~10 national; PJM-only value
    will differ but should be large).
  - did_event_study: event-time path of within-POI post-first-withdrawal
    withdrawal rate minus a matched no-peer-wd control (target: null at k=1,
    positive at k=4-8 quarters). Lightweight version; not a full matched DiD.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def completion_rate(panel: pd.DataFrame) -> float:
    term = panel[panel["status"].isin(["completed", "withdrawn"])]
    if len(term) == 0:
        return float("nan")
    return (term["status"] == "completed").mean()


def poi_summary(panel: pd.DataFrame, min_depth: int = 2) -> pd.DataFrame:
    g = panel.groupby("poi_id").agg(
        depth=("project_id", "count"),
        n_withdrawn=("status", lambda s: (s == "withdrawn").sum()),
        n_completed=("status", lambda s: (s == "completed").sum()),
    )
    g["wd_rate"] = g["n_withdrawn"] / g["depth"]
    return g[g["depth"] >= min_depth].copy()


def withdrawal_variance_ratio(panel: pd.DataFrame, min_depth: int = 2):
    """VR = observed var(wd_rate) / expected var under Binomial-with-pooled-p.

    Exactly the statistic used in the overdispersion test. See
    contagion/descriptive.py for the empirical implementation.
    """
    g = poi_summary(panel, min_depth=min_depth)
    if len(g) < 5:
        return {"vr": float("nan"), "n_pois": len(g)}
    n_i = g["depth"].to_numpy()
    k_i = g["n_withdrawn"].to_numpy()
    p_hat = k_i.sum() / n_i.sum()
    # Observed cross-POI variance of the rate
    obs_var = g["wd_rate"].var(ddof=1)
    # Expected variance of rate under independent binomial with common p
    exp_var = np.mean(p_hat * (1 - p_hat) / n_i)
    vr = obs_var / exp_var if exp_var > 0 else float("nan")
    return {"vr": vr, "n_pois": len(g), "p_hat": p_hat}


def dose_response(panel: pd.DataFrame):
    g = poi_summary(panel, min_depth=2)
    if g.empty:
        return pd.DataFrame()
    g["depth_bin"] = pd.cut(g["depth"], bins=[1, 2, 3, 5, 9, 100],
                             labels=["2", "3", "4-5", "6-9", "10+"], include_lowest=False)
    out = g.groupby("depth_bin", observed=True).agg(
        n_pois=("depth", "count"),
        mean_wd_rate=("wd_rate", "mean"),
        mean_depth=("depth", "mean"),
    )
    return out


def cross_sectional_peer_effect(panel: pd.DataFrame):
    """Crude cross-sectional peer-effect test.

    For each project at a multi-project POI, compute the peer-withdrawal rate
    (fraction of peers that withdrew). Then compare withdrawal rate among
    projects with high peer-wd-rate (>=median) vs low (<median).
    Returns a relative-risk-style statistic and a naive logistic OR.
    """
    g = poi_summary(panel, min_depth=2)
    if g.empty:
        return {}
    m = panel.merge(g[["depth", "n_withdrawn"]], left_on="poi_id", right_index=True, how="inner")
    # peer withdrawal rate: exclude self
    m["peer_wd"] = np.where(m["status"] == "withdrawn", m["n_withdrawn"] - 1, m["n_withdrawn"])
    m["peer_wd_rate"] = m["peer_wd"] / (m["depth"] - 1).clip(lower=1)
    m["y"] = (m["status"] == "withdrawn").astype(int)
    med = m["peer_wd_rate"].median()
    hi = m[m["peer_wd_rate"] > med]
    lo = m[m["peer_wd_rate"] <= med]
    p_hi = hi["y"].mean() if len(hi) else float("nan")
    p_lo = lo["y"].mean() if len(lo) else float("nan")
    rr = p_hi / p_lo if p_lo > 0 else float("nan")
    # Naive OR from contingency
    def _odds(p): return p / (1 - p) if p not in (0, 1) else float("nan")
    or_ = _odds(p_hi) / _odds(p_lo) if _odds(p_lo) and not np.isnan(_odds(p_lo)) else float("nan")
    return {"n": len(m), "p_hi": p_hi, "p_lo": p_lo, "rr": rr, "or": or_}


def mini_event_study(panel: pd.DataFrame, horizon_months: int,
                      k_quarters: tuple = (1, 2, 3, 4, 5, 6, 7, 8)):
    """For each POI that ever has a first withdrawal at time t0 (and has >=1
    other project still active at t0), compute the fraction of those other
    projects that withdraw in quarter k (3-month window [t0+3(k-1), t0+3k]).

    Compare to a matched-null: POIs that had no withdrawal as of t0 (sampled
    at the same calendar time) and compute the same quarterly wd rates among
    their active projects.

    Returns a DataFrame with k, treated_rate, control_rate, diff.
    """
    # Precompute: per-POI first-withdrawal time (NaN if no withdrawal).
    proj = panel[["project_id", "poi_id", "t_entry", "t_exit", "status"]].copy()
    proj["t_exit_filled"] = proj["t_exit"].where(proj["t_exit"].notna(), horizon_months + 1)

    first_wd_by_poi = (
        proj[proj["status"] == "withdrawn"].groupby("poi_id")["t_exit"].min()
    )

    # Numpy-backed per-project arrays for fast filtering
    proj_pid = proj["poi_id"].to_numpy()
    proj_entry = proj["t_entry"].to_numpy()
    proj_exit_f = proj["t_exit_filled"].to_numpy()
    proj_status = proj["status"].to_numpy()
    proj_exit_raw = proj["t_exit"].to_numpy()  # NaN for unexited

    def _peers_active_at(poi_id: int, t_ref: int):
        m = (proj_pid == poi_id) & (proj_entry <= t_ref) & (proj_exit_f >= t_ref)
        return np.where(m)[0]

    def _frac_wd_in_window(idx: np.ndarray, lo: int, hi: int):
        if len(idx) == 0:
            return None
        t_exit_sub = proj_exit_raw[idx]
        st_sub = proj_status[idx]
        wd_mask = (st_sub == "withdrawn") & (~np.isnan(t_exit_sub)) & \
                  (t_exit_sub >= lo) & (t_exit_sub < hi)
        return float(wd_mask.sum() / len(idx))

    treated_pois = first_wd_by_poi.index.tolist()
    treated_set = set(treated_pois)

    # Eligible-control POIs per t0: those with first-withdrawal > t0 (or no withdrawal).
    # We use t_ref to filter; we'll find them via first_wd_by_poi lookup.
    all_pois = proj["poi_id"].unique()
    first_wd_arr = {int(pid): int(first_wd_by_poi.get(pid, horizon_months + 1))
                     for pid in all_pois}

    rng = np.random.default_rng(12345)

    rows = []
    for k in k_quarters:
        lo_k = 3 * (k - 1)
        hi_k = 3 * k
        treated_vals, control_vals = [], []
        for pid in treated_pois:
            t0 = int(first_wd_by_poi.loc[pid])
            # Treated peers: active at t0 at this POI, excluding the project that withdrew at t0
            idx = _peers_active_at(pid, t0)
            if len(idx) == 0:
                continue
            # remove the trigger (project that withdrew at t0 at this POI)
            trig = (proj_status[idx] == "withdrawn") & (proj_exit_raw[idx] == t0)
            idx = idx[~trig]
            v = _frac_wd_in_window(idx, t0 + lo_k, t0 + hi_k)
            if v is not None:
                treated_vals.append(v)

            # Matched control: a POI with first_wd > t0 and ≥2 active projects at t0
            eligible = [p for p in all_pois
                         if (p != pid) and (first_wd_arr[int(p)] > t0)]
            if not eligible:
                continue
            # Try several draws to find one with ≥2 active peers
            tries = 5
            for _ in range(tries):
                cpid = int(rng.choice(eligible))
                cidx = _peers_active_at(cpid, t0)
                if len(cidx) >= 2:
                    break
            else:
                continue
            v = _frac_wd_in_window(cidx, t0 + lo_k, t0 + hi_k)
            if v is not None:
                control_vals.append(v)
        rows.append({
            "k_quarter": k,
            "treated_rate": float(np.mean(treated_vals)) if treated_vals else float("nan"),
            "control_rate": float(np.mean(control_vals)) if control_vals else float("nan"),
            "n_treated": len(treated_vals),
            "n_control": len(control_vals),
        })
    df = pd.DataFrame(rows)
    df["diff"] = df["treated_rate"] - df["control_rate"]
    return df


def full_report(panel: pd.DataFrame, horizon_months: int):
    return {
        "completion_rate": completion_rate(panel),
        "variance_ratio": withdrawal_variance_ratio(panel),
        "dose_response": dose_response(panel),
        "peer_effect": cross_sectional_peer_effect(panel),
        "event_study": mini_event_study(panel, horizon_months),
    }
