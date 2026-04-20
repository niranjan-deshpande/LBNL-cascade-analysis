"""PJM-only bootstrap samplers for the multi-POI ABM.

Exports:
  sample_mw(rng, n)
  sample_duration_months(rng, n)
  sample_dollars_per_kw(rng, n)        # log-uniform between completion-avg and withdrawn-avg
  empirical_completion_rate_pjm()      # target for baseline calibration
"""
from __future__ import annotations
import os
import pickle
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(BASE_DIR, "_calib_cache_pjm.pkl")
EXCEL_PATH = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)),
                           "lbnl_ix_queue_data_file_thru2024_v2.xlsx")
SHEET_NAME = "03. Complete Queue Data"
EXCEL_EPOCH = pd.Timestamp("1899-12-30")


def _excel_to_dt(s):
    n = pd.to_numeric(s, errors="coerce")
    return pd.to_timedelta(n, unit="D") + EXCEL_EPOCH


def _load_pjm_panel():
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, header=None)
    hdr = None
    for i in range(min(20, len(df))):
        if any("q_id" in str(v).lower() for v in df.iloc[i]):
            hdr = i; break
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, header=hdr)
    df.columns = df.columns.str.strip()
    df = df[df["entity"].astype(str).str.upper() == "PJM"].copy()
    for c in ("q_date", "on_date", "wd_date"):
        if c in df.columns:
            df[c] = _excel_to_dt(df[c])
    df["mw1"] = pd.to_numeric(df.get("mw1"), errors="coerce")
    return df


def _build_cache():
    df = _load_pjm_panel()
    status = df["q_status"].astype(str).str.lower()
    completed_mask = status.isin(["operational", "complete", "completed"])
    withdrawn_mask = status.isin(["withdrawn"])

    mw_all = df["mw1"]
    mw_all = mw_all[(mw_all > 1) & (mw_all < 2000)].dropna().to_numpy()

    comp = df[completed_mask].copy()
    dur_days = (comp["on_date"] - comp["q_date"]).dt.days
    dur_months = dur_days.dropna()
    dur_months = dur_months[(dur_months > 180) & (dur_months < 365 * 10)] / 30.44
    dur_months = dur_months.to_numpy()

    # Empirical completion rate in PJM (terminal outcomes only — drop still-active)
    terminal = completed_mask | withdrawn_mask
    pjm_completion_rate = completed_mask.sum() / max(terminal.sum(), 1)

    # Empirical monthly project arrival rate in PJM (q_date counts / month span)
    qd = df["q_date"].dropna()
    span_months = (qd.max() - qd.min()).days / 30.44
    pjm_arrivals_per_month = len(qd) / span_months if span_months > 0 else np.nan

    cache = {
        "mw": mw_all,
        "dur_months": dur_months,
        "pjm_completion_rate": float(pjm_completion_rate),
        "pjm_arrivals_per_month": float(pjm_arrivals_per_month),
        "n_pjm_projects_total": int(len(df)),
    }
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)
    return cache


def _cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    return _build_cache()


def sample_mw(rng, n):
    return rng.choice(_cache()["mw"], size=n, replace=True)


def sample_duration_months(rng, n):
    return rng.choice(_cache()["dur_months"], size=n, replace=True)


def sample_dollars_per_kw(rng, n, low=71.0, high=563.0):
    return np.exp(rng.uniform(np.log(low), np.log(high), size=n))


def pjm_targets():
    c = _cache()
    return {
        "completion_rate": c["pjm_completion_rate"],
        "arrivals_per_month": c["pjm_arrivals_per_month"],
        "n_pjm_projects_total": c["n_pjm_projects_total"],
    }


if __name__ == "__main__":
    c = _cache()
    print({k: v if not hasattr(v, "__len__") else f"array(len={len(v)})" for k, v in c.items()})
