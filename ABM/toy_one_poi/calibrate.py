"""Bootstrap samplers calibrated from LBNL queue data (PJM focus).

Loads the Excel once, caches to a local pickle for faster re-runs, and exposes:
  - sample_mw(rng, n): nameplate MW draws (from completed + withdrawn PJM projects)
  - sample_duration_months(rng, n): queue-to-COD duration, in months, from completed PJM projects
  - sample_upgrade_dollars_per_kw(rng): $/kW for total POI upgrade (between completed and withdrawn avgs)
"""

from __future__ import annotations
import os
import pickle
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(BASE_DIR, "_calib_cache.pkl")
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
    mw = df["mw1"].dropna()
    mw = mw[(mw > 1) & (mw < 2000)].to_numpy()

    comp = df[df["q_status"].astype(str).str.lower().isin(["operational", "complete", "completed"])].copy()
    dur_days = (comp["on_date"] - comp["q_date"]).dt.days
    dur_months = dur_days.dropna()
    dur_months = dur_months[(dur_months > 180) & (dur_months < 365 * 10)] / 30.44
    dur_months = dur_months.to_numpy()

    cache = {"mw": mw, "dur_months": dur_months}
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)
    return cache


def _cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    return _build_cache()


def sample_mw(rng, n):
    mw = _cache()["mw"]
    return rng.choice(mw, size=n, replace=True)


def sample_duration_months(rng, n):
    d = _cache()["dur_months"]
    return rng.choice(d, size=n, replace=True)


def sample_upgrade_dollars_per_kw(rng, low=71.0, high=563.0):
    """Log-uniform between the completion-avg and withdrawn-avg figures from the brief."""
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


if __name__ == "__main__":
    c = _cache()
    print(f"mw: n={len(c['mw'])}, median={np.median(c['mw']):.1f}, p90={np.percentile(c['mw'],90):.1f}")
    print(f"dur_months: n={len(c['dur_months'])}, median={np.median(c['dur_months']):.1f}, p90={np.percentile(c['dur_months'],90):.1f}")
