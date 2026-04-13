"""Data loading, cleaning, and sample construction for contagion analysis."""

import pandas as pd
import numpy as np
from datetime import timedelta
from config import (
    EXCEL_PATH, SHEET_NAME, EXCEL_EPOCH, END_OF_STUDY, COX_ORIGIN,
    TIER2_ENTITIES, DEPTH_BINS, DEPTH_LABELS
)


def load_raw_data():
    """Load queue data from Excel sheet 03. Handles header detection."""
    # Try reading with default header
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, header=None)

    # Find the header row containing 'q_id'
    header_row = None
    for i in range(min(20, len(df))):
        row_vals = df.iloc[i].astype(str).str.lower().tolist()
        if any("q_id" in v for v in row_vals):
            header_row = i
            break

    if header_row is None:
        raise ValueError("Could not find header row with 'q_id' in sheet")

    # Re-read with correct header
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, header=header_row)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    return df


def convert_excel_date(series):
    """Convert Excel serial numbers to datetime. Non-numeric values become NaT."""
    numeric = pd.to_numeric(series, errors="coerce")
    return pd.to_timedelta(numeric, unit="D") + EXCEL_EPOCH


def convert_dates(df):
    """Convert date columns to datetime. All dates in Excel are serial numbers."""
    for col in ["q_date", "ia_date", "on_date", "wd_date", "prop_date"]:
        if col in df.columns:
            df[col] = convert_excel_date(df[col])

    return df


import re

# Abbreviation standardization map (order matters: longer patterns first)
_POI_ABBREVS = [
    # Substation variants (multi-word first to prevent "sub station" -> "substation station")
    (r'\bsub\s+station\b', 'substation'),
    (r'\bsubsta\b', 'substation'),
    (r'\bsub\b', 'substation'),
    (r'\bss\b', 'substation'),
    (r'\bsubs\b', 'substation'),
    # Junction
    (r'\bjct\b', 'junction'),
    (r'\bjctn\b', 'junction'),
    (r'\bjunc\b', 'junction'),
    # Road
    (r'\brd\b', 'road'),
    # Mountain
    (r'\bmt\b', 'mount'),
    (r'\bmtn\b', 'mountain'),
    # Transmission / switching station
    (r'\btrans\b', 'transmission'),
    (r'\bsw\b', 'switching'),
    (r'\bsta\b', 'station'),
    # Kilovolt
    (r'\bkv\b', 'kv'),
    # NOTE: The following were REMOVED due to excessive false positives:
    # - \bst\b -> "street": corrupts "St" (Saint) in place names (St. Louis, St. Francis)
    # - \bn\b, \bs\b, \be\b, \bw\b: single-letter expansions match possessive "'s",
    #   coordinate fragments, and other non-directional uses (403+ false positives)
    # - \bco\b -> "county": ambiguous with "Company"
]


def normalize_poi_name(series):
    """Normalize POI names: lowercase, standardize abbreviations, clean whitespace."""
    s = series.astype(str).str.lower().str.strip()

    # Remove Excel carriage return artifacts (_x000D_ and similar)
    s = s.str.replace(r'_x[0-9a-f]{4}_', '', regex=True)

    # Remove trailing punctuation
    s = s.str.replace(r'[.,;:\-/\\]+$', '', regex=True)

    # Normalize internal whitespace (tabs, multiple spaces)
    s = s.str.replace(r'\s+', ' ', regex=True)

    # Remove periods (e.g., "St." -> "St", "N." -> "N") for consistency
    s = s.str.replace(r'\.', '', regex=True)

    # Normalize hyphens/dashes to spaces
    s = s.str.replace(r'[-–—]', ' ', regex=True)

    # Normalize "NUMBERkv" -> "NUMBER kv" (e.g., "138kv" -> "138 kv")
    # This is the single largest source of unmerged duplicates (~930 POIs)
    s = s.str.replace(r'(\d)kv\b', r'\1 kv', regex=True)

    # Normalize whitespace again after above replacements
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()

    # Mark uninformative POI names as NA:
    # - Standard NA values (nan, n/a, unknown, none, empty)
    # - "tbd"/"tba" (to be determined/announced) — these group unrelated projects
    # - Bare voltage levels ("138 kv", "345", "69 kv") — no substation identity
    na_values = {"nan", "na", "n/a", "unknown", "none", "", "tbd", "tba"}
    s = s.where(~s.isin(na_values), other=np.nan)
    # Also mark bare voltage levels (just a number, or number + "kv")
    bare_voltage = s.str.fullmatch(r'\d+(\.\d+)?\s*(kv)?', na=False)
    s = s.where(~bare_voltage, other=np.nan)

    # Standardize abbreviations (only on non-NA values)
    for pattern, replacement in _POI_ABBREVS:
        s = s.str.replace(pattern, replacement, regex=True)

    # Final whitespace cleanup after replacements
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()

    return s


def clean_data(df):
    """Clean and prepare the dataset."""
    # Fix q_year anomalies
    if "q_year" in df.columns:
        df.loc[df["q_year"] == 1970, "q_year"] = np.nan

    # Flag negative mw1
    if "mw1" in df.columns:
        df["mw1_negative"] = df["mw1"] < 0
        df["mw1_log"] = np.log1p(df["mw1"].clip(lower=0))

    # Normalize poi_name
    if "poi_name" in df.columns:
        df["poi_name_clean"] = normalize_poi_name(df["poi_name"])

    # Create entity_poi key
    df["entity_poi"] = df["entity"].astype(str) + "||" + df["poi_name_clean"].astype(str)
    df.loc[df["poi_name_clean"].isna(), "entity_poi"] = np.nan

    # Binary withdrawal indicator
    df["withdrawn"] = (df["q_status"] == "withdrawn").astype(int)

    return df


def build_tier1_sample(df):
    """Build Tier 1 logistic regression sample: multi-project POIs."""
    # Drop rows without valid entity_poi
    t1 = df.dropna(subset=["entity_poi"]).copy()

    # Count projects per entity_poi
    poi_counts = t1.groupby("entity_poi").size().rename("poi_depth")
    t1 = t1.merge(poi_counts, on="entity_poi", how="left")

    # Filter to multi-project POIs
    t1 = t1[t1["poi_depth"] >= 2].copy()
    print(f"Tier 1 sample: {len(t1)} projects in {t1['entity_poi'].nunique()} multi-project POIs")

    # Compute leave-one-out peer withdrawal rate and count
    poi_wd = t1.groupby("entity_poi")["withdrawn"].agg(["sum", "count"]).rename(
        columns={"sum": "poi_wd_total", "count": "poi_n"}
    )
    t1 = t1.merge(poi_wd, on="entity_poi", how="left")

    # Leave-one-out: exclude the focal project
    t1["peer_wd_count"] = t1["poi_wd_total"] - t1["withdrawn"]
    t1["peer_n"] = t1["poi_n"] - 1
    t1["peer_wd_rate"] = t1["peer_wd_count"] / t1["peer_n"]

    # Type diversity at POI
    type_diversity = t1.groupby("entity_poi")["type_clean"].nunique().rename("poi_type_diversity")
    t1 = t1.merge(type_diversity, on="entity_poi", how="left")

    return t1


def _build_tier2_base(df, min_poi_depth=2):
    """Shared base sample construction for Tier 2 Cox models."""
    t2 = df[df["entity"].isin(TIER2_ENTITIES)].copy()
    t2 = t2.dropna(subset=["entity_poi"]).copy()
    t2 = t2.dropna(subset=["q_date"]).copy()

    t2["exit_date"] = END_OF_STUDY
    t2["event"] = 0

    wd_mask = (t2["q_status"] == "withdrawn") & t2["wd_date"].notna()
    t2.loc[wd_mask, "exit_date"] = t2.loc[wd_mask, "wd_date"]
    t2.loc[wd_mask, "event"] = 1

    op_mask = (t2["q_status"] == "operational") & t2["on_date"].notna()
    t2.loc[op_mask, "exit_date"] = t2.loc[op_mask, "on_date"]

    t2["start"] = (t2["q_date"] - COX_ORIGIN).dt.days
    t2["stop"] = (t2["exit_date"] - COX_ORIGIN).dt.days

    t2 = t2[t2["stop"] > t2["start"]].copy()
    t2 = t2[t2["start"].notna() & t2["stop"].notna()].copy()

    poi_counts = t2.groupby("entity_poi").size()
    eligible_pois = poi_counts[poi_counts >= min_poi_depth].index
    t2 = t2[t2["entity_poi"].isin(eligible_pois)].copy()

    return t2


def build_tier2_sample(df, lag_days=0, min_poi_depth=2):
    """Build Tier 2 Cox model sample: counting-process format with time-varying peer withdrawals.

    Args:
        lag_days: Number of days to lag peer withdrawal exposure. 0 = immediate,
                  183 = 6 months, 365 = 12 months.
        min_poi_depth: Minimum number of projects at a POI to include (default 2).
    """
    t2 = _build_tier2_base(df, min_poi_depth=min_poi_depth)

    # Create unique subject ID (q_id is entity-local, not globally unique)
    t2["subject_id"] = t2["entity"].astype(str) + "_" + t2["q_id"].astype(str)
    n_dupes = len(t2) - t2["subject_id"].nunique()
    if n_dupes > 0:
        # Same entity+q_id at multiple POIs is rare but possible; use row index
        t2["subject_id"] = t2["subject_id"] + "_" + t2.index.astype(str)

    print(f"Tier 2 base sample: {len(t2)} projects in {t2['entity_poi'].nunique()} POIs "
          f"(min depth={min_poi_depth})")

    # Build counting-process format: split intervals at peer withdrawal times
    # Track withdrawals by subject_id to avoid the same-day bug
    records = []
    n_deduped = 0
    for epoi, group in t2.groupby("entity_poi"):
        # Collect (withdrawal_time, subject_id) pairs for this POI
        wd_info = group.loc[group["event"] == 1, ["subject_id", "stop"]].copy()
        wd_info = wd_info.sort_values("stop")

        for _, row in group.iterrows():
            # Peer withdrawals: exclude this project by subject_id
            peer_wd = wd_info[wd_info["subject_id"] != row["subject_id"]]

            # Apply lag: peer withdrawal exposure time = wd_date + lag_days
            peer_exposure_times = (peer_wd["stop"].values + lag_days).astype(float)

            # Keep only exposures within this project's at-risk window
            # Use strict < for the upper bound to avoid duplicating the stop time
            peer_exposure_times = peer_exposure_times[
                (peer_exposure_times > row["start"]) & (peer_exposure_times < row["stop"])
            ]
            peer_exposure_times = np.sort(peer_exposure_times)

            # Count peer exposures exactly at stop (these are peers, not splits)
            n_at_stop = int(np.sum(
                ((peer_wd["stop"].values + lag_days) == row["stop"])
            ))
            if n_at_stop > 0:
                n_deduped += n_at_stop

            # Count peer withdrawals that happened before this project entered
            cum_peer_wd = int(np.sum(
                (peer_wd["stop"].values + lag_days) <= row["start"]
            ))

            interval_start = row["start"]
            split_points = list(peer_exposure_times) + [row["stop"]]

            for sp in split_points:
                if sp <= interval_start:
                    cum_peer_wd += 1
                    continue
                is_final = (sp == row["stop"])
                event_here = row["event"] if is_final else 0
                records.append({
                    "q_id": row["subject_id"],  # use globally unique ID
                    "entity_poi": epoi,
                    "entity": row["entity"],
                    "start": interval_start,
                    "stop": sp,
                    "event": event_here,
                    "cumulative_peer_wd": cum_peer_wd,
                    "type_clean": row["type_clean"],
                    "mw1_log": row.get("mw1_log", 0),
                    "q_year": row.get("q_year", np.nan),
                })
                interval_start = sp
                if not is_final:
                    cum_peer_wd += 1

    t2_cp = pd.DataFrame(records)
    lag_label = f" (lag={lag_days}d)" if lag_days > 0 else ""
    depth_label = f", depth>={min_poi_depth}" if min_poi_depth > 2 else ""
    print(f"Tier 2 counting-process{lag_label}{depth_label}: {len(t2_cp)} intervals "
          f"from {t2_cp['q_id'].nunique()} projects")
    if n_deduped > 0:
        print(f"  Edge cases: {n_deduped} peer exposures at exact stop time (handled via strict < filter)")
    return t2_cp


def build_tier3_features(t1):
    """Build feature matrix for Tier 3 ML model from Tier 1 sample."""
    feature_cols = ["peer_wd_count", "poi_depth", "mw1_log", "q_year", "poi_type_diversity"]

    t3 = t1.copy()

    # One-hot encode type_clean (top types + other)
    top_types = t3["type_clean"].value_counts().head(8).index.tolist()
    t3["type_group"] = t3["type_clean"].where(t3["type_clean"].isin(top_types), "Other")
    type_dummies = pd.get_dummies(t3["type_group"], prefix="type", drop_first=True, dtype=int)

    # One-hot encode entity (top entities + other)
    top_entities = t3["entity"].value_counts().head(8).index.tolist()
    t3["entity_group"] = t3["entity"].where(t3["entity"].isin(top_entities), "Other")
    entity_dummies = pd.get_dummies(t3["entity_group"], prefix="entity", drop_first=True, dtype=int)

    # Region dummies
    region_dummies = pd.get_dummies(t3["region"], prefix="region", drop_first=True, dtype=int)

    X = pd.concat([t3[feature_cols], type_dummies, entity_dummies, region_dummies], axis=1)
    X = X.fillna(0)
    y = t3["withdrawn"]

    return X, y, t3
