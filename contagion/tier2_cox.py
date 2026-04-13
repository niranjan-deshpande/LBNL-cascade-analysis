"""Tier 2: Cox model with time-varying peer withdrawal covariate."""

import pandas as pd
import numpy as np
from lifelines import CoxTimeVaryingFitter
from config import TABLES_DIR
import os


def run_cox(t2_cp):
    """Fit Cox model with time-varying cumulative peer withdrawal count."""
    df = t2_cp.copy()

    # Create type dummies
    top_types = df["type_clean"].value_counts().head(6).index.tolist()
    df["type_group"] = df["type_clean"].where(df["type_clean"].isin(top_types), "Other")
    type_dummies = pd.get_dummies(df["type_group"], prefix="type", drop_first=True, dtype=int)

    # Entity dummies
    entity_dummies = pd.get_dummies(df["entity"], prefix="entity", drop_first=True, dtype=int)

    # Build model dataframe
    model_df = pd.concat([
        df[["q_id", "start", "stop", "event", "cumulative_peer_wd", "mw1_log", "q_year"]],
        type_dummies,
        entity_dummies,
    ], axis=1).fillna(0)

    # Remove zero-length intervals
    model_df = model_df[model_df["stop"] > model_df["start"]].copy()

    print(f"\nTier 2 Cox model: {len(model_df)} intervals, {model_df['q_id'].nunique()} subjects")
    print(f"Events: {model_df.groupby('q_id')['event'].max().sum()}")

    # Fit Cox time-varying model
    ctv = CoxTimeVaryingFitter(penalizer=0.01)
    ctv.fit(
        model_df,
        id_col="q_id",
        start_col="start",
        stop_col="stop",
        event_col="event",
    )

    ctv.print_summary()

    # Extract hazard ratios
    summary = ctv.summary.copy()
    summary["hazard_ratio"] = np.exp(summary["coef"])
    summary["hr_ci_lower"] = np.exp(summary["coef"] - 1.96 * summary["se(coef)"])
    summary["hr_ci_upper"] = np.exp(summary["coef"] + 1.96 * summary["se(coef)"])

    print("\n=== Key Hazard Ratios ===")
    key_vars = ["cumulative_peer_wd", "mw1_log", "q_year"]
    for v in key_vars:
        if v in summary.index:
            row = summary.loc[v]
            print(f"  {v}: HR={row['hazard_ratio']:.3f} "
                  f"[{row['hr_ci_lower']:.3f}, {row['hr_ci_upper']:.3f}] "
                  f"p={row['p']:.4f}")

    summary.to_csv(os.path.join(TABLES_DIR, "tier2_cox_results.csv"))
    return ctv, summary
