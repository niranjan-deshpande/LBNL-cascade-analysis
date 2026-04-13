"""Tier 1: Logistic regression of withdrawal on peer withdrawal rate."""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from config import TABLES_DIR
import os


def run_logistic(t1):
    """Logistic regression with clustered standard errors at entity_poi level."""
    df = t1.copy()

    # Prepare state variable: top 10 + Other
    top_states = df["state"].value_counts().head(10).index.tolist()
    df["state_group"] = df["state"].where(df["state"].isin(top_states), "Other")

    # Prepare type: top types + Other
    top_types = df["type_clean"].value_counts().head(6).index.tolist()
    df["type_group"] = df["type_clean"].where(df["type_clean"].isin(top_types), "Other")

    # Prepare entity dummies
    top_entities = df["entity"].value_counts().head(8).index.tolist()
    df["entity_group"] = df["entity"].where(df["entity"].isin(top_entities), "Other")

    # Build design matrix (convert bool dummies to int for statsmodels)
    type_dummies = pd.get_dummies(df["type_group"], prefix="type", drop_first=True, dtype=int)
    entity_dummies = pd.get_dummies(df["entity_group"], prefix="entity", drop_first=True, dtype=int)
    state_dummies = pd.get_dummies(df["state_group"], prefix="state", drop_first=True, dtype=int)

    X = pd.concat([
        df[["peer_wd_rate", "mw1_log", "q_year"]],
        type_dummies,
        entity_dummies,
        state_dummies,
    ], axis=1).fillna(0)

    # Ensure all columns are numeric
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    y = df["withdrawn"].astype(float)

    # Drop rows with any remaining NaN
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]
    cluster_var = df.loc[mask, "entity_poi"]

    # Add constant
    X = sm.add_constant(X)

    # Fit logistic regression with clustered SEs
    model = sm.Logit(y, X)
    try:
        result = model.fit(cov_type="cluster", cov_kwds={"groups": cluster_var}, disp=False)
    except Exception as e:
        print(f"Clustered SE failed ({e}), falling back to HC1")
        result = model.fit(cov_type="HC1", disp=False)

    # Extract results
    summary_df = pd.DataFrame({
        "coef": result.params,
        "se": result.bse,
        "z": result.tvalues,
        "p_value": result.pvalues,
        "odds_ratio": np.exp(result.params),
        "ci_lower": np.exp(result.conf_int()[0]),
        "ci_upper": np.exp(result.conf_int()[1]),
    })

    print("\n=== Tier 1: Logistic Regression ===")
    print(f"N = {result.nobs:.0f}, Pseudo R² = {result.prsquared:.4f}")
    print(summary_df[["coef", "odds_ratio", "ci_lower", "ci_upper", "p_value"]].round(4).to_string())

    summary_df.to_csv(os.path.join(TABLES_DIR, "tier1_logistic_results.csv"))

    return result, summary_df
