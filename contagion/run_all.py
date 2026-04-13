"""Main pipeline script for contagion analysis."""

import os
import sys
import time

# Ensure contagion/ is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import OUTPUT_DIR, FIGURES_DIR, TABLES_DIR


def main():
    start = time.time()

    # Create output directories
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    # ── Step 1: Load and prepare data ──────────────────────────────────
    print("=" * 60)
    print("STEP 1: DATA PREPARATION")
    print("=" * 60)

    from data_prep import load_raw_data, convert_dates, clean_data, build_tier1_sample, build_tier2_sample, build_tier3_features

    df = load_raw_data()
    assert len(df) > 30000, f"Expected ~36,441 rows, got {len(df)}"
    print(f"✓ Row count: {len(df)}")

    df = convert_dates(df)
    df = clean_data(df)

    # Tier 1
    t1 = build_tier1_sample(df)
    print(f"✓ Tier 1 sample ready: {len(t1)} projects")

    # Tier 2
    t2_cp = build_tier2_sample(df)
    print(f"✓ Tier 2 counting-process ready: {len(t2_cp)} intervals")

    # Tier 3
    X_ml, y_ml, t3 = build_tier3_features(t1)
    print(f"✓ Tier 3 features ready: {X_ml.shape}")

    # ── Step 2: Descriptive statistics ─────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: DESCRIPTIVE STATISTICS")
    print("=" * 60)

    from descriptive import run_descriptive
    poi, od, ent = run_descriptive(t1)

    # ── Step 3: Tier 1 — Logistic regression ──────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: TIER 1 — LOGISTIC REGRESSION")
    print("=" * 60)

    from tier1_logistic import run_logistic
    t1_model, t1_summary = run_logistic(t1)

    # ── Step 4: Tier 2 — Cox model ────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: TIER 2 — COX MODEL")
    print("=" * 60)

    from tier2_cox import run_cox
    cox_model, cox_summary = run_cox(t2_cp)

    # ── Step 5: Robustness checks ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: ROBUSTNESS CHECKS")
    print("=" * 60)

    from robustness import run_robustness
    rob = run_robustness(t1)

    # ── Step 6: Tier 3 — ML + SHAP ────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6: TIER 3 — GRADIENT BOOSTING + SHAP")
    print("=" * 60)

    from tier3_ml import run_ml
    gbc, shap_values, X_shap, feat_imp, cv_scores = run_ml(X_ml, y_ml)

    # ── Step 7: Visualizations ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 7: VISUALIZATIONS")
    print("=" * 60)

    from visualizations import run_visualizations
    perm_results, null_stats = rob["permutation"]
    run_visualizations(
        poi=poi,
        ent=ent,
        t1_summary=t1_summary,
        cox_summary=cox_summary,
        t2_cp=t2_cp,
        perm_results=perm_results,
        null_stats=null_stats,
        eh=rob["entity_heterogeneity"],
        dr=rob["dose_response"],
        temporal=rob["temporal"],
        shap_values=shap_values,
        X_ml=X_shap,
    )

    # ── Done ───────────────────────────────────────────────────────────
    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"PIPELINE COMPLETE — {elapsed:.1f}s")
    print(f"Figures: {FIGURES_DIR}")
    print(f"Tables:  {TABLES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
