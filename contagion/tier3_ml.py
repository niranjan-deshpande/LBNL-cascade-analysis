"""Tier 3: Gradient boosting + SHAP feature importance."""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.inspection import permutation_importance
import shap
from config import TABLES_DIR, FIGURES_DIR, RANDOM_SEED
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_ml(X, y):
    """Train gradient boosting classifier with cross-validation and SHAP analysis."""
    print(f"\nTier 3 ML: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class balance: {y.mean():.3f} withdrawal rate")

    # 5-fold stratified CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    gbc = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=RANDOM_SEED,
    )

    scores = cross_val_score(gbc, X, y, cv=cv, scoring="roc_auc")
    print(f"Cross-validated AUC: {scores.mean():.3f} ± {scores.std():.3f}")

    # Fit on full data for feature importance
    gbc.fit(X, y)

    # Built-in feature importance
    feat_imp = pd.DataFrame({
        "feature": X.columns,
        "importance": gbc.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\n=== Top 15 Features (built-in importance) ===")
    print(feat_imp.head(15).to_string(index=False))

    # Permutation importance
    perm_imp = permutation_importance(gbc, X, y, n_repeats=10,
                                       random_state=RANDOM_SEED, scoring="roc_auc")
    feat_imp["perm_importance_mean"] = perm_imp.importances_mean
    feat_imp["perm_importance_std"] = perm_imp.importances_std

    # SHAP
    explainer = shap.TreeExplainer(gbc)
    shap_values = explainer.shap_values(X)

    feat_imp["shap_mean_abs"] = np.abs(shap_values).mean(axis=0)
    feat_imp = feat_imp.sort_values("shap_mean_abs", ascending=False)

    print("\n=== Top 15 Features (SHAP) ===")
    print(feat_imp[["feature", "shap_mean_abs"]].head(15).to_string(index=False))

    feat_imp.to_csv(os.path.join(TABLES_DIR, "tier3_feature_importance.csv"), index=False)

    # Save CV results
    cv_results = pd.DataFrame({
        "fold": range(1, 6),
        "auc": scores,
    })
    cv_results.to_csv(os.path.join(TABLES_DIR, "tier3_cv_results.csv"), index=False)

    return gbc, shap_values, X, feat_imp, scores
