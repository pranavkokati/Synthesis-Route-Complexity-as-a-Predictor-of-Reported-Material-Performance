"""
Statistical analysis: multivariate regression, correlation, and subgroup tests.

Analyses performed:
    1. Multivariate OLS: perf_norm ~ complexity features
    2. Pearson correlation matrix across features + performance
    3. Per-metric-class subgroup regressions
    4. Per-element-family subgroup regressions (based on dominant cation)
    5. VIF (variance inflation factors) for multicollinearity check
"""

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.feature_extractor import FEATURE_COLS, FEATURE_LABELS

logger = logging.getLogger(__name__)

REGRESSION_TARGET = "perf_norm"


def _clean_regression_data(
    df: pd.DataFrame,
    features: list[str],
    target: str = REGRESSION_TARGET,
    winsorise: float = 0.01,
) -> pd.DataFrame:
    """
    Return a clean subset: rows where all features and the target are finite,
    with optional winsorisation of the target at ``winsorise`` tails.
    """
    cols = features + [target]
    sub = df[cols].replace([np.inf, -np.inf], np.nan).dropna()

    if winsorise > 0:
        lo = sub[target].quantile(winsorise)
        hi = sub[target].quantile(1.0 - winsorise)
        sub = sub[(sub[target] >= lo) & (sub[target] <= hi)]

    return sub


def run_ols(
    df: pd.DataFrame,
    features: Optional[list[str]] = None,
    target: str = REGRESSION_TARGET,
    label: str = "Full dataset",
) -> dict:
    """
    Fit an OLS model: *target* ~ *features* (with intercept).

    Returns a dict with keys:
        model, summary_df, r_squared, adj_r_squared, n, label
    """
    if features is None:
        features = FEATURE_COLS

    sub = _clean_regression_data(df, features, target)
    n = len(sub)

    if n < 20:
        logger.warning("Skipping OLS for '%s': only %d observations.", label, n)
        return {"label": label, "n": n, "r_squared": np.nan, "model": None, "summary_df": pd.DataFrame()}

    # Standardise features to reduce condition number before fitting
    X_raw = sub[features].astype(float)
    X_mean = X_raw.mean()
    X_std = X_raw.std().replace(0, 1)
    X_scaled = (X_raw - X_mean) / X_std
    X = sm.add_constant(X_scaled, has_constant="add")
    y = sub[target].astype(float)

    model = sm.OLS(y, X).fit(cov_type="HC3")  # heteroscedasticity-robust SEs

    coef = model.params
    se = model.bse
    pval = model.pvalues
    ci = model.conf_int()

    summary_df = pd.DataFrame(
        {
            "feature": coef.index,
            "coefficient": coef.values,
            "std_error": se.values,
            "p_value": pval.values,
            "ci_lower": ci[0].values,
            "ci_upper": ci[1].values,
        }
    )
    summary_df["significant"] = summary_df["p_value"] < 0.05
    summary_df["feature_label"] = summary_df["feature"].map(
        {**FEATURE_LABELS, "const": "Intercept"}
    ).fillna(summary_df["feature"])

    logger.info(
        "[%s] n=%d  R²=%.4f  adj-R²=%.4f",
        label,
        n,
        model.rsquared,
        model.rsquared_adj,
    )

    return {
        "label": label,
        "n": n,
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "model": model,
        "summary_df": summary_df,
    }


def compute_correlation_matrix(
    df: pd.DataFrame,
    features: Optional[list[str]] = None,
    target: str = REGRESSION_TARGET,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute Pearson correlation matrix and associated p-value matrix.

    Returns (corr_matrix, pval_matrix) as DataFrames.
    """
    if features is None:
        features = FEATURE_COLS
    cols = features + [target]
    sub = df[cols].replace([np.inf, -np.inf], np.nan).dropna()

    n_cols = len(cols)
    corr_vals = np.eye(n_cols)
    pval_vals = np.zeros((n_cols, n_cols))

    for i in range(n_cols):
        for j in range(n_cols):
            if i == j:
                continue
            r, p = pearsonr(sub.iloc[:, i], sub.iloc[:, j])
            corr_vals[i, j] = r
            pval_vals[i, j] = p

    col_labels = [FEATURE_LABELS.get(c, c) for c in cols]
    corr_df = pd.DataFrame(corr_vals, index=col_labels, columns=col_labels)
    pval_df = pd.DataFrame(pval_vals, index=col_labels, columns=col_labels)
    return corr_df, pval_df


def compute_vif(df: pd.DataFrame, features: Optional[list[str]] = None) -> pd.DataFrame:
    """Compute variance inflation factors for the feature set."""
    if features is None:
        features = FEATURE_COLS
    sub = df[features].replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    X = sm.add_constant(sub)
    vif_data = pd.DataFrame(
        {
            "feature": X.columns,
            "vif": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
        }
    )
    return vif_data[vif_data["feature"] != "const"]


def subgroup_regression_by_metric(df: pd.DataFrame) -> list[dict]:
    """Run OLS within each extracted metric class."""
    results = []
    for metric in sorted(df["metric_name"].dropna().unique()):
        sub = df[df["metric_name"] == metric]
        res = run_ols(sub, label=f"metric={metric}")
        res["group_key"] = metric
        results.append(res)
    return results


def subgroup_regression_by_family(df: pd.DataFrame) -> list[dict]:
    """
    Run OLS within coarse material families, defined by the dominant cation
    element derived from the target formula.

    Families are assembled by counting elements in the formula string and
    taking the first capital-letter token as a proxy for the cation.
    """
    df = df.copy()
    df["material_family"] = df["target_formula"].apply(_infer_family)

    results = []
    for family, gdf in df.groupby("material_family"):
        if gdf["perf_norm"].notna().sum() < 20:
            continue
        res = run_ols(gdf, label=f"family={family}")
        res["group_key"] = family
        results.append(res)
    return results


def _infer_family(formula: str) -> str:
    """
    Map a chemical formula to a broad material family based on the first
    capital-letter element token.  Returns 'Other' if unparseable.
    """
    if not formula:
        return "Other"

    # Element symbols are one uppercase letter optionally followed by lowercase
    elements = re.findall(r"[A-Z][a-z]?", formula)
    if not elements:
        return "Other"

    # Map first element to broad families
    _FAMILY_MAP = {
        "Li": "Li-ion",
        "Na": "Na-ion",
        "K": "K-ion",
        "Ba": "Perovskite/BaTiO3",
        "Sr": "Perovskite/SrTiO3",
        "La": "Lanthanide oxide",
        "Y": "Yttrium oxide",
        "Bi": "Bismuth oxide",
        "Zn": "ZnO family",
        "Ti": "Titanate",
        "Fe": "Iron oxide",
        "Mn": "Manganese oxide",
        "Co": "Cobalt oxide",
        "Ni": "Nickel oxide",
        "Cu": "Copper oxide",
        "Al": "Aluminate",
        "Si": "Silicate",
        "Zr": "Zirconate",
        "Ce": "Ceria",
        "Ca": "Calcium compound",
    }
    return _FAMILY_MAP.get(elements[0], "Other")


def feature_importance_ranking(ols_result: dict) -> pd.DataFrame:
    """
    Rank complexity features by absolute standardised coefficient
    from a fitted OLS result.
    """
    sdf = ols_result.get("summary_df", pd.DataFrame())
    if sdf.empty:
        return sdf
    sub = sdf[sdf["feature"] != "const"].copy()
    sub["abs_coef"] = sub["coefficient"].abs()
    return sub.sort_values("abs_coef", ascending=False).reset_index(drop=True)


