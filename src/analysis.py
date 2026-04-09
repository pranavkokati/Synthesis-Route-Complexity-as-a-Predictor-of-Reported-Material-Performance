"""
Statistical analysis: multivariate regression, correlation, and subgroup tests.

For each Materials Project property (band_gap, formation_energy_per_atom,
energy_above_hull, density) a separate OLS model is fit:

    property ~ precursor_count + max_temperature_C + total_time_h
               + n_steps + precursor_diversity

All features are z-score standardised before regression to yield comparable
standardised coefficients.  HC3 heteroscedasticity-robust standard errors
are used throughout.

Analyses:
    1. Per-property multivariate OLS
    2. Pearson correlation matrix (features + all MP properties)
    3. VIF diagnostics
    4. Subgroup OLS by dominant cation family
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

# Materials Project properties used as performance / outcome variables
MP_PROPERTIES = {
    "band_gap":                   "Band Gap (eV)",
    "formation_energy_per_atom":  "Formation Energy (eV/atom)",
    "energy_above_hull":          "Energy Above Hull (eV/atom)",
    "density":                    "Crystal Density (g/cm³)",
}


def _clean_for_regression(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    winsorise: float = 0.01,
) -> pd.DataFrame:
    """
    Return rows where all *features* and *target* are finite, with optional
    Winsorisation of the target at ``winsorise`` tails.
    """
    sub = df[features + [target]].replace([np.inf, -np.inf], np.nan).dropna()
    if winsorise > 0 and len(sub) > 20:
        lo = sub[target].quantile(winsorise)
        hi = sub[target].quantile(1.0 - winsorise)
        sub = sub[(sub[target] >= lo) & (sub[target] <= hi)]
    return sub


def run_ols(
    df: pd.DataFrame,
    target: str,
    features: Optional[list[str]] = None,
    label: str = "",
) -> dict:
    """
    Fit OLS with standardised features.  Returns a result dict with keys:
    model, summary_df, r_squared, adj_r_squared, n, label, target.
    """
    if features is None:
        features = FEATURE_COLS

    sub = _clean_for_regression(df, features, target)
    n = len(sub)

    if n < 20:
        logger.warning(
            "Skipping OLS for target='%s' label='%s': only %d observations.",
            target, label, n,
        )
        return {
            "label": label, "target": target, "n": n,
            "r_squared": np.nan, "adj_r_squared": np.nan,
            "model": None, "summary_df": pd.DataFrame(),
        }

    # Standardise features to reduce condition number
    X_raw = sub[features].astype(float)
    X_mean = X_raw.mean()
    X_std  = X_raw.std().replace(0, 1.0)
    X_sc   = (X_raw - X_mean) / X_std
    X      = sm.add_constant(X_sc, has_constant="add")
    y      = sub[target].astype(float)

    model = sm.OLS(y, X).fit(cov_type="HC3")

    coef = model.params
    ci   = model.conf_int()

    summary_df = pd.DataFrame({
        "feature":       coef.index,
        "coefficient":   coef.values,
        "std_error":     model.bse.values,
        "p_value":       model.pvalues.values,
        "ci_lower":      ci[0].values,
        "ci_upper":      ci[1].values,
    })
    summary_df["significant"]   = summary_df["p_value"] < 0.05
    summary_df["feature_label"] = summary_df["feature"].map(
        {**FEATURE_LABELS, "const": "Intercept"}
    ).fillna(summary_df["feature"])
    summary_df["target"] = target

    logger.info(
        "[%s | target=%s] n=%d  R²=%.4f  adj-R²=%.4f",
        label, target, n, model.rsquared, model.rsquared_adj,
    )

    return {
        "label":        label,
        "target":       target,
        "n":            n,
        "r_squared":    model.rsquared,
        "adj_r_squared":model.rsquared_adj,
        "model":        model,
        "summary_df":   summary_df,
    }


def run_all_property_regressions(
    df: pd.DataFrame,
    features: Optional[list[str]] = None,
) -> dict[str, dict]:
    """
    Run OLS for each MP property.

    Returns a dict keyed by property name.
    """
    if features is None:
        features = FEATURE_COLS
    results = {}
    for prop in MP_PROPERTIES:
        if prop not in df.columns:
            continue
        results[prop] = run_ols(df, target=prop, features=features, label="Full dataset")
    return results


def compute_correlation_matrix(
    df: pd.DataFrame,
    features: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pearson correlation matrix over features + all available MP properties.
    Returns (corr_df, pval_df).
    """
    if features is None:
        features = FEATURE_COLS
    mp_cols = [p for p in MP_PROPERTIES if p in df.columns]
    cols = features + mp_cols

    sub = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    n_c = len(cols)
    corr_v = np.eye(n_c)
    pval_v = np.zeros((n_c, n_c))

    for i in range(n_c):
        for j in range(n_c):
            if i == j:
                continue
            r, p = pearsonr(sub.iloc[:, i], sub.iloc[:, j])
            corr_v[i, j] = r
            pval_v[i, j] = p

    labels = [FEATURE_LABELS.get(c, MP_PROPERTIES.get(c, c)) for c in cols]
    corr_df = pd.DataFrame(corr_v, index=labels, columns=labels)
    pval_df = pd.DataFrame(pval_v, index=labels, columns=labels)
    return corr_df, pval_df


def compute_vif(df: pd.DataFrame, features: Optional[list[str]] = None) -> pd.DataFrame:
    """Variance inflation factors for the feature set."""
    if features is None:
        features = FEATURE_COLS
    sub = df[features].replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    X = sm.add_constant(sub)
    rows = []
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        rows.append({"feature": col, "vif": variance_inflation_factor(X.values, i)})
    return pd.DataFrame(rows)


def subgroup_regression_by_family(
    df: pd.DataFrame,
    target: str = "band_gap",
    features: Optional[list[str]] = None,
    min_n: int = 15,
) -> list[dict]:
    """
    Run per-family OLS for *target*.  Families are inferred from the first
    recognisable element in the target formula.
    """
    if features is None:
        features = FEATURE_COLS
    df = df.copy()
    df["material_family"] = df["target_formula"].apply(_infer_family)

    results = []
    for family, gdf in df.groupby("material_family"):
        sub = gdf.dropna(subset=[target] + features)
        if len(sub) < min_n:
            continue
        res = run_ols(gdf, target=target, features=features, label=f"family={family}")
        res["group_key"] = family
        results.append(res)
    return results


def feature_importance_ranking(ols_result: dict) -> pd.DataFrame:
    """Rank features by absolute standardised coefficient."""
    sdf = ols_result.get("summary_df", pd.DataFrame())
    if sdf.empty:
        return sdf
    sub = sdf[sdf["feature"] != "const"].copy()
    sub["abs_coef"] = sub["coefficient"].abs()
    return sub.sort_values("abs_coef", ascending=False).reset_index(drop=True)


def _infer_family(formula: str) -> str:
    """Coarse material-family label from the first element in the formula."""
    if not formula:
        return "Other"
    elements = re.findall(r"[A-Z][a-z]?", formula)
    if not elements:
        return "Other"
    _FAMILY_MAP = {
        "Li": "Li-ion (Li)", "Na": "Na-ion (Na)", "K": "Alkali (K)",
        "Ba": "Ba-titanate", "Sr": "Sr-titanate", "Pb": "Pb-perovskite",
        "La": "La-oxide",    "Y": "Y-oxide",      "Ce": "Ce-oxide",
        "Bi": "Bi-oxide",    "Zn": "ZnO family",  "Ti": "Titanate",
        "Fe": "Fe-oxide",    "Mn": "Mn-oxide",    "Co": "Co-oxide",
        "Ni": "Ni-oxide",    "Cu": "Cu-oxide",    "Al": "Aluminate",
        "Si": "Silicate",    "Zr": "Zirconate",   "Ca": "Ca-compound",
        "Mg": "Mg-compound", "In": "In-oxide",    "Ga": "Ga-oxide",
        "Nd": "Nd-compound", "Sm": "Sm-compound",
    }
    return _FAMILY_MAP.get(elements[0], "Other")
