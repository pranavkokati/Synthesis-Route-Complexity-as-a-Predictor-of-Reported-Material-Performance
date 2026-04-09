"""
Pre-manuscript diagnostic analyses.

Three checks required before any writing:

    1. Subgroup FDR correction (Benjamini-Hochberg) across all material-family
       regressions.  Separates robust findings (large n, FDR-significant) from
       exploratory findings (small n, preliminary).

    2. Precursor-Diversity confound check.  Adds target_n_elements (number of
       distinct elements in the target formula) as a covariate and measures how
       much of the precursor_diversity coefficient is explained by compositional
       complexity.

    3. Bootstrap confidence intervals for small-subgroup R² values (n < 100).
       Addresses inflated R² from small-sample post-hoc selection.

Results are written to output/results/diagnostics/ and used in the manuscript.
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import f as f_dist
from statsmodels.stats.multitest import multipletests

from src.analysis import _infer_family, MP_PROPERTIES
from src.feature_extractor import FEATURE_COLS, FEATURE_LABELS

logger = logging.getLogger(__name__)

_BOOTSTRAP_REPS = 1000
_SMALL_N_THRESHOLD = 100  # subgroups below this get bootstrap CI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _standardise(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    X = df[cols].astype(float)
    return (X - X.mean()) / X.std().replace(0, 1.0)


def _ols(X_sc: pd.DataFrame, y: pd.Series) -> sm.regression.linear_model.RegressionResultsWrapper:
    return sm.OLS(y, sm.add_constant(X_sc, has_constant="add")).fit(cov_type="HC3")


def count_target_elements(formula: str) -> float:
    """Number of distinct element symbols in a formula string."""
    if not formula:
        return np.nan
    return float(len(set(re.findall(r"[A-Z][a-z]?", formula))))


# ---------------------------------------------------------------------------
# 1. FDR-corrected subgroup analysis
# ---------------------------------------------------------------------------

def fdr_corrected_subgroup(
    df: pd.DataFrame,
    target: str = "band_gap",
    min_n: int = 15,
) -> pd.DataFrame:
    """
    Run OLS within every material family, collect F-test p-values, and apply
    Benjamini-Hochberg FDR correction.

    Returns a DataFrame with columns:
        family, n, r_squared, adj_r_squared, f_pvalue, p_fdr, sig_fdr,
        reliability_tier
    """
    df = df.copy()
    df["material_family"] = df["target_formula"].apply(_infer_family)

    rows = []
    for fam, gdf in df.groupby("material_family"):
        sub = gdf.dropna(subset=FEATURE_COLS + [target])
        n = len(sub)
        if n < min_n:
            continue
        X_sc = _standardise(sub, FEATURE_COLS)
        y = sub[target].astype(float)
        m = sm.OLS(y, sm.add_constant(X_sc)).fit()
        if not np.isfinite(m.f_pvalue):
            continue
        rows.append({
            "family": fam, "n": n,
            "r_squared": m.rsquared, "adj_r_squared": m.rsquared_adj,
            "f_pvalue": m.f_pvalue,
        })

    fdf = pd.DataFrame(rows).dropna(subset=["f_pvalue"])
    reject, p_corr, _, _ = multipletests(fdf["f_pvalue"].values, method="fdr_bh")
    fdf["p_fdr"] = p_corr
    fdf["sig_fdr"] = reject
    fdf["reliability_tier"] = fdf["n"].apply(
        lambda n: "robust" if n >= 200 else ("moderate" if n >= 50 else "exploratory")
    )

    return fdf.sort_values("r_squared", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Bootstrap R² CIs for small subgroups
# ---------------------------------------------------------------------------

def bootstrap_r2(
    df: pd.DataFrame,
    family: str,
    target: str = "band_gap",
    n_boot: int = _BOOTSTRAP_REPS,
    seed: int = 42,
) -> dict:
    """
    Parametric bootstrap of R² for a specific subgroup.

    Returns a dict with observed_r2, boot_mean, ci_lower, ci_upper, n.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["material_family"] = df["target_formula"].apply(_infer_family)
    sub = df[df["material_family"] == family].dropna(subset=FEATURE_COLS + [target])
    n = len(sub)

    X_sc = _standardise(sub, FEATURE_COLS).values
    y = sub[target].astype(float).values
    X = np.column_stack([np.ones(n), X_sc])

    observed = sm.OLS(y, X).fit().rsquared
    boot_r2 = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Xb, yb = X[idx], y[idx]
        try:
            r2 = sm.OLS(yb, Xb).fit().rsquared
            if np.isfinite(r2):
                boot_r2.append(r2)
        except Exception:
            pass

    boot_r2 = np.array(boot_r2)
    return {
        "family": family,
        "target": target,
        "n": n,
        "observed_r2": observed,
        "boot_mean": float(np.mean(boot_r2)),
        "ci_lower": float(np.percentile(boot_r2, 2.5)),
        "ci_upper": float(np.percentile(boot_r2, 97.5)),
        "n_boot": n_boot,
    }


# ---------------------------------------------------------------------------
# 3. Precursor Diversity confound check
# ---------------------------------------------------------------------------

def precursor_diversity_confound_check(
    df: pd.DataFrame,
    targets: list[str] | None = None,
) -> pd.DataFrame:
    """
    For each target property, compare the precursor_diversity coefficient
    in two models:
        A — baseline (FEATURE_COLS only)
        B — controlled (FEATURE_COLS + target_n_elements)

    Returns a DataFrame documenting coefficient change and retained significance.
    """
    if targets is None:
        targets = [p for p in MP_PROPERTIES if p != "energy_above_hull" and p in df.columns]

    df = df.copy()
    df["target_n_elements"] = df["target_formula"].apply(count_target_elements)

    rows = []
    for prop in targets:
        sub = df[FEATURE_COLS + ["target_n_elements", prop]].replace([np.inf, -np.inf], np.nan).dropna()
        n = len(sub)

        # Model A — without control
        X_a = _standardise(sub, FEATURE_COLS)
        mA = _ols(X_a, sub[prop].astype(float))

        # Model B — with target_n_elements
        X_b = _standardise(sub, FEATURE_COLS + ["target_n_elements"])
        mB = _ols(X_b, sub[prop].astype(float))

        beta_a = mA.params.get("precursor_diversity", np.nan)
        beta_b = mB.params.get("precursor_diversity", np.nan)
        p_a    = mA.pvalues.get("precursor_diversity", np.nan)
        p_b    = mB.pvalues.get("precursor_diversity", np.nan)
        beta_ctrl = mB.params.get("target_n_elements", np.nan)
        p_ctrl    = mB.pvalues.get("target_n_elements", np.nan)

        pct_change = 100.0 * (beta_b - beta_a) / abs(beta_a) if beta_a != 0 else np.nan

        rows.append({
            "property": prop,
            "property_label": MP_PROPERTIES.get(prop, prop),
            "n": n,
            "beta_no_ctrl": beta_a,
            "p_no_ctrl": p_a,
            "beta_with_ctrl": beta_b,
            "p_with_ctrl": p_b,
            "pct_beta_change": pct_change,
            "beta_target_n_elements": beta_ctrl,
            "p_target_n_elements": p_ctrl,
            "r2_model_a": mA.rsquared,
            "r2_model_b": mB.rsquared,
            "interpretation": _confound_interpretation(pct_change, p_b),
        })

    return pd.DataFrame(rows)


def _confound_interpretation(pct_change: float, p_with_ctrl: float) -> str:
    if not np.isfinite(pct_change) or not np.isfinite(p_with_ctrl):
        return "insufficient data"
    if p_with_ctrl >= 0.05:
        return "loses significance — likely proxy for composition"
    if abs(pct_change) < 15:
        return "robust — minimal confounding (<15% change)"
    if abs(pct_change) < 40:
        return "partial confound — retains significance despite 15–40% β change"
    return "substantial confound — >40% β change, interpret with caution"


# ---------------------------------------------------------------------------
# 4. Per-feature partial correlations controlling for target_n_elements
# ---------------------------------------------------------------------------

def partial_r2_decomposition(
    df: pd.DataFrame,
    target: str = "band_gap",
) -> pd.DataFrame:
    """
    Compute each feature's semi-partial R² contribution (incremental R²
    when added last to a model containing all other features).
    """
    df = df.copy()
    df["target_n_elements"] = df["target_formula"].apply(count_target_elements)
    all_feats = FEATURE_COLS + ["target_n_elements"]

    sub = df[all_feats + [target]].replace([np.inf, -np.inf], np.nan).dropna()
    n = len(sub)

    X_full_sc = _standardise(sub, all_feats)
    m_full = sm.OLS(sub[target].astype(float), sm.add_constant(X_full_sc)).fit()
    r2_full = m_full.rsquared

    rows = []
    for feat in all_feats:
        others = [f for f in all_feats if f != feat]
        X_red_sc = _standardise(sub, others)
        m_red = sm.OLS(sub[target].astype(float), sm.add_constant(X_red_sc)).fit()
        semi_partial_r2 = r2_full - m_red.rsquared
        label = FEATURE_LABELS.get(feat, feat)
        rows.append({
            "feature": feat,
            "feature_label": label,
            "semi_partial_r2": semi_partial_r2,
            "r2_without_feature": m_red.rsquared,
        })

    return pd.DataFrame(rows).sort_values("semi_partial_r2", ascending=False)


# ---------------------------------------------------------------------------
# Master runner
# ---------------------------------------------------------------------------

def run_all_diagnostics(
    df: pd.DataFrame,
    out_dir: Path,
    primary_target: str = "band_gap",
    bootstrap_families: list[str] | None = None,
) -> dict:
    """
    Run all three diagnostic analyses and save results.

    Returns a dict of DataFrames for use in visualisation.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1 — FDR subgroup
    logger.info("Running FDR-corrected subgroup analysis...")
    fdr_df = fdr_corrected_subgroup(df, target=primary_target)
    fdr_df.to_csv(out_dir / "subgroup_fdr.csv", index=False)
    logger.info("\n%s", fdr_df[["family","n","r_squared","adj_r_squared","p_fdr","sig_fdr","reliability_tier"]].to_string(index=False))

    # 2 — Bootstrap CIs for small subgroups
    if bootstrap_families is None:
        bootstrap_families = fdr_df[fdr_df["n"] < _SMALL_N_THRESHOLD]["family"].tolist()

    boot_rows = []
    for fam in bootstrap_families:
        logger.info("Bootstrapping R² for family=%s...", fam)
        b = bootstrap_r2(df, fam, target=primary_target)
        boot_rows.append(b)
        logger.info(
            "  %s: R²=%.3f  95%%CI=[%.3f, %.3f]  (n=%d)",
            fam, b["observed_r2"], b["ci_lower"], b["ci_upper"], b["n"],
        )

    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(out_dir / "bootstrap_r2.csv", index=False)

    # 3 — Confound check
    logger.info("Running precursor diversity confound check...")
    confound_df = precursor_diversity_confound_check(df)
    confound_df.to_csv(out_dir / "confound_check.csv", index=False)
    for _, row in confound_df.iterrows():
        logger.info(
            "  %-30s  β: %.3f→%.3f (%+.1f%%)  p_ctrl=%.4f  → %s",
            row["property_label"],
            row["beta_no_ctrl"], row["beta_with_ctrl"], row["pct_beta_change"],
            row["p_with_ctrl"], row["interpretation"],
        )

    # 4 — Semi-partial R²
    logger.info("Computing semi-partial R² decomposition for %s...", primary_target)
    partial_df = partial_r2_decomposition(df, target=primary_target)
    partial_df.to_csv(out_dir / "semi_partial_r2.csv", index=False)
    logger.info("\n%s", partial_df.to_string(index=False))

    return {
        "fdr_df": fdr_df,
        "boot_df": boot_df,
        "confound_df": confound_df,
        "partial_df": partial_df,
    }
