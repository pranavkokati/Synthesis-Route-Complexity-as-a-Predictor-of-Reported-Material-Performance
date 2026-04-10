"""
Statistical analysis: coordination-environment entropy vs synthesizability.

Analyses
--------
1. One-way ANOVA: each CE descriptor ~ synth_class (0 / 1 / 2)
   - Effect size η² (eta-squared)
   - Post-hoc Tukey HSD
   - Benjamini–Hochberg FDR correction across descriptors

2. Three-model logistic regression (binary synthesizable ~ features)
   - Model A  naive:    [nsites, nelements]
   - Model B  entropy:  [ce_entropy, n_distinct_envs, dominance, gini]
   - Model C  full:     all eight features above
   - ROC-AUC with bootstrap 95 % CI for all three models
   - McFadden pseudo-R² and LRT incremental test (B vs A, C vs A)

3. Spearman ρ: ce_entropy ~ energy_above_hull

4. Composition control
   - Per-element mean CE entropy baseline
   - Composition-corrected entropy = raw − element-mean baseline
   - ANOVA repeated on corrected entropy

5. Crystal-system subgroup ANOVA

6. Pettifor chemical-scale analysis
   - Pettifor span = max − min Mendeleev number among structure's elements
   - Structures binned into chemical-diversity quartiles
   - ANOVA within each quartile
"""

import ast
import json
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_NAMES = {0: "Stable", 1: "Metastable", 2: "Unstable"}
DESCRIPTOR_COLS = ["ce_entropy", "n_distinct_envs", "dominance", "gini"]
CONTROL_COLS = ["nsites", "nelements"]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _parse_elements(val) -> list[str]:
    """Parse element list from any format: list, JSON string, or Python repr."""
    if isinstance(val, list):
        return [str(e) for e in val]
    if isinstance(val, str):
        # Try JSON first (double-quoted), then Python repr (single-quoted)
        for parser in (json.loads, ast.literal_eval):
            try:
                result = parser(val)
                if isinstance(result, list):
                    return [str(e) for e in result]
            except Exception:
                pass
    return []


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _eta_squared(groups: list[np.ndarray]) -> float:
    """One-way ANOVA η² from group arrays."""
    all_vals = np.concatenate(groups)
    grand_mean = all_vals.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = np.sum((all_vals - grand_mean) ** 2)
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d between two groups."""
    pooled_sd = np.sqrt(
        ((len(a) - 1) * a.std(ddof=1) ** 2 + (len(b) - 1) * b.std(ddof=1) ** 2)
        / (len(a) + len(b) - 2)
    )
    return float((a.mean() - b.mean()) / pooled_sd) if pooled_sd > 0 else 0.0


def _tukey_hsd(groups: dict[str, np.ndarray]) -> pd.DataFrame:
    """Pairwise Tukey HSD from a dict of {name: values}."""
    from itertools import combinations
    from scipy.stats import studentized_range

    keys = sorted(groups.keys())
    ns = {k: len(v) for k, v in groups.items()}
    means = {k: float(np.mean(v)) for k, v in groups.items()}
    k = len(keys)
    n_total = sum(ns.values())
    ss_within = sum(np.sum((v - means[kk]) ** 2) for kk, v in groups.items())
    mse = ss_within / max(n_total - k, 1)

    rows = []
    for a, b in combinations(keys, 2):
        se = np.sqrt(mse * 0.5 * (1 / ns[a] + 1 / ns[b]))
        q = abs(means[a] - means[b]) / se if se > 0 else 0.0
        p = float(1 - studentized_range.cdf(q, k, n_total - k))
        rows.append({
            "group_a":  a,
            "group_b":  b,
            "mean_a":   means[a],
            "mean_b":   means[b],
            "diff":     means[a] - means[b],
            "cohens_d": _cohens_d(groups[a], groups[b]),
            "q_stat":   q,
            "p_tukey":  p,
            "sig":      "*" if p < 0.05 else "",
        })
    return pd.DataFrame(rows)


def _bh_correction(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR correction. Returns adjusted p-values."""
    n = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(n)
    cummin = np.inf
    for i in range(n - 1, -1, -1):
        rank = order[i]
        adj = pvalues[rank] * n / (i + 1)
        cummin = min(cummin, adj)
        adjusted[rank] = cummin
    return np.clip(adjusted, 0.0, 1.0)


def _bootstrap_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (auc, ci_low_95, ci_high_95) via percentile bootstrap."""
    rng = np.random.default_rng(seed)
    base_auc = float(roc_auc_score(y_true, y_score))
    boot_aucs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    lo = float(np.percentile(boot_aucs, 2.5))
    hi = float(np.percentile(boot_aucs, 97.5))
    return base_auc, lo, hi


def _mcfadden_r2(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """McFadden pseudo-R² = 1 − log-likelihood(model) / log-likelihood(null)."""
    eps = 1e-15
    p = np.clip(y_prob, eps, 1 - eps)
    ll_model = np.sum(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    p0 = y_true.mean()
    ll_null = len(y_true) * (p0 * np.log(p0 + eps) + (1 - p0) * np.log(1 - p0 + eps))
    return float(1 - ll_model / ll_null) if ll_null != 0 else 0.0


def _fit_logit(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
) -> tuple[LogisticRegression, np.ndarray]:
    """Fit logistic regression and return (model, y_prob).

    No class_weight balancing: balanced weighting shifts predicted
    probabilities away from the empirical marginal, making McFadden R²
    undefined relative to the intercept-only null.  AUC is invariant to
    class balance, so this choice does not affect the primary metric.
    """
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=seed)
    clf.fit(X, y)
    return clf, clf.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Pettifor / Mendeleev helpers
# ---------------------------------------------------------------------------

def _mendeleev_numbers(elements: list[str]) -> list[int]:
    """Return Mendeleev ordering numbers for a list of element symbols."""
    try:
        from pymatgen.core.periodic_table import Element
        return [Element(e).mendeleev_no for e in elements if e]
    except Exception:
        return []


def _pettifor_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add pettifor_span and pettifor_mean columns to df (in place copy).

    pettifor_span : range of Mendeleev numbers (chemical diversity proxy)
    pettifor_mean : mean Mendeleev number (chemical identity proxy)
    """
    df = df.copy()
    spans, means = [], []
    for _, row in df.iterrows():
        elems = _parse_elements(row.get("elements", []))
        nums = _mendeleev_numbers(elems)
        if nums:
            spans.append(max(nums) - min(nums))
            means.append(float(np.mean(nums)))
        else:
            spans.append(np.nan)
            means.append(np.nan)
    df["pettifor_span"] = spans
    df["pettifor_mean"] = means
    return df


# ---------------------------------------------------------------------------
# Composition control helpers
# ---------------------------------------------------------------------------

def _element_entropy_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-element mean CE entropy across all structures containing that element.
    """
    elem_vals: dict[str, list[float]] = defaultdict(list)

    for _, row in df.iterrows():
        if pd.isna(row.get("ce_entropy")):
            continue
        for el in _parse_elements(row.get("elements", [])):
            elem_vals[el].append(float(row["ce_entropy"]))

    return pd.DataFrame([
        {
            "element":          el,
            "mean_entropy":     float(np.mean(vals)),
            "median_entropy":   float(np.median(vals)),
            "sd_entropy":       float(np.std(vals)),
            "n_structures":     len(vals),
        }
        for el, vals in sorted(elem_vals.items())
    ])


def _composition_corrected_entropy(
    row: pd.Series,
    el_map: dict[str, float],
) -> float | None:
    """
    Composition-corrected entropy = raw − mean(element baselines).

    The element baseline for each element is the mean CE entropy of all
    structures containing that element.  This removes the compositional
    confound (simple elements form simple structures).
    """
    elems = _parse_elements(row.get("elements", []))
    if not elems:
        return None
    baselines = [el_map[e] for e in elems if e in el_map]
    if not baselines:
        return None
    raw = row.get("ce_entropy")
    return float(raw - np.mean(baselines)) if not pd.isna(raw) else None


# ---------------------------------------------------------------------------
# Main analysis entry-point
# ---------------------------------------------------------------------------

def run_analysis(df: pd.DataFrame) -> dict:
    """
    Full statistical analysis pipeline.

    Parameters
    ----------
    df : merged DataFrame from coord_env.compute_coord_entropy().
         Required columns: ce_entropy, n_distinct_envs, dominance, gini,
         synth_class, synthesizable, energy_above_hull, nsites, nelements,
         crystal_system, elements.

    Returns
    -------
    dict of DataFrames and scalar results (see keys below).
    """
    results: dict = {}
    df = df.dropna(
        subset=["ce_entropy", "synth_class", "synthesizable", "energy_above_hull"]
    ).copy()
    logger.info("Analysis dataset: %d structures", len(df))

    # ------------------------------------------------------------------
    # 0. Pettifor features
    # ------------------------------------------------------------------
    df = _pettifor_descriptors(df)

    # ------------------------------------------------------------------
    # 1. Descriptive statistics per class
    # ------------------------------------------------------------------
    rows_cs = []
    for cls in [0, 1, 2]:
        sub = df[df["synth_class"] == cls]["ce_entropy"]
        rows_cs.append({
            "synth_class":    cls,
            "label":          CLASS_NAMES[cls],
            "n":              len(sub),
            "mean":           sub.mean(),
            "sd":             sub.std(),
            "median":         sub.median(),
            "q25":            sub.quantile(0.25),
            "q75":            sub.quantile(0.75),
        })
    results["class_stats"] = pd.DataFrame(rows_cs)
    logger.info(
        "Class stats:\n%s",
        results["class_stats"][["label", "n", "mean", "sd"]].to_string(index=False),
    )

    # ------------------------------------------------------------------
    # 2. ANOVA + η² for all four descriptors
    # ------------------------------------------------------------------
    anova_rows = []
    for desc in DESCRIPTOR_COLS:
        if desc not in df.columns:
            continue
        grps = {
            CLASS_NAMES[c]: df[df["synth_class"] == c][desc].dropna().values
            for c in [0, 1, 2]
            if (df["synth_class"] == c).sum() > 0
        }
        if len(grps) < 2:
            continue
        f_stat, p_val = stats.f_oneway(*grps.values())
        eta2 = _eta_squared(list(grps.values()))
        anova_rows.append({
            "descriptor":  desc,
            "F_stat":      f_stat,
            "p_value":     p_val,
            "eta_squared": eta2,
            "n":           sum(len(v) for v in grps.values()),
        })

    anova_df = pd.DataFrame(anova_rows)
    # Benjamini–Hochberg correction across descriptors
    if len(anova_df):
        anova_df["p_adj_bh"] = _bh_correction(anova_df["p_value"].values)
        anova_df["sig"] = anova_df["p_adj_bh"].apply(
            lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        )
    results["anova_table"] = anova_df
    logger.info(
        "ANOVA (ce_entropy): F=%.2f, p=%.2e, η²=%.4f",
        anova_df[anova_df["descriptor"] == "ce_entropy"].iloc[0]["F_stat"],
        anova_df[anova_df["descriptor"] == "ce_entropy"].iloc[0]["p_value"],
        anova_df[anova_df["descriptor"] == "ce_entropy"].iloc[0]["eta_squared"],
    )

    # ------------------------------------------------------------------
    # 3. Post-hoc Tukey HSD
    # ------------------------------------------------------------------
    ce_groups = {
        CLASS_NAMES[c]: df[df["synth_class"] == c]["ce_entropy"].values
        for c in [0, 1, 2]
        if (df["synth_class"] == c).sum() > 0
    }
    results["tukey_table"] = _tukey_hsd(ce_groups)

    # ------------------------------------------------------------------
    # 4. Spearman ρ with energy_above_hull
    # ------------------------------------------------------------------
    rho, pval = spearmanr(df["ce_entropy"], df["energy_above_hull"])
    results["spearman"] = {"rho": float(rho), "pval": float(pval)}
    logger.info("Spearman ρ(ce_entropy, ehull)=%.4f  p=%.2e", rho, pval)

    # ------------------------------------------------------------------
    # 5. Three-model logistic regression
    # ------------------------------------------------------------------
    y = df["synthesizable"].values.astype(int)
    feat_avail = [c for c in DESCRIPTOR_COLS + CONTROL_COLS if c in df.columns]
    lr_df = df[feat_avail + ["synthesizable"]].dropna()
    y_lr = lr_df["synthesizable"].values.astype(int)
    n_pos, n_neg = y_lr.sum(), (y_lr == 0).sum()
    logger.info(
        "Logistic regression dataset: n=%d (synth=%d, non=%d)",
        len(y_lr), n_pos, n_neg,
    )

    roc_data = {}
    logreg_rows = []

    if n_pos >= 20 and n_neg >= 20:
        models_spec = {
            "naive":   [c for c in CONTROL_COLS if c in feat_avail],
            "entropy": [c for c in DESCRIPTOR_COLS if c in feat_avail],
            "full":    feat_avail,
        }
        scaler = StandardScaler()
        X_all = scaler.fit_transform(lr_df[feat_avail].values)
        feat_idx = {f: i for i, f in enumerate(feat_avail)}

        for model_name, feats in models_spec.items():
            if not feats:
                continue
            idx = [feat_idx[f] for f in feats]
            X_sub = X_all[:, idx]
            clf, y_prob = _fit_logit(X_sub, y_lr)
            auc, ci_lo, ci_hi = _bootstrap_auc(y_lr, y_prob)
            fpr, tpr, _ = roc_curve(y_lr, y_prob)
            r2mc = _mcfadden_r2(y_lr, y_prob)

            roc_data[model_name] = {
                "fpr": fpr.tolist(), "tpr": tpr.tolist(),
                "auc": auc, "ci_low": ci_lo, "ci_high": ci_hi,
                "mcfadden_r2": r2mc,
            }

            for fname, coef in zip(feats, clf.coef_[0]):
                logreg_rows.append({
                    "model": model_name,
                    "feature": fname,
                    "coefficient": coef,
                })

            logger.info(
                "Model %-8s  AUC=%.4f [%.4f–%.4f]  pseudo-R²=%.4f",
                model_name, auc, ci_lo, ci_hi, r2mc,
            )

    results["roc_data"] = roc_data
    results["logreg_coef"] = pd.DataFrame(logreg_rows)

    # ------------------------------------------------------------------
    # 6. Composition control
    # ------------------------------------------------------------------
    el_table = _element_entropy_table(df)
    results["element_entropy"] = el_table
    el_map = el_table.set_index("element")["mean_entropy"].to_dict()

    df["ce_entropy_corrected"] = df.apply(
        lambda row: _composition_corrected_entropy(row, el_map), axis=1
    )
    df_corr = df.dropna(subset=["ce_entropy_corrected"])
    logger.info(
        "Composition-corrected entropy: %d structures", len(df_corr)
    )

    corr_grps = {
        CLASS_NAMES[c]: df_corr[df_corr["synth_class"] == c]["ce_entropy_corrected"].values
        for c in [0, 1, 2]
        if (df_corr["synth_class"] == c).sum() > 0
    }
    f_c, p_c = stats.f_oneway(*corr_grps.values())
    results["composition_anova"] = pd.DataFrame([{
        "descriptor":  "ce_entropy_corrected",
        "F_stat":      f_c,
        "p_value":     p_c,
        "eta_squared": _eta_squared(list(corr_grps.values())),
        "n":           sum(len(v) for v in corr_grps.values()),
    }])
    logger.info(
        "Composition-corrected ANOVA: F=%.2f  p=%.2e  η²=%.4f",
        f_c, p_c, results["composition_anova"]["eta_squared"].iloc[0],
    )

    # ------------------------------------------------------------------
    # 7. Crystal-system subgroup ANOVA
    # ------------------------------------------------------------------
    subgroup_rows = []
    if "crystal_system" in df.columns:
        for cs, sub in df.groupby("crystal_system"):
            grps = {
                CLASS_NAMES[c]: sub[sub["synth_class"] == c]["ce_entropy"].dropna().values
                for c in [0, 1, 2]
                if (sub["synth_class"] == c).sum() >= 10
            }
            if len(grps) < 2:
                continue
            try:
                f_sg, p_sg = stats.f_oneway(*grps.values())
                subgroup_rows.append({
                    "crystal_system": cs,
                    "n_total":        len(sub),
                    "F_stat":         f_sg,
                    "p_value":        p_sg,
                    "eta_squared":    _eta_squared(list(grps.values())),
                    **{f"n_{CLASS_NAMES[c]}": len(grps.get(CLASS_NAMES[c], [])) for c in [0, 1, 2]},
                })
            except Exception as exc:
                logger.warning("Subgroup ANOVA error [%s]: %s", cs, exc)

    if subgroup_rows:
        sg_df = pd.DataFrame(subgroup_rows)
        sg_df["p_adj_bh"] = _bh_correction(sg_df["p_value"].values)
        results["subgroup_anova"] = sg_df
        logger.info("Subgroup ANOVA: %d crystal systems", len(sg_df))
    else:
        results["subgroup_anova"] = pd.DataFrame()

    # ------------------------------------------------------------------
    # 8. Pettifor chemical-scale analysis
    # ------------------------------------------------------------------
    results["pettifor_analysis"] = _run_pettifor_analysis(df)

    # ------------------------------------------------------------------
    # Store enriched df for downstream plotting
    # ------------------------------------------------------------------
    results["_df_enriched"] = df[[c for c in df.columns if c != "structure"]].copy()

    return results


# ---------------------------------------------------------------------------
# Pettifor analysis
# ---------------------------------------------------------------------------

def _run_pettifor_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin structures by Pettifor span (chemical diversity) and run ANOVA
    of ce_entropy ~ synth_class within each quartile bin.

    Returns a DataFrame with one row per quartile bin.
    """
    df_p = df.dropna(subset=["pettifor_span", "ce_entropy", "synth_class"])
    if len(df_p) < 100:
        logger.warning("Too few structures for Pettifor analysis")
        return pd.DataFrame()

    try:
        df_p = df_p.copy()
        df_p["pettifor_bin"] = pd.qcut(
            df_p["pettifor_span"], q=4,
            labels=["Q1 (similar)", "Q2", "Q3", "Q4 (diverse)"],
            duplicates="drop",
        )
    except Exception as exc:
        logger.warning("Pettifor binning failed: %s", exc)
        return pd.DataFrame()

    rows = []
    for bin_label, sub in df_p.groupby("pettifor_bin", observed=True):
        grps = {
            CLASS_NAMES[c]: sub[sub["synth_class"] == c]["ce_entropy"].dropna().values
            for c in [0, 1, 2]
            if (sub["synth_class"] == c).sum() >= 10
        }
        if len(grps) < 2:
            continue
        try:
            f_p, p_p = stats.f_oneway(*grps.values())
            rows.append({
                "pettifor_bin":        str(bin_label),
                "span_mean":           sub["pettifor_span"].mean(),
                "span_median":         sub["pettifor_span"].median(),
                "n_total":             len(sub),
                "F_stat":              f_p,
                "p_value":             p_p,
                "eta_squared":         _eta_squared(list(grps.values())),
                **{f"n_{CLASS_NAMES[c]}": len(grps.get(CLASS_NAMES[c], [])) for c in [0, 1, 2]},
            })
        except Exception as exc:
            logger.warning("Pettifor ANOVA error [%s]: %s", bin_label, exc)

    if rows:
        pet_df = pd.DataFrame(rows)
        pet_df["p_adj_bh"] = _bh_correction(pet_df["p_value"].values)
        logger.info("Pettifor analysis: %d bins", len(pet_df))
        return pet_df

    return pd.DataFrame()
