"""
Statistical analysis: coordination-environment entropy vs synthesizability.

Analyses performed
------------------
1. One-way ANOVA: ce_entropy ~ synth_class (0/1/2)
   - Effect size: eta-squared (η²)
   - Post-hoc: Tukey HSD

2. Binary logistic regression: synthesizable ~ [ce_entropy, n_distinct_envs,
   dominance, gini] controlling for nsites + nelements
   - ROC-AUC with bootstrap 95% CI
   - Coefficient table

3. Composition control
   - Compute per-element mean CE entropy (weighted by site count)
   - Subtract element-specific baseline → composition-corrected entropy
   - Repeat ANOVA on corrected entropy

4. Crystal-system subgroup analysis
   - Repeat ANOVA within each crystal system

5. Spearman correlation: ce_entropy ~ energy_above_hull

All numeric results are returned as a dict of DataFrames + scalar values.
"""

import json
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eta_squared(groups: list[np.ndarray]) -> float:
    """Compute eta-squared from group arrays."""
    all_vals = np.concatenate(groups)
    grand_mean = np.mean(all_vals)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_total = np.sum((all_vals - grand_mean) ** 2)
    return ss_between / ss_total if ss_total > 0 else 0.0


def _tukey_hsd(groups: dict[str, np.ndarray]) -> pd.DataFrame:
    """Pairwise Tukey HSD (manual implementation)."""
    from itertools import combinations
    from scipy.stats import studentized_range

    keys = sorted(groups.keys())
    ns = {k: len(v) for k, v in groups.items()}
    means = {k: np.mean(v) for k, v in groups.items()}
    # Pooled MSE
    k = len(keys)
    n_total = sum(ns.values())
    ss_within = sum(np.sum((v - means[kk]) ** 2) for kk, v in groups.items())
    mse = ss_within / (n_total - k)
    rows = []
    for a, b in combinations(keys, 2):
        se = np.sqrt(mse * 0.5 * (1 / ns[a] + 1 / ns[b]))
        q = abs(means[a] - means[b]) / se if se > 0 else 0
        p = 1 - studentized_range.cdf(q, k, n_total - k)
        rows.append({
            "group_a": a,
            "group_b": b,
            "mean_a": means[a],
            "mean_b": means[b],
            "diff": means[a] - means[b],
            "q_stat": q,
            "p_tukey": p,
        })
    return pd.DataFrame(rows)


def _bootstrap_auc(y_true, y_score, n_boot=1000, seed=42) -> tuple[float, float, float]:
    """Return (auc, ci_low, ci_high) via bootstrap."""
    rng = np.random.default_rng(seed)
    auc_base = roc_auc_score(y_true, y_score)
    boot_aucs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    lo = float(np.percentile(boot_aucs, 2.5))
    hi = float(np.percentile(boot_aucs, 97.5))
    return float(auc_base), lo, hi


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def run_analysis(df: pd.DataFrame) -> dict:
    """
    Run all statistical analyses on the merged dataset.

    Parameters
    ----------
    df : DataFrame produced by coord_env.compute_coord_entropy(), must contain:
         material_id, ce_entropy, n_distinct_envs, dominance, gini,
         synth_class, synthesizable, energy_above_hull, nsites, nelements,
         crystal_system, elements, ce_symbols

    Returns
    -------
    dict with keys:
        anova_table         : pd.DataFrame
        tukey_table         : pd.DataFrame
        logreg_coef         : pd.DataFrame
        roc_data            : dict {fpr, tpr, auc, ci_low, ci_high}
        spearman            : dict {rho, pval}
        composition_anova   : pd.DataFrame
        subgroup_anova      : pd.DataFrame
        element_entropy     : pd.DataFrame  (per-element mean entropy)
        class_stats         : pd.DataFrame  (mean/sd entropy per class)
    """
    results = {}
    df = df.copy()

    # Drop rows with missing key columns
    required = ["ce_entropy", "synth_class", "synthesizable", "energy_above_hull"]
    df = df.dropna(subset=required)
    logger.info("Analysis dataset: %d records", len(df))

    # -----------------------------------------------------------------------
    # 1. Class statistics
    # -----------------------------------------------------------------------
    class_names = {0: "Stable", 1: "Metastable", 2: "Unstable"}
    class_stats_rows = []
    for cls in [0, 1, 2]:
        sub = df[df["synth_class"] == cls]["ce_entropy"]
        class_stats_rows.append({
            "synth_class": cls,
            "label": class_names[cls],
            "n": len(sub),
            "mean_entropy": sub.mean(),
            "sd_entropy": sub.std(),
            "median_entropy": sub.median(),
            "q25": sub.quantile(0.25),
            "q75": sub.quantile(0.75),
        })
    results["class_stats"] = pd.DataFrame(class_stats_rows)
    logger.info("Class statistics:\n%s", results["class_stats"].to_string(index=False))

    # -----------------------------------------------------------------------
    # 2. One-way ANOVA: ce_entropy ~ synth_class
    # -----------------------------------------------------------------------
    groups = {
        class_names[c]: df[df["synth_class"] == c]["ce_entropy"].values
        for c in [0, 1, 2]
        if (df["synth_class"] == c).sum() > 0
    }
    f_stat, p_anova = stats.f_oneway(*groups.values())
    eta2 = _eta_squared(list(groups.values()))

    results["anova_table"] = pd.DataFrame([{
        "descriptor": "ce_entropy",
        "F_stat": f_stat,
        "p_value": p_anova,
        "eta_squared": eta2,
        "n_total": len(df),
    }])
    logger.info(
        "ANOVA ce_entropy ~ synth_class: F=%.4f, p=%.2e, η²=%.4f",
        f_stat, p_anova, eta2,
    )

    # Repeat for other descriptors
    extra_rows = []
    for desc in ["n_distinct_envs", "dominance", "gini"]:
        if desc not in df.columns:
            continue
        g = {class_names[c]: df[df["synth_class"] == c][desc].dropna().values
             for c in [0, 1, 2] if (df["synth_class"] == c).sum() > 0}
        f, p = stats.f_oneway(*g.values())
        e2 = _eta_squared(list(g.values()))
        extra_rows.append({
            "descriptor": desc,
            "F_stat": f,
            "p_value": p,
            "eta_squared": e2,
            "n_total": len(df),
        })
    if extra_rows:
        results["anova_table"] = pd.concat(
            [results["anova_table"], pd.DataFrame(extra_rows)],
            ignore_index=True,
        )

    # -----------------------------------------------------------------------
    # 3. Post-hoc Tukey HSD
    # -----------------------------------------------------------------------
    results["tukey_table"] = _tukey_hsd(groups)
    logger.info("Tukey HSD done (%d pairwise comparisons)", len(results["tukey_table"]))

    # -----------------------------------------------------------------------
    # 4. Spearman correlation with energy_above_hull
    # -----------------------------------------------------------------------
    rho, pval = spearmanr(df["ce_entropy"], df["energy_above_hull"])
    results["spearman"] = {"rho": float(rho), "pval": float(pval)}
    logger.info("Spearman ρ(ce_entropy, e_hull) = %.4f, p=%.2e", rho, pval)

    # -----------------------------------------------------------------------
    # 5. Logistic regression: synthesizable ~ entropy features + nsites/nelems
    # -----------------------------------------------------------------------
    feature_cols = ["ce_entropy", "n_distinct_envs", "dominance", "gini"]
    control_cols = ["nsites", "nelements"]
    all_feat = [c for c in feature_cols + control_cols if c in df.columns]

    lr_df = df[["synthesizable"] + all_feat].dropna()
    X = lr_df[all_feat].values
    y = lr_df["synthesizable"].values.astype(int)

    # Check class balance
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    logger.info("Logistic regression dataset: n=%d, synth=%d, non-synth=%d", len(y), n_pos, n_neg)

    if n_pos > 10 and n_neg > 10:
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)

        clf = LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced")
        clf.fit(X_sc, y)
        y_prob = clf.predict_proba(X_sc)[:, 1]

        auc_val, auc_lo, auc_hi = _bootstrap_auc(y, y_prob)
        fpr, tpr, _ = roc_curve(y, y_prob)

        results["roc_data"] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": auc_val,
            "ci_low": auc_lo,
            "ci_high": auc_hi,
        }

        # Coefficient table
        coef_rows = []
        for name, coef in zip(all_feat, clf.coef_[0]):
            coef_rows.append({"feature": name, "coefficient": coef})
        results["logreg_coef"] = pd.DataFrame(coef_rows)

        logger.info(
            "Logistic regression AUC=%.4f [%.4f–%.4f]",
            auc_val, auc_lo, auc_hi,
        )
    else:
        logger.warning("Skipping logistic regression: insufficient class balance")
        results["roc_data"] = {}
        results["logreg_coef"] = pd.DataFrame()

    # -----------------------------------------------------------------------
    # 6. Composition control
    # -----------------------------------------------------------------------
    element_entropy = _compute_element_entropy(df)
    results["element_entropy"] = element_entropy

    # Compute composition-corrected entropy per structure
    df["ce_entropy_corrected"] = df.apply(
        lambda row: _corrected_entropy(row, element_entropy),
        axis=1,
    )
    df_corr = df.dropna(subset=["ce_entropy_corrected"])
    logger.info(
        "Composition-corrected entropy computed for %d structures",
        df_corr["ce_entropy_corrected"].notna().sum(),
    )

    groups_corr = {
        class_names[c]: df_corr[df_corr["synth_class"] == c]["ce_entropy_corrected"].values
        for c in [0, 1, 2]
        if (df_corr["synth_class"] == c).sum() > 0
    }
    f_corr, p_corr = stats.f_oneway(*groups_corr.values())
    eta2_corr = _eta_squared(list(groups_corr.values()))

    results["composition_anova"] = pd.DataFrame([{
        "descriptor": "ce_entropy_corrected",
        "F_stat": f_corr,
        "p_value": p_corr,
        "eta_squared": eta2_corr,
        "n_total": len(df_corr),
    }])
    logger.info(
        "Composition-corrected ANOVA: F=%.4f, p=%.2e, η²=%.4f",
        f_corr, p_corr, eta2_corr,
    )

    # -----------------------------------------------------------------------
    # 7. Crystal-system subgroup analysis
    # -----------------------------------------------------------------------
    subgroup_rows = []
    if "crystal_system" in df.columns:
        for cs, sub_df in df.groupby("crystal_system"):
            if len(sub_df) < 30:
                continue
            sg = {
                class_names[c]: sub_df[sub_df["synth_class"] == c]["ce_entropy"].values
                for c in [0, 1, 2]
                if (sub_df["synth_class"] == c).sum() >= 5
            }
            if len(sg) < 2:
                continue
            try:
                f_sg, p_sg = stats.f_oneway(*sg.values())
                eta2_sg = _eta_squared(list(sg.values()))
                subgroup_rows.append({
                    "crystal_system": cs,
                    "n": len(sub_df),
                    "F_stat": f_sg,
                    "p_value": p_sg,
                    "eta_squared": eta2_sg,
                    **{f"n_{k}": len(v) for k, v in sg.items()},
                })
            except Exception as exc:
                logger.warning("Subgroup ANOVA failed for %s: %s", cs, exc)

    results["subgroup_anova"] = pd.DataFrame(subgroup_rows)
    if len(subgroup_rows):
        logger.info(
            "Subgroup ANOVA completed for %d crystal systems",
            len(subgroup_rows),
        )

    # Store corrected entropy back for visualization
    df_out_cols = [c for c in df.columns if c != "structure"]
    results["_df_with_corrected"] = df[df_out_cols].copy()

    return results


# ---------------------------------------------------------------------------
# Composition control helpers
# ---------------------------------------------------------------------------

def _compute_element_entropy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean ce_entropy for structures grouped by each element present.

    Returns a DataFrame with columns: element, mean_entropy, n_structures.
    """
    elem_entropy: dict[str, list[float]] = defaultdict(list)

    for _, row in df.iterrows():
        if pd.isna(row.get("ce_entropy")):
            continue
        elements = row.get("elements", [])
        if isinstance(elements, str):
            try:
                elements = json.loads(elements)
            except Exception:
                elements = []
        for el in elements:
            elem_entropy[el].append(row["ce_entropy"])

    rows = [
        {
            "element": el,
            "mean_entropy": np.mean(vals),
            "median_entropy": np.median(vals),
            "n_structures": len(vals),
        }
        for el, vals in sorted(elem_entropy.items())
    ]
    return pd.DataFrame(rows)


def _corrected_entropy(row, element_entropy: pd.DataFrame) -> float | None:
    """
    Subtract composition-expected entropy from raw ce_entropy.

    Expected entropy = average of per-element mean entropies in the formula.
    """
    elements = row.get("elements", [])
    if isinstance(elements, str):
        try:
            elements = json.loads(elements)
        except Exception:
            return None

    if not elements:
        return None

    el_map = element_entropy.set_index("element")["mean_entropy"].to_dict()
    baselines = [el_map.get(el) for el in elements if el in el_map]
    if not baselines:
        return None

    baseline = np.mean(baselines)
    raw = row.get("ce_entropy")
    if pd.isna(raw):
        return None
    return raw - baseline
