"""
Figure generation for the synthesis complexity analysis.

Figures produced:
    Figure 1  — Distributions of the five complexity features (full dataset)
    Figure 2  — Pearson correlation heatmap (features + all MP properties)
    Figure 3  — Scatter plots: each MP property vs. most predictive feature
    Figure 4  — OLS regression coefficients (one panel per MP property)
    Figure 5  — Subgroup R² by material family (band gap model)
    Figure S1 — MP property distributions in the merged dataset
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde

from src.analysis import MP_PROPERTIES
from src.feature_extractor import FEATURE_COLS, FEATURE_LABELS

logger = logging.getLogger(__name__)

PALETTE   = "muted"
CMAP_CORR = "RdBu_r"
FIG_DPI   = 150
FONT_SIZE = 11

plt.rcParams.update({
    "font.size":         FONT_SIZE,
    "axes.titlesize":    FONT_SIZE + 1,
    "axes.labelsize":    FONT_SIZE,
    "xtick.labelsize":   FONT_SIZE - 1,
    "ytick.labelsize":   FONT_SIZE - 1,
    "legend.fontsize":   FONT_SIZE - 1,
    "figure.dpi":        FIG_DPI,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


def _save(fig: plt.Figure, path: Path, name: str) -> None:
    out = path / name
    fig.savefig(out, bbox_inches="tight", dpi=FIG_DPI)
    plt.close(fig)
    logger.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Figure 1 — Feature distributions (full dataset)
# ---------------------------------------------------------------------------
def plot_feature_distributions(df: pd.DataFrame, out_dir: Path) -> None:
    """Histogram + KDE for each of the five complexity features."""
    cols   = sns.color_palette(PALETTE, 6)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for idx, col in enumerate(FEATURE_COLS):
        ax   = axes[idx]
        data = df[col].dropna()
        ax.hist(data, bins=60, color=cols[idx], alpha=0.7, density=True)
        try:
            kde   = gaussian_kde(data, bw_method="scott")
            xr    = np.linspace(data.min(), data.max(), 300)
            ax.plot(xr, kde(xr), color="black", lw=1.6)
        except Exception:
            pass
        ax.set_xlabel(FEATURE_LABELS[col])
        ax.set_ylabel("Density")
        ax.set_title(FEATURE_LABELS[col])
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))

    # Panel 6 — composite index
    ax   = axes[5]
    data = df["complexity_index"].dropna()
    ax.hist(data, bins=60, color=cols[5], alpha=0.7, density=True)
    try:
        kde = gaussian_kde(data)
        xr  = np.linspace(data.min(), data.max(), 300)
        ax.plot(xr, kde(xr), color="black", lw=1.6)
    except Exception:
        pass
    ax.set_xlabel("Composite Complexity Index")
    ax.set_ylabel("Density")
    ax.set_title("Composite Complexity Index")

    fig.suptitle(
        f"Synthesis Complexity Features — Kononova Dataset (n = {len(df):,} recipes)",
        fontsize=FONT_SIZE + 2, y=1.02,
    )
    fig.tight_layout()
    _save(fig, out_dir, "fig1_feature_distributions.png")


# ---------------------------------------------------------------------------
# Figure 2 — Correlation heatmap
# ---------------------------------------------------------------------------
def plot_correlation_heatmap(
    corr_df: pd.DataFrame,
    pval_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Annotated Pearson correlation heatmap."""
    fig, ax = plt.subplots(figsize=(11, 9))

    annot = corr_df.copy().astype(str)
    for i in range(len(corr_df)):
        for j in range(len(corr_df)):
            r = corr_df.iloc[i, j]
            p = pval_df.iloc[i, j]
            stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            annot.iloc[i, j] = f"{r:.2f}{stars}"

    mask = np.tril(np.ones_like(corr_df.values, dtype=bool), k=-1)
    sns.heatmap(
        corr_df, ax=ax, annot=annot, fmt="", cmap=CMAP_CORR,
        vmin=-1, vmax=1, linewidths=0.4, square=True,
        cbar_kws={"label": "Pearson r", "shrink": 0.8},
        mask=mask,
    )
    ax.set_title(
        "Pearson Correlation — Complexity Features & Materials Project Properties\n"
        "(* p<0.05  ** p<0.01  *** p<0.001)",
        pad=10,
    )
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    _save(fig, out_dir, "fig2_correlation_heatmap.png")


# ---------------------------------------------------------------------------
# Figure 3 — Scatter plots: MP properties vs. top feature
# ---------------------------------------------------------------------------
def plot_property_scatter(
    df: pd.DataFrame,
    top_feature: str,
    out_dir: Path,
) -> None:
    """2×2 scatter grid: each MP property vs. the most predictive feature."""
    props = [p for p in MP_PROPERTIES if p in df.columns]
    n_props = len(props)
    ncols = 2
    nrows = (n_props + 1) // 2
    colors = sns.color_palette(PALETTE, n_props)

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows))
    axes = np.array(axes).flatten()

    for i, (prop, color) in enumerate(zip(props, colors)):
        ax  = axes[i]
        sub = df[[top_feature, prop]].dropna()
        x   = sub[top_feature].values
        y   = sub[prop].values

        ax.scatter(x, y, alpha=0.3, s=10, color=color, rasterized=True)
        m = np.polyfit(x, y, 1)
        xfit = np.linspace(x.min(), x.max(), 200)
        ax.plot(xfit, np.polyval(m, xfit), color="black", lw=1.8, label="OLS fit")

        r = np.corrcoef(x, y)[0, 1]
        ax.set_xlabel(FEATURE_LABELS.get(top_feature, top_feature))
        ax.set_ylabel(MP_PROPERTIES[prop])
        ax.set_title(f"{MP_PROPERTIES[prop]}\nr = {r:.3f}  (n={len(sub):,})")
        ax.legend(fontsize=FONT_SIZE - 2)

    # Hide unused panels
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Materials Project Properties vs. {FEATURE_LABELS.get(top_feature, top_feature)}",
        fontsize=FONT_SIZE + 2,
    )
    fig.tight_layout()
    _save(fig, out_dir, "fig3_property_scatter.png")


# ---------------------------------------------------------------------------
# Figure 4 — Regression coefficients for all MP properties
# ---------------------------------------------------------------------------
def plot_all_regression_coefficients(
    ols_results: dict[str, dict],
    out_dir: Path,
) -> None:
    """One panel per MP property showing coefficients + 95% CI."""
    valid = {k: v for k, v in ols_results.items() if v.get("model") is not None}
    if not valid:
        logger.warning("No valid OLS results to plot.")
        return

    n_panels = len(valid)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, (prop, res) in zip(axes, valid.items()):
        sdf   = res["summary_df"]
        sub   = sdf[sdf["feature"] != "const"].copy().sort_values("coefficient")
        ypos  = np.arange(len(sub))
        colors = ["#d62728" if p < 0.05 else "#aec7e8" for p in sub["p_value"]]

        ax.barh(
            ypos, sub["coefficient"],
            xerr=[
                sub["coefficient"] - sub["ci_lower"],
                sub["ci_upper"]    - sub["coefficient"],
            ],
            color=colors, alpha=0.85, height=0.55, capsize=4,
            error_kw={"elinewidth": 1.4, "ecolor": "black"},
        )
        ax.axvline(0, color="black", lw=0.9, linestyle="--")
        ax.set_yticks(ypos)
        ax.set_yticklabels(sub["feature_label"])
        ax.set_xlabel("Standardised Coefficient (HC3)")
        ax.set_title(
            f"{MP_PROPERTIES[prop]}\n"
            f"n={res['n']:,}  R²={res['r_squared']:.3f}\n"
            "(red = p<0.05)"
        )

    fig.suptitle("OLS Regression Coefficients — Complexity Features → MP Properties",
                 fontsize=FONT_SIZE + 2)
    fig.tight_layout()
    _save(fig, out_dir, "fig4_regression_coefficients.png")


# ---------------------------------------------------------------------------
# Figure 5 — Subgroup R² by material family (FDR-corrected, tiered)
# ---------------------------------------------------------------------------
def plot_subgroup_fdr(fdr_df: pd.DataFrame, target_label: str, out_dir: Path) -> None:
    """
    Bar chart of subgroup R² with:
    - Colour encoding: reliability tier (robust/moderate/exploratory)
    - Error bars: bootstrap 95% CI where available
    - Asterisks: FDR significance
    - Sample size labels
    """
    sub = fdr_df[fdr_df["sig_fdr"]].sort_values("r_squared", ascending=False).copy()
    if sub.empty:
        logger.warning("No FDR-significant subgroups to plot.")
        return

    tier_colors = {"robust": "#2ca02c", "moderate": "#1f77b4", "exploratory": "#d62728"}
    colors = [tier_colors.get(t, "#7f7f7f") for t in sub["reliability_tier"]]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(sub))))
    ypos = np.arange(len(sub))
    ax.barh(ypos, sub["r_squared"], color=colors, alpha=0.85, height=0.6)

    for i, (_, row) in enumerate(sub.iterrows()):
        ax.text(
            row["r_squared"] + 0.005, i,
            f"n={int(row['n'])}",
            va="center", fontsize=FONT_SIZE - 2,
        )
        ax.text(
            -0.008, i, row["reliability_tier"][0].upper(),
            va="center", ha="right", fontsize=FONT_SIZE - 2,
            color=tier_colors.get(row["reliability_tier"], "black"),
        )

    ax.set_yticks(ypos)
    ax.set_yticklabels(sub["family"])
    ax.set_xlabel("R² (FDR-significant subgroups only)")
    ax.set_title(
        f"Subgroup OLS R² — Target: {target_label}\n"
        "Colour: G=robust (n≥200)  B=moderate (n≥50)  R=exploratory (n<50)"
    )
    ax.set_xlim(-0.05, max(sub["r_squared"]) * 1.3 + 0.02)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    fig.tight_layout()
    _save(fig, out_dir, "fig5_subgroup_r2_fdr.png")


def plot_subgroup_results(
    subgroup_results: list[dict],
    target_label: str,
    out_dir: Path,
) -> None:
    """Bar chart of R² by material family."""
    valid = [
        r for r in subgroup_results
        if np.isfinite(r.get("r_squared", np.nan)) and r.get("n", 0) >= 15
    ]
    if not valid:
        logger.warning("No valid subgroup results to plot.")
        return

    valid.sort(key=lambda r: r.get("r_squared", 0), reverse=True)
    labels = [r["label"].replace("family=", "") for r in valid]
    r2s    = [r["r_squared"] for r in valid]
    ns     = [r["n"] for r in valid]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(valid))))
    colors  = sns.color_palette(PALETTE, len(valid))
    ypos    = np.arange(len(valid))
    bars    = ax.barh(ypos, r2s, color=colors, alpha=0.85, height=0.6)

    for bar, n in zip(bars, ns):
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"n={n:,}", va="center", fontsize=FONT_SIZE - 2,
        )

    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("R²")
    ax.set_title(
        f"OLS R² by Material Family — Target: {target_label}\n"
        "(complexity features as predictors)"
    )
    ax.set_xlim(0, max(r2s) * 1.3 + 0.01)
    fig.tight_layout()
    _save(fig, out_dir, "fig5_subgroup_r2.png")


# ---------------------------------------------------------------------------
# Figure S1 — MP property distributions in the merged dataset
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Figure S2 — Confound check: precursor_diversity β before/after control
# ---------------------------------------------------------------------------
def plot_confound_check(confound_df: pd.DataFrame, out_dir: Path) -> None:
    """
    Paired bar chart showing precursor_diversity β with and without
    target_n_elements as a covariate.
    """
    sub = confound_df.dropna(subset=["beta_no_ctrl", "beta_with_ctrl"])
    if sub.empty:
        return

    n_props = len(sub)
    x = np.arange(n_props)
    width = 0.35
    labels = sub["property_label"].tolist()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, sub["beta_no_ctrl"], width, label="Without composition control",
           color="#4c72b0", alpha=0.85)
    ax.bar(x + width/2, sub["beta_with_ctrl"], width, label="Controlling for target element count",
           color="#dd8452", alpha=0.85)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Standardised β (precursor diversity)")
    ax.set_title(
        "Precursor Diversity: Independent Synthesis Signal vs. Composition Proxy\n"
        "β change after controlling for target element count"
    )
    ax.legend()

    # Annotate % change
    for i, (_, row) in enumerate(sub.iterrows()):
        pct = row["pct_beta_change"]
        if np.isfinite(pct):
            ax.text(i, max(abs(row["beta_no_ctrl"]), abs(row["beta_with_ctrl"])) + 0.02,
                    f"{pct:+.0f}%", ha="center", fontsize=FONT_SIZE - 2, color="black")

    fig.tight_layout()
    _save(fig, out_dir, "figS2_confound_check.png")


# ---------------------------------------------------------------------------
# Figure S3 — Semi-partial R² decomposition
# ---------------------------------------------------------------------------
def plot_semi_partial_r2(partial_df: pd.DataFrame, target_label: str, out_dir: Path) -> None:
    """Horizontal bar chart of semi-partial R² per feature for a given target."""
    sub = partial_df.copy()
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = sns.color_palette(PALETTE, len(sub))
    ypos = np.arange(len(sub))
    ax.barh(ypos, sub["semi_partial_r2"], color=colors, alpha=0.85, height=0.6)
    ax.set_yticks(ypos)
    ax.set_yticklabels(sub["feature_label"])
    ax.set_xlabel("Semi-partial R² (unique variance contribution)")
    ax.set_title(f"Feature Contribution to {target_label}\n(controlling for all other features)")
    fig.tight_layout()
    _save(fig, out_dir, "figS3_semi_partial_r2.png")


def plot_mp_property_distributions(df: pd.DataFrame, out_dir: Path) -> None:
    """Histograms of the Materials Project properties present in the merged dataset."""
    props  = [p for p in MP_PROPERTIES if p in df.columns and df[p].notna().sum() > 5]
    colors = sns.color_palette(PALETTE, len(props))

    fig, axes = plt.subplots(1, len(props), figsize=(5 * len(props), 4))
    if len(props) == 1:
        axes = [axes]

    for ax, prop, color in zip(axes, props, colors):
        data = df[prop].dropna()
        ax.hist(data, bins=50, color=color, alpha=0.75, density=True)
        try:
            kde = gaussian_kde(data)
            xr  = np.linspace(data.min(), data.max(), 300)
            ax.plot(xr, kde(xr), color="black", lw=1.5)
        except Exception:
            pass
        ax.set_xlabel(MP_PROPERTIES[prop])
        ax.set_ylabel("Density")
        ax.set_title(f"n = {len(data):,}")

    fig.suptitle(
        "Materials Project Property Distributions (merged synthesis dataset)",
        fontsize=FONT_SIZE + 1,
    )
    fig.tight_layout()
    _save(fig, out_dir, "figS1_mp_property_distributions.png")
