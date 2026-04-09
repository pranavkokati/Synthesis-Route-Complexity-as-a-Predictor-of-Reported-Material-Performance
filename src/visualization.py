"""
Figure generation for the synthesis complexity analysis.

Figures produced:
    Figure 1  — Distributions of the five complexity features
    Figure 2  — Pearson correlation heatmap (features + performance)
    Figure 3  — Scatter plots: performance vs. the two most predictive features
    Figure 4  — OLS regression coefficients with 95% CI
    Figure 5  — Subgroup regression R² and dominant coefficient by material family
    Figure S1 — Performance metric distribution by class
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.feature_extractor import FEATURE_COLS, FEATURE_LABELS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
PALETTE = "muted"
CMAP_CORR = "RdBu_r"
FIG_DPI = 150
FONT_SIZE = 11

plt.rcParams.update(
    {
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE + 1,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
        "legend.fontsize": FONT_SIZE - 1,
        "figure.dpi": FIG_DPI,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _save(fig: plt.Figure, path: Path, name: str) -> None:
    out = path / name
    fig.savefig(out, bbox_inches="tight", dpi=FIG_DPI)
    plt.close(fig)
    logger.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Figure 1 — Feature distributions
# ---------------------------------------------------------------------------
def plot_feature_distributions(df: pd.DataFrame, out_dir: Path) -> None:
    """Histogram + KDE for each of the five complexity features."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for idx, col in enumerate(FEATURE_COLS):
        ax = axes[idx]
        data = df[col].dropna()
        ax.hist(data, bins=50, color=sns.color_palette(PALETTE)[idx], alpha=0.75, density=True)

        # KDE overlay
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data, bw_method="scott")
            x_range = np.linspace(data.min(), data.max(), 300)
            ax.plot(x_range, kde(x_range), color="black", linewidth=1.5)
        except Exception:
            pass

        ax.set_xlabel(FEATURE_LABELS[col])
        ax.set_ylabel("Density")
        ax.set_title(FEATURE_LABELS[col])
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))

    # Sixth panel: composite complexity index
    ax = axes[5]
    ci_data = df["complexity_index"].dropna()
    ax.hist(ci_data, bins=50, color=sns.color_palette(PALETTE)[5], alpha=0.75, density=True)
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(ci_data)
        xr = np.linspace(ci_data.min(), ci_data.max(), 300)
        ax.plot(xr, kde(xr), color="black", linewidth=1.5)
    except Exception:
        pass
    ax.set_xlabel("Composite Complexity Index")
    ax.set_ylabel("Density")
    ax.set_title("Composite Complexity Index")

    fig.suptitle(
        "Distribution of Synthesis Complexity Features\n"
        f"(n = {len(df):,} solid-state synthesis recipes)",
        fontsize=FONT_SIZE + 2,
        y=1.02,
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
    """Annotated Pearson correlation heatmap with significance asterisks."""
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr_df.values, dtype=bool), k=1)

    annot = corr_df.round(2).astype(str)
    for i in range(len(corr_df)):
        for j in range(len(corr_df)):
            p = pval_df.iloc[i, j]
            r = corr_df.iloc[i, j]
            stars = ""
            if p < 0.001:
                stars = "***"
            elif p < 0.01:
                stars = "**"
            elif p < 0.05:
                stars = "*"
            annot.iloc[i, j] = f"{r:.2f}{stars}"

    sns.heatmap(
        corr_df,
        ax=ax,
        annot=annot,
        fmt="",
        cmap=CMAP_CORR,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        square=True,
        cbar_kws={"label": "Pearson r", "shrink": 0.8},
        mask=np.tril(np.ones_like(corr_df.values, dtype=bool), k=-1),
    )
    ax.set_title(
        "Pearson Correlation Matrix — Complexity Features & Performance\n"
        "(* p<0.05, ** p<0.01, *** p<0.001)",
        pad=10,
    )
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    _save(fig, out_dir, "fig2_correlation_heatmap.png")


# ---------------------------------------------------------------------------
# Figure 3 — Scatter plots vs. two most predictive features
# ---------------------------------------------------------------------------
def plot_performance_scatter(
    df: pd.DataFrame,
    top_features: list[str],
    out_dir: Path,
) -> None:
    """Scatter plots of normalised performance vs. the two most predictive features."""
    target = "perf_norm"
    sub = df[[target] + top_features].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = sns.color_palette(PALETTE, 2)

    for ax, feat, color in zip(axes, top_features[:2], colors):
        x = sub[feat].values
        y = sub[target].values

        ax.scatter(x, y, alpha=0.25, s=8, color=color, rasterized=True)

        # Regression line
        m = np.polyfit(x, y, 1)
        xfit = np.linspace(x.min(), x.max(), 200)
        ax.plot(xfit, np.polyval(m, xfit), color="black", linewidth=1.8, label="OLS fit")

        r = np.corrcoef(x, y)[0, 1]
        ax.set_xlabel(FEATURE_LABELS.get(feat, feat))
        ax.set_ylabel("Normalised Performance (z-score)")
        ax.set_title(f"r = {r:.3f}")
        ax.legend(fontsize=FONT_SIZE - 2)

    fig.suptitle(
        "Performance vs. Top-2 Complexity Predictors",
        fontsize=FONT_SIZE + 2,
    )
    fig.tight_layout()
    _save(fig, out_dir, "fig3_performance_scatter.png")


# ---------------------------------------------------------------------------
# Figure 4 — Regression coefficients with 95% CI
# ---------------------------------------------------------------------------
def plot_regression_coefficients(ols_result: dict, out_dir: Path) -> None:
    """Forest plot of OLS coefficients (excluding intercept) with 95% CI."""
    sdf = ols_result.get("summary_df", pd.DataFrame())
    if sdf.empty:
        return

    sub = sdf[sdf["feature"] != "const"].copy()
    sub = sub.sort_values("coefficient")
    colors = ["#d62728" if p < 0.05 else "#aec7e8" for p in sub["p_value"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    y_pos = np.arange(len(sub))

    ax.barh(
        y_pos,
        sub["coefficient"],
        xerr=[
            sub["coefficient"] - sub["ci_lower"],
            sub["ci_upper"] - sub["coefficient"],
        ],
        color=colors,
        alpha=0.85,
        height=0.55,
        capsize=4,
        error_kw={"elinewidth": 1.4, "ecolor": "black"},
    )
    ax.axvline(0, color="black", linewidth=0.9, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub["feature_label"])
    ax.set_xlabel("OLS Coefficient (HC3 robust SE)")
    ax.set_title(
        f"Regression Coefficients — {ols_result.get('label', '')}\n"
        f"n={ols_result.get('n', 0):,}  R²={ols_result.get('r_squared', np.nan):.4f}  "
        "(red = p<0.05)"
    )
    fig.tight_layout()
    _save(fig, out_dir, "fig4_regression_coefficients.png")


# ---------------------------------------------------------------------------
# Figure 5 — Subgroup R² by material family
# ---------------------------------------------------------------------------
def plot_subgroup_results(subgroup_results: list[dict], out_dir: Path) -> None:
    """Bar chart of R² per material family subgroup."""
    valid = [r for r in subgroup_results if np.isfinite(r.get("r_squared", np.nan)) and r.get("n", 0) >= 20]
    if not valid:
        logger.warning("No valid subgroup results to plot.")
        return

    valid.sort(key=lambda r: r.get("r_squared", 0), reverse=True)
    labels = [r["label"].replace("family=", "").replace("metric=", "") for r in valid]
    r2s = [r["r_squared"] for r in valid]
    ns = [r["n"] for r in valid]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(valid))))
    colors = sns.color_palette(PALETTE, len(valid))
    y_pos = np.arange(len(valid))
    bars = ax.barh(y_pos, r2s, color=colors, alpha=0.85, height=0.6)

    for bar, n in zip(bars, ns):
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"n={n:,}",
            va="center",
            fontsize=FONT_SIZE - 2,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("R²")
    ax.set_title("OLS R² by Material Family / Metric Subgroup")
    ax.set_xlim(0, max(r2s) * 1.25 + 0.01)
    fig.tight_layout()
    _save(fig, out_dir, "fig5_subgroup_r2.png")


# ---------------------------------------------------------------------------
# Figure S1 — Performance distribution by metric class
# ---------------------------------------------------------------------------
def plot_performance_by_class(df: pd.DataFrame, out_dir: Path) -> None:
    """Box plots of raw_value by metric class (log scale)."""
    sub = df[df["metric_name"].notna() & (df["raw_value"] > 0)].copy()
    if sub.empty:
        return

    sub["log_value"] = np.log10(sub["raw_value"])
    order = sub.groupby("metric_name")["log_value"].median().sort_values(ascending=False).index

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(
        data=sub,
        x="metric_name",
        y="log_value",
        hue="metric_name",
        order=order,
        palette=PALETTE,
        legend=False,
        ax=ax,
        flierprops={"marker": ".", "alpha": 0.3, "markersize": 3},
    )
    ax.set_xlabel("Performance Metric Class")
    ax.set_ylabel("log₁₀(value)")
    ax.set_title("Distribution of Extracted Performance Values by Metric Class")
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    _save(fig, out_dir, "figS1_performance_by_class.png")
