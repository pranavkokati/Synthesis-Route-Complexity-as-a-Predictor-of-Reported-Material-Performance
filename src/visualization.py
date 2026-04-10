"""
Figures for the Information Entropy of Local Coordination Environments paper.

Figure 1 — Violin plots: CE entropy distribution by synthesizability class
Figure 2 — ROC curve for binary synthesizability prediction
Figure 3 — Periodic table heatmap of per-element mean CE entropy
Figure 4 — Composition-corrected entropy vs energy_above_hull (scatter + lowess)
Figure 5 — Crystal-system subgroup ANOVA η² bar chart
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Consistent colour palette
PALETTE = {
    "Stable":      "#2ecc71",
    "Metastable":  "#f39c12",
    "Unstable":    "#e74c3c",
}
CLASS_ORDER = ["Stable", "Metastable", "Unstable"]
CLASS_MAP   = {0: "Stable", 1: "Metastable", 2: "Unstable"}

# Periodic table layout (row, col) for elements 1–86
_PT_LAYOUT = {
    "H": (1, 1), "He": (1, 18),
    "Li": (2, 1), "Be": (2, 2),
    "B": (2, 13), "C": (2, 14), "N": (2, 15), "O": (2, 16), "F": (2, 17), "Ne": (2, 18),
    "Na": (3, 1), "Mg": (3, 2),
    "Al": (3, 13), "Si": (3, 14), "P": (3, 15), "S": (3, 16), "Cl": (3, 17), "Ar": (3, 18),
    "K": (4, 1), "Ca": (4, 2),
    "Sc": (4, 3), "Ti": (4, 4), "V": (4, 5), "Cr": (4, 6), "Mn": (4, 7),
    "Fe": (4, 8), "Co": (4, 9), "Ni": (4, 10), "Cu": (4, 11), "Zn": (4, 12),
    "Ga": (4, 13), "Ge": (4, 14), "As": (4, 15), "Se": (4, 16), "Br": (4, 17), "Kr": (4, 18),
    "Rb": (5, 1), "Sr": (5, 2),
    "Y": (5, 3), "Zr": (5, 4), "Nb": (5, 5), "Mo": (5, 6), "Tc": (5, 7),
    "Ru": (5, 8), "Rh": (5, 9), "Pd": (5, 10), "Ag": (5, 11), "Cd": (5, 12),
    "In": (5, 13), "Sn": (5, 14), "Sb": (5, 15), "Te": (5, 16), "I": (5, 17), "Xe": (5, 18),
    "Cs": (6, 1), "Ba": (6, 2),
    "La": (8, 3), "Ce": (8, 4), "Pr": (8, 5), "Nd": (8, 6), "Pm": (8, 7),
    "Sm": (8, 8), "Eu": (8, 9), "Gd": (8, 10), "Tb": (8, 11), "Dy": (8, 12),
    "Ho": (8, 13), "Er": (8, 14), "Tm": (8, 15), "Yb": (8, 16), "Lu": (8, 17),
    "Hf": (6, 4), "Ta": (6, 5), "W": (6, 6), "Re": (6, 7),
    "Os": (6, 8), "Ir": (6, 9), "Pt": (6, 10), "Au": (6, 11), "Hg": (6, 12),
    "Tl": (6, 13), "Pb": (6, 14), "Bi": (6, 15), "Po": (6, 16), "At": (6, 17), "Rn": (6, 18),
    "Fr": (7, 1), "Ra": (7, 2),
    "Ac": (9, 3), "Th": (9, 4), "Pa": (9, 5), "U": (9, 6), "Np": (9, 7),
    "Pu": (9, 8), "Am": (9, 9), "Cm": (9, 10), "Bk": (9, 11), "Cf": (9, 12),
    "Es": (9, 13), "Fm": (9, 14), "Md": (9, 15), "No": (9, 16), "Lr": (9, 17),
}


def _set_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def figure1_violin(df: pd.DataFrame, out_path: str | Path) -> Path:
    """
    Fig 1: Violin plots of CE entropy by synthesizability class.
    Includes strip jitter overlay and class-mean markers.
    """
    _set_style()
    df = df.copy()
    df["Class"] = df["synth_class"].map(CLASS_MAP)
    df = df[df["Class"].notna()]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: raw entropy
    ax = axes[0]
    sns.violinplot(
        data=df, x="Class", y="ce_entropy", order=CLASS_ORDER,
        palette=PALETTE, inner=None, ax=ax, linewidth=1.2,
    )
    sns.stripplot(
        data=df.sample(min(3000, len(df)), random_state=42),
        x="Class", y="ce_entropy", order=CLASS_ORDER,
        color="black", alpha=0.15, size=1.5, ax=ax, jitter=True,
    )
    # Add mean markers
    for i, cls in enumerate(CLASS_ORDER):
        mean_val = df[df["Class"] == cls]["ce_entropy"].mean()
        ax.plot(i, mean_val, "w^", markersize=8, markeredgecolor="black", zorder=5)
    ax.set_xlabel("Synthesizability Class")
    ax.set_ylabel("Shannon Entropy of CE Distribution (nats)")
    ax.set_title("(A) CE Entropy by Synthesizability Class")

    # Right: n_distinct_envs
    ax2 = axes[1]
    if "n_distinct_envs" in df.columns:
        sns.violinplot(
            data=df, x="Class", y="n_distinct_envs", order=CLASS_ORDER,
            palette=PALETTE, inner=None, ax=ax2, linewidth=1.2,
        )
        ax2.set_xlabel("Synthesizability Class")
        ax2.set_ylabel("No. Distinct CE Types")
        ax2.set_title("(B) Distinct Environments by Synthesizability Class")

    fig.suptitle("Coordination Environment Diversity vs. Synthesizability", fontsize=14)
    plt.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved Figure 1 → %s", out)
    return out


def figure2_roc(roc_data: dict, out_path: str | Path) -> Path:
    """
    Fig 2: ROC curve for binary synthesizability classifier.
    """
    _set_style()
    if not roc_data or "fpr" not in roc_data:
        logger.warning("No ROC data available, skipping Figure 2")
        return Path(out_path)

    fpr = np.array(roc_data["fpr"])
    tpr = np.array(roc_data["tpr"])
    auc = roc_data["auc"]
    ci_lo = roc_data.get("ci_low", auc)
    ci_hi = roc_data.get("ci_high", auc)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#2980b9", lw=2,
            label=f"Logistic Regression\nAUC = {auc:.3f} [{ci_lo:.3f}–{ci_hi:.3f}]")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#2980b9")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve: Predicting Synthesizability from CE Entropy")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved Figure 2 → %s", out)
    return out


def figure3_periodic_table(element_entropy: pd.DataFrame, out_path: str | Path) -> Path:
    """
    Fig 3: Periodic table heatmap — per-element mean CE entropy.
    """
    _set_style()
    if element_entropy is None or len(element_entropy) == 0:
        logger.warning("No element entropy data, skipping Figure 3")
        return Path(out_path)

    el_map = element_entropy.set_index("element")["mean_entropy"].to_dict()

    # Build grid (9 rows x 18 cols)
    grid = np.full((9, 18), np.nan)
    for el, (row, col) in _PT_LAYOUT.items():
        if el in el_map:
            grid[row - 1, col - 1] = el_map[el]

    fig, ax = plt.subplots(figsize=(16, 7))
    im = ax.imshow(grid, cmap="viridis", aspect="auto",
                   vmin=np.nanmin(grid), vmax=np.nanmax(grid))

    # Label each element cell
    for el, (row, col) in _PT_LAYOUT.items():
        r, c = row - 1, col - 1
        val = el_map.get(el)
        if val is not None:
            ax.text(c, r, f"{el}\n{val:.2f}", ha="center", va="center",
                    fontsize=5.5, color="white" if val < np.nanmean(grid) + 0.5 else "black")
        else:
            ax.text(c, r, el, ha="center", va="center", fontsize=6, color="#cccccc")

    plt.colorbar(im, ax=ax, label="Mean CE Shannon Entropy (nats)", shrink=0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Per-Element Mean Coordination Environment Entropy", fontsize=14)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved Figure 3 → %s", out)
    return out


def figure4_scatter(df: pd.DataFrame, out_path: str | Path) -> Path:
    """
    Fig 4: Composition-corrected CE entropy vs energy_above_hull.
    Scatter (colour = synth_class) + LOWESS trend line.
    """
    _set_style()
    col = "ce_entropy_corrected" if "ce_entropy_corrected" in df.columns else "ce_entropy"
    plot_df = df[["energy_above_hull", col, "synth_class"]].dropna()
    # Cap e_hull at 2 eV for readability
    plot_df = plot_df[plot_df["energy_above_hull"] <= 2.0]

    if len(plot_df) == 0:
        logger.warning("No data for Figure 4")
        return Path(out_path)

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = plot_df["synth_class"].map({0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"})
    ax.scatter(
        plot_df["energy_above_hull"], plot_df[col],
        c=colors, alpha=0.2, s=5, rasterized=True,
    )

    # LOWESS trend
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        sample = plot_df.sample(min(5000, len(plot_df)), random_state=42)
        smooth = lowess(
            sample[col].values,
            sample["energy_above_hull"].values,
            frac=0.2, it=2,
        )
        smooth = smooth[smooth[:, 0].argsort()]
        ax.plot(smooth[:, 0], smooth[:, 1], "k-", lw=2.5, label="LOWESS trend")
    except ImportError:
        logger.warning("statsmodels not available, skipping LOWESS line")

    patches = [
        mpatches.Patch(color="#2ecc71", label="Stable (class 0)"),
        mpatches.Patch(color="#f39c12", label="Metastable (class 1)"),
        mpatches.Patch(color="#e74c3c", label="Unstable (class 2)"),
    ]
    ax.legend(handles=patches, fontsize=9)
    ax.axvline(0.0, color="grey", ls="--", lw=0.8)
    ax.axvline(0.1, color="grey", ls="--", lw=0.8)

    label = "Composition-Corrected CE Entropy" if col == "ce_entropy_corrected" else "CE Shannon Entropy (nats)"
    ax.set_xlabel("Energy Above Hull (eV/atom)")
    ax.set_ylabel(label)
    ax.set_title("CE Entropy vs. Thermodynamic Stability")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved Figure 4 → %s", out)
    return out


def figure5_subgroup(subgroup_anova: pd.DataFrame, out_path: str | Path) -> Path:
    """
    Fig 5: Crystal-system subgroup η² bar chart.
    """
    _set_style()
    if subgroup_anova is None or len(subgroup_anova) == 0:
        logger.warning("No subgroup data, skipping Figure 5")
        return Path(out_path)

    sg = subgroup_anova.sort_values("eta_squared", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4, 0.6 * len(sg))))

    colors = ["#c0392b" if p < 0.05 else "#7f8c8d" for p in sg["p_value"]]
    bars = ax.barh(sg["crystal_system"], sg["eta_squared"], color=colors, edgecolor="white")

    # Annotate with n and p-value
    for bar, (_, row) in zip(bars, sg.iterrows()):
        ax.text(
            bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
            f"n={row['n']:,}  p={row['p_value']:.2e}",
            va="center", fontsize=9,
        )

    ax.set_xlabel("η² (Effect Size)")
    ax.set_title("Subgroup ANOVA: CE Entropy vs. Synthesizability\nby Crystal System")

    sig_patch = mpatches.Patch(color="#c0392b", label="p < 0.05")
    ns_patch  = mpatches.Patch(color="#7f8c8d", label="p ≥ 0.05")
    ax.legend(handles=[sig_patch, ns_patch], fontsize=9)

    plt.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved Figure 5 → %s", out)
    return out


def save_all(results: dict, df: pd.DataFrame, fig_dir: str | Path = "figures") -> list[Path]:
    """
    Convenience wrapper: produce all five figures.

    Parameters
    ----------
    results : output from analysis.run_analysis()
    df      : merged DataFrame (may include ce_entropy_corrected column)
    fig_dir : output directory

    Returns list of saved paths.
    """
    fig_dir = Path(fig_dir)
    saved = []

    saved.append(figure1_violin(df, fig_dir / "fig1_violin.png"))
    saved.append(figure2_roc(results.get("roc_data", {}), fig_dir / "fig2_roc.png"))
    saved.append(
        figure3_periodic_table(results.get("element_entropy"), fig_dir / "fig3_periodic.png")
    )

    plot_df = results.get("_df_with_corrected", df)
    saved.append(figure4_scatter(plot_df, fig_dir / "fig4_scatter.png"))
    saved.append(figure5_subgroup(results.get("subgroup_anova"), fig_dir / "fig5_subgroup.png"))

    return saved
