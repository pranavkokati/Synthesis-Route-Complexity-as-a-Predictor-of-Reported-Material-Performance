"""
Publication figures for:
"Information Entropy of Local Coordination Environments Predicts Synthesizability"

Figure 1 — Violin + box plots: CE entropy by synthesizability class (×2 panels)
Figure 2 — Three-curve ROC: naive vs. entropy-only vs. full model
Figure 3 — Periodic table heatmap: per-element mean CE entropy
Figure 4 — Composition-corrected entropy vs. energy_above_hull
             (scatter with hexbin density + LOWESS trend)
Figure 5 — Two-panel subgroup analysis:
             (A) crystal-system η² bar chart
             (B) Pettifor chemical-diversity quartile η² bar chart
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

CLASS_ORDER  = ["Stable", "Metastable", "Unstable"]
CLASS_COLORS = {"Stable": "#27ae60", "Metastable": "#e67e22", "Unstable": "#c0392b"}
MODEL_COLORS = {"naive": "#7f8c8d", "entropy": "#2980b9", "full": "#8e44ad"}
MODEL_LABELS = {
    "naive":   "Naive (nsites, nelements)",
    "entropy": "CE entropy descriptors",
    "full":    "Full model (entropy + naive)",
}

# Periodic table (row, col) — 1-indexed
_PT_POS: dict[str, tuple[int, int]] = {
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


def _style():
    """Apply publication-ready rcParams."""
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.labelsize":    12,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.05,
    })


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved → %s", path)
    return path


# ---------------------------------------------------------------------------
# Figure 1 — Violin plots
# ---------------------------------------------------------------------------

def figure1_violin(df: pd.DataFrame, out_path: str | Path) -> Path:
    """
    Two-panel violin/box plot of CE entropy and n_distinct_envs by class.
    """
    _style()
    try:
        import seaborn as sns
        has_sns = True
    except ImportError:
        has_sns = False

    df = df.copy()
    df["Class"] = df["synth_class"].map({0: "Stable", 1: "Metastable", 2: "Unstable"})
    df = df[df["Class"].notna()]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)

    panels = [
        ("ce_entropy", "Shannon Entropy of CE Distribution (nats)", "(A)"),
        ("n_distinct_envs", "Number of Distinct CE Types", "(B)"),
    ]

    for ax, (col, ylabel, tag) in zip(axes, panels):
        if col not in df.columns:
            ax.set_visible(False)
            continue

        if has_sns:
            sns.violinplot(
                data=df, x="Class", y=col, order=CLASS_ORDER,
                palette=CLASS_COLORS, inner=None, ax=ax,
                linewidth=1.0, cut=0,
            )
            # Overlay boxplot
            sns.boxplot(
                data=df, x="Class", y=col, order=CLASS_ORDER,
                width=0.15, ax=ax, showfliers=False,
                boxprops=dict(facecolor="white", zorder=2),
                medianprops=dict(color="black", linewidth=2),
                whiskerprops=dict(linewidth=1.2),
                capprops=dict(linewidth=1.2),
            )
        else:
            data_by_class = [
                df[df["Class"] == c][col].dropna().values
                for c in CLASS_ORDER
                if c in df["Class"].values
            ]
            positions = range(len(data_by_class))
            parts = ax.violinplot(data_by_class, positions=positions, showmedians=True)
            for pc, cls in zip(parts["bodies"], CLASS_ORDER):
                pc.set_facecolor(CLASS_COLORS[cls])
                pc.set_alpha(0.7)

        # Annotate sample sizes
        for i, cls in enumerate(CLASS_ORDER):
            n = (df["Class"] == cls).sum()
            ax.text(
                i, ax.get_ylim()[0] if ax.get_ylim() else 0,
                f"n={n:,}", ha="center", va="top", fontsize=8.5, color="#555555",
            )

        ax.set_xlabel("Synthesizability Class")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{tag} CE {col.replace('_', ' ').title()} by Synthesizability")

    fig.suptitle(
        "Coordination Environment Diversity vs. Synthesizability",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    return _save(fig, Path(out_path))


# ---------------------------------------------------------------------------
# Figure 2 — Three-curve ROC
# ---------------------------------------------------------------------------

def figure2_roc(roc_data: dict, out_path: str | Path) -> Path:
    """
    Three ROC curves: naive / entropy-only / full model, each with bootstrap CI.
    """
    _style()
    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random classifier (AUC = 0.500)")

    for model_name in ["naive", "entropy", "full"]:
        rd = roc_data.get(model_name)
        if not rd or "fpr" not in rd:
            continue
        fpr = np.array(rd["fpr"])
        tpr = np.array(rd["tpr"])
        auc = rd["auc"]
        ci_lo = rd.get("ci_low", auc)
        ci_hi = rd.get("ci_high", auc)
        r2mc = rd.get("mcfadden_r2", 0.0)
        color = MODEL_COLORS[model_name]
        label = MODEL_LABELS[model_name]

        ax.plot(
            fpr, tpr, color=color, lw=2.0,
            label=f"{label}\nAUC = {auc:.3f} [{ci_lo:.3f}–{ci_hi:.3f}]  R²_McF = {r2mc:.3f}",
        )
        ax.fill_between(fpr, tpr, alpha=0.08, color=color)

    ax.set_xlabel("False Positive Rate (1 − Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title(
        "ROC Curves: Predicting Synthesizability\nfrom CE Entropy Descriptors",
        fontsize=13,
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.02)
    ax.set_aspect("equal")

    return _save(fig, Path(out_path))


# ---------------------------------------------------------------------------
# Figure 3 — Periodic table heatmap
# ---------------------------------------------------------------------------

def figure3_periodic_table(element_entropy: pd.DataFrame, out_path: str | Path) -> Path:
    """
    Periodic table coloured by per-element mean CE entropy.
    """
    _style()
    if element_entropy is None or len(element_entropy) == 0:
        logger.warning("No element entropy data — skipping Figure 3")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return _save(fig, Path(out_path))

    el_map = element_entropy.set_index("element")["mean_entropy"].to_dict()
    all_vals = list(el_map.values())
    vmin, vmax = min(all_vals), max(all_vals)
    cmap = plt.get_cmap("plasma")
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(18, 8))
    ax.set_xlim(0, 19)
    ax.set_ylim(0, 10.5)
    ax.axis("off")

    cell_w, cell_h = 0.92, 0.88

    for el, (row, col) in _PT_POS.items():
        x = col - 0.5
        y = 10 - row + 0.1

        val = el_map.get(el)
        facecolor = cmap(norm(val)) if val is not None else "#e0e0e0"
        rect = plt.Rectangle(
            (x, y), cell_w, cell_h,
            facecolor=facecolor, edgecolor="white", linewidth=0.5,
        )
        ax.add_patch(rect)

        # Element symbol
        txt_color = "white" if val is not None and val > (vmin + vmax) * 0.4 else "#222222"
        ax.text(
            x + cell_w / 2, y + cell_h * 0.62, el,
            ha="center", va="center", fontsize=6.5,
            fontweight="bold", color=txt_color,
        )
        if val is not None:
            ax.text(
                x + cell_w / 2, y + cell_h * 0.25, f"{val:.2f}",
                ha="center", va="center", fontsize=5, color=txt_color,
            )

    # Lanthanide / Actinide row markers
    for row_y, label in [(8 - 10 + 0.1 + 0.44, "Ln"), (9 - 10 + 0.1 + 0.44, "An")]:
        ax.text(2.5, row_y, label, ha="center", va="center", fontsize=7, color="#555555")

    # Colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal",
                        fraction=0.025, pad=0.02, aspect=40)
    cbar.set_label("Mean CE Shannon Entropy (nats)", fontsize=11)

    ax.set_title(
        "Per-Element Mean Coordination Environment Entropy",
        fontsize=14, fontweight="bold", pad=8,
    )
    return _save(fig, Path(out_path))


# ---------------------------------------------------------------------------
# Figure 4 — Composition-corrected entropy vs. e_above_hull
# ---------------------------------------------------------------------------

def figure4_scatter(df: pd.DataFrame, out_path: str | Path) -> Path:
    """
    Scatter with hexbin density overlay and LOWESS trend line.
    Colour encodes synthesizability class.
    """
    _style()
    col = "ce_entropy_corrected" if "ce_entropy_corrected" in df.columns else "ce_entropy"
    plot_df = df[["energy_above_hull", col, "synth_class"]].dropna().copy()
    plot_df = plot_df[plot_df["energy_above_hull"] <= 1.5]

    if len(plot_df) == 0:
        logger.warning("No data for Figure 4")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return _save(fig, Path(out_path))

    fig, ax = plt.subplots(figsize=(9, 6.5))

    # Hexbin for density (full dataset)
    hb = ax.hexbin(
        plot_df["energy_above_hull"], plot_df[col],
        gridsize=60, cmap="Greys", mincnt=1, alpha=0.6, linewidths=0.2,
    )
    cb = fig.colorbar(hb, ax=ax, label="Count per hex bin", shrink=0.7)

    # Scatter overlay coloured by class (subsample to avoid overplotting)
    sample = plot_df.sample(min(5000, len(plot_df)), random_state=42)
    for cls in [0, 1, 2]:
        sub = sample[sample["synth_class"] == cls]
        label = CLASS_ORDER[cls]
        ax.scatter(
            sub["energy_above_hull"], sub[col],
            c=CLASS_COLORS[label], s=6, alpha=0.35,
            label=label, rasterized=True, linewidths=0,
        )

    # LOWESS trend line
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        spl = plot_df.sample(min(8000, len(plot_df)), random_state=1)
        smooth = lowess(spl[col].values, spl["energy_above_hull"].values,
                        frac=0.15, it=2, return_sorted=True)
        ax.plot(smooth[:, 0], smooth[:, 1], "r-", lw=2.5,
                label="LOWESS trend", zorder=5)
    except Exception:
        pass

    # Threshold lines
    ax.axvline(0.0, color="#555555", ls="--", lw=0.9, alpha=0.7)
    ax.axvline(0.1, color="#555555", ls="--", lw=0.9, alpha=0.7)
    ax.text(0.0, ax.get_ylim()[1] if ax.get_ylim() else 0,
            "stable\nthreshold", ha="left", va="top", fontsize=8, color="#555555")
    ax.text(0.1, ax.get_ylim()[1] if ax.get_ylim() else 0,
            "metastable\nthreshold", ha="left", va="top", fontsize=8, color="#555555")

    xlabel = "Energy Above Hull (eV/atom)"
    ylabel = ("Composition-Corrected CE Entropy (nats)"
              if col == "ce_entropy_corrected" else "CE Shannon Entropy (nats)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(
        "CE Entropy vs. Thermodynamic Stability\n"
        "(composition-corrected; density shading)",
        fontsize=13,
    )

    patches = [mpatches.Patch(color=CLASS_COLORS[c], label=c) for c in CLASS_ORDER]
    ax.legend(handles=patches + [plt.Line2D([0], [0], color="red", lw=2, label="LOWESS")],
              fontsize=9, loc="upper right")

    return _save(fig, Path(out_path))


# ---------------------------------------------------------------------------
# Figure 5 — Subgroup analysis (crystal system + Pettifor)
# ---------------------------------------------------------------------------

def figure5_subgroup(
    subgroup_anova: pd.DataFrame,
    pettifor_analysis: pd.DataFrame,
    out_path: str | Path,
) -> Path:
    """
    Two-panel figure:
      (A) Crystal-system ANOVA η² bar chart
      (B) Pettifor chemical-diversity quartile ANOVA η² bar chart
    """
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A — crystal system
    ax = axes[0]
    if subgroup_anova is not None and len(subgroup_anova) > 0:
        sg = subgroup_anova.sort_values("eta_squared", ascending=True)
        colors = ["#c0392b" if p < 0.05 else "#95a5a6" for p in sg["p_value"]]
        bars = ax.barh(
            range(len(sg)), sg["eta_squared"].values,
            color=colors, edgecolor="white", height=0.65,
        )
        ax.set_yticks(range(len(sg)))
        ax.set_yticklabels(sg["crystal_system"].tolist(), fontsize=10)

        for bar, (_, row) in zip(bars, sg.iterrows()):
            sig = "***" if row["p_value"] < 0.001 else ("**" if row["p_value"] < 0.01
                   else ("*" if row["p_value"] < 0.05 else "ns"))
            ax.text(
                bar.get_width() + 0.0005,
                bar.get_y() + bar.get_height() / 2,
                f"n={int(row['n_total']):,}  {sig}",
                va="center", fontsize=8.5,
            )
    else:
        ax.text(0.5, 0.5, "No subgroup data", ha="center", va="center")

    ax.set_xlabel("η² (Effect Size)")
    ax.set_title("(A) Crystal-System Subgroup ANOVA\nCE Entropy vs. Synthesizability")

    sig_p = mpatches.Patch(color="#c0392b", label="p < 0.05")
    ns_p  = mpatches.Patch(color="#95a5a6", label="p ≥ 0.05")
    ax.legend(handles=[sig_p, ns_p], fontsize=9)

    # Panel B — Pettifor quartiles
    ax2 = axes[1]
    if pettifor_analysis is not None and len(pettifor_analysis) > 0:
        pet = pettifor_analysis.sort_values("eta_squared", ascending=True)
        colors2 = ["#2980b9" if p < 0.05 else "#95a5a6" for p in pet["p_value"]]
        bars2 = ax2.barh(
            range(len(pet)), pet["eta_squared"].values,
            color=colors2, edgecolor="white", height=0.65,
        )
        ax2.set_yticks(range(len(pet)))
        ax2.set_yticklabels(pet["pettifor_bin"].tolist(), fontsize=10)

        for bar, (_, row) in zip(bars2, pet.iterrows()):
            sig = "***" if row["p_value"] < 0.001 else ("**" if row["p_value"] < 0.01
                   else ("*" if row["p_value"] < 0.05 else "ns"))
            ax2.text(
                bar.get_width() + 0.0005,
                bar.get_y() + bar.get_height() / 2,
                f"n={int(row['n_total']):,}  {sig}",
                va="center", fontsize=8.5,
            )
    else:
        ax2.text(0.5, 0.5, "No Pettifor data", ha="center", va="center")

    sig_p2 = mpatches.Patch(color="#2980b9", label="p < 0.05")
    ax2.legend(handles=[sig_p2, ns_p], fontsize=9)
    ax2.set_xlabel("η² (Effect Size)")
    ax2.set_title(
        "(B) Pettifor Chemical-Diversity Subgroup ANOVA\n"
        "CE Entropy vs. Synthesizability"
    )

    plt.tight_layout()
    return _save(fig, Path(out_path))


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def save_all(
    results: dict,
    df: pd.DataFrame,
    fig_dir: str | Path = "figures",
) -> list[Path]:
    """
    Produce all five figures.

    Parameters
    ----------
    results : output of analysis.run_analysis()
    df      : enriched DataFrame (results["_df_enriched"])
    fig_dir : output directory

    Returns list of saved paths.
    """
    fig_dir = Path(fig_dir)
    saved = []

    plot_df = results.get("_df_enriched", df)

    saved.append(figure1_violin(plot_df, fig_dir / "fig1_violin.png"))
    saved.append(figure2_roc(results.get("roc_data", {}), fig_dir / "fig2_roc.png"))
    saved.append(
        figure3_periodic_table(
            results.get("element_entropy"), fig_dir / "fig3_periodic_table.png"
        )
    )
    saved.append(figure4_scatter(plot_df, fig_dir / "fig4_scatter.png"))
    saved.append(
        figure5_subgroup(
            results.get("subgroup_anova"),
            results.get("pettifor_analysis"),
            fig_dir / "fig5_subgroups.png",
        )
    )
    return saved
