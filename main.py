"""
main.py — Synthesis Route Complexity as a Predictor of Reported Material Performance
=====================================================================================

Pipeline:
    1. Load the Kononova et al. solid-state synthesis dataset (31 K recipes)
    2. Extract five operationally defined synthesis complexity features
    3. Query the Materials Project API for real DFT-computed properties
       (band_gap, formation_energy_per_atom, energy_above_hull, density)
    4. Merge synthesis complexity features with MP properties
    5. Multivariate OLS regression for each MP property
    6. Pearson correlation matrix + VIF diagnostics
    7. Subgroup OLS by material family
    8. Publication-quality figures

Usage:
    python main.py [--data PATH] [--output DIR] [--mp-key KEY]

Environment variables:
    MP_API_KEY — Materials Project API key
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data",   default="data/solid_state_synthesis.json")
    p.add_argument("--output", default="output")
    p.add_argument("--mp-key", default=None, help="Materials Project API key (overrides env)")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(args.log_level)

    mp_key = args.mp_key or os.environ.get("MP_API_KEY") or os.environ.get("MAPI_KEY")
    if not mp_key:
        logger.error(
            "No Materials Project API key found. "
            "Set MP_API_KEY or pass --mp-key."
        )
        sys.exit(1)

    out_root = Path(args.output)
    fig_dir  = out_root / "figures"
    res_dir  = out_root / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 — Load dataset
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 1  Loading dataset")
    logger.info("=" * 60)

    from src.data_loader import load_dataset, parse_reactions
    reactions = load_dataset(args.data)
    df_raw    = parse_reactions(reactions)

    # ------------------------------------------------------------------
    # Step 2 — Complexity features
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 2  Computing complexity features")
    logger.info("=" * 60)

    from src.feature_extractor import compute_features, get_feature_summary, FEATURE_COLS
    df       = compute_features(df_raw)
    summary  = get_feature_summary(df)
    logger.info("\n%s", summary.to_string())
    summary.to_csv(res_dir / "feature_summary.csv")

    # ------------------------------------------------------------------
    # Step 3 — Materials Project query
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 3  Querying Materials Project API")
    logger.info("=" * 60)

    from src.materials_project import fetch_mp_properties
    unique_formulas = df["target_formula"].dropna().unique().tolist()
    mp_df = fetch_mp_properties(
        formulas=unique_formulas,
        api_key=mp_key,
        cache_path="data/mp_properties_cache.json",
    )
    mp_df.to_csv(res_dir / "mp_properties.csv", index=False)
    logger.info("MP properties retrieved for %d unique formulas", len(mp_df))

    # ------------------------------------------------------------------
    # Step 4 — Merge
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 4  Merging synthesis features with MP properties")
    logger.info("=" * 60)

    df_merged = df.merge(mp_df, left_on="target_formula", right_on="formula", how="inner")
    logger.info(
        "Merged dataset: %d records (%.1f%% of full dataset)",
        len(df_merged), 100.0 * len(df_merged) / len(df),
    )

    from src.analysis import MP_PROPERTIES
    for prop, label in MP_PROPERTIES.items():
        n = df_merged[prop].notna().sum() if prop in df_merged.columns else 0
        logger.info("  %-35s : %d records", label, n)

    df_merged.to_csv(res_dir / "merged_dataset.csv", index=False)

    # ------------------------------------------------------------------
    # Step 5 — OLS regressions
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 5  Multivariate OLS — one model per MP property")
    logger.info("=" * 60)

    from src.analysis import (
        run_all_property_regressions,
        compute_correlation_matrix,
        compute_vif,
        subgroup_regression_by_family,
        feature_importance_ranking,
    )

    ols_results = run_all_property_regressions(df_merged)

    all_summaries = []
    for prop, res in ols_results.items():
        if res.get("model") is not None:
            logger.info("\n%s", res["model"].summary())
            res["summary_df"].to_csv(
                res_dir / f"ols_{prop}.csv", index=False
            )
            all_summaries.append(res["summary_df"])

    if all_summaries:
        pd.concat(all_summaries).to_csv(res_dir / "ols_all_properties.csv", index=False)

    # ------------------------------------------------------------------
    # Step 6 — Correlation matrix + VIF
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 6  Correlation matrix and VIF")
    logger.info("=" * 60)

    corr_df, pval_df = compute_correlation_matrix(df_merged)
    corr_df.to_csv(res_dir / "correlation_matrix.csv")
    pval_df.to_csv(res_dir / "pvalue_matrix.csv")
    logger.info("\n%s", corr_df.to_string())

    vif_df = compute_vif(df_merged)
    vif_df.to_csv(res_dir / "vif.csv", index=False)
    logger.info("\nVIF:\n%s", vif_df.to_string())

    # Feature importance across all models (by mean |β|)
    importance_rows = []
    for prop, res in ols_results.items():
        if res.get("model") is not None:
            imp = feature_importance_ranking(res)
            imp["mp_property"] = prop
            importance_rows.append(imp)
    if importance_rows:
        imp_all = pd.concat(importance_rows)
        imp_all.to_csv(res_dir / "feature_importance_all.csv", index=False)

        # Top feature per property
        logger.info("\nTop predictor per MP property:")
        for prop, res in ols_results.items():
            if res.get("model") is not None:
                imp = feature_importance_ranking(res)
                if not imp.empty:
                    top = imp.iloc[0]
                    logger.info(
                        "  %-35s  top=%s  β=%+.4f  p=%.4f",
                        MP_PROPERTIES[prop],
                        top["feature_label"],
                        top["coefficient"],
                        top["p_value"],
                    )

    # Determine the single most predictive feature (highest mean |β| across models)
    if importance_rows:
        imp_agg = (
            imp_all[imp_all["feature"] != "const"]
            .groupby("feature")["abs_coef"]
            .mean()
            .sort_values(ascending=False)
        )
        top_feature = imp_agg.index[0]
        logger.info("Most predictive feature overall: %s", top_feature)
    else:
        top_feature = FEATURE_COLS[0]

    # ------------------------------------------------------------------
    # Step 7 — Subgroup regressions
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 7  Subgroup regressions by material family")
    logger.info("=" * 60)

    # Use band_gap as the primary subgroup target (most data); fall back to best covered
    primary_target = next(
        (p for p in ["band_gap", "formation_energy_per_atom", "energy_above_hull", "density"]
         if p in df_merged.columns and df_merged[p].notna().sum() >= 30),
        None,
    )

    sub_family = []
    if primary_target:
        sub_family = subgroup_regression_by_family(df_merged, target=primary_target)
        sg_rows = [{"label": r["label"], "n": r["n"], "r_squared": r["r_squared"],
                    "adj_r_squared": r.get("adj_r_squared", np.nan),
                    "target": r["target"]} for r in sub_family]
        pd.DataFrame(sg_rows).to_csv(res_dir / "subgroup_results.csv", index=False)

    # ------------------------------------------------------------------
    # Step 8 — Figures
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 8  Generating figures")
    logger.info("=" * 60)

    from src.visualization import (
        plot_feature_distributions,
        plot_correlation_heatmap,
        plot_property_scatter,
        plot_all_regression_coefficients,
        plot_subgroup_results,
        plot_mp_property_distributions,
    )

    plot_feature_distributions(df, fig_dir)
    plot_correlation_heatmap(corr_df, pval_df, fig_dir)
    plot_property_scatter(df_merged, top_feature, fig_dir)
    plot_all_regression_coefficients(ols_results, fig_dir)

    if sub_family and primary_target:
        plot_subgroup_results(
            sub_family,
            target_label=MP_PROPERTIES.get(primary_target, primary_target),
            out_dir=fig_dir,
        )

    plot_mp_property_distributions(df_merged, fig_dir)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)

    print("\n" + "=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)
    print(f"  Full dataset size             : {len(df):,} recipes")
    print(f"  MP-matched records (merged)   : {len(df_merged):,} ({100.0*len(df_merged)/len(df):.1f}%)")
    print()

    for prop, label in MP_PROPERTIES.items():
        if prop not in df_merged.columns:
            continue
        n  = df_merged[prop].notna().sum()
        res = ols_results.get(prop, {})
        r2  = res.get("r_squared", np.nan)
        print(f"  [{label}]")
        print(f"    n = {n:,}   R² = {r2:.4f}" if np.isfinite(r2) else f"    n = {n:,}")
        sdf = res.get("summary_df", pd.DataFrame())
        if not sdf.empty:
            sig = sdf[(sdf["significant"]) & (sdf["feature"] != "const")]
            if not sig.empty:
                for _, row in sig.iterrows():
                    print(f"    ** {row['feature_label']:<38s} β={row['coefficient']:+.4f}  p={row['p_value']:.4f}")
        print()

    print(f"  Figures  → {fig_dir.resolve()}")
    print(f"  Results  → {res_dir.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    main()
