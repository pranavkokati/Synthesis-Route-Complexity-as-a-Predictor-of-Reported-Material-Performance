"""
main.py — Synthesis Route Complexity as a Predictor of Reported Material Performance
=====================================================================================

Pipeline:
    1. Load the Kononova et al. solid-state synthesis dataset
    2. Extract five operationally-defined complexity features per recipe
    3. Extract quantitative performance metrics from synthesis paragraph text
    4. Normalise performance metrics within each class
    5. Multivariate OLS regression: perf_norm ~ complexity features
    6. Correlation matrix and VIF diagnostics
    7. Subgroup regressions (by metric class and by material family)
    8. Generate all publication-quality figures

Usage:
    python main.py [--data PATH] [--output DIR] [--log-level LEVEL]

Environment variables:
    MP_API_KEY    — (optional) Materials Project API key for E_hull retrieval
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--data",
        default="data/solid_state_synthesis.json",
        help="Path to the Kononova dataset JSON file.",
    )
    p.add_argument(
        "--output",
        default="output",
        help="Root directory for figures and result files.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(args.log_level)

    out_root = Path(args.output)
    fig_dir = out_root / "figures"
    res_dir = out_root / "results"
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
    df_raw = parse_reactions(reactions)

    # ------------------------------------------------------------------
    # Step 2 — Complexity features
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 2  Computing complexity features")
    logger.info("=" * 60)

    from src.feature_extractor import compute_features, get_feature_summary, FEATURE_COLS

    df = compute_features(df_raw)
    summary = get_feature_summary(df)
    logger.info("\n%s", summary.to_string())
    summary.to_csv(res_dir / "feature_summary.csv")

    # ------------------------------------------------------------------
    # Step 3 — Materials Project (optional)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 3  Materials Project integration (optional)")
    logger.info("=" * 60)

    from src.materials_project import fetch_mp_properties

    unique_formulas = df["target_formula"].dropna().unique().tolist()
    mp_df = fetch_mp_properties(unique_formulas[:500])  # cap to avoid rate limits
    if not mp_df.empty:
        df = df.merge(mp_df, left_on="target_formula", right_on="formula", how="left")
        logger.info("Merged MP data for %d records", df["energy_above_hull"].notna().sum())

    # ------------------------------------------------------------------
    # Step 4 — Performance extraction
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 4  Extracting performance metrics from text")
    logger.info("=" * 60)

    from src.performance_extractor import extract_all_performance, normalise_performance

    df = extract_all_performance(df)
    df = normalise_performance(df)

    metric_counts = df["metric_name"].value_counts()
    logger.info("\nMetric class counts:\n%s", metric_counts.to_string())
    metric_counts.to_csv(res_dir / "metric_counts.csv", header=["count"])

    # Subset with both complexity features and performance metric
    analysis_cols = FEATURE_COLS + ["perf_norm", "metric_name", "target_formula", "doi"]
    df_analysis = df[analysis_cols].dropna(subset=["perf_norm"] + FEATURE_COLS)
    logger.info(
        "Analysis-ready subset: %d records (%.1f%% of full dataset)",
        len(df_analysis),
        100.0 * len(df_analysis) / len(df),
    )
    df_analysis.to_csv(res_dir / "analysis_dataset.csv", index=False)

    # ------------------------------------------------------------------
    # Step 5 — Multivariate OLS
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 5  Multivariate OLS regression")
    logger.info("=" * 60)

    from src.analysis import run_ols, compute_correlation_matrix, compute_vif

    ols_full = run_ols(df_analysis, label="Full analysis dataset")
    if ols_full["model"] is not None:
        logger.info("\n%s", ols_full["model"].summary())
        ols_full["summary_df"].to_csv(res_dir / "ols_full_results.csv", index=False)

    # ------------------------------------------------------------------
    # Step 6 — Correlation matrix + VIF
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 6  Correlation matrix and VIF")
    logger.info("=" * 60)

    corr_df, pval_df = compute_correlation_matrix(df_analysis)
    corr_df.to_csv(res_dir / "correlation_matrix.csv")
    pval_df.to_csv(res_dir / "pvalue_matrix.csv")
    logger.info("\nCorrelation matrix:\n%s", corr_df.to_string())

    vif_df = compute_vif(df_analysis)
    vif_df.to_csv(res_dir / "vif.csv", index=False)
    logger.info("\nVIF:\n%s", vif_df.to_string())

    # ------------------------------------------------------------------
    # Step 7 — Subgroup regressions
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 7  Subgroup regressions")
    logger.info("=" * 60)

    from src.analysis import (
        subgroup_regression_by_metric,
        subgroup_regression_by_family,
        feature_importance_ranking,
    )

    sub_metric = subgroup_regression_by_metric(df)
    sub_family = subgroup_regression_by_family(df)

    subgroup_summary = []
    for r in sub_metric + sub_family:
        subgroup_summary.append(
            {
                "label": r["label"],
                "n": r["n"],
                "r_squared": r["r_squared"],
                "adj_r_squared": r.get("adj_r_squared", np.nan),
            }
        )
    pd.DataFrame(subgroup_summary).to_csv(res_dir / "subgroup_results.csv", index=False)

    # Feature importance
    if ols_full["model"] is not None:
        imp = feature_importance_ranking(ols_full)
        logger.info("\nFeature importance (by |coefficient|):\n%s", imp[["feature_label", "coefficient", "p_value"]].to_string())
        imp.to_csv(res_dir / "feature_importance.csv", index=False)
        top_features = imp["feature"].tolist()[:2]
    else:
        top_features = FEATURE_COLS[:2]

    # ------------------------------------------------------------------
    # Step 8 — Figures
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 8  Generating figures")
    logger.info("=" * 60)

    from src.visualization import (
        plot_feature_distributions,
        plot_correlation_heatmap,
        plot_performance_scatter,
        plot_regression_coefficients,
        plot_subgroup_results,
        plot_performance_by_class,
    )

    plot_feature_distributions(df, fig_dir)
    plot_correlation_heatmap(corr_df, pval_df, fig_dir)

    if len(top_features) >= 2:
        plot_performance_scatter(df_analysis, top_features, fig_dir)

    if ols_full["model"] is not None:
        plot_regression_coefficients(ols_full, fig_dir)

    all_subgroups = sub_metric + sub_family
    plot_subgroup_results(all_subgroups, fig_dir)
    plot_performance_by_class(df, fig_dir)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Dataset size              : {len(df):,} reactions")
    print(f"  With max temperature      : {df['max_temperature_C'].notna().sum():,}")
    print(f"  With total time           : {df['total_time_h'].notna().sum():,}")
    print(f"  Performance extracted     : {df['perf_norm'].notna().sum():,}")
    print(f"  Analysis-ready records    : {len(df_analysis):,}")
    print()

    if ols_full["model"] is not None:
        print(f"  OLS R²                    : {ols_full['r_squared']:.4f}")
        print(f"  OLS adj-R²                : {ols_full['adj_r_squared']:.4f}")
        print(f"  OLS n                     : {ols_full['n']:,}")
        print()
        sdf = ols_full["summary_df"]
        sig = sdf[sdf["significant"] & (sdf["feature"] != "const")]
        if not sig.empty:
            print("  Significant predictors (p<0.05):")
            for _, row in sig.iterrows():
                print(
                    f"    {row['feature_label']:<40s}  β={row['coefficient']:+.4f}  p={row['p_value']:.4f}"
                )

    print()
    print(f"  Figures saved to  : {fig_dir.resolve()}")
    print(f"  Results saved to  : {res_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
