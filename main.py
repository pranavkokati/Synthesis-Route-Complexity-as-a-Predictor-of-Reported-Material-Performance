#!/usr/bin/env python3
"""
Information Entropy of Local Coordination Environments Predicts Synthesizability
=================================================================================

Full pipeline:
    1. Download stratified crystal structures from Materials Project
    2. Compute Shannon entropy of coordination environment distributions
    3. Run statistical analyses (ANOVA, logistic regression, composition control)
    4. Generate all five figures
    5. Save summary tables to results/

Usage
-----
    python main.py --api-key YOUR_KEY [options]

Options
-------
    --api-key          Materials Project API key (required)
    --n-per-class      Structures per synthesizability class (default: 4000)
    --workers          Parallel worker processes for ChemEnv (default: 8)
    --cache-dir        Directory for JSON/CSV caches (default: data/)
    --fig-dir          Figure output directory (default: figures/)
    --results-dir      Results tables output directory (default: results/)
    --seed             Random seed (default: 42)
    --skip-download    Skip MP download, use existing cache
    --skip-chemenv     Skip ChemEnv computation, use existing CSV cache
    --timeout          Per-structure ChemEnv timeout in seconds (default: 120)
    --log-level        Logging level: DEBUG/INFO/WARNING (default: INFO)
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="CE Entropy vs Synthesizability pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--api-key", required=True,
                   help="Materials Project API key")
    p.add_argument("--n-per-class", type=int, default=4000,
                   help="Target structures per synthesizability class")
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel ChemEnv worker processes")
    p.add_argument("--cache-dir", default="data",
                   help="Directory for intermediate caches")
    p.add_argument("--fig-dir", default="figures",
                   help="Directory for output figures")
    p.add_argument("--results-dir", default="results",
                   help="Directory for result tables")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-download", action="store_true",
                   help="Load existing MP cache instead of downloading")
    p.add_argument("--skip-chemenv", action="store_true",
                   help="Load existing ChemEnv CSV cache instead of computing")
    p.add_argument("--timeout", type=int, default=120,
                   help="Per-structure ChemEnv timeout (seconds)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def setup_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("pipeline.log", mode="a"),
        ],
    )


def ensure_dirs(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger("main")

    cache_dir   = Path(args.cache_dir)
    fig_dir     = Path(args.fig_dir)
    results_dir = Path(args.results_dir)
    ensure_dirs(cache_dir, fig_dir, results_dir)

    mp_cache   = cache_dir / "mp_structures_cache.json"
    ce_cache   = cache_dir / "coord_entropy_results.csv"

    # -----------------------------------------------------------------------
    # Step 1: Download or load MP structures
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 1: Materials Project download")
    logger.info("=" * 60)

    from src.mp_downloader import download_structures

    if args.skip_download and mp_cache.exists():
        logger.info("--skip-download set; loading existing MP cache from %s", mp_cache)
        # We still need the DataFrame; call download_structures which will
        # return immediately from cache for already-fetched IDs.

    df_mp = download_structures(
        api_key=args.api_key,
        n_per_class=args.n_per_class,
        cache_path=mp_cache,
        seed=args.seed,
    )

    logger.info(
        "MP dataset: %d structures | class dist: %s",
        len(df_mp),
        df_mp["synth_class"].value_counts().sort_index().to_dict(),
    )

    # -----------------------------------------------------------------------
    # Step 2: Compute or load coordination environment entropy
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 2: Coordination environment entropy computation")
    logger.info("=" * 60)

    from src.coord_env import compute_coord_entropy

    df_ce = compute_coord_entropy(
        df=df_mp,
        cache_path=ce_cache,
        n_workers=args.workers,
        timeout_per_structure=args.timeout,
    )

    logger.info(
        "CE dataset: %d structures with entropy descriptors",
        len(df_ce),
    )

    # Save merged dataset
    meta_cols = [c for c in df_ce.columns if c != "structure"]
    df_ce[meta_cols].to_csv(results_dir / "dataset_with_entropy.csv", index=False)
    logger.info("Saved full dataset to results/dataset_with_entropy.csv")

    # -----------------------------------------------------------------------
    # Step 3: Statistical analysis
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 3: Statistical analysis")
    logger.info("=" * 60)

    from src.analysis import run_analysis

    results = run_analysis(df_ce)

    # Save all tables
    _save_table(results["class_stats"],      results_dir / "class_stats.csv")
    _save_table(results["anova_table"],      results_dir / "anova_results.csv")
    _save_table(results["tukey_table"],      results_dir / "tukey_hsd.csv")
    _save_table(results["logreg_coef"],      results_dir / "logreg_coefficients.csv")
    _save_table(results["composition_anova"],results_dir / "composition_corrected_anova.csv")
    _save_table(results["subgroup_anova"],   results_dir / "subgroup_anova.csv")
    _save_table(results["element_entropy"],  results_dir / "element_entropy.csv")

    _print_summary(results, logger)

    # -----------------------------------------------------------------------
    # Step 4: Generate all figures
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 4: Figure generation")
    logger.info("=" * 60)

    from src.visualization import save_all

    plot_df = results.get("_df_with_corrected", df_ce)
    saved_figs = save_all(results, plot_df, fig_dir=fig_dir)

    for p in saved_figs:
        logger.info("Figure saved: %s", p)

    # -----------------------------------------------------------------------
    # Done
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Pipeline complete.")
    logger.info("Figures    → %s/", fig_dir)
    logger.info("Results    → %s/", results_dir)
    logger.info("Log file   → pipeline.log")
    logger.info("=" * 60)


def _save_table(df, path):
    if df is not None and len(df) > 0:
        df.to_csv(path, index=False)
        logging.getLogger("main").info("Saved → %s", path)


def _print_summary(results: dict, logger):
    """Print key findings to console/log."""
    logger.info("-" * 50)
    logger.info("KEY FINDINGS")
    logger.info("-" * 50)

    cs = results.get("class_stats")
    if cs is not None:
        for _, row in cs.iterrows():
            logger.info(
                "  %s (class %d): n=%d  H̄=%.4f ± %.4f",
                row["label"], row["synth_class"], row["n"],
                row["mean_entropy"], row["sd_entropy"],
            )

    at = results.get("anova_table")
    if at is not None and len(at):
        row = at.iloc[0]
        logger.info(
            "  ANOVA ce_entropy: F=%.2f, p=%.2e, η²=%.4f",
            row["F_stat"], row["p_value"], row["eta_squared"],
        )

    sp = results.get("spearman", {})
    if sp:
        logger.info(
            "  Spearman(ce_entropy, e_hull): ρ=%.4f, p=%.2e",
            sp.get("rho", float("nan")), sp.get("pval", float("nan")),
        )

    roc = results.get("roc_data", {})
    if roc.get("auc"):
        logger.info(
            "  Binary logistic AUC: %.4f [%.4f–%.4f]",
            roc["auc"], roc.get("ci_low", roc["auc"]), roc.get("ci_high", roc["auc"]),
        )

    ca = results.get("composition_anova")
    if ca is not None and len(ca):
        row = ca.iloc[0]
        logger.info(
            "  Composition-corrected ANOVA: F=%.2f, p=%.2e, η²=%.4f",
            row["F_stat"], row["p_value"], row["eta_squared"],
        )

    sa = results.get("subgroup_anova")
    if sa is not None and len(sa):
        logger.info("  Subgroup ANOVA (crystal systems):")
        for _, row in sa.sort_values("eta_squared", ascending=False).iterrows():
            sig = "*" if row["p_value"] < 0.05 else " "
            logger.info(
                "    %s %-15s n=%5d  F=%8.2f  p=%.2e  η²=%.4f",
                sig, row["crystal_system"], row["n"],
                row["F_stat"], row["p_value"], row["eta_squared"],
            )

    logger.info("-" * 50)


if __name__ == "__main__":
    main()
