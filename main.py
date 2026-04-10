#!/usr/bin/env python3
"""
Information Entropy of Local Coordination Environments Predicts Synthesizability
=================================================================================
Pipeline orchestrator — runs the full analysis end-to-end.

Steps
-----
1. Download stratified crystal structures from Materials Project (3 classes ×
   n_per_class, with 2–6 elements and 4–80 atoms per unit cell)
2. Compute Shannon entropy of coordination environment (CE) distributions
   using pymatgen ChemEnv
3. Statistical analyses: ANOVA, logistic regression (3 models), Spearman ρ,
   composition control, crystal-system subgroups, Pettifor chemical-scale
   subgroups
4. Generate all five publication figures

Usage
-----
    python main.py --api-key YOUR_KEY [options]

Required
--------
    --api-key    Materials Project API key

Optional
--------
    --n-per-class    Structures per synthesizability class  [4000]
    --workers        Parallel ChemEnv workers               [8]
    --timeout        Per-structure ChemEnv timeout (s)      [120]
    --cache-dir      Cache directory                        [data/]
    --fig-dir        Figure output directory                [figures/]
    --results-dir    Results tables directory               [results/]
    --seed           Random seed                            [42]
    --skip-download  Re-use existing MP structure cache
    --skip-chemenv   Re-use existing CE entropy CSV cache
    --log-level      DEBUG / INFO / WARNING                 [INFO]
"""

import argparse
import logging
import sys
import textwrap
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=textwrap.dedent(__doc__),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--api-key", required=True,
                   help="Materials Project API key")
    p.add_argument("--n-per-class", type=int, default=4000,
                   help="Target structures per synthesizability class")
    p.add_argument("--workers", type=int, default=8,
                   help="ChemEnv worker processes")
    p.add_argument("--timeout", type=int, default=120,
                   help="Per-structure ChemEnv timeout (seconds)")
    p.add_argument("--cache-dir",    default="data",    help="Cache directory")
    p.add_argument("--fig-dir",      default="figures", help="Figure output directory")
    p.add_argument("--results-dir",  default="results", help="Results directory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-download", action="store_true",
                   help="Use existing MP structure cache instead of re-downloading")
    p.add_argument("--skip-chemenv", action="store_true",
                   help="Use existing CE entropy CSV instead of recomputing")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(level: str, log_file: str = "pipeline.log"):
    fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    datefmt = "%H:%M:%S"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="a"),
    ]
    logging.basicConfig(level=getattr(logging, level), format=fmt,
                        datefmt=datefmt, handlers=handlers)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mkdirs(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def _save_csv(df: pd.DataFrame | None, path: Path, logger):
    if df is not None and len(df) > 0:
        df.to_csv(path, index=False)
        logger.info("Saved → %s  (%d rows)", path, len(df))


def _section(logger, title: str):
    bar = "=" * 60
    logger.info(bar)
    logger.info(title)
    logger.info(bar)


def _print_key_findings(results: dict, logger):
    logger.info("-" * 55)
    logger.info("KEY FINDINGS")
    logger.info("-" * 55)

    cs = results.get("class_stats")
    if cs is not None and len(cs):
        for _, r in cs.iterrows():
            logger.info(
                "  %-12s  n=%5d  H̄=%.4f ± %.4f (nats)",
                r["label"], r["n"], r["mean"], r["sd"],
            )

    at = results.get("anova_table")
    if at is not None and len(at):
        r = at[at["descriptor"] == "ce_entropy"].squeeze()
        if isinstance(r, pd.Series):
            logger.info(
                "  ANOVA ce_entropy: F=%.2f  p=%.2e  η²=%.4f  p_adj=%.2e  %s",
                r["F_stat"], r["p_value"], r["eta_squared"],
                r.get("p_adj_bh", r["p_value"]), r.get("sig", ""),
            )

    sp = results.get("spearman", {})
    if sp:
        logger.info("  Spearman ρ(entropy, ehull): %.4f  p=%.2e", sp["rho"], sp["pval"])

    roc = results.get("roc_data", {})
    for name in ["naive", "entropy", "full"]:
        rd = roc.get(name, {})
        if rd.get("auc"):
            logger.info(
                "  ROC %-8s  AUC=%.4f [%.4f–%.4f]  pseudo-R²=%.4f",
                name, rd["auc"], rd.get("ci_low", 0), rd.get("ci_high", 0),
                rd.get("mcfadden_r2", 0),
            )

    ca = results.get("composition_anova")
    if ca is not None and len(ca):
        r = ca.iloc[0]
        logger.info(
            "  Composition-corrected ANOVA: F=%.2f  p=%.2e  η²=%.4f",
            r["F_stat"], r["p_value"], r["eta_squared"],
        )

    sa = results.get("subgroup_anova")
    if sa is not None and len(sa):
        logger.info("  Crystal-system subgroup results:")
        for _, r in sa.sort_values("eta_squared", ascending=False).iterrows():
            sig = r.get("p_adj_bh", r["p_value"])
            logger.info(
                "    %-16s  n=%5d  F=%8.2f  p=%.2e  η²=%.4f",
                r["crystal_system"], r["n_total"], r["F_stat"],
                r["p_value"], r["eta_squared"],
            )

    pa = results.get("pettifor_analysis")
    if pa is not None and len(pa):
        logger.info("  Pettifor chemical-diversity subgroups:")
        for _, r in pa.sort_values("eta_squared", ascending=False).iterrows():
            logger.info(
                "    %-20s  n=%5d  F=%8.2f  p=%.2e  η²=%.4f",
                r["pettifor_bin"], r["n_total"], r["F_stat"],
                r["p_value"], r["eta_squared"],
            )

    logger.info("-" * 55)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    _setup_logging(args.log_level)
    logger = logging.getLogger("main")

    cache_dir   = Path(args.cache_dir)
    fig_dir     = Path(args.fig_dir)
    results_dir = Path(args.results_dir)
    _mkdirs(cache_dir, fig_dir, results_dir)

    mp_cache = cache_dir / "mp_structures_cache.json"
    ce_cache = cache_dir / "coord_entropy_results.csv"

    # ---------------------------------------------------------------
    # Step 1 — Download structures
    # ---------------------------------------------------------------
    _section(logger, "STEP 1 — Materials Project download")

    from src.mp_downloader import download_structures

    df_mp = download_structures(
        api_key=args.api_key,
        n_per_class=args.n_per_class,
        num_elements=(2, 6),
        num_sites=(4, 80),
        cache_path=mp_cache,
        seed=args.seed,
    )

    class_dist = df_mp["synth_class"].value_counts().sort_index().to_dict()
    logger.info(
        "MP dataset ready: %d structures | class dist: %s",
        len(df_mp), class_dist,
    )

    # ---------------------------------------------------------------
    # Step 2 — Coordination environment entropy
    # ---------------------------------------------------------------
    _section(logger, "STEP 2 — ChemEnv coordination entropy")

    from src.coord_env import compute_coord_entropy

    df_ce = compute_coord_entropy(
        df=df_mp,
        cache_path=ce_cache,
        n_workers=args.workers,
        timeout_per_structure=args.timeout,
    )

    logger.info(
        "CE dataset ready: %d structures with entropy descriptors",
        len(df_ce),
    )

    # Save dataset
    meta_cols = [c for c in df_ce.columns if c != "structure"]
    df_ce[meta_cols].to_csv(results_dir / "dataset_with_entropy.csv", index=False)
    logger.info("Dataset saved → results/dataset_with_entropy.csv")

    # ---------------------------------------------------------------
    # Step 3 — Statistical analysis
    # ---------------------------------------------------------------
    _section(logger, "STEP 3 — Statistical analysis")

    from src.analysis import run_analysis

    results = run_analysis(df_ce)

    _save_csv(results.get("class_stats"),       results_dir / "class_stats.csv", logger)
    _save_csv(results.get("anova_table"),        results_dir / "anova_results.csv", logger)
    _save_csv(results.get("tukey_table"),        results_dir / "tukey_hsd.csv", logger)
    _save_csv(results.get("logreg_coef"),        results_dir / "logreg_coefficients.csv", logger)
    _save_csv(results.get("composition_anova"),  results_dir / "composition_anova.csv", logger)
    _save_csv(results.get("subgroup_anova"),     results_dir / "crystal_system_anova.csv", logger)
    _save_csv(results.get("pettifor_analysis"),  results_dir / "pettifor_anova.csv", logger)
    _save_csv(results.get("element_entropy"),    results_dir / "element_entropy.csv", logger)

    # ROC data as CSV
    roc_rows = []
    for model, rd in results.get("roc_data", {}).items():
        roc_rows.append({
            "model":        model,
            "auc":          rd.get("auc"),
            "ci_low":       rd.get("ci_low"),
            "ci_high":      rd.get("ci_high"),
            "mcfadden_r2":  rd.get("mcfadden_r2"),
        })
    if roc_rows:
        _save_csv(pd.DataFrame(roc_rows), results_dir / "roc_summary.csv", logger)

    _print_key_findings(results, logger)

    # ---------------------------------------------------------------
    # Step 4 — Figures
    # ---------------------------------------------------------------
    _section(logger, "STEP 4 — Figure generation")

    from src.visualization import save_all

    saved_figs = save_all(results, results.get("_df_enriched", df_ce), fig_dir=fig_dir)
    for p in saved_figs:
        logger.info("Figure → %s", p)

    # ---------------------------------------------------------------
    # Done
    # ---------------------------------------------------------------
    _section(logger, "PIPELINE COMPLETE")
    logger.info("Figures  → %s/", fig_dir)
    logger.info("Results  → %s/", results_dir)
    logger.info("Log      → pipeline.log")


if __name__ == "__main__":
    main()
