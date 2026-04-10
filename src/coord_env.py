"""
Coordination environment entropy computation.

For each pymatgen Structure, classifies every atomic site into one of ~30
coordination environment (CE) types using ChemEnv (LocalGeometryFinder +
SimplestChemenvStrategy), then derives four structural complexity descriptors:

  ce_entropy       Shannon entropy H = −Σ p_i log(p_i) of CE-symbol frequencies
  n_distinct_envs  Number of unique CE symbols present across all sites
  dominance        Fraction of sites occupied by the most common CE type
  gini             Gini coefficient of CE-symbol frequency distribution
                   (0 = perfectly uniform; approaches 1 = fully dominated)

CE symbols use pymatgen notation, e.g. O:6 (octahedral), T:4 (tetrahedral),
C:12 (cuboctahedral), L:2 (linear).

Parallel processing is handled by concurrent.futures.ProcessPoolExecutor.
Per-structure wall-clock timeout is enforced via SIGALRM on Linux.
Results are cached incrementally to data/coord_entropy_results.csv.
"""

import json
import logging
import math
import os
import signal
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
# numpy is also needed for the shuffle rng above

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Timeout helpers (Linux SIGALRM)
# ---------------------------------------------------------------------------

class _TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _TimeoutError()


# ---------------------------------------------------------------------------
# Worker function — runs in a subprocess
# ---------------------------------------------------------------------------

def _compute_one(args: tuple) -> dict | None:
    """
    Compute CE descriptors for a single structure.

    Parameters
    ----------
    args : (material_id: str, structure_dict: dict, timeout_s: int)

    Returns
    -------
    dict with descriptor fields, or dict with ``_error`` key on failure.
    """
    material_id, struct_dict, timeout_s = args

    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_s)

    try:
        from pymatgen.core import Structure
        from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import (
            SimplestChemenvStrategy,
        )
        from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import (
            LocalGeometryFinder,
        )
        from pymatgen.analysis.chemenv.coordination_environments.structure_environments import (
            LightStructureEnvironments,
        )

        struct = Structure.from_dict(struct_dict)

        lgf = LocalGeometryFinder()
        lgf.setup_structure(structure=struct)

        se = lgf.compute_structure_environments(
            maximum_distance_factor=1.41,
            only_cations=False,
        )
        strategy = SimplestChemenvStrategy(
            distance_cutoff=1.4,
            angle_cutoff=0.3,
        )
        lse = LightStructureEnvironments.from_structure_environments(
            strategy=strategy,
            structure_environments=se,
        )

        # Tally CE symbols across all sites (take highest-csm candidate)
        ce_counts: dict[str, int] = {}
        for site_envs in lse.coordination_environments:
            if not site_envs:
                continue
            best = site_envs[0].get("ce_symbol", "UNKNOWN")
            ce_counts[best] = ce_counts.get(best, 0) + 1

        if not ce_counts:
            return {"material_id": material_id, "_error": "no_ce_assigned"}

        total = sum(ce_counts.values())
        probs = sorted(v / total for v in ce_counts.values())  # ascending

        # Shannon entropy (nats)
        ce_entropy = -sum(p * math.log(p) for p in probs if p > 0)

        n_distinct = len(ce_counts)
        dominance = probs[-1]  # largest probability

        # Gini coefficient — Lorenz curve area formula
        # G = (2/n) * Σ_{i=1}^n i*p_i - (n+1)/n  (p sorted ascending, 1-indexed)
        n = len(probs)
        if n == 1:
            gini = 0.0
        else:
            weighted_sum = sum((i + 1) * p for i, p in enumerate(probs))
            gini = (2.0 * weighted_sum / n) - (n + 1) / n
            # Normalise to [0, 1] range for any n
            gini = max(0.0, min(1.0, gini))

        return {
            "material_id":     material_id,
            "ce_entropy":      ce_entropy,
            "n_distinct_envs": n_distinct,
            "dominance":       dominance,
            "gini":            gini,
            "ce_symbols":      json.dumps(ce_counts),
            "n_sites_ce":      total,
        }

    except _TimeoutError:
        return {"material_id": material_id, "_error": "timeout"}
    except Exception as exc:
        return {"material_id": material_id, "_error": str(exc)[:300]}
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_coord_entropy(
    df: pd.DataFrame,
    cache_path: str | Path = "data/coord_entropy_results.csv",
    n_workers: int | None = None,
    timeout_per_structure: int = 120,
    save_every: int = 200,
) -> pd.DataFrame:
    """
    Compute coordination-environment entropy descriptors for all structures.

    Structures already present in ``cache_path`` are skipped.  Results are
    saved incrementally every ``save_every`` completions to guard against
    interruptions.

    Parameters
    ----------
    df                    : DataFrame with ``material_id`` + ``structure``
                            columns (pymatgen Structure objects).
    cache_path            : CSV output / cache file.
    n_workers             : worker processes (default: min(8, cpu_count)).
    timeout_per_structure : per-structure wall-clock timeout in seconds.
    save_every            : save after this many newly completed structures.

    Returns
    -------
    pd.DataFrame — input df columns (minus ``structure``) joined with
    CE descriptor columns.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load existing cache
    # ------------------------------------------------------------------
    if cache_path.exists():
        cached = pd.read_csv(cache_path, dtype={"material_id": str})
        done_ids = set(cached["material_id"].tolist())
        logger.info("Cache has %d CE results", len(cached))
    else:
        cached = pd.DataFrame()
        done_ids = set()

    # ------------------------------------------------------------------
    # Filter to pending structures
    # ------------------------------------------------------------------
    pending = df[
        df["material_id"].notna() &
        df["structure"].notna() &
        ~df["material_id"].isin(done_ids)
    ][["material_id", "structure"]].copy()

    logger.info(
        "%d structures pending; %d already cached",
        len(pending), len(done_ids),
    )

    new_rows: list[dict] = []
    error_count = 0

    if len(pending) > 0:
        if n_workers is None:
            n_workers = min(8, os.cpu_count() or 4)

        args_list = [
            (row.material_id, row.structure.as_dict(), timeout_per_structure)
            for row in pending.itertuples()
            if row.structure is not None
        ]
        # Shuffle so all synthesizability classes are interleaved from the
        # start; this ensures intermediate cache saves contain a representative
        # mix of classes and intermediate analyses are meaningful.
        rng_shuffle = np.random.default_rng(seed=0)
        rng_shuffle.shuffle(args_list)

        logger.info(
            "Launching ChemEnv: %d workers × %d structures (timeout=%ds)",
            n_workers, len(args_list), timeout_per_structure,
        )
        t0 = time.time()

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            fut_map = {pool.submit(_compute_one, a): a[0] for a in args_list}
            n_total = len(fut_map)

            for done_count, fut in enumerate(as_completed(fut_map), start=1):
                mid = fut_map[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    result = {"material_id": mid, "_error": str(exc)[:200]}

                if result is None:
                    error_count += 1
                elif "_error" in result:
                    error_count += 1
                    logger.debug(
                        "CE error [%s]: %s", result["material_id"], result["_error"]
                    )
                else:
                    new_rows.append(result)

                if done_count % 100 == 0:
                    elapsed = time.time() - t0
                    rate = done_count / elapsed if elapsed > 0 else 1
                    eta_min = (n_total - done_count) / rate / 60
                    logger.info(
                        "Progress: %d/%d  (%.1f/s)  ETA %.0f min  errors: %d",
                        done_count, n_total, rate, eta_min, error_count,
                    )

                if new_rows and len(new_rows) % save_every == 0:
                    cached = _write_cache(new_rows, cached, cache_path)
                    logger.info(
                        "Saved %d new results to cache (errors: %d)",
                        len(new_rows), error_count,
                    )

        total_time = time.time() - t0
        logger.info(
            "ChemEnv complete: %d succeeded, %d errors in %.0f s (%.2f/s avg)",
            len(new_rows), error_count, total_time,
            len(new_rows) / total_time if total_time > 0 else 0,
        )

    # Final save
    if new_rows:
        cached = _write_cache(new_rows, cached, cache_path)

    # ------------------------------------------------------------------
    # Merge CE descriptors back onto input df
    # ------------------------------------------------------------------
    full_cache = (
        pd.read_csv(cache_path, dtype={"material_id": str})
        if cache_path.exists() else
        (pd.DataFrame(new_rows) if new_rows else pd.DataFrame())
    )

    meta_cols = [c for c in df.columns if c != "structure"]
    merged = df[meta_cols].merge(full_cache, on="material_id", how="inner")

    logger.info(
        "Merged dataset: %d structures with CE descriptors", len(merged)
    )
    return merged


def _write_cache(
    new_rows: list[dict],
    existing: pd.DataFrame,
    path: Path,
) -> pd.DataFrame:
    """Append new rows to cache CSV, deduplicating on material_id."""
    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["material_id"], keep="last")
    combined.to_csv(path, index=False)
    return combined
