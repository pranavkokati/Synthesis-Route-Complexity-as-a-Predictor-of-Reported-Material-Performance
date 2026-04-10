"""
Coordination environment entropy computation.

For each pymatgen Structure, uses ChemEnv (LocalGeometryFinder +
SimplestChemenvStrategy) to classify every atomic site into one of ~30
coordination environment types (CE symbols such as O:6, T:4, C:12, ...).

Descriptors computed per structure
-----------------------------------
ce_entropy      : Shannon entropy H = −Σ p_i log(p_i) of CE-symbol frequencies
n_distinct_envs : number of unique CE symbols present
dominance       : fraction of most-common CE symbol
gini            : Gini coefficient of CE-symbol frequency distribution
ce_symbols      : JSON string of {symbol: count} dict (for downstream use)

Parallel processing is handled by concurrent.futures.ProcessPoolExecutor.
Per-structure timeout is enforced via a wrapper signal approach (Linux only).
Results are cached to data/coord_entropy_results.csv.
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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timeout sentinel
# ---------------------------------------------------------------------------

class _TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _TimeoutError()


# ---------------------------------------------------------------------------
# Core per-structure computation (runs in worker process)
# ---------------------------------------------------------------------------

def _compute_one(args):
    """
    Compute CE descriptors for a single structure.

    Parameters
    ----------
    args : tuple of (material_id: str, structure_dict: dict, timeout_s: int)

    Returns
    -------
    dict with material_id + descriptor fields, or None on failure
    """
    material_id, struct_dict, timeout_s = args

    # Set per-process alarm (Linux only)
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

        # Collect CE symbols across all sites
        ce_counts: dict[str, int] = {}
        for site_envs in lse.coordination_environments:
            if not site_envs:
                continue
            # Each site may have multiple CE candidates; take the best (index 0)
            best = site_envs[0].get("ce_symbol", "UNKNOWN")
            ce_counts[best] = ce_counts.get(best, 0) + 1

        if not ce_counts:
            return None

        total = sum(ce_counts.values())
        probs = [v / total for v in ce_counts.values()]

        # Shannon entropy (nats → bits optional, keep nats)
        ce_entropy = -sum(p * math.log(p) for p in probs if p > 0)

        n_distinct_envs = len(ce_counts)
        dominance = max(probs)

        # Gini coefficient
        sorted_p = sorted(probs)
        n = len(sorted_p)
        if n == 1:
            gini = 0.0
        else:
            cumsum = 0.0
            gini_num = 0.0
            for i, p in enumerate(sorted_p):
                gini_num += (2 * (i + 1) - n - 1) * p
            gini = gini_num / (n * sum(sorted_p))

        return {
            "material_id":    material_id,
            "ce_entropy":     ce_entropy,
            "n_distinct_envs": n_distinct_envs,
            "dominance":      dominance,
            "gini":           gini,
            "ce_symbols":     json.dumps(ce_counts),
        }

    except _TimeoutError:
        return {"material_id": material_id, "_error": "timeout"}
    except Exception as exc:
        return {"material_id": material_id, "_error": str(exc)[:200]}
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
    chunk_save_every: int = 200,
) -> pd.DataFrame:
    """
    Compute coordination-environment entropy descriptors for all structures.

    Parameters
    ----------
    df                      : DataFrame with columns 'material_id' and 'structure'
                              (pymatgen Structure objects).
    cache_path              : CSV file for caching / incremental saves.
    n_workers               : worker processes (default: min(8, cpu_count)).
    timeout_per_structure   : per-structure wall-clock timeout in seconds.
    chunk_save_every        : save cache after this many completed structures.

    Returns
    -------
    pd.DataFrame joined back to input df columns (minus 'structure').
    """
    cache_path = Path(cache_path)

    # Load existing cache
    if cache_path.exists():
        cached = pd.read_csv(cache_path, dtype={"material_id": str})
        logger.info("Cache contains %d CE results", len(cached))
        done_ids = set(cached["material_id"].tolist())
    else:
        cached = pd.DataFrame()
        done_ids = set()

    # Filter to structures that need processing
    todo = df[
        df["material_id"].notna() &
        df["structure"].notna() &
        ~df["material_id"].isin(done_ids)
    ][["material_id", "structure"]].copy()

    logger.info(
        "%d structures to process (%d already cached)",
        len(todo), len(done_ids),
    )

    new_rows: list[dict] = []
    error_count = 0

    if len(todo) > 0:
        if n_workers is None:
            n_workers = min(8, (os.cpu_count() or 4))

        # Build args list: (material_id, struct_dict, timeout)
        args_list = [
            (row.material_id, row.structure.as_dict(), timeout_per_structure)
            for row in todo.itertuples()
            if row.structure is not None
        ]

        logger.info(
            "Starting parallel ChemEnv computation: %d workers, %d structures",
            n_workers, len(args_list),
        )
        t0 = time.time()

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_compute_one, a): a[0] for a in args_list}
            completed = 0

            for fut in as_completed(futures):
                mid = futures[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    result = {"material_id": mid, "_error": str(exc)[:200]}

                if result is None:
                    error_count += 1
                elif "_error" in result:
                    error_count += 1
                    logger.debug("Error for %s: %s", result["material_id"], result["_error"])
                else:
                    new_rows.append(result)

                completed += 1
                if completed % 100 == 0:
                    elapsed = time.time() - t0
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = (len(args_list) - completed) / rate if rate > 0 else 0
                    logger.info(
                        "Progress: %d/%d (%.1f/s) — ETA %.0f min",
                        completed, len(args_list), rate, remaining / 60,
                    )

                # Incremental save
                if len(new_rows) > 0 and len(new_rows) % chunk_save_every == 0:
                    _append_cache(new_rows, cached, cache_path)
                    logger.info(
                        "Intermediate save: %d new results (errors so far: %d)",
                        len(new_rows), error_count,
                    )

        elapsed_total = time.time() - t0
        logger.info(
            "ChemEnv done: %d succeeded, %d errors/timeouts in %.1f s",
            len(new_rows), error_count, elapsed_total,
        )

    # Final save
    if new_rows:
        _append_cache(new_rows, cached, cache_path)

    # Reload full cache
    if cache_path.exists():
        full_cache = pd.read_csv(cache_path, dtype={"material_id": str})
    else:
        full_cache = pd.DataFrame(new_rows) if new_rows else pd.DataFrame()

    # Merge back with input df (drop structure column to keep df slim)
    meta_cols = [c for c in df.columns if c != "structure"]
    merged = df[meta_cols].merge(full_cache, on="material_id", how="inner")

    logger.info(
        "Merged dataset: %d records with CE descriptors", len(merged)
    )
    return merged


def _append_cache(
    new_rows: list[dict],
    existing: pd.DataFrame,
    cache_path: Path,
) -> pd.DataFrame:
    """Append new_rows to cache file."""
    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["material_id"], keep="last")
    combined.to_csv(cache_path, index=False)
    return combined
