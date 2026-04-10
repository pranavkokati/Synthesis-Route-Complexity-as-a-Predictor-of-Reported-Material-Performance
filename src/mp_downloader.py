"""
Materials Project data downloader.

Downloads crystal structures with synthesizability labels from the Materials
Project REST API, applies stratified sampling across three synthesizability
classes, and caches everything to disk.

Synthesizability labels:
    0 — Stable      : energy_above_hull == 0  (on convex hull)
    1 — Metastable  : 0 < energy_above_hull ≤ 0.1 eV/atom
    2 — Unstable    : energy_above_hull > 0.1 eV/atom

Binary synthesizability (for logistic regression):
    synthesizable = 1  if  NOT theoretical  (experimentally observed)
    synthesizable = 0  if  theoretical      (purely computational)

MP access date is stored in the cache for reproducibility.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from mp_api.client import MPRester

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resilience patch: _get_database_version hits sub-rester heartbeat endpoints
# that may return non-JSON in some network environments.  Wrap it to return a
# fallback version string rather than crashing.
# ---------------------------------------------------------------------------
def _patch_mp_api():
    try:
        from mp_api.client.core.client import BaseRester
        import requests as _req
        from functools import cache as _cache

        @staticmethod  # type: ignore[misc]
        @_cache
        def _safe_get_database_version(endpoint):
            try:
                r = _req.get(url=endpoint + "heartbeat", timeout=15)
                data = r.json()
                return data.get("db_version", "unknown")
            except Exception:
                return "unknown"

        BaseRester._get_database_version = _safe_get_database_version
    except Exception:
        pass

_patch_mp_api()

_FIELDS = [
    "material_id", "formula_pretty", "energy_above_hull",
    "is_stable", "theoretical", "database_IDs",
    "nsites", "nelements", "symmetry", "structure",
    "chemsys", "elements",
]

_SYNTH_THRESHOLDS = (0.0, 0.1)   # boundaries for class 0/1/2


def synthesizability_class(ehull: float) -> int:
    """Map energy_above_hull to a 3-class synthesizability label."""
    if ehull <= _SYNTH_THRESHOLDS[0]:
        return 0
    if ehull <= _SYNTH_THRESHOLDS[1]:
        return 1
    return 2


def _doc_to_record(doc) -> dict:
    """Extract flat record from an mp-api SummaryDoc."""
    sym = doc.symmetry
    has_icsd = bool(
        doc.database_IDs and (
            "icsd" in doc.database_IDs or
            any("icsd" in k.lower() for k in doc.database_IDs)
        )
    )
    return {
        "material_id":        doc.material_id,
        "formula_pretty":     doc.formula_pretty,
        "energy_above_hull":  doc.energy_above_hull,
        "is_stable":          doc.is_stable,
        "theoretical":        doc.theoretical,
        "has_icsd":           has_icsd,
        "nsites":             doc.nsites,
        "nelements":          doc.nelements,
        "crystal_system":     sym.crystal_system.value if sym else "unknown",
        "space_group":        sym.symbol if sym else "unknown",
        "chemsys":            doc.chemsys,
        "elements":           [str(e) for e in (doc.elements or [])],
        "synth_class":        synthesizability_class(
                                  doc.energy_above_hull if doc.energy_above_hull is not None else 1.0
                              ),
        "synthesizable":      int(not doc.theoretical),
        "structure":          doc.structure,
    }


def download_structures(
    api_key: str,
    n_per_class: int = 4000,
    num_elements: tuple[int, int] = (2, 6),
    num_sites: tuple[int, int] = (4, 25),
    cache_path: str | Path = "data/mp_structures_cache.json",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Download a stratified sample of crystal structures from Materials Project.

    Parameters
    ----------
    api_key      : MP API key
    n_per_class  : target records per synthesizability class
    num_elements : (min, max) distinct element types
    num_sites    : (min, max) atoms per unit cell
    cache_path   : JSON cache file path
    seed         : random seed for reproducible stratified sampling

    Returns
    -------
    pd.DataFrame with columns including structure (pymatgen Structure objects).
    """
    cache_path = Path(cache_path)
    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------
    # Step 1 — download metadata only (fast, no structure objects)
    # ---------------------------------------------------------------
    logger.info(
        "Querying MP for metadata: %d–%d elements, %d–%d sites ...",
        *num_elements, *num_sites,
    )
    # Clear cached emmet-version check in case it was poisoned by a prior
    # failed network call within the same process.
    try:
        MPRester.get_emmet_version.cache_clear()
    except AttributeError:
        pass
    with MPRester(api_key) as mpr:
        meta_docs = mpr.materials.summary.search(
            num_elements=num_elements,
            num_sites=num_sites,
            fields=["material_id", "energy_above_hull", "theoretical"],
            all_fields=False,
        )

    logger.info("Retrieved %d metadata records", len(meta_docs))

    # Assign synth class from metadata
    # NOTE: use `if X is not None` not `X or default`, since ehull=0.0 is
    #       falsy in Python and `0.0 or 1.0` would wrongly return 1.0.
    meta = pd.DataFrame([{
        "material_id":       d.material_id,
        "energy_above_hull": d.energy_above_hull if d.energy_above_hull is not None else 0.0,
        "theoretical":       d.theoretical,
        "synth_class":       synthesizability_class(
                                 d.energy_above_hull if d.energy_above_hull is not None else 1.0
                             ),
    } for d in meta_docs])

    class_counts = meta["synth_class"].value_counts().sort_index()
    logger.info("Class distribution in full dataset: %s", class_counts.to_dict())

    # ---------------------------------------------------------------
    # Step 2 — stratified sample per class
    # ---------------------------------------------------------------
    sampled_ids: list[str] = []
    for cls in [0, 1, 2]:
        pool = meta[meta["synth_class"] == cls]["material_id"].tolist()
        n_sample = min(n_per_class, len(pool))
        chosen = rng.choice(pool, size=n_sample, replace=False).tolist()
        sampled_ids.extend(chosen)
        logger.info("Class %d: sampling %d / %d", cls, n_sample, len(pool))

    logger.info("Total sampled: %d structures", len(sampled_ids))

    # ---------------------------------------------------------------
    # Step 3 — load from cache or download full structures
    # ---------------------------------------------------------------
    if cache_path.exists():
        logger.info("Loading cached structures from %s", cache_path)
        with open(cache_path, encoding="utf-8") as fh:
            raw_cache = json.load(fh)
        cached_ids = set(raw_cache.get("records", {}).keys())
        logger.info("Cache contains %d structures", len(cached_ids))
    else:
        raw_cache = {"access_date": datetime.now(timezone.utc).isoformat(), "records": {}}
        cached_ids = set()

    ids_to_fetch = [mid for mid in sampled_ids if mid not in cached_ids]
    logger.info("Fetching %d structures from MP (not in cache)", len(ids_to_fetch))

    if ids_to_fetch:
        chunk_size = 500
        chunks = [ids_to_fetch[i:i+chunk_size] for i in range(0, len(ids_to_fetch), chunk_size)]
        access_date = datetime.now(timezone.utc).isoformat()

        for chunk_idx, chunk in enumerate(chunks):
            logger.info("Fetching chunk %d/%d (%d IDs)...", chunk_idx+1, len(chunks), len(chunk))
            with MPRester(api_key) as mpr:
                docs = mpr.materials.summary.search(
                    material_ids=chunk,
                    fields=_FIELDS,
                    all_fields=False,
                )

            for doc in docs:
                rec = _doc_to_record(doc)
                struct = rec.pop("structure")
                # Store structure as JSON-serialisable dict
                raw_cache["records"][doc.material_id] = {
                    **rec,
                    "structure_json": struct.as_dict() if struct else None,
                }

            # Save intermediate cache
            raw_cache["access_date"] = access_date
            with open(cache_path, "w", encoding="utf-8") as fh:
                json.dump(raw_cache, fh)
            logger.info("Cache updated (%d records total)", len(raw_cache["records"]))

    # ---------------------------------------------------------------
    # Step 4 — assemble DataFrame
    # ---------------------------------------------------------------
    from pymatgen.core import Structure as PMGStructure

    rows = []
    for mid in sampled_ids:
        rec = raw_cache["records"].get(mid)
        if rec is None:
            continue
        struct_json = rec.get("structure_json")
        structure = None
        if struct_json:
            try:
                structure = PMGStructure.from_dict(struct_json)
            except Exception:
                pass
        rows.append({**{k: v for k, v in rec.items() if k != "structure_json"},
                     "structure": structure})

    df = pd.DataFrame(rows)
    logger.info("Final dataset: %d records", len(df))
    logger.info("MP access date: %s", raw_cache.get("access_date", "unknown"))
    return df
