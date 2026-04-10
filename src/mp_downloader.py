"""
Materials Project data downloader.

Downloads crystal structures with synthesizability labels from the Materials
Project REST API, applies stratified sampling across three synthesizability
classes, and caches everything to disk.

Synthesizability labels
-----------------------
  Class 0 — Stable      : energy_above_hull == 0  (on convex hull)
  Class 1 — Metastable  : 0 < energy_above_hull ≤ 0.1 eV/atom
  Class 2 — Unstable    : energy_above_hull > 0.1 eV/atom

Binary synthesizability
-----------------------
  synthesizable = 1  if  NOT theoretical  (experimentally observed / ICSD)
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
# Resilience patch: heartbeat endpoints return non-JSON in some environments.
# Patch both MPRester.get_emmet_version and BaseRester._get_database_version
# to degrade gracefully instead of crashing.
# ---------------------------------------------------------------------------
def _patch_mp_api():
    import requests as _req
    from functools import cache as _cache
    from packaging import version as _version

    # --- patch get_emmet_version on MPRester ---
    try:
        from mp_api.client.mprester import MPRester as _MPR

        @staticmethod  # type: ignore[misc]
        @_cache
        def _safe_emmet_version(endpoint):
            try:
                r = _req.get(url=endpoint + "heartbeat", timeout=15)
                v = r.json().get("version", "0.0.0")
                return _version.parse(v)
            except Exception:
                return _version.parse("0.0.0")

        _MPR.get_emmet_version = _safe_emmet_version
    except Exception:
        pass

    # --- patch _get_database_version on BaseRester ---
    try:
        from mp_api.client.core.client import BaseRester as _BR

        @staticmethod  # type: ignore[misc]
        @_cache
        def _safe_db_version(endpoint):
            try:
                r = _req.get(url=endpoint + "heartbeat", timeout=15)
                return r.json().get("db_version", "unknown")
            except Exception:
                return "unknown"

        _BR._get_database_version = _safe_db_version
    except Exception:
        pass


_patch_mp_api()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FIELDS = [
    "material_id", "formula_pretty", "energy_above_hull",
    "is_stable", "theoretical", "database_IDs",
    "nsites", "nelements", "symmetry", "structure",
    "chemsys", "elements",
]

_SYNTH_THRESHOLDS = (0.0, 0.1)   # boundaries for class 0 / 1 / 2


def synthesizability_class(ehull: float) -> int:
    """Map energy_above_hull (eV/atom) to a 3-class synthesizability label.

    Returns
    -------
    0  if ehull == 0   (thermodynamically stable, on convex hull)
    1  if 0 < ehull ≤ 0.1   (metastable, accessible via kinetics)
    2  if ehull > 0.1   (unstable / hypothetical)
    """
    if ehull <= _SYNTH_THRESHOLDS[0]:
        return 0
    if ehull <= _SYNTH_THRESHOLDS[1]:
        return 1
    return 2


def _ehull_safe(val) -> float:
    """Return float ehull, defaulting to 1.0 (unstable) when None.

    Critical: must NOT use ``val or default`` because 0.0 is falsy in Python
    and ``0.0 or 1.0`` returns 1.0, misclassifying all stable structures.
    """
    return val if val is not None else 1.0


def _doc_to_record(doc) -> dict:
    """Extract a flat, JSON-serialisable record from an mp-api SummaryDoc."""
    sym = doc.symmetry
    has_icsd = bool(
        doc.database_IDs and (
            "icsd" in doc.database_IDs or
            any("icsd" in k.lower() for k in doc.database_IDs)
        )
    )
    ehull = _ehull_safe(doc.energy_above_hull)
    return {
        "material_id":        doc.material_id,
        "formula_pretty":     doc.formula_pretty,
        "energy_above_hull":  ehull,
        "is_stable":          doc.is_stable,
        "theoretical":        doc.theoretical,
        "has_icsd":           has_icsd,
        "nsites":             doc.nsites,
        "nelements":          doc.nelements,
        "crystal_system":     sym.crystal_system.value if sym else "unknown",
        "space_group":        sym.symbol if sym else "unknown",
        "chemsys":            doc.chemsys,
        "elements":           [str(e) for e in (doc.elements or [])],
        "synth_class":        synthesizability_class(ehull),
        "synthesizable":      int(not doc.theoretical),
        "structure":          doc.structure,
    }


def download_structures(
    api_key: str,
    n_per_class: int = 4000,
    num_elements: tuple[int, int] = (2, 6),
    num_sites: tuple[int, int] = (4, 80),
    cache_path: str | Path = "data/mp_structures_cache.json",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Download a stratified sample of crystal structures from Materials Project.

    Parameters
    ----------
    api_key      : MP API key
    n_per_class  : target records per synthesizability class (0/1/2)
    num_elements : (min, max) distinct element types — spec: (2, 6)
    num_sites    : (min, max) atoms per unit cell — spec: (4, 80)
    cache_path   : JSON cache file; intermediate saves guard against restarts
    seed         : random seed for reproducible stratified sampling

    Returns
    -------
    pd.DataFrame with one row per structure.  Columns include ``structure``
    (pymatgen Structure objects), metadata, and synthesizability labels.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Fast-path: if the cache already has enough full-structure records,
    # skip the metadata query (which can fail with 503 in some network
    # environments) and use the cached IDs directly.
    # ------------------------------------------------------------------
    _sufficient_cache = False
    sampled_ids: list[str] = []

    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as _fh:
            _peek = json.load(_fh)
        _cached_recs = _peek.get("records", {})
        _n_needed = n_per_class * 3
        if len(_cached_recs) >= _n_needed:
            # Reconstruct sampled_ids from existing cache (all cached IDs)
            sampled_ids = list(_cached_recs.keys())
            _sufficient_cache = True
            logger.info(
                "Cache has %d records (≥ %d needed) — skipping metadata query.",
                len(_cached_recs), _n_needed,
            )

    if not _sufficient_cache:
        # ------------------------------------------------------------------
        # Phase 1 — lightweight metadata scan (no structure objects)
        # ------------------------------------------------------------------
        logger.info(
            "Querying MP for metadata: %d–%d elements, %d–%d sites ...",
            *num_elements, *num_sites,
        )
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

        # NOTE: use `is not None` guard — 0.0 is falsy, so `val or default`
        #       misclassifies all stable structures (ehull=0) as unstable.
        meta = pd.DataFrame([{
            "material_id":       d.material_id,
            "energy_above_hull": _ehull_safe(d.energy_above_hull),
            "theoretical":       d.theoretical,
            "synth_class":       synthesizability_class(_ehull_safe(d.energy_above_hull)),
        } for d in meta_docs])

        class_counts = meta["synth_class"].value_counts().sort_index()
        logger.info("Full dataset class distribution: %s", class_counts.to_dict())

        # ------------------------------------------------------------------
        # Phase 2 — stratified sampling
        # ------------------------------------------------------------------
        for cls in [0, 1, 2]:
            pool = meta[meta["synth_class"] == cls]["material_id"].tolist()
            n_sample = min(n_per_class, len(pool))
            if n_sample == 0:
                logger.warning("No structures available for class %d", cls)
                continue
            chosen = rng.choice(pool, size=n_sample, replace=False).tolist()
            sampled_ids.extend(chosen)
            logger.info("Class %d: sampling %d / %d", cls, n_sample, len(pool))

        logger.info("Total sampled: %d structures", len(sampled_ids))

    # ------------------------------------------------------------------
    # Phase 3 — load or download full structures (with caching)
    # ------------------------------------------------------------------
    if cache_path.exists():
        logger.info("Loading existing structure cache from %s", cache_path)
        with open(cache_path, encoding="utf-8") as fh:
            raw_cache = json.load(fh)
        cached_ids = set(raw_cache.get("records", {}).keys())
        logger.info("Cache contains %d structures", len(cached_ids))
    else:
        raw_cache = {
            "access_date": datetime.now(timezone.utc).isoformat(),
            "records": {},
        }
        cached_ids = set()

    ids_to_fetch = [mid for mid in sampled_ids if mid not in cached_ids]
    logger.info(
        "Need to fetch %d structures from MP (already cached: %d)",
        len(ids_to_fetch), len(cached_ids),
    )

    if ids_to_fetch:
        chunk_size = 500
        chunks = [
            ids_to_fetch[i: i + chunk_size]
            for i in range(0, len(ids_to_fetch), chunk_size)
        ]
        access_date = datetime.now(timezone.utc).isoformat()

        for idx, chunk in enumerate(chunks):
            logger.info(
                "Fetching chunk %d/%d (%d IDs) ...", idx + 1, len(chunks), len(chunk)
            )
            try:
                MPRester.get_emmet_version.cache_clear()
            except AttributeError:
                pass

            with MPRester(api_key) as mpr:
                docs = mpr.materials.summary.search(
                    material_ids=chunk,
                    fields=_FIELDS,
                    all_fields=False,
                )

            for doc in docs:
                rec = _doc_to_record(doc)
                struct = rec.pop("structure")
                raw_cache["records"][doc.material_id] = {
                    **rec,
                    "structure_json": struct.as_dict() if struct else None,
                }

            raw_cache["access_date"] = access_date
            with open(cache_path, "w", encoding="utf-8") as fh:
                json.dump(raw_cache, fh)
            logger.info(
                "Cache saved: %d records total", len(raw_cache["records"])
            )

    # ------------------------------------------------------------------
    # Phase 4 — assemble DataFrame
    # ------------------------------------------------------------------
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
        rows.append({
            **{k: v for k, v in rec.items() if k != "structure_json"},
            "structure": structure,
        })

    df = pd.DataFrame(rows)
    logger.info(
        "Final dataset: %d records | access date: %s",
        len(df),
        raw_cache.get("access_date", "unknown"),
    )
    return df
