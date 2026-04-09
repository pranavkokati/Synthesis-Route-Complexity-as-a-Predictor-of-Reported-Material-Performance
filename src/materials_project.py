"""
Materials Project API integration.

Fetches DFT-computed properties for target material formulas using the
Materials Project REST API v2 (next-gen).  For each formula, the entry with
the lowest energy_above_hull is selected (ground-state / most stable polymorph).

Properties retrieved per formula:
    band_gap                  — electronic band gap (eV)
    formation_energy_per_atom — thermodynamic formation energy (eV/atom)
    energy_above_hull         — distance from the convex hull (eV/atom)
    density                   — crystal density (g/cm³)
    volume                    — unit cell volume (Å³)

Results are cached to data/mp_properties_cache.json to avoid redundant API
calls on repeated runs.

API reference: https://next-gen.materialsproject.org/api
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_MP_BASE = "https://api.materialsproject.org"
_FIELDS = "formula_pretty,band_gap,formation_energy_per_atom,energy_above_hull,density,volume"
_BATCH_SIZE = 20        # formulas per API request
_SLEEP_BETWEEN = 0.25   # seconds between requests

# Elements for formula validation
_ELEMENTS = {
    "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P",
    "S","Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh",
    "Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd",
    "Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re",
    "Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th",
    "Pa","U","Np","Pu",
}


def is_queryable(formula: str) -> bool:
    """
    Return True if *formula* is a clean stoichiometric formula that can be
    submitted directly to the MP API (no variable placeholders, no operators,
    only recognised element symbols).
    """
    if not formula or len(formula) < 2:
        return False
    # Allow only alphanumeric, dots, and parentheses
    if not re.match(r"^[A-Za-z0-9\(\)\.]+$", formula):
        return False
    # Reject parenthesised groups (complex nested expressions)
    if "(" in formula or ")" in formula:
        return False
    # Extract element tokens and validate every one
    tokens = re.findall(r"[A-Z][a-z]?", formula)
    if not tokens:
        return False
    return all(t in _ELEMENTS for t in tokens)


def _query_batch(
    formulas: list[str], api_key: str, session: requests.Session
) -> list[dict[str, Any]]:
    """Query the MP summary endpoint for a batch of formulas."""
    url = f"{_MP_BASE}/materials/summary/"
    params = {
        "formula": ",".join(formulas),
        "_fields": _FIELDS,
        "_limit": _BATCH_SIZE * 50,  # generous upper bound for polymorphs
    }
    headers = {"X-API-KEY": api_key}
    resp = session.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json().get("data", [])


def _best_entry(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the ground-state entry (minimum energy_above_hull)."""
    valid = [e for e in entries if e.get("energy_above_hull") is not None]
    if not valid:
        return entries[0]
    return min(valid, key=lambda e: e["energy_above_hull"])


def fetch_mp_properties(
    formulas: list[str],
    api_key: str,
    cache_path: str | Path = "data/mp_properties_cache.json",
) -> pd.DataFrame:
    """
    Fetch Materials Project properties for a list of target formulas.

    Results are cached to *cache_path*.  Only formulas absent from the cache
    are queried.  After querying, the cache is updated.

    Parameters
    ----------
    formulas : list[str]
        Raw target formulas from the Kononova dataset.
    api_key : str
        Materials Project API key.
    cache_path : path-like
        Path to the local JSON cache file.

    Returns
    -------
    pd.DataFrame
        One row per formula with columns:
        formula, band_gap, formation_energy_per_atom,
        energy_above_hull, density, volume.
    """
    cache_path = Path(cache_path)

    # Load existing cache
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as fh:
            cache: dict[str, dict] = json.load(fh)
        logger.info("Loaded %d cached MP entries from %s", len(cache), cache_path)
    else:
        cache = {}

    # Filter to queryable formulas not yet in cache
    queryable = [f for f in formulas if is_queryable(f)]
    unique_queryable = sorted(set(queryable))
    to_fetch = [f for f in unique_queryable if f not in cache]

    logger.info(
        "Formulas total=%d  queryable=%d  cached=%d  to_fetch=%d",
        len(formulas),
        len(unique_queryable),
        len(unique_queryable) - len(to_fetch),
        len(to_fetch),
    )

    if to_fetch:
        session = requests.Session()
        batches = [
            to_fetch[i : i + _BATCH_SIZE] for i in range(0, len(to_fetch), _BATCH_SIZE)
        ]
        logger.info("Querying MP API: %d batches of up to %d formulas", len(batches), _BATCH_SIZE)

        fetched = 0
        failed = 0
        for idx, batch in enumerate(batches):
            try:
                entries = _query_batch(batch, api_key, session)
                # Group by formula_pretty
                by_formula: dict[str, list[dict]] = {}
                for entry in entries:
                    fp = entry.get("formula_pretty", "")
                    by_formula.setdefault(fp, []).append(entry)

                for formula in batch:
                    if formula in by_formula:
                        best = _best_entry(by_formula[formula])
                        cache[formula] = {
                            "band_gap": best.get("band_gap"),
                            "formation_energy_per_atom": best.get("formation_energy_per_atom"),
                            "energy_above_hull": best.get("energy_above_hull"),
                            "density": best.get("density"),
                            "volume": best.get("volume"),
                        }
                        fetched += 1
                    else:
                        # Formula not found in MP
                        cache[formula] = None

            except requests.RequestException as exc:
                logger.warning("Batch %d failed: %s", idx, exc)
                failed += 1

            if (idx + 1) % 50 == 0:
                logger.info(
                    "Progress: %d/%d batches | fetched=%d",
                    idx + 1, len(batches), fetched,
                )
                # Save intermediate cache
                with open(cache_path, "w", encoding="utf-8") as fh:
                    json.dump(cache, fh)

            time.sleep(_SLEEP_BETWEEN)

        # Final cache save
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump(cache, fh)

        logger.info(
            "MP query complete: fetched=%d  not_found=%d  failed_batches=%d",
            fetched,
            len(to_fetch) - fetched - failed * _BATCH_SIZE,
            failed,
        )

    # Build result DataFrame
    records = []
    for formula in unique_queryable:
        entry = cache.get(formula)
        if entry is not None:
            records.append({"formula": formula, **entry})

    df = pd.DataFrame(records) if records else pd.DataFrame(
        columns=["formula", "band_gap", "formation_energy_per_atom", "energy_above_hull", "density", "volume"]
    )

    # Drop rows where all properties are null
    prop_cols = ["band_gap", "formation_energy_per_atom", "energy_above_hull", "density"]
    df = df[df[prop_cols].notna().any(axis=1)].reset_index(drop=True)

    logger.info("MP DataFrame: %d formulas with at least one property", len(df))
    return df
