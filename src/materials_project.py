"""
Optional Materials Project integration.

Attempts to retrieve energy above hull (E_hull, eV/atom) for target formulas
via the Materials Project REST API (v2).  Requires a valid MP_API_KEY environment
variable.  If no key is present, this module returns an empty Series and logs a
warning — all downstream analyses degrade gracefully.

API reference: https://next-gen.materialsproject.org/api
"""

import logging
import os
import time
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_MP_BASE = "https://api.materialsproject.org"
_DEFAULT_FIELDS = "formula_pretty,energy_above_hull,band_gap,formation_energy_per_atom"
_RATE_LIMIT_SLEEP = 0.5  # seconds between requests


def _get_api_key() -> Optional[str]:
    return os.environ.get("MP_API_KEY") or os.environ.get("MAPI_KEY")


def query_formula(formula: str, api_key: str, session: requests.Session) -> dict | None:
    """Query MP for the lowest-energy entry matching *formula*."""
    url = f"{_MP_BASE}/materials/summary/"
    params = {
        "formula": formula,
        "fields": _DEFAULT_FIELDS,
        "_limit": 1,
    }
    headers = {"X-API-KEY": api_key}
    try:
        resp = session.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if data:
            return data[0]
    except requests.RequestException as exc:
        logger.debug("MP query failed for %s: %s", formula, exc)
    return None


def fetch_mp_properties(
    formulas: list[str],
    max_queries: int = 2000,
) -> pd.DataFrame:
    """
    Fetch Materials Project properties for a list of target formulas.

    Parameters
    ----------
    formulas : list[str]
        Unique target material formulas.
    max_queries : int
        Cap on the number of API calls (rate-limit protection).

    Returns
    -------
    pd.DataFrame
        Columns: formula, energy_above_hull, band_gap, formation_energy_per_atom.
        Empty DataFrame if no API key is available.
    """
    api_key = _get_api_key()
    if not api_key:
        logger.warning(
            "MP_API_KEY not set — skipping Materials Project integration. "
            "Set the environment variable to enable E_hull retrieval."
        )
        return pd.DataFrame(
            columns=["formula", "energy_above_hull", "band_gap", "formation_energy_per_atom"]
        )

    records = []
    seen: set[str] = set()
    session = requests.Session()

    for formula in formulas:
        if formula in seen or not formula:
            continue
        seen.add(formula)
        if len(seen) > max_queries:
            logger.warning("MP query limit (%d) reached.", max_queries)
            break

        entry = query_formula(formula, api_key, session)
        if entry:
            records.append(
                {
                    "formula": formula,
                    "energy_above_hull": entry.get("energy_above_hull"),
                    "band_gap": entry.get("band_gap"),
                    "formation_energy_per_atom": entry.get("formation_energy_per_atom"),
                }
            )
        time.sleep(_RATE_LIMIT_SLEEP)

    df = pd.DataFrame(records) if records else pd.DataFrame(
        columns=["formula", "energy_above_hull", "band_gap", "formation_energy_per_atom"]
    )
    logger.info("Retrieved MP data for %d formulas", len(df))
    return df
