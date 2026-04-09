"""
Data loading and parsing for the Kononova et al. solid-state synthesis dataset.

Dataset reference:
    Kononova et al., Scientific Data 6, 203 (2019).
    https://doi.org/10.1038/s41597-019-0224-1
"""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Load the raw JSON dataset and return the list of reaction records."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    logger.info("Loading dataset from %s", path)
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    if isinstance(raw, dict) and "reactions" in raw:
        reactions = raw["reactions"]
    elif isinstance(raw, list):
        reactions = raw
    else:
        raise ValueError("Unexpected dataset format.")

    logger.info("Loaded %d reactions", len(reactions))
    return reactions


def _to_celsius(value: float, unit: str) -> float | None:
    """Convert a temperature value to degrees Celsius."""
    unit = (unit or "").strip().lower()
    if unit in ("°c", "c", "deg c", "celsius", ""):
        return value
    if unit in ("k", "kelvin"):
        return value - 273.15
    if unit in ("°f", "f", "fahrenheit"):
        return (value - 32.0) * 5.0 / 9.0
    return None


def _to_hours(value: float, unit: str) -> float | None:
    """Convert a time value to hours."""
    unit = (unit or "").strip().lower()
    if unit in ("h", "hr", "hrs", "hour", "hours", ""):
        return value
    if unit in ("min", "mins", "minute", "minutes"):
        return value / 60.0
    if unit in ("s", "sec", "second", "seconds"):
        return value / 3600.0
    if unit in ("d", "day", "days"):
        return value * 24.0
    return None


def parse_reactions(reactions: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Parse the raw reaction records into a structured DataFrame.

    Returns one row per reaction with fields:
        doi, target_formula, paragraph_string, n_precursors,
        precursor_formulas, precursor_elements, n_operations,
        temperatures_C (list), times_h (list), synthesis_type
    """
    records = []
    for rxn in reactions:
        target = rxn.get("target") or {}
        formula = target.get("material_formula", "")
        precursors = rxn.get("precursors") or []

        # Collect unique elements across all precursors
        elements: set[str] = set()
        precursor_formulas: list[str] = []
        for prec in precursors:
            pf = prec.get("material_formula", "")
            if pf:
                precursor_formulas.append(pf)
            for comp in prec.get("composition") or []:
                elements.update((comp.get("elements") or {}).keys())

        # Parse temperatures and times from operations
        operations = rxn.get("operations") or []
        temperatures_C: list[float] = []
        times_h: list[float] = []

        for op in operations:
            cond = op.get("conditions") or {}

            for t_entry in cond.get("heating_temperature") or []:
                raw_vals = t_entry.get("values") or []
                unit = t_entry.get("units", "°C")
                min_v = t_entry.get("min_value")
                max_v = t_entry.get("max_value")

                if raw_vals:
                    vals = raw_vals
                elif min_v is not None and max_v is not None:
                    vals = [(min_v + max_v) / 2.0]
                else:
                    continue

                for v in vals:
                    if v is not None:
                        c = _to_celsius(float(v), unit)
                        if c is not None and -100 < c < 4000:
                            temperatures_C.append(c)

            for time_entry in cond.get("heating_time") or []:
                raw_vals = time_entry.get("values") or []
                unit = time_entry.get("units", "h")
                min_v = time_entry.get("min_value")
                max_v = time_entry.get("max_value")

                if raw_vals:
                    vals = raw_vals
                elif min_v is not None and max_v is not None:
                    vals = [(min_v + max_v) / 2.0]
                else:
                    continue

                for v in vals:
                    if v is not None:
                        h = _to_hours(float(v), unit)
                        if h is not None and 0 < h < 10000:
                            times_h.append(h)

        records.append(
            {
                "doi": rxn.get("doi", ""),
                "target_formula": formula,
                "paragraph_string": rxn.get("paragraph_string", ""),
                "synthesis_type": rxn.get("synthesis_type", ""),
                "n_precursors": len(precursor_formulas),
                "precursor_formulas": precursor_formulas,
                "precursor_elements": sorted(elements),
                "n_operations": len(operations),
                "temperatures_C": temperatures_C,
                "times_h": times_h,
            }
        )

    df = pd.DataFrame(records)
    logger.info("Parsed %d reaction records", len(df))
    return df
