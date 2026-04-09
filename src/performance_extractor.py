"""
Performance metric extraction from synthesis paragraph text.

Strategy: regex-based pattern matching for quantitative performance metrics
reported in solid-state synthesis literature.  All patterns are documented
and ordered by specificity.  No external API or model is required.

Metrics extracted (in priority order):
    1. Discharge / specific capacity      [mAh/g]
    2. Ionic / electrical conductivity    [S/cm]
    3. BET surface area                   [m²/g]
    4. Sintered bulk density              [g/cm³]
    5. Relative sintered density          [% theoretical]
    6. Resistivity                        [Ω·cm]
    7. Energy density                     [Wh/kg]

For cross-metric comparison, each value is log₁₀-transformed and z-score
normalised within its metric class.

Note on band-gap / eV: the Kononova dataset paragraphs that contain eV values
predominantly refer to XPS binding-energy calibration (C1s at 284.6 eV) rather
than band-gap measurements; this class is excluded to avoid systematic noise.
"""

import logging
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_NUMBER = r"[-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?"

# ---------------------------------------------------------------------------
# XPS exclusion filter — skip paragraphs that only mention eV in a calibration
# context so we do not confuse binding energies with band gaps.
# ---------------------------------------------------------------------------
_XPS_FILTER = re.compile(
    r"(?:calibrat|C\s*1s|binding\s+energ|photoelectron|XPS|ESCA)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Pattern registry
# Each entry: (metric_name, unit_label, pattern)
# The first capture group must yield the numeric value.
# ---------------------------------------------------------------------------
_PATTERNS: list[tuple[str, str, re.Pattern]] = [
    # ------------------------------------------------------------------ capacity
    (
        "capacity",
        "mAh/g",
        re.compile(
            rf"({_NUMBER})\s*(?:m[Aa]\s*[Hh]|mAh)\s*[·/]?\s*g(?:[-−]1|\^[-−]?1)?",
            re.IGNORECASE,
        ),
    ),
    (
        "capacity",
        "mAh/g",
        re.compile(
            rf"({_NUMBER})\s*(?:m[Aa]\s*[Hh]|mAh)\s*g[-−]1",
            re.IGNORECASE,
        ),
    ),
    # -------------------------------------------------------------- conductivity
    (
        "conductivity",
        "S/cm",
        re.compile(
            rf"({_NUMBER})\s*(?:[×xX\*]\s*10\s*[-−]\s*\d+\s*)?S\s*(?:cm|·\s*cm)[-−]?1",
            re.IGNORECASE,
        ),
    ),
    (
        "conductivity",
        "S/cm",
        re.compile(
            rf"({_NUMBER})\s*S\s*/\s*cm\b",
            re.IGNORECASE,
        ),
    ),
    # ---------------------------------------------------------------- surface area
    (
        "surface_area",
        "m2/g",
        re.compile(
            rf"({_NUMBER})\s*m[²2]\s*g[-−]?1",
            re.IGNORECASE,
        ),
    ),
    (
        "surface_area",
        "m2/g",
        re.compile(
            rf"({_NUMBER})\s*m[²2]\s*/\s*g",
            re.IGNORECASE,
        ),
    ),
    (
        "surface_area",
        "m2/g",
        re.compile(
            rf"({_NUMBER})\s*m\s*2\s*g\s*[-−]?\s*1",
            re.IGNORECASE,
        ),
    ),
    # -------------------------------------------------------------------- density
    (
        "density",
        "g/cm3",
        re.compile(
            rf"(?:sintered?|measured|theoretical|bulk|relative)\s+density\D{{0,30}}?({_NUMBER})\s*g\s*(?:cm[-−]?3|/\s*cm[³3])",
            re.IGNORECASE,
        ),
    ),
    (
        "density",
        "g/cm3",
        re.compile(
            rf"density\s+(?:of|=|:)\s*({_NUMBER})\s*g\s*(?:cm[-−]?3|/\s*cm[³3])",
            re.IGNORECASE,
        ),
    ),
    (
        "density",
        "g/cm3",
        re.compile(
            rf"({_NUMBER})\s*g\s*/\s*cm[³3](?!\s*solution)",
            re.IGNORECASE,
        ),
    ),
    (
        "density",
        "g/cm3",
        re.compile(
            rf"({_NUMBER})\s*g\s*cm[-−]?3\b",
            re.IGNORECASE,
        ),
    ),
    # ------------------------------------------------------------ relative density
    (
        "rel_density",
        "% theoretical",
        re.compile(
            rf"({_NUMBER})\s*%\s+(?:of\s+)?theoretical\s+density",
            re.IGNORECASE,
        ),
    ),
    (
        "rel_density",
        "% theoretical",
        re.compile(
            rf"relative\s+density\s+(?:of|=|:)\s*({_NUMBER})\s*%",
            re.IGNORECASE,
        ),
    ),
    # ---------------------------------------------------------------- resistivity
    (
        "resistivity",
        "Ω·cm",
        re.compile(
            rf"({_NUMBER})\s*(?:[×xX\*]\s*10\s*[-−]?\s*\d+\s*)?[ΩΩ]\s*cm",
            re.IGNORECASE,
        ),
    ),
    # ---------------------------------------------------------------- energy density
    (
        "energy_density",
        "Wh/kg",
        re.compile(
            rf"({_NUMBER})\s*W\s*h\s*/\s*(?:kg|g)",
            re.IGNORECASE,
        ),
    ),
    (
        "energy_density",
        "Wh/kg",
        re.compile(
            rf"({_NUMBER})\s*W\s*h\s*(?:kg|g)[-−]?1",
            re.IGNORECASE,
        ),
    ),
]

# Scientific-notation multiplier pattern  e.g. "3.2 × 10⁻³ S/cm"
_SCI_PREFIX = re.compile(
    rf"({_NUMBER})\s*[×xX\*]\s*10\s*\^?\s*([-−+]?\d+)",
    re.IGNORECASE,
)


def _resolve_sci(text: str, pos: int) -> float | None:
    """Look backwards up to 35 characters for a sci-notation prefix."""
    window = text[max(0, pos - 35): pos]
    m = _SCI_PREFIX.search(window)
    if m:
        mantissa = float(m.group(1))
        exp_str = m.group(2).replace("−", "-")
        return mantissa * (10 ** int(exp_str))
    return None


def extract_performance(paragraph: str) -> dict | None:
    """
    Extract the primary quantitative performance metric from a paragraph.

    Returns ``{"metric_name": str, "value": float, "unit": str}`` or ``None``.
    """
    if not paragraph:
        return None

    for metric_name, unit_label, pattern in _PATTERNS:
        for match in pattern.finditer(paragraph):
            try:
                raw_val = float(match.group(1))
            except (ValueError, IndexError):
                continue

            # Resolve preceding scientific-notation multiplier
            sci = _resolve_sci(paragraph, match.start())
            value = sci if sci is not None else raw_val

            if not (np.isfinite(value) and value > 0):
                continue

            # Sanity bounds per metric class
            if metric_name == "capacity" and not (1 < value < 5000):
                continue
            if metric_name == "conductivity" and not (1e-15 < value < 1e6):
                continue
            if metric_name == "surface_area" and not (0.01 < value < 5000):
                continue
            if metric_name == "density" and not (0.5 < value < 25):
                continue
            if metric_name == "rel_density" and not (50 < value <= 100):
                continue
            if metric_name == "resistivity" and not (1e-8 < value < 1e10):
                continue
            if metric_name == "energy_density" and not (1 < value < 50000):
                continue

            return {"metric_name": metric_name, "value": value, "unit": unit_label}

    return None


def extract_all_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply ``extract_performance`` to every row and add columns:
    ``metric_name``, ``raw_value``, ``unit``.
    """
    df = df.copy()
    results = df["paragraph_string"].apply(extract_performance)

    df["metric_name"] = results.apply(lambda r: r["metric_name"] if r else None)
    df["raw_value"] = results.apply(lambda r: float(r["value"]) if r else np.nan)
    df["unit"] = results.apply(lambda r: r["unit"] if r else None)

    n_extracted = df["metric_name"].notna().sum()
    logger.info(
        "Performance metric extracted for %d / %d records (%.1f%%)",
        n_extracted,
        len(df),
        100.0 * n_extracted / len(df),
    )
    logger.info("Metric breakdown:\n%s", df["metric_name"].value_counts().to_string())
    return df


def normalise_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Within each metric class: log₁₀-transform the raw value, then z-score
    normalise.  Stores the result in ``perf_norm``.  Also assigns a numeric
    ``metric_class_id`` for optional subgroup encodings.
    """
    df = df.copy()
    df["perf_norm"] = np.nan
    df["metric_class_id"] = np.nan

    classes = sorted(df["metric_name"].dropna().unique())
    class_map = {name: i for i, name in enumerate(classes)}

    for metric, gidx in df.groupby("metric_name").groups.items():
        vals = df.loc[gidx, "raw_value"]
        pos_mask = vals > 0
        if pos_mask.sum() < 3:
            continue
        log_vals = np.log10(vals[pos_mask])
        z = (log_vals - log_vals.mean()) / (log_vals.std() + 1e-12)
        df.loc[gidx[pos_mask], "perf_norm"] = z.values
        df.loc[gidx, "metric_class_id"] = class_map[metric]

    n_norm = df["perf_norm"].notna().sum()
    logger.info("Normalised performance available for %d records", n_norm)
    return df
