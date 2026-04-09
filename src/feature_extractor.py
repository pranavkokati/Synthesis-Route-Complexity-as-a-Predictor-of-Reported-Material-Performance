"""
Synthesis complexity feature computation.

Five operationally defined complexity features (Kononova dataset):
    (a) precursor_count       — number of distinct precursor compounds
    (b) max_temperature_C     — maximum sintering/calcination temperature (°C)
    (c) total_time_h          — cumulative heating time (hours)
    (d) n_steps               — total number of described processing operations
    (e) precursor_diversity   — number of distinct elements across all precursors

A composite complexity index (z-score mean of all five features) is also computed.
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import zscore

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "precursor_count",
    "max_temperature_C",
    "total_time_h",
    "n_steps",
    "precursor_diversity",
]

FEATURE_LABELS = {
    "precursor_count": "Precursor Count",
    "max_temperature_C": "Max Temperature (°C)",
    "total_time_h": "Total Heating Time (h)",
    "n_steps": "Number of Steps",
    "precursor_diversity": "Precursor Diversity (# elements)",
    "complexity_index": "Composite Complexity Index",
}


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive the five complexity features from the parsed DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``data_loader.parse_reactions``.

    Returns
    -------
    pd.DataFrame
        Original columns plus the five complexity feature columns
        and a composite ``complexity_index``.
    """
    df = df.copy()

    df["precursor_count"] = df["n_precursors"].astype(float)
    df["precursor_diversity"] = df["precursor_elements"].apply(lambda x: float(len(x)))
    df["n_steps"] = df["n_operations"].astype(float)

    df["max_temperature_C"] = df["temperatures_C"].apply(
        lambda temps: float(max(temps)) if temps else np.nan
    )
    df["total_time_h"] = df["times_h"].apply(
        lambda times: float(sum(times)) if times else np.nan
    )

    # Drop rows with no precursors (uninformative)
    initial = len(df)
    df = df[df["precursor_count"] > 0].reset_index(drop=True)
    logger.info(
        "Dropped %d records with zero precursors; %d remain",
        initial - len(df),
        len(df),
    )

    # Composite complexity index: mean of per-feature z-scores
    valid_mask = df[FEATURE_COLS].notna().all(axis=1)
    z = df.loc[valid_mask, FEATURE_COLS].apply(zscore, nan_policy="omit")
    df.loc[valid_mask, "complexity_index"] = z.mean(axis=1)

    logger.info(
        "Feature coverage — max_temp: %d, total_time: %d, complexity_index: %d",
        df["max_temperature_C"].notna().sum(),
        df["total_time_h"].notna().sum(),
        df["complexity_index"].notna().sum(),
    )
    return df


def get_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for all five complexity features."""
    return df[FEATURE_COLS].describe().T.rename(index=FEATURE_LABELS)
