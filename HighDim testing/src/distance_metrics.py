"""
distance_metrics.py
===================

Distance helpers and the table builders that compare processes -- both with
classical estimators and with signatures.

Why normalisation matters in high dimensions
--------------------------------------------
A level-``k`` signature block has ``d**k`` coordinates, and each is essentially a
``k``-th-order product of increments, so higher levels are both **much more
numerous** and **much larger in magnitude**. A raw L2 distance therefore mixes
"real geometric difference" with "this level just has bigger/more numbers". We
report three views so the reader can separate the two:

    1. RAW distance                -- straight L2, no rescaling.
    2. LEVEL-NORMALISED distance   -- divide a level's distance by sqrt(number of
                                      coordinates in that level), i.e. a per-
                                      coordinate RMS. This removes the "more
                                      coordinates" effect and makes levels of
                                      different width comparable.
    3. STANDARDISED distance (opt) -- divide each coordinate by the Process-A
                                      signature coordinate std before measuring
                                      distance. This removes the "bigger numbers"
                                      scale effect, so the distance reflects
                                      information in units of A's own variability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils import level_coord_count, level_slices


# ---------------------------------------------------------------------------
# Basic distances
# ---------------------------------------------------------------------------
def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean (L2) distance between two vectors."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b))


def frobenius_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Frobenius distance between two matrices (L2 over all entries)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b))


# ---------------------------------------------------------------------------
# Statistical distance table
# ---------------------------------------------------------------------------
def build_statistical_distance_table(
    config_name: str,
    summaries: dict[str, dict],
    comparisons: list[tuple[str, str, str]],
) -> pd.DataFrame:
    """Table of mean/covariance distances between processes (classical view).

    Parameters
    ----------
    config_name : str
        Label of the current configuration.
    summaries : dict
        ``summaries[process] = ProcessStatistics.summary()`` output.
    comparisons : list of (label, first, second)
        Which process pairs to compare.
    """
    rows = []
    for label, first, second in comparisons:
        s1, s2 = summaries[first], summaries[second]
        rows.append(
            {
                "config_name": config_name,
                "comparison": label,
                "mean_vector_l2_distance": l2_distance(
                    s1["terminal_mean"], s2["terminal_mean"]
                ),
                "covariance_frobenius_distance": frobenius_distance(
                    s1["terminal_cov"], s2["terminal_cov"]
                ),
                "increment_mean_l2_distance": l2_distance(
                    s1["increment_mean"], s2["increment_mean"]
                ),
                "increment_covariance_frobenius_distance": frobenius_distance(
                    s1["increment_cov"], s2["increment_cov"]
                ),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Signature distance tables
# ---------------------------------------------------------------------------
def _signature_interpretation(comparison: str) -> str:
    """Short interpretation string for the whole-signature distance rows."""
    if comparison == "A_vs_B":
        return "Mean shift -> expect a strong level-1 contribution."
    if comparison == "A_vs_C":
        return "Correlation blocks -> expect the signal at level 2+ (cross terms)."
    if comparison == "A_vs_D":
        return "Common market factor -> expect co-movement at level 2+ (cross terms)."
    if comparison == "B_vs_C":
        return "Mixed: mean shift (level 1) plus correlation geometry (level 2+)."
    return "L2 distance between the two expected signatures."


def _level_interpretation(comparison: str, level: int) -> str:
    """Interpretation for a single (comparison, level) row."""
    if comparison == "A_vs_B":
        if level == 1:
            return "Mean shift shows up strongly here: drift is a level-1 property."
        return "Little new info: the mean difference is already captured at level 1."
    if comparison in ("A_vs_C", "A_vs_D"):
        if level == 1:
            return "Small: both processes have zero mean, so level 1 barely differs."
        if level == 2:
            return "Cross-channel terms S^{i,j} switch on: covariance geometry appears here."
        return "Higher-order cross terms; watch the normalised value, not the raw scale."
    if comparison == "B_vs_C":
        if level == 1:
            return "Reflects B's mean shift (C has none)."
        if level == 2:
            return "Adds C's correlation-block geometry on top."
        return "Combined higher-order differences; compare via the normalised value."
    return "Level-wise L2 distance between expected signatures."


def build_signature_distance_table(
    config_name: str,
    depth: int,
    dimension: int,
    expected_sigs: dict[str, np.ndarray],
    comparisons: list[tuple[str, str, str]],
) -> pd.DataFrame:
    """Whole-signature distance table (raw and level-normalised).

    The level-normalised whole-signature distance divides the raw distance by the
    sqrt of the TOTAL number of signature coordinates, giving a per-coordinate RMS
    that is comparable across configs of different width.
    """
    total_coords = sum(level_coord_count(dimension, k) for k in range(1, depth + 1))
    rows = []
    for label, first, second in comparisons:
        raw = l2_distance(expected_sigs[first], expected_sigs[second])
        rows.append(
            {
                "config_name": config_name,
                "comparison": label,
                "depth": depth,
                "raw_signature_l2_distance": raw,
                "level_normalized_signature_distance": raw / np.sqrt(total_coords),
                "interpretation": _signature_interpretation(label),
            }
        )
    return pd.DataFrame(rows)


def build_levelwise_distance_table(
    config_name: str,
    depth: int,
    dimension: int,
    expected_sigs: dict[str, np.ndarray],
    comparisons: list[tuple[str, str, str]],
    coordinate_std: np.ndarray | None = None,
) -> pd.DataFrame:
    """Per-level signature distance table with two (optionally three) normalisations.

    Columns:
        level, number_of_level_coordinates, raw_level_l2_distance,
        normalized_level_distance (+ standardized_level_distance if a std is given).

    Parameters
    ----------
    coordinate_std : np.ndarray, optional
        Per-coordinate std (from Process A). When provided, an extra
        ``standardized_level_distance`` column divides each coordinate difference
        by A's std before measuring the level distance, removing the raw magnitude
        scale effect. Coordinates with ~0 std are skipped to avoid dividing by 0.
    """
    slices = level_slices(dimension, depth)
    rows = []
    for label, first, second in comparisons:
        exp1 = expected_sigs[first]
        exp2 = expected_sigs[second]
        for level in range(1, depth + 1):
            sl = slices[level]
            n_coords = level_coord_count(dimension, level)
            diff = exp1[sl] - exp2[sl]

            raw = float(np.linalg.norm(diff))
            # Per-coordinate RMS: removes the "this level has more coordinates" effect.
            normalized = raw / np.sqrt(n_coords)

            row = {
                "config_name": config_name,
                "comparison": label,
                "level": level,
                "number_of_level_coordinates": n_coords,
                "raw_level_l2_distance": raw,
                "normalized_level_distance": normalized,
                "interpretation": _level_interpretation(label, level),
            }

            if coordinate_std is not None:
                std_slice = coordinate_std[sl]
                # Only use coordinates whose std is meaningfully non-zero.
                usable = std_slice > 1e-12
                if np.any(usable):
                    standardized = float(
                        np.linalg.norm(diff[usable] / std_slice[usable])
                    ) / np.sqrt(np.count_nonzero(usable))
                else:
                    standardized = 0.0
                row["standardized_level_distance"] = standardized

            rows.append(row)
    return pd.DataFrame(rows)
