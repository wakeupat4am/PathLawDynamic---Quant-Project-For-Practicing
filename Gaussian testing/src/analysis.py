"""
analysis.py
===========

This module compares processes using their signatures and also computes plain
empirical moments (mean, variance, covariance) directly from the data.

Two views of the same question:
    1. `compute_sample_moments` -> the "ground truth" moments of the raw data.
    2. `compute_signature_distances` -> how far apart the signature
       "fingerprints" of two processes are, level by level.

If signatures are doing their job, the signature distances should reflect the
moment differences we deliberately built into processes A, B and C.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return the Euclidean (L2) distance between two vectors.

    The L2 distance is the straight-line distance:
        sqrt( sum_i (a_i - b_i)^2 ).
    We use it to measure how different two expected signatures are. A larger
    distance means the two processes look more different to the signature.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b))


def _interpretation(comparison: str, level: int) -> str:
    """Return a short, beginner-friendly explanation for one table row.

    The text depends on WHICH processes are being compared and on the
    truncation level, so that a beginner reading the CSV understands what the
    number is supposed to mean.
    """
    if comparison == "A_vs_B":
        # A and B differ only in their MEAN (a first-order property).
        if level == 1:
            return "Mean difference already appears at level 1 (first-order / drift info)."
        return "Mean is a level-1 effect and is already captured by level 1."

    if comparison == "A_vs_C":
        # A and C differ only in their VARIANCE (a second-order property).
        if level == 1:
            return "Almost zero: with equal means, level 1 barely separates the two variances."
        if level == 2:
            return "Big jump: variance is a level-2 (second-order) effect, so it switches on here."
        return (
            "Variance is already captured at level 2. Raw distance still grows only because "
            "higher-order signature terms are numerically larger (a scale effect, see README)."
        )

    if comparison == "B_vs_C":
        # B and C differ in BOTH mean and variance.
        if level == 1:
            return "Level 1 mainly reflects the mean gap between B and C."
        if level == 2:
            return "Level 2 adds the variance gap on top of the mean gap."
        return (
            "Both differences are already captured by level 2; higher levels mostly add "
            "numerical scale, not new information (see README)."
        )

    return "Distance between the two expected signatures at this level."


def interpretation_higher_moment(comparison: str, level: int) -> str:
    """Beginner-friendly interpretation for the NON-Gaussian (stage 2) test.

    Here the processes are G (Gaussian), S (skew-normal) and T (Student-t), all
    with matched mean and variance. So differences should only appear at HIGHER
    levels: level 3 for skewness, level 4 for kurtosis.
    """
    if comparison == "G_vs_S":
        # G and S differ only in the THIRD moment (skewness).
        if level <= 2:
            return "Near zero: mean and variance are matched, so levels 1-2 cannot separate them."
        if level == 3:
            return "Jump expected here: skewness is a third-moment (level-3) effect."
        return "Skewness already captured at level 3; higher levels add mostly numerical scale."

    if comparison == "G_vs_T":
        # G and T differ only in the FOURTH moment (kurtosis / heavy tails).
        if level <= 2:
            return "Near zero: matched mean and variance keep levels 1-2 indistinguishable."
        if level == 3:
            return "Still small: Student-t is symmetric, so it has (almost) no third moment."
        if level == 4:
            return "Jump expected here: heavy tails / kurtosis is a fourth-moment (level-4) effect."
        return "Kurtosis already captured at level 4; higher levels add mostly numerical scale."

    if comparison == "S_vs_T":
        # S and T differ in BOTH the third and fourth moments.
        if level <= 2:
            return "Near zero: both share the same mean and variance."
        if level == 3:
            return "Skewness of S starts to separate them at level 3."
        if level == 4:
            return "Kurtosis of T adds further separation at level 4."
        return "Combined higher-moment difference already captured by level 4."

    return "Distance between the two expected signatures at this level."


def compute_signature_distances(
    signature_results: dict,
    comparisons: list | None = None,
    interpretation_fn=None,
) -> pd.DataFrame:
    """Build a table of L2 distances between expected signatures.

    For every truncation level we compare a set of process pairs using the L2
    distance between their expected signatures.

    Parameters
    ----------
    signature_results : dict
        Output of `compute_signatures_for_levels`, i.e.
        signature_results[name][level]["expected"] is available.
    comparisons : list of (label, first, second) tuples, optional
        Which process pairs to compare. Defaults to the Gaussian-stage pairs
        A_vs_B, A_vs_C, B_vs_C.
    interpretation_fn : callable, optional
        Function (comparison_label, level) -> str used to fill the
        "interpretation" column. Defaults to the Gaussian-stage interpretations.

    Returns
    -------
    pd.DataFrame
        Columns: level, comparison, l2_distance_between_expected_signatures,
        interpretation. One row per (level, comparison).
    """
    # Default to the original Gaussian-stage comparisons for backward compatibility.
    if comparisons is None:
        comparisons = [
            ("A_vs_B", "A", "B"),
            ("A_vs_C", "A", "C"),
            ("B_vs_C", "B", "C"),
        ]
    if interpretation_fn is None:
        interpretation_fn = _interpretation

    # Read the levels from the first process referenced in `comparisons`.
    any_process = comparisons[0][1]
    levels = sorted(signature_results[any_process].keys())

    rows = []
    for level in levels:
        for label, first, second in comparisons:
            # The expected (averaged) signatures for the two processes.
            expected_first = signature_results[first][level]["expected"]
            expected_second = signature_results[second][level]["expected"]

            distance = l2_distance(expected_first, expected_second)

            rows.append(
                {
                    "level": level,
                    "comparison": label,
                    "l2_distance_between_expected_signatures": distance,
                    "interpretation": interpretation_fn(label, level),
                }
            )

    return pd.DataFrame(rows)


def compute_sample_moments(paths: np.ndarray) -> dict:
    """Compute the empirical first two moments of a process's increments.

    We measure moments on the INCREMENTS (the per-step changes), not on the
    cumulative path. This is because we generated the process by choosing the
    mean and volatility of the increments, so the increments are where the
    "true" moments live and are easiest to sanity-check.

    Parameters
    ----------
    paths : np.ndarray
        Paths of shape (n_paths, path_length, dimension).

    Returns
    -------
    dict
        {
          "empirical_mean_of_increments":      per-channel mean (np.ndarray),
          "empirical_variance_of_increments":  per-channel variance (np.ndarray),
          "empirical_covariance_of_increments": covariance matrix if dimension>1,
                                                 else None,
        }
    """
    # Recover the increments by differencing along the time axis.
    # np.diff gives x[t] - x[t-1]. We prepend the first time step (which equals
    # the first increment, since the path started from that increment) so that
    # the number of increments matches path_length.
    first_step = paths[:, :1, :]                      # shape (n_paths, 1, dim)
    later_steps = np.diff(paths, axis=1)              # shape (n_paths, len-1, dim)
    increments = np.concatenate([first_step, later_steps], axis=1)

    # Flatten path and time together so every single increment is one sample.
    # Result shape: (n_paths * path_length, dimension).
    dimension = increments.shape[2]
    flat_increments = increments.reshape(-1, dimension)

    # First moment (mean) and second central moment (variance), per channel.
    empirical_mean = flat_increments.mean(axis=0)
    empirical_variance = flat_increments.var(axis=0)

    # Third and fourth STANDARDISED moments, computed by hand (no SciPy needed):
    #   skewness        = E[(x-mu)^3] / sigma^3         (0 for a Gaussian)
    #   excess kurtosis = E[(x-mu)^4] / sigma^4  - 3    (0 for a Gaussian)
    # These are exactly the higher moments that stage 2 (skew / heavy tails)
    # is designed to probe. `sigma` is the standard deviation per channel.
    centered = flat_increments - empirical_mean
    sigma = np.sqrt(empirical_variance)
    empirical_skewness = (centered**3).mean(axis=0) / sigma**3
    empirical_excess_kurtosis = (centered**4).mean(axis=0) / sigma**4 - 3.0

    # Covariance only makes sense with more than one channel.
    if dimension > 1:
        # rowvar=False => each column is a variable (channel).
        empirical_covariance = np.cov(flat_increments, rowvar=False)
    else:
        empirical_covariance = None

    return {
        "empirical_mean_of_increments": empirical_mean,
        "empirical_variance_of_increments": empirical_variance,
        "empirical_skewness_of_increments": empirical_skewness,
        "empirical_excess_kurtosis_of_increments": empirical_excess_kurtosis,
        "empirical_covariance_of_increments": empirical_covariance,
    }
