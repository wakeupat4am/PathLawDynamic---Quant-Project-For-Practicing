"""
simulate_nongaussian.py
=======================

Stage 2 of the experiment: NON-Gaussian processes.

The Gaussian stage (see `simulate_gaussian.py`) showed that signatures recover
the mean (level 1) and the variance (level 2). But a Gaussian has NO independent
information beyond those first two moments. To test whether signatures can
recover HIGHER moments, we need processes that differ from a Gaussian only in
their third or fourth moment.

We build three processes, all with the SAME mean (0) and the SAME variance (1),
so that levels 1 and 2 cannot tell them apart. They differ only higher up:

    G : Gaussian        -> skewness 0,   excess kurtosis 0   (baseline)
    S : Skew-normal     -> skewness > 0, excess kurtosis ~0  (3rd moment)
    T : Student-t       -> skewness 0,   excess kurtosis > 0 (4th moment)

Because the first two moments are matched on purpose:
    G vs S  should switch on at LEVEL 3  (skewness / third moment).
    G vs T  should switch on at LEVEL 4  (heavy tails / kurtosis / 4th moment).
    S vs T  is a combined higher-moment difference.

Why "matched moments" matters
-----------------------------
If we did not match the variance, the processes would already differ at level 2
and we would learn nothing about higher moments. Matching the low moments is the
whole trick that isolates the third and fourth moments.

Note on the Central Limit Theorem
---------------------------------
A path here is a random walk: a running sum of independent increments. Summing
many increments makes the total look MORE Gaussian (the CLT), which dilutes the
higher moments. That is why this stage uses more paths (4000) than the Gaussian
stage -- the higher-moment signal is smaller and needs more samples to stand out.

Only NumPy is used (no SciPy), so the skew-normal and Student-t samplers are
written out explicitly and standardised by hand to mean 0 / variance 1.
"""

from __future__ import annotations

import numpy as np

# Reuse the Gaussian sampler for the baseline process G.
from simulate_gaussian import simulate_gaussian_paths


def _standardised_skewnormal_increments(
    rng: np.random.Generator, alpha: float, size: tuple
) -> np.ndarray:
    """Draw skew-normal increments standardised to mean 0 and variance 1.

    We use the classic Azzalini construction: if U0, U1 are independent standard
    normals and delta = alpha / sqrt(1 + alpha^2), then

        Z = delta * |U0| + sqrt(1 - delta^2) * U1

    is skew-normal with shape parameter `alpha`. A larger `alpha` = more skew.
    Z has a known (non-zero) mean and variance, so we subtract the mean and
    divide by the standard deviation to get increments with mean 0, variance 1,
    but non-zero SKEWNESS (a third-moment effect).
    """
    delta = alpha / np.sqrt(1.0 + alpha**2)

    u0 = np.abs(rng.standard_normal(size))   # half-normal part (creates the skew)
    u1 = rng.standard_normal(size)           # symmetric part
    z = delta * u0 + np.sqrt(1.0 - delta**2) * u1

    # Theoretical mean and variance of Z for this construction.
    mean_z = delta * np.sqrt(2.0 / np.pi)
    var_z = 1.0 - (2.0 * delta**2 / np.pi)

    # Standardise: now mean 0, variance 1, skewness preserved.
    return (z - mean_z) / np.sqrt(var_z)


def _standardised_student_t_increments(
    rng: np.random.Generator, df: float, size: tuple
) -> np.ndarray:
    """Draw Student-t increments standardised to variance 1.

    A standard Student-t with `df` degrees of freedom is symmetric (skewness 0)
    but has HEAVY TAILS -> positive excess kurtosis (a fourth-moment effect).
    Its variance is df / (df - 2), so we divide by sqrt(df / (df - 2)) to rescale
    it to variance 1 while keeping the heavy tails. `df = 5` gives finite,
    strong excess kurtosis; smaller df = even heavier tails.
    """
    t = rng.standard_t(df, size=size)
    scale = np.sqrt(df / (df - 2.0))  # so that variance becomes 1
    return t / scale


def simulate_skewnormal_paths(
    n_paths: int,
    path_length: int,
    dimension: int,
    volatility: float,
    skew_alpha: float,
    seed: int,
    mean: float = 0.0,
) -> np.ndarray:
    """Simulate skew-normal random-walk paths (mean 0, variance = volatility^2).

    Same recipe as the Gaussian sampler (draw increments, then cumulative sum),
    but the increments are skew-normal instead of normal. Returns an array of
    shape (n_paths, path_length, dimension).
    """
    rng = np.random.default_rng(seed)

    # Standardised skew-normal increments (mean 0, variance 1), then rescaled to
    # the requested mean and volatility so the first two moments are controlled.
    standardised = _standardised_skewnormal_increments(
        rng, alpha=skew_alpha, size=(n_paths, path_length, dimension)
    )
    increments = mean + volatility * standardised

    paths = np.cumsum(increments, axis=1)
    return np.ascontiguousarray(paths, dtype=np.float64)


def simulate_student_t_paths(
    n_paths: int,
    path_length: int,
    dimension: int,
    volatility: float,
    df: float,
    seed: int,
    mean: float = 0.0,
) -> np.ndarray:
    """Simulate Student-t random-walk paths (mean 0, variance = volatility^2).

    Increments are heavy-tailed Student-t, rescaled to the requested variance.
    Returns an array of shape (n_paths, path_length, dimension).
    """
    rng = np.random.default_rng(seed)

    standardised = _standardised_student_t_increments(
        rng, df=df, size=(n_paths, path_length, dimension)
    )
    increments = mean + volatility * standardised

    paths = np.cumsum(increments, axis=1)
    return np.ascontiguousarray(paths, dtype=np.float64)


def create_higher_moment_processes(
    n_paths: int = 4000,
    path_length: int = 20,
    dimension: int = 1,
    seed: int = 42,
    skew_alpha: float = 8.0,
    t_df: float = 5.0,
) -> dict:
    """Create the three matched-variance processes G, S, T for stage 2.

        G : Gaussian     (baseline; skew 0, excess kurtosis 0)
        S : Skew-normal  (tests the THIRD moment / skewness)
        T : Student-t    (tests the FOURTH moment / kurtosis / tail risk)

    All three share mean = 0 and volatility = 1, so any difference the signatures
    detect must come from a HIGHER moment.

    We deliberately use more paths (default 4000) than the Gaussian stage because
    higher-moment signals are weaker and need more samples to rise above noise.

    Returns
    -------
    dict
        Maps "G"/"S"/"T" -> dict with keys 'paths', 'mean', 'volatility',
        'label', 'description'.
    """
    mean = 0.0
    volatility = 1.0

    # Distinct seeds keep the three processes independent but reproducible.
    gaussian_paths = simulate_gaussian_paths(
        n_paths=n_paths, path_length=path_length, dimension=dimension,
        mean=mean, volatility=volatility, seed=seed + 0,
    )
    skew_paths = simulate_skewnormal_paths(
        n_paths=n_paths, path_length=path_length, dimension=dimension,
        volatility=volatility, skew_alpha=skew_alpha, seed=seed + 1, mean=mean,
    )
    t_paths = simulate_student_t_paths(
        n_paths=n_paths, path_length=path_length, dimension=dimension,
        volatility=volatility, df=t_df, seed=seed + 2, mean=mean,
    )

    return {
        "G": {
            "paths": gaussian_paths,
            "mean": mean,
            "volatility": volatility,
            "label": "Gaussian (skew 0, kurt 0)",
            "description": "baseline Gaussian: matched mean 0, variance 1",
        },
        "S": {
            "paths": skew_paths,
            "mean": mean,
            "volatility": volatility,
            "label": f"Skew-normal (alpha={skew_alpha})",
            "description": "skewed: same mean/variance, non-zero third moment",
        },
        "T": {
            "paths": t_paths,
            "mean": mean,
            "volatility": volatility,
            "label": f"Student-t (df={t_df})",
            "description": "heavy-tailed: same mean/variance, high fourth moment",
        },
    }
