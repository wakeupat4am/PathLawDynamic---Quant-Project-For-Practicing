"""
simulate_gaussian.py
====================

This module creates SYNTHETIC Gaussian processes that we use to test path
signatures. We deliberately do NOT touch the real SPY/QQQ/TLT finance data here
-- the whole point of this first experiment is to work with data whose "true"
mean and variance we already know, so we can check whether signatures recover
them.

Key idea (in plain words)
-------------------------
A "path" is just a line that moves over time (like a stock price chart).
We build a path in two steps:

    1. Draw random *increments* (the small step taken at each time point).
       Here every increment is Gaussian (normally distributed) with a chosen
       mean and volatility.
    2. Add the increments up with a cumulative sum. This turns the steps into
       a running total -- the path itself (this is a "random walk").

Because the increments are Gaussian, the resulting process is a discrete
Gaussian process, and a Gaussian is *fully described* by only its first two
moments: the mean and the variance. That is exactly why it is the perfect
starting point for the professor's question.
"""

from __future__ import annotations

import numpy as np


def simulate_gaussian_paths(
    n_paths: int,
    path_length: int,
    dimension: int,
    mean: float,
    volatility: float,
    seed: int,
) -> np.ndarray:
    """Simulate a batch of Gaussian random-walk paths.

    Parameters
    ----------
    n_paths : int
        How many independent example paths to generate.
    path_length : int
        Number of time steps in each path (how "long" each path is).
    dimension : int
        Number of channels/coordinates. 1 = a single time series.
    mean : float
        Average size of each Gaussian increment (the "drift").
    volatility : float
        Standard deviation of each Gaussian increment (how noisy the steps are).
    seed : int
        Random seed. Fixing this makes the experiment fully reproducible.

    Returns
    -------
    np.ndarray
        Array of shape (n_paths, path_length, dimension) containing the paths.
    """
    # A dedicated random generator seeded with `seed` => deterministic output.
    rng = np.random.default_rng(seed)

    # Step 1: draw the random increments.
    # `normal` gives Gaussian numbers with the requested mean and volatility.
    # Shape (n_paths, path_length, dimension) so every path/time/channel gets
    # its own independent increment.
    increments = rng.normal(
        loc=mean,
        scale=volatility,
        size=(n_paths, path_length, dimension),
    )

    # Step 2: turn increments into a path by cumulative summation along time.
    # axis=1 is the time axis, so each path becomes a running total of its steps.
    paths = np.cumsum(increments, axis=1)

    # Make the array C-contiguous. pysiglib prefers contiguous arrays and will
    # otherwise print a "non-contiguous array" warning and copy internally.
    return np.ascontiguousarray(paths, dtype=np.float64)


def create_gaussian_test_processes(
    n_paths: int = 1000,
    path_length: int = 20,
    dimension: int = 1,
    seed: int = 42,
) -> dict:
    """Create the three Gaussian processes used in this experiment.

    We build three processes that differ in a controlled way so that each
    comparison isolates ONE property:

        Process A: mean = 0.00, volatility = 1.0   (baseline)
        Process B: mean = 0.05, volatility = 1.0   (only the MEAN changed)
        Process C: mean = 0.00, volatility = 2.0   (only the VOLATILITY changed)

    Because only one knob is turned at a time, we can attribute any difference
    in signatures to that specific knob:

        A vs B  ->  detects a MEAN difference.
        A vs C  ->  detects a VARIANCE / VOLATILITY difference.
        B vs C  ->  a combined (mean + variance) difference.

    Each process gets its own seed (derived from the base seed) so that the
    three processes are independent but the whole experiment stays reproducible.

    Returns
    -------
    dict
        Maps process name ("A", "B", "C") to a dict with keys:
        'paths', 'mean', 'volatility', 'description'.
    """
    # (mean, volatility, human-readable description) for each process.
    specs = {
        "A": (0.00, 1.0, "baseline: mean 0, volatility 1"),
        "B": (0.05, 1.0, "mean shifted to 0.05 (tests mean recovery)"),
        "C": (0.00, 2.0, "volatility doubled to 2 (tests variance recovery)"),
    }

    processes = {}
    for offset, (name, (mean, volatility, description)) in enumerate(specs.items()):
        # Give each process a distinct seed so they are not accidentally
        # correlated, while keeping everything deterministic.
        process_seed = seed + offset

        paths = simulate_gaussian_paths(
            n_paths=n_paths,
            path_length=path_length,
            dimension=dimension,
            mean=mean,
            volatility=volatility,
            seed=process_seed,
        )

        processes[name] = {
            "paths": paths,
            "mean": mean,
            "volatility": volatility,
            "description": description,
        }

    return processes
