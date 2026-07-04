"""
simulate_highdim_gaussian.py
============================

High-dimensional synthetic Gaussian processes for the signature benchmark.

The key conceptual point (and the whole reason this stage exists)
-----------------------------------------------------------------
Each path here is **multi-dimensional**:

    X_t = ( r_t^1 , r_t^2 , ... , r_t^d )

where ``r_t^i`` is the simulated return/increment of *synthetic asset i* at time
``t``. So the **path dimension d is the number of synthetic asset/factor return
channels evolving together** -- 20 assets for the d=20 config, 10 assets for the
d=10 config. This is completely different from the *signature depth/level*, which
is the order of the iterated-integral features. In 1D there is only ever one
channel, so level 2+ signature terms mostly behave like powers of the endpoint.
Once d > 1, level 2 already contains genuine **cross-channel** terms S^{i,j}
(covariance / co-movement geometry), which is exactly what we want to probe.

How a path is built
-------------------
For every path and every time step we draw a Gaussian **increment**

    dX_t ~ N( mu * dt , Sigma * dt )

and then form the path by cumulative sum

    X_t = sum_{s <= t} dX_s      (a multivariate Gaussian random walk).

We keep ``dt = 1.0`` by default so the numbers stated in the experiment design
(e.g. a drift of 0.05, an identity covariance) are exactly the per-step increment
mean and covariance -- no hidden rescaling.

The four processes
------------------
A  Baseline independent Gaussian .... mu = 0, Sigma = I.
B  Mean-shift Gaussian ............... Sigma = I, first 25% of channels drift +0.05.
C  Correlation-block Gaussian ........ mu = 0, Sigma has correlated blocks.
D  Common-shock Gaussian (optional) .. a shared market factor + idiosyncratic noise.

To stay within a strict memory budget we NEVER build all paths at once. Instead
this module exposes ``simulate_batch`` (one batch at a time) plus a
``ProcessSpec`` describing each process, and the runner drives the batching.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Process specification
# ---------------------------------------------------------------------------
@dataclass
class ProcessSpec:
    """Everything needed to simulate one process at a given dimension.

    Attributes
    ----------
    name : str
        Short process name, e.g. "A".
    mean : np.ndarray
        Increment mean vector ``mu`` of length ``dimension``.
    cov : np.ndarray
        Increment covariance matrix ``Sigma`` of shape (dimension, dimension).
    description : str
        Human-readable interpretation, used in tables and the README.
    mean_structure : str
        Short label of the mean structure (for the config table).
    cov_structure : str
        Short label of the covariance structure (for the config table).
    kind : str
        "gaussian" for A/B/C (drawn via a Cholesky factor of ``cov``) or
        "common_shock" for D (built from a shared factor + idiosyncratic noise).
    common_shock : dict
        Extra parameters used only when ``kind == 'common_shock'``.
    """

    name: str
    mean: np.ndarray
    cov: np.ndarray
    description: str
    mean_structure: str
    cov_structure: str
    kind: str = "gaussian"
    common_shock: dict = field(default_factory=dict)


def _block_correlation_cov(dimension: int, blocks: list[tuple[int, int, float]]) -> np.ndarray:
    """Build an identity covariance with correlated diagonal blocks.

    Every channel has unit variance (diagonal = 1). Each block ``(start, stop,
    rho)`` sets the pairwise correlation between distinct channels in
    ``[start, stop)`` to ``rho`` (which, with unit variances, equals the
    covariance). Channels outside every block stay independent (identity).

    This produces a valid (positive-definite) covariance as long as each
    ``rho`` is in ``(-1/(block_size-1), 1)`` -- comfortably satisfied by the
    0.3 / 0.5 values used here.
    """
    cov = np.eye(dimension)
    for start, stop, rho in blocks:
        for i in range(start, stop):
            for j in range(start, stop):
                if i != j:
                    cov[i, j] = rho
    return cov


def build_process_specs(dimension: int, include_common_shock: bool = True) -> dict[str, ProcessSpec]:
    """Create the process specifications A, B, C (and optionally D) for a dimension.

    The mean/covariance structures follow the experiment design and are scaled to
    whatever ``dimension`` is passed (d=20 or d=10).
    """
    specs: dict[str, ProcessSpec] = {}

    # -- Process A: baseline independent Gaussian, mu = 0, Sigma = I ---------
    specs["A"] = ProcessSpec(
        name="A",
        mean=np.zeros(dimension),
        cov=np.eye(dimension),
        description="Baseline: d independent assets, no drift, unit volatility.",
        mean_structure="zero mean (all channels 0)",
        cov_structure="identity (independent, unit variance)",
    )

    # -- Process B: mean-shift Gaussian -------------------------------------
    # First 25% of channels get a positive drift of 0.05; the rest stay at 0.
    n_shift = max(1, dimension // 4)
    mean_b = np.zeros(dimension)
    mean_b[:n_shift] = 0.05
    specs["B"] = ProcessSpec(
        name="B",
        mean=mean_b,
        cov=np.eye(dimension),
        description=(
            f"Mean-shift: first {n_shift} of {dimension} assets drift +0.05, "
            "same identity covariance as A."
        ),
        mean_structure=f"first {n_shift} channels = 0.05, rest = 0",
        cov_structure="identity (independent, unit variance)",
    )

    # -- Process C: correlation-block Gaussian ------------------------------
    # For d=20: channels 0-4 rho=0.5, channels 5-9 rho=0.3, channels 10-19 free.
    # For d=10: channels 0-4 rho=0.5, channels 5-9 rho=0.3.
    blocks = []
    if dimension >= 5:
        blocks.append((0, 5, 0.5))
    if dimension >= 10:
        blocks.append((5, 10, 0.3))
    cov_c = _block_correlation_cov(dimension, blocks)
    block_desc = "; ".join(f"ch{a}-{b - 1} rho={r}" for a, b, r in blocks)
    specs["C"] = ProcessSpec(
        name="C",
        mean=np.zeros(dimension),
        cov=cov_c,
        description=(
            "Correlation blocks: sector/block co-movement between some assets, "
            f"zero drift. Blocks: {block_desc}."
        ),
        mean_structure="zero mean (all channels 0)",
        cov_structure=f"identity + correlated blocks ({block_desc})",
    )

    # -- Process D (optional): common-shock Gaussian ------------------------
    if include_common_shock:
        # Every channel = beta * shared_factor + idiosyncratic noise.
        # With beta=1, factor_vol=0.3, idio_vol=1.0 the implied covariance is
        # Sigma = idio_vol^2 * I + (beta*factor_vol)^2 * 1 1^T, i.e. a small
        # POSITIVE correlation shared by ALL channels (a market-wide factor).
        beta = 1.0
        factor_vol = 0.3
        idio_vol = 1.0
        implied_cov = (idio_vol ** 2) * np.eye(dimension) + (
            (beta * factor_vol) ** 2
        ) * np.ones((dimension, dimension))
        specs["D"] = ProcessSpec(
            name="D",
            mean=np.zeros(dimension),
            cov=implied_cov,  # stored for the statistical "truth" reference
            description=(
                "Common shock: a shared market factor (vol 0.3) plus unit "
                "idiosyncratic noise drives all assets to co-move."
            ),
            mean_structure="zero mean (all channels 0)",
            cov_structure="market factor + idiosyncratic (all channels co-move)",
            kind="common_shock",
            common_shock={"beta": beta, "factor_vol": factor_vol, "idio_vol": idio_vol},
        )

    return specs


# ---------------------------------------------------------------------------
# Batch simulation
# ---------------------------------------------------------------------------
def _simulate_increments(
    spec: ProcessSpec,
    batch_size: int,
    time_steps: int,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one batch of Gaussian increments, shape (batch, time_steps, dim)."""
    dimension = spec.mean.shape[0]

    if spec.kind == "common_shock":
        # Build the increments directly from a shared factor + idiosyncratic noise
        # so the "common market factor" interpretation is explicit in the code.
        p = spec.common_shock
        beta, factor_vol, idio_vol = p["beta"], p["factor_vol"], p["idio_vol"]
        scale = np.sqrt(dt)
        # Shared factor: one value per (path, time), broadcast across channels.
        factor = rng.normal(0.0, factor_vol * scale, size=(batch_size, time_steps, 1))
        idio = rng.normal(0.0, idio_vol * scale, size=(batch_size, time_steps, dimension))
        increments = beta * factor + idio
        # Add the (zero) drift for completeness / symmetry with the Gaussian case.
        increments += spec.mean * dt
        return increments

    # -- Standard Gaussian case (A, B, C) -----------------------------------
    # Draw standard normals and colour them with a Cholesky factor of Sigma so
    # that Cov(increment) = Sigma * dt exactly, then add the mean drift mu * dt.
    chol = np.linalg.cholesky(spec.cov)  # L with L L^T = Sigma
    z = rng.standard_normal(size=(batch_size, time_steps, dimension))
    # (batch, time, dim) @ (dim, dim)^T -> correlated increments.
    increments = z @ chol.T * np.sqrt(dt) + spec.mean * dt
    return increments


def simulate_batch(
    spec: ProcessSpec,
    batch_size: int,
    time_steps: int,
    dt: float,
    rng: np.random.Generator,
    time_aug: bool = False,
    end_time: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate ONE batch of paths and return (paths, increments).

    Parameters
    ----------
    spec : ProcessSpec
        Which process to simulate.
    batch_size : int
        Number of paths in this batch.
    time_steps : int
        Number of time steps per path.
    dt : float
        Time increment. Kept at 1.0 by default so mu/Sigma are the per-step
        increment mean/covariance directly.
    rng : np.random.Generator
        The (seeded) generator; passing the SAME generator across batches keeps
        successive batches independent while the whole run stays reproducible.
    time_aug : bool
        If True, append a monotone time channel, turning a d-dim path into a
        (d+1)-dim path. Off by default so d stays exactly 20 or 10.
    end_time : float
        The final time value for the appended time channel when ``time_aug``.

    Returns
    -------
    (paths, increments)
        ``paths`` has shape (batch, time_steps[, +1 if time_aug], dim) and is the
        cumulative sum of ``increments``. ``increments`` is returned separately so
        the statistical estimators can measure increment moments without
        re-differencing.
    """
    increments = _simulate_increments(spec, batch_size, time_steps, dt, rng)

    # Cumulative sum along the time axis turns increments into the random-walk path.
    paths = np.cumsum(increments, axis=1)

    if time_aug:
        # Append a deterministic, strictly increasing time channel in [0, end_time].
        # This changes the EFFECTIVE signature dimension to d+1 (reported by the
        # runner). We add it to the path only, not to the increments used for the
        # statistical estimators, so the asset-return statistics stay pure.
        t = np.linspace(end_time / time_steps, end_time, time_steps)
        t = np.broadcast_to(t.reshape(1, time_steps, 1), (batch_size, time_steps, 1))
        paths = np.concatenate([paths, t], axis=2)

    # pysiglib prefers C-contiguous float64 arrays (else it warns and copies).
    paths = np.ascontiguousarray(paths, dtype=np.float64)
    increments = np.ascontiguousarray(increments, dtype=np.float64)
    return paths, increments
