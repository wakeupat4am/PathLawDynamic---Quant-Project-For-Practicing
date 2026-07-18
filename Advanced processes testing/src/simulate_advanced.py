"""
simulate_advanced.py
====================

Stage 3 of the project: **advanced financial processes** whose paths break the
two assumptions the earlier stages relied on -- constant volatility and
independent (Markov) Gaussian increments.

The six processes
-----------------
All six are 1-D log-price paths ``log S_t`` (a single synthetic asset) built by
cumulatively summing per-step log-returns. They are deliberately calibrated to
share the SAME baseline diffusive volatility (``sigma = 0.2`` annualised) so that
levels 1-2 of the signature are broadly comparable and the interesting
differences live in the *higher moments* and the *path memory*:

    BS    Black-Scholes GBM ............ constant vol, i.i.d. Gaussian returns.
                                         The null model / baseline.
    HEST  Heston stochastic volatility . variance follows a mean-reverting CIR
                                         process, correlated with the price
                                         (leverage). No jumps.
    MERT  Merton jump-diffusion ........ constant-vol diffusion PLUS compound-
                                         Poisson lognormal jumps (skew + fat tails).
    BATES Bates (combined) ............. Heston stochastic vol AND Merton jumps
                                         together -- the full jump-diffusion.
    FBM   Fractional Brownian motion ... non-Markov: increments are fractional
                                         Gaussian noise with Hurst H (long-range
                                         autocorrelation). Endpoint variance is
                                         matched to BS on purpose.
    RVOL  Rough Bergomi volatility ..... "rough" (very autocorrelated / H<0.5)
                                         volatility driving the price, with
                                         leverage. The rough-vol model.

Mapping to the request
----------------------
    stochastic vol (Heston) .............. HEST
    jumps (Merton) ....................... MERT
    jump-diffusion (combined) ............ BATES
    non-Markov (fractional BM) ........... FBM
    autocorrelated / rough volatility .... RVOL

Batched, streaming API
----------------------
Exactly like Stage 2, we never build all paths at once. Each process is
described by an :class:`AdvancedProcessSpec`, and :func:`simulate_batch` returns
ONE batch of ``(paths, increments)`` at a time so the runner can stream toward
large ``num_paths`` at a flat memory cost. ``paths`` is the cumulative-sum log-
price (shape ``(batch, time_steps[, +1 if time_aug], 1)``) and ``increments`` is
the per-step log-returns (shape ``(batch, time_steps, 1)``), matching the shapes
the signature and statistical machinery in ``HighDim testing/src`` expect.

Only NumPy is used; the schemes (Euler full-truncation for CIR, compound-Poisson
jumps, Cholesky-factored fractional Gaussian noise) are written out explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Process specification
# ---------------------------------------------------------------------------
@dataclass
class AdvancedProcessSpec:
    """Everything needed to simulate one advanced 1-D log-price process.

    Attributes
    ----------
    name : str
        Short process code, e.g. ``"HEST"``.
    kind : str
        Which simulator branch to use: one of ``"bs"``, ``"heston"``,
        ``"merton"``, ``"bates"``, ``"fbm"``, ``"rough_bergomi"``.
    params : dict
        Model parameters (see each simulator for the keys it reads).
    description : str
        Human-readable interpretation, used in tables and the README.
    family : str
        Short label of the modelling feature (for the config table), e.g.
        ``"stochastic volatility"`` or ``"jumps"``.
    _cache : dict
        Internal scratch space for quantities that are expensive to build once
        and can be reused across batches (e.g. a Cholesky factor). Not part of
        the public description.
    """

    name: str
    kind: str
    params: dict
    description: str
    family: str
    _cache: dict = field(default_factory=dict, repr=False)

    # Convenience labels so the Stage-3 runner can fill the same config-table
    # columns the Stage-2 runner used (mean_structure / cov_structure) without a
    # special case. For these processes the "structure" of interest is the
    # volatility / jump behaviour, so we surface ``family`` there.
    @property
    def mean_structure(self) -> str:
        return "zero diffusive drift (jumps/leverage may add skew)"

    @property
    def cov_structure(self) -> str:
        return self.family


def build_advanced_specs(
    sigma: float = 0.2,
    include_rough: bool = True,
) -> dict[str, AdvancedProcessSpec]:
    """Create the six process specifications, sharing baseline volatility ``sigma``.

    Parameters
    ----------
    sigma : float
        Baseline annualised diffusive volatility shared by every process. For
        Heston/Bates this is the square root of the initial and long-run
        variance; for FBM it is the scale that matches the terminal variance to
        BS; for RVOL it is the square root of the forward variance ``xi0``.
    include_rough : bool
        If False, drop the rough Bergomi process (``RVOL``). On by default.
    """
    var = sigma ** 2  # baseline variance level shared by the stochastic-vol models

    specs: dict[str, AdvancedProcessSpec] = {}

    # -- BS: Black-Scholes geometric Brownian motion ------------------------
    specs["BS"] = AdvancedProcessSpec(
        name="BS",
        kind="bs",
        params={"sigma": sigma, "mu": 0.0},
        description=(
            "Black-Scholes GBM: constant volatility, i.i.d. Gaussian log-returns. "
            "The null model every other process is compared against."
        ),
        family="constant volatility (baseline)",
    )

    # -- HEST: Heston stochastic volatility ---------------------------------
    # CIR variance v_t with mean reversion kappa to long-run theta, vol-of-vol
    # xi, and price/vol correlation rho (negative = leverage effect).
    specs["HEST"] = AdvancedProcessSpec(
        name="HEST",
        kind="heston",
        params={
            "v0": var, "theta": var, "kappa": 2.0, "xi": 0.3, "rho": -0.7, "mu": 0.0,
        },
        description=(
            "Heston stochastic volatility: variance is a mean-reverting CIR "
            "process (kappa=2, vol-of-vol=0.3) correlated -0.7 with the price. "
            "Volatility clustering and mild skew, no jumps."
        ),
        family="stochastic volatility",
    )

    # -- MERT: Merton jump-diffusion ----------------------------------------
    # Constant-vol diffusion plus compound-Poisson lognormal jumps. Negative
    # mean jump size -> negative skew and heavy (fat) tails.
    specs["MERT"] = AdvancedProcessSpec(
        name="MERT",
        kind="merton",
        params={
            "sigma": sigma, "mu": 0.0,
            "jump_intensity": 1.0, "jump_mean": -0.10, "jump_std": 0.15,
        },
        description=(
            "Merton jump-diffusion: constant-vol diffusion plus ~1 lognormal "
            "jump/year (mean -0.10, std 0.15). Adds negative skew and fat tails "
            "on top of Black-Scholes."
        ),
        family="jumps",
    )

    # -- BATES: Heston stochastic vol + Merton jumps ------------------------
    specs["BATES"] = AdvancedProcessSpec(
        name="BATES",
        kind="bates",
        params={
            "v0": var, "theta": var, "kappa": 2.0, "xi": 0.3, "rho": -0.7, "mu": 0.0,
            "jump_intensity": 1.0, "jump_mean": -0.10, "jump_std": 0.15,
        },
        description=(
            "Bates (combined): Heston stochastic volatility AND Merton jumps "
            "together -- stochastic-vol clustering plus jump skew/tails. The full "
            "jump-diffusion model."
        ),
        family="stochastic volatility + jumps",
    )

    # -- FBM: fractional Brownian motion ------------------------------------
    # Non-Markov: increments are fractional Gaussian noise with Hurst H>0.5, so
    # returns are POSITIVELY autocorrelated (trending / long memory). The scale
    # is chosen so Var(log S_T) matches BS -- the endpoint distribution is the
    # same, only the path MEMORY differs.
    specs["FBM"] = AdvancedProcessSpec(
        name="FBM",
        kind="fbm",
        params={"hurst": 0.70, "sigma": sigma, "mu": 0.0},
        description=(
            "Fractional Brownian motion (Hurst=0.70): non-Markov log-price whose "
            "increments have long-range autocorrelation. Terminal variance is "
            "matched to BS, so only the path memory differs."
        ),
        family="non-Markov (long memory)",
    )

    # -- RVOL: rough Bergomi rough volatility -------------------------------
    if include_rough:
        specs["RVOL"] = AdvancedProcessSpec(
            name="RVOL",
            kind="rough_bergomi",
            params={
                "hurst": 0.10, "xi0": var, "eta": 0.8, "rho": -0.7, "mu": 0.0,
            },
            description=(
                "Rough Bergomi: volatility driven by a rough (Hurst=0.10) "
                "fractional process (vol-of-vol eta=0.8) correlated -0.7 with the "
                "price. Very autocorrelated, jagged volatility -- the rough-vol "
                "model. Vol-of-vol is kept moderate so the ROUGHNESS, not raw "
                "scale, is what distinguishes it from Black-Scholes."
            ),
            family="rough volatility (autocorrelated)",
        )

    return specs


# ---------------------------------------------------------------------------
# Fractional Gaussian noise helper (shared by FBM and RVOL)
# ---------------------------------------------------------------------------
def _fgn_cholesky(time_steps: int, hurst: float) -> np.ndarray:
    """Lower-Cholesky factor of the fractional-Gaussian-noise covariance.

    Fractional Gaussian noise (the increments of fBM on a unit grid) has the
    stationary autocovariance

        gamma(k) = 0.5 * (|k+1|^{2H} - 2|k|^{2H} + |k-1|^{2H}).

    We build the ``time_steps x time_steps`` Toeplitz covariance from ``gamma``
    and return its Cholesky factor ``L`` (with ``L L^T = Sigma``). Multiplying a
    matrix of i.i.d. standard normals by ``L^T`` then yields exact fGn samples
    with unit per-step variance and the correct long-range correlation. The
    factor depends only on ``(time_steps, hurst)`` so the caller caches it and
    reuses it across every batch.
    """
    k = np.arange(time_steps)
    # Autocovariance gamma(k) of unit-variance fractional Gaussian noise.
    gamma = 0.5 * (
        np.abs(k + 1) ** (2 * hurst)
        - 2 * np.abs(k) ** (2 * hurst)
        + np.abs(k - 1) ** (2 * hurst)
    )
    # Symmetric Toeplitz covariance matrix Sigma[i, j] = gamma(|i - j|).
    cov = gamma[np.abs(np.subtract.outer(k, k))]
    # A tiny jitter keeps the Cholesky stable for very rough H (near-singular).
    cov[np.diag_indices(time_steps)] += 1e-12
    return np.linalg.cholesky(cov)


def _fgn_batch(
    spec: AdvancedProcessSpec,
    batch_size: int,
    time_steps: int,
    hurst: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw a batch of unit-variance fGn samples, shape (batch, time_steps).

    The Cholesky factor for ``(time_steps, hurst)`` is built once and cached on
    the spec so repeated batches are cheap.
    """
    cache_key = f"fgn_chol_{time_steps}_{hurst}"
    chol = spec._cache.get(cache_key)
    if chol is None:
        chol = _fgn_cholesky(time_steps, hurst)
        spec._cache[cache_key] = chol
    z = rng.standard_normal(size=(batch_size, time_steps))
    # (batch, T) @ (T, T)^T -> correlated fGn rows with unit per-step variance.
    return z @ chol.T


# ---------------------------------------------------------------------------
# Per-process increment simulators (all return log-returns, shape (batch, T, 1))
# ---------------------------------------------------------------------------
def _bs_increments(spec, batch_size, time_steps, dt, rng):
    """Black-Scholes: i.i.d. Gaussian log-returns (-0.5 sigma^2 dt drift)."""
    sigma, mu = spec.params["sigma"], spec.params["mu"]
    drift = (mu - 0.5 * sigma ** 2) * dt
    z = rng.standard_normal(size=(batch_size, time_steps))
    incr = drift + sigma * np.sqrt(dt) * z
    return incr[:, :, None]


def _cir_variance_and_price_diffusion(params, batch_size, time_steps, dt, rng):
    """Shared Heston/Bates engine: CIR variance path + its diffusive log-return.

    Returns ``(diffusion_increments, )`` of shape (batch, time_steps) -- the
    stochastic-volatility diffusion part of the log-return, BEFORE any jumps are
    added. Uses an Euler full-truncation scheme: the variance is floored at 0
    wherever it is used, which keeps the well-known Feller-violation-safe
    behaviour of the standard scheme.
    """
    v0, theta = params["v0"], params["theta"]
    kappa, xi, rho, mu = params["kappa"], params["xi"], params["rho"], params["mu"]

    sqrt_dt = np.sqrt(dt)
    # Two correlated Brownian innovations per step: price z_s, variance z_v.
    z_v = rng.standard_normal(size=(batch_size, time_steps))
    z_perp = rng.standard_normal(size=(batch_size, time_steps))
    z_s = rho * z_v + np.sqrt(1.0 - rho ** 2) * z_perp

    incr = np.empty((batch_size, time_steps), dtype=np.float64)
    v = np.full(batch_size, v0, dtype=np.float64)
    for t in range(time_steps):
        v_pos = np.maximum(v, 0.0)
        sqrt_v = np.sqrt(v_pos)
        # Log-price increment with the stochastic (Ito) -0.5 v dt correction.
        incr[:, t] = (mu - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * z_s[:, t]
        # CIR variance update (full truncation: use v_pos inside drift & vol).
        v = v + kappa * (theta - v_pos) * dt + xi * sqrt_v * sqrt_dt * z_v[:, t]
    return incr


def _heston_increments(spec, batch_size, time_steps, dt, rng):
    """Heston stochastic volatility, no jumps."""
    incr = _cir_variance_and_price_diffusion(
        spec.params, batch_size, time_steps, dt, rng
    )
    return incr[:, :, None]


def _jump_increments(params, batch_size, time_steps, dt, rng):
    """Compound-Poisson lognormal jump contribution to the log-return.

    Per step the number of jumps is Poisson(lambda*dt); the total log jump size
    over those jumps is Normal(N*m, N*s^2). A deterministic compensator
    ``-lambda*k_bar*dt`` (with ``k_bar = exp(m + s^2/2) - 1``) is subtracted so
    the jumps do not add spurious drift to the mean price.
    """
    lam = params["jump_intensity"]
    m, s = params["jump_mean"], params["jump_std"]
    k_bar = np.exp(m + 0.5 * s ** 2) - 1.0

    counts = rng.poisson(lam * dt, size=(batch_size, time_steps))
    # Sum of `counts` i.i.d. Normal(m, s^2) jumps == Normal(counts*m, counts*s^2).
    jump_z = rng.standard_normal(size=(batch_size, time_steps))
    jumps = counts * m + np.sqrt(counts) * s * jump_z
    # Martingale compensation so E[jump return] contributes no net drift.
    jumps = jumps - lam * k_bar * dt
    return jumps


def _merton_increments(spec, batch_size, time_steps, dt, rng):
    """Merton jump-diffusion: BS diffusion + compound-Poisson jumps."""
    sigma, mu = spec.params["sigma"], spec.params["mu"]
    drift = (mu - 0.5 * sigma ** 2) * dt
    z = rng.standard_normal(size=(batch_size, time_steps))
    diffusion = drift + sigma * np.sqrt(dt) * z
    jumps = _jump_increments(spec.params, batch_size, time_steps, dt, rng)
    return (diffusion + jumps)[:, :, None]


def _bates_increments(spec, batch_size, time_steps, dt, rng):
    """Bates: Heston stochastic-vol diffusion + Merton jumps."""
    diffusion = _cir_variance_and_price_diffusion(
        spec.params, batch_size, time_steps, dt, rng
    )
    jumps = _jump_increments(spec.params, batch_size, time_steps, dt, rng)
    return (diffusion + jumps)[:, :, None]


def _fbm_increments(spec, batch_size, time_steps, dt, rng):
    """Fractional Brownian motion log-returns (fractional Gaussian noise).

    The unit-variance fGn is scaled so that the TERMINAL variance of the log-
    price equals ``sigma^2 * T`` (T = time_steps * dt), i.e. it matches the
    Black-Scholes endpoint variance. Because fBM is self-similar, fBM at step
    ``n = time_steps`` has variance ``n^{2H}`` in grid units, so the scale is
    ``sigma * sqrt(T) / n^H``. Only the autocorrelation (memory) then
    distinguishes FBM from BS.
    """
    hurst, sigma, mu = spec.params["hurst"], spec.params["sigma"], spec.params["mu"]
    total_time = time_steps * dt
    fgn = _fgn_batch(spec, batch_size, time_steps, hurst, rng)
    scale = sigma * np.sqrt(total_time) / (time_steps ** hurst)
    incr = mu * dt + scale * fgn
    return incr[:, :, None]


def _rough_bergomi_increments(spec, batch_size, time_steps, dt, rng):
    """Rough Bergomi rough-volatility log-returns (simplified hybrid scheme).

    A rough (Hurst<0.5) fractional process ``W_hat`` drives the instantaneous
    variance

        v_t = xi0 * exp( eta * W_hat_t - 0.5 * eta^2 * t^{2H} )

    which is the standard rough-Bergomi martingale normalisation (so E[v_t]=xi0).
    The price is then

        dlogS_t = -0.5 * v_t * dt + sqrt(v_t) * dB_t,

    with the spot Brownian increment ``dB_t`` correlated (``rho``, the leverage
    effect) with the per-step innovation used to build ``W_hat``. Building the
    fractional process on the same grid via a Cholesky factor keeps it exact in
    distribution; correlating the spot with the per-step vol innovation is a
    documented simplification of the full Volterra construction, adequate for
    this synthetic signature testbed.
    """
    hurst, xi0 = spec.params["hurst"], spec.params["xi0"]
    eta, rho, mu = spec.params["eta"], spec.params["rho"], spec.params["mu"]
    sqrt_dt = np.sqrt(dt)

    # Raw i.i.d. innovations for the vol process; W_hat is a linear map of them.
    cache_key = f"fgn_chol_{time_steps}_{hurst}"
    chol = spec._cache.get(cache_key)
    if chol is None:
        chol = _fgn_cholesky(time_steps, hurst)
        spec._cache[cache_key] = chol
    z_vol = rng.standard_normal(size=(batch_size, time_steps))
    fgn = z_vol @ chol.T  # unit-variance fractional Gaussian noise

    # W_hat_t: fractional process with Var(W_hat_t) = t^{2H} (self-similar).
    total_time = time_steps * dt
    fbm_scale = np.sqrt(total_time) / (time_steps ** hurst)
    w_hat = fbm_scale * np.cumsum(fgn, axis=1)
    t_grid = np.arange(1, time_steps + 1) * dt  # times t_1..t_T
    variance = xi0 * np.exp(eta * w_hat - 0.5 * eta ** 2 * t_grid[None, :] ** (2 * hurst))

    # Spot Brownian correlated with the per-step vol innovation z_vol (leverage).
    z_perp = rng.standard_normal(size=(batch_size, time_steps))
    dB = (rho * z_vol + np.sqrt(1.0 - rho ** 2) * z_perp) * sqrt_dt
    incr = (mu - 0.5 * variance) * dt + np.sqrt(variance) * dB
    return incr[:, :, None]


_SIMULATORS = {
    "bs": _bs_increments,
    "heston": _heston_increments,
    "merton": _merton_increments,
    "bates": _bates_increments,
    "fbm": _fbm_increments,
    "rough_bergomi": _rough_bergomi_increments,
}


# ---------------------------------------------------------------------------
# Public batch API (matches HighDim testing/src/simulate_highdim_gaussian.py)
# ---------------------------------------------------------------------------
def simulate_batch(
    spec: AdvancedProcessSpec,
    batch_size: int,
    time_steps: int,
    dt: float,
    rng: np.random.Generator,
    time_aug: bool = False,
    end_time: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate ONE batch of log-price paths and return (paths, increments).

    Parameters
    ----------
    spec : AdvancedProcessSpec
        Which process to simulate.
    batch_size : int
        Number of paths in this batch.
    time_steps : int
        Number of time steps per path.
    dt : float
        Time increment (years per step). For the annualised parameters used
        here, pass ``dt = T / time_steps`` with ``T`` the horizon in years.
    rng : np.random.Generator
        Seeded generator; passing the SAME generator across batches keeps
        successive batches independent while the whole run stays reproducible.
    time_aug : bool
        If True, append a monotone time channel, turning the 1-D log-price into
        a 2-D path. Recommended for these 1-D processes: it makes the signature
        sensitive to the ORDER of the returns (essential for the non-Markov and
        rough-vol paths). The extra channel goes on ``paths`` only, not on the
        increments used by the statistical estimators.
    end_time : float
        Final value of the appended time channel when ``time_aug``.

    Returns
    -------
    (paths, increments)
        ``paths`` has shape (batch, time_steps[, +1 if time_aug], 1) and is the
        cumulative sum of the log-returns (the log-price relative to 0).
        ``increments`` has shape (batch, time_steps, 1) and is the per-step
        log-returns, returned separately for the statistical estimators.
    """
    simulate = _SIMULATORS[spec.kind]
    increments = simulate(spec, batch_size, time_steps, dt, rng)

    # Cumulative sum along the time axis -> the log-price path (starts near 0).
    paths = np.cumsum(increments, axis=1)

    if time_aug:
        # Deterministic, strictly increasing time channel in [end_time/T, end_time].
        t = np.linspace(end_time / time_steps, end_time, time_steps)
        t = np.broadcast_to(t.reshape(1, time_steps, 1), (batch_size, time_steps, 1))
        paths = np.concatenate([paths, t], axis=2)

    # pysiglib prefers C-contiguous float64 arrays.
    paths = np.ascontiguousarray(paths, dtype=np.float64)
    increments = np.ascontiguousarray(increments, dtype=np.float64)
    return paths, increments
