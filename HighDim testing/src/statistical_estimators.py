"""
statistical_estimators.py
=========================

Classical moment estimators computed the *same* batched, streaming way as the
signatures, so both sides of the comparison scale to large ``num_paths`` without
storing all the data.

For each process we estimate, over all paths:

    1. terminal mean vector        E[X_T]           (mean of the endpoint)
    2. terminal covariance matrix  Cov[X_T]
    3. increment mean vector       E[dX]
    4. increment covariance matrix Cov[dX]
    5. average marginal variance   (mean of the diagonal of the covariance)

These are the "textbook" summaries a quant would reach for. The benchmark then
asks whether signature-based summaries see the same structure (mean shift ->
level 1; correlation blocks -> level 2+).

Streaming covariance
--------------------
A covariance needs E[x], E[x x^T]. We keep running sums of ``x`` and of the outer
products ``x x^T`` (both tiny: length d and d-by-d), then at the end form

    mean = sum_x / n
    cov  = sum_xx / n - mean mean^T      (population covariance).

For the terminal moments each path contributes ONE sample (its endpoint). For the
increment moments each (path, time step) contributes one sample.
"""

from __future__ import annotations

import numpy as np


class RunningVectorStats:
    """Streaming mean and covariance for vector samples, one batch at a time.

    Holds only ``sum_x`` (length d) and ``sum_xx`` (d-by-d) plus a count, so its
    memory is independent of the number of samples.
    """

    def __init__(self, dimension: int) -> None:
        self._sum = np.zeros(dimension, dtype=np.float64)
        self._sum_outer = np.zeros((dimension, dimension), dtype=np.float64)
        self._count = 0

    def update(self, samples: np.ndarray) -> None:
        """Fold in a batch of vector samples of shape (n_samples, dimension)."""
        samples = np.asarray(samples, dtype=np.float64)
        self._sum += samples.sum(axis=0)
        # (n, d)^T @ (n, d) -> (d, d) sum of outer products, done in one BLAS call.
        self._sum_outer += samples.T @ samples
        self._count += samples.shape[0]

    @property
    def count(self) -> int:
        return self._count

    def mean(self) -> np.ndarray:
        """Estimated mean vector E[x]."""
        if self._count == 0:
            raise ValueError("No samples accumulated yet.")
        return self._sum / self._count

    def covariance(self) -> np.ndarray:
        """Estimated population covariance Cov[x] = E[x x^T] - E[x] E[x]^T."""
        if self._count == 0:
            raise ValueError("No samples accumulated yet.")
        mean = self.mean()
        return self._sum_outer / self._count - np.outer(mean, mean)


class ProcessStatistics:
    """Bundle the terminal and increment streaming estimators for one process."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.terminal = RunningVectorStats(dimension)   # samples = X_T (one per path)
        self.increment = RunningVectorStats(dimension)  # samples = dX (many per path)

    def update(self, paths: np.ndarray, increments: np.ndarray) -> None:
        """Fold one batch into both estimators.

        Parameters
        ----------
        paths : np.ndarray
            Shape (batch, time_steps, dimension). Only the endpoint X_T (last time
            step, original d channels) is used for the terminal statistics.
        increments : np.ndarray
            Shape (batch, time_steps, dimension). Flattened over (path, time) so
            every single increment is one sample of dX.
        """
        d = self.dimension
        # Endpoint of each path: take the last time step, first d channels (in case
        # time augmentation added an extra channel to `paths`).
        terminal = paths[:, -1, :d]
        self.terminal.update(terminal)

        # Every increment (over all paths and all time steps) is a sample of dX.
        flat_increments = increments[:, :, :d].reshape(-1, d)
        self.increment.update(flat_increments)

    def summary(self) -> dict:
        """Return the scalar/vector/matrix summaries required by the tables."""
        terminal_mean = self.terminal.mean()
        terminal_cov = self.terminal.covariance()
        increment_mean = self.increment.mean()
        increment_cov = self.increment.covariance()
        return {
            "terminal_mean": terminal_mean,
            "terminal_cov": terminal_cov,
            "increment_mean": increment_mean,
            "increment_cov": increment_cov,
            # Norm of the terminal mean vector: 0 when there is no drift, grows
            # with the number/size of drifting channels (so B stands out).
            "terminal_mean_norm": float(np.linalg.norm(terminal_mean)),
            # Average marginal variance = mean of the covariance diagonal.
            "avg_terminal_variance": float(np.mean(np.diag(terminal_cov))),
            "avg_increment_variance": float(np.mean(np.diag(increment_cov))),
        }
