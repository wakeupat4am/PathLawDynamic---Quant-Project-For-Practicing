"""
signature_highdim.py
=====================

Signature computation for the high-dimensional benchmark, built around a strict
rule from the experiment design:

    **Never store all signatures for all paths at once.**

Instead we process the data in batches and keep only *running* summaries. The
central object is ``RunningSignatureStats``, which consumes one batch of
signatures at a time and maintains:

    - a running SUM of signature coordinates      (to get the expected signature)
    - a running SUM of SQUARED coordinates        (to get per-coordinate std)
    - a running count of paths

From those three running quantities we can recover, at the end and without ever
holding more than one batch in memory:

    - the **expected (mean) signature** = sum / count
    - the per-coordinate **standard deviation** (used to standardise higher
      levels, whose raw coordinates are numerically huge).

pysiglib does the heavy lifting; this module only wraps it and accumulates.
"""

from __future__ import annotations

import numpy as np

# Import pysiglib defensively with an actionable message (same style as stage 1).
try:
    import pysiglib
except ImportError as exc:  # pragma: no cover - only triggers when not installed
    raise ImportError(
        "\n\nThe 'pysiglib' library is required for this experiment but was not found.\n"
        "Install it with:\n\n"
        "    pip install pysiglib\n\n"
        "(Or install everything at once with:\n"
        "    pip install -r 'HighDim testing/requirements.txt')\n"
    ) from exc


def compute_batch_signatures(
    paths: np.ndarray,
    depth: int,
    time_aug: bool = False,
    end_time: float = 1.0,
    n_jobs: int = -1,
) -> np.ndarray:
    """Compute truncated signatures for one batch of paths.

    Parameters
    ----------
    paths : np.ndarray
        Batch of paths, shape (batch, time_steps, dimension). If ``time_aug`` was
        already applied upstream (extra channel present) leave ``time_aug=False``
        here to avoid adding it twice.
    depth : int
        Signature truncation depth N.
    time_aug : bool
        Ask pysiglib to append a time channel itself. In this project we usually
        add time augmentation in the simulator instead, so this stays False.
    end_time : float
        End time for pysiglib's time augmentation (only used if ``time_aug``).
    n_jobs : int
        Threads for pysiglib; -1 uses all available cores.

    Returns
    -------
    np.ndarray
        Signatures of shape (batch, signature_length), float64, WITHOUT the
        leading constant "1" term (pysiglib default ``scalar_term=False``).
    """
    paths = np.ascontiguousarray(paths, dtype=np.float64)
    sigs = pysiglib.sig(
        paths, degree=depth, time_aug=time_aug, end_time=end_time, n_jobs=n_jobs
    )
    return np.asarray(sigs, dtype=np.float64)


class RunningSignatureStats:
    """Accumulate expected-signature statistics one batch at a time.

    Memory footprint is just three vectors of length ``signature_length`` (plus a
    scalar count) -- independent of how many paths are processed. This is what
    lets us scale ``num_paths`` toward 100000 without ever materialising every
    signature.
    """

    def __init__(self, signature_length: int) -> None:
        # Running sums are kept in float64 to limit accumulation error across many
        # batches (higher-level signature coordinates can be numerically large).
        self._sum = np.zeros(signature_length, dtype=np.float64)
        self._sum_sq = np.zeros(signature_length, dtype=np.float64)
        self._count = 0

    def update(self, batch_signatures: np.ndarray) -> None:
        """Fold one batch of signatures into the running statistics.

        Parameters
        ----------
        batch_signatures : np.ndarray
            Shape (batch, signature_length). Consumed and then discarded by the
            caller; only the running sums survive.
        """
        self._sum += batch_signatures.sum(axis=0)
        self._sum_sq += np.square(batch_signatures).sum(axis=0)
        self._count += batch_signatures.shape[0]

    @property
    def count(self) -> int:
        """How many paths have been folded in so far."""
        return self._count

    def expected_signature(self) -> np.ndarray:
        """The expected (mean) signature = running sum / count."""
        if self._count == 0:
            raise ValueError("No signatures accumulated yet.")
        return self._sum / self._count

    def coordinate_std(self) -> np.ndarray:
        """Per-coordinate standard deviation across paths.

        Uses the running sums: var = E[x^2] - E[x]^2, clipped at 0 to avoid tiny
        negative values from floating-point cancellation. Used to standardise
        signature coordinates when comparing levels of very different scale.
        """
        if self._count == 0:
            raise ValueError("No signatures accumulated yet.")
        mean = self._sum / self._count
        mean_sq = self._sum_sq / self._count
        var = np.clip(mean_sq - np.square(mean), a_min=0.0, a_max=None)
        return np.sqrt(var)
