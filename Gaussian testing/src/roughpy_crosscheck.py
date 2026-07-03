"""
roughpy_crosscheck.py
=====================

Independent validation of our signature pipeline using a SECOND library.

Why this file exists
--------------------
All of our results are produced by `pysiglib`. Signatures are subtle to compute
(truncation conventions, whether the constant "1" term is included, how the path
is parametrised, ...). The safest way to trust the numbers is to compute the
SAME signature with a completely different library and check they agree.

Here that second library is `roughpy`. If both libraries produce the same
signatures, we can be confident our pipeline is correct before moving on to
harder (non-Gaussian) processes or the real SPY/QQQ/TLT data.

How the two libraries differ (important detail)
-----------------------------------------------
- `pysiglib.sig(path, degree)` takes the PATH POINTS directly (shape
  (length, dimension)) and, by default, returns the signature WITHOUT the
  leading constant "1" term.
- `roughpy` is built around "streams" of INCREMENTS. So we must:
      1. difference the path into consecutive increments (np.diff),
      2. build a `LieIncrementStream` from those increments,
      3. ask for the signature over the whole interval,
      4. drop the leading "1" term so it lines up with pysiglib.

Both describe the exact same piecewise-linear path, so the signatures match.
"""

from __future__ import annotations

import numpy as np

# Import both libraries defensively with clear install instructions.
try:
    import pysiglib
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "\n\n'pysiglib' is required. Install it with:\n\n    pip install pysiglib\n"
    ) from exc

try:
    import roughpy as rp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "\n\n'roughpy' is required for the cross-check. Install it with:\n\n"
        "    pip install roughpy\n"
    ) from exc


def roughpy_signature_single(path: np.ndarray, degree: int) -> np.ndarray:
    """Compute one path's truncated signature with roughpy.

    The result is aligned with `pysiglib.sig`: same ordering of terms and the
    leading constant "1" term removed.

    Parameters
    ----------
    path : np.ndarray
        A single path, shape (path_length, dimension).
    degree : int
        Truncation level.

    Returns
    -------
    np.ndarray
        Signature terms (without the scalar term), shape (signature_length,).
    """
    path = np.ascontiguousarray(path, dtype=np.float64)
    dimension = path.shape[1]

    # Step 1: consecutive increments describe the same piecewise-linear path.
    increments = np.ascontiguousarray(np.diff(path, axis=0), dtype=np.float64)

    # Step 2: a roughpy "context" fixes the algebra (width = dimension,
    # depth = truncation degree, real-valued coefficients).
    ctx = rp.get_context(width=dimension, depth=degree, coeffs=rp.DPReal)

    # Step 3: build the increment stream and take its signature over the full
    # parameter range [0, number_of_increments].
    stream = rp.LieIncrementStream.from_increments(increments, ctx=ctx)
    free_tensor = stream.signature(rp.RealInterval(0, len(increments)))

    # Step 4: the FreeTensor array starts with the constant "1" term at index 0;
    # drop it so the layout matches pysiglib's default output.
    return np.asarray(free_tensor)[1:]


def roughpy_expected_signature(paths: np.ndarray, degree: int) -> np.ndarray:
    """Compute the expected (average) signature of a batch of paths with roughpy.

    roughpy works one path at a time, so we loop and average. This is slower
    than pysiglib's batched call -- which is fine, because this file is only a
    validation check, not the main workhorse.

    Parameters
    ----------
    paths : np.ndarray
        Batch of paths, shape (n_paths, path_length, dimension).
    degree : int
        Truncation level.

    Returns
    -------
    np.ndarray
        The averaged signature, shape (signature_length,).
    """
    per_path = [roughpy_signature_single(paths[i], degree) for i in range(paths.shape[0])]
    return np.mean(per_path, axis=0)


def compare_libraries(paths: np.ndarray, degree: int) -> dict:
    """Compare pysiglib and roughpy expected signatures for one process/level.

    Parameters
    ----------
    paths : np.ndarray
        Batch of paths, shape (n_paths, path_length, dimension).
    degree : int
        Truncation level.

    Returns
    -------
    dict
        {
          "max_abs_difference": largest absolute gap between the two expected
                                signatures (should be ~1e-12 or smaller),
          "expected_pysiglib":  the pysiglib expected signature,
          "expected_roughpy":   the roughpy expected signature,
        }
    """
    # pysiglib: batched, then average over paths.
    pysig = np.asarray(pysiglib.sig(np.ascontiguousarray(paths, dtype=np.float64), degree=degree))
    expected_pysiglib = pysig.mean(axis=0)

    # roughpy: per-path, then average.
    expected_roughpy = roughpy_expected_signature(paths, degree)

    max_abs_difference = float(np.max(np.abs(expected_pysiglib - expected_roughpy)))

    return {
        "max_abs_difference": max_abs_difference,
        "expected_pysiglib": expected_pysiglib,
        "expected_roughpy": expected_roughpy,
    }
