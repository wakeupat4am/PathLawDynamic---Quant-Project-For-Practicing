"""
signature_features.py
=====================

This module turns paths into *signature features* using the `pysiglib` library.

What is a signature (very short version)?
-----------------------------------------
The path signature is a list of numbers that summarises the SHAPE of a path.
    - Level 1 numbers  ~ how much the path moved overall (first-order info).
    - Level 2 numbers  ~ areas / interactions between channels (second-order).
    - Higher levels     ~ increasingly fine detail about the path.

A full signature is infinite, so in practice we always TRUNCATE it at some
finite "degree" (also called the truncation level). This module computes those
truncated signatures and their averages.
"""

from __future__ import annotations

import numpy as np

# Import pysiglib defensively. Beginners often forget to install it, so we give
# a clear, actionable error message instead of a confusing ImportError later.
try:
    import pysiglib
except ImportError as exc:  # pragma: no cover - only triggers when not installed
    raise ImportError(
        "\n\nThe 'pysiglib' library is required for this experiment but was not found.\n"
        "Install it with:\n\n"
        "    pip install pysiglib\n\n"
        "(Or install everything at once with:\n"
        "    pip install -r 'Gaussian testing/requirements.txt')\n"
    ) from exc


def compute_signatures(paths: np.ndarray, degree: int) -> np.ndarray:
    """Compute truncated signatures for a batch of paths.

    Parameters
    ----------
    paths : np.ndarray
        Batch of paths, shape (n_paths, path_length, dimension).
    degree : int
        Truncation level N. Higher N = more (and higher-order) signature terms.

    Returns
    -------
    np.ndarray
        Signature features, shape (n_paths, signature_length). The signature
        length depends on the dimension and the degree. For dimension = 1 it is
        simply equal to `degree` (one term per level).

    Notes
    -----
    We call `pysiglib.sig(paths, degree=degree)`. By default pysiglib omits the
    constant leading "1" term (scalar_term=False), which is what we want: that
    term is the same for every path and carries no information.
    """
    # Ensure a clean, contiguous float64 array so pysiglib does not warn/copy.
    paths = np.ascontiguousarray(paths, dtype=np.float64)

    # The core call: compute the truncated signature of every path in the batch.
    signatures = pysiglib.sig(paths, degree=degree)

    # pysiglib returns a numpy array for numpy input; make sure of the type.
    return np.asarray(signatures, dtype=np.float64)


def compute_expected_signature(signatures: np.ndarray) -> np.ndarray:
    """Compute the *expected signature*: the average signature over all paths.

    The "expected signature" is simply the empirical mean of the signature
    vectors across all simulated paths. It is a single feature vector that
    represents the whole process (its "signature fingerprint"). Two processes
    with different moments should have different expected signatures.

    Parameters
    ----------
    signatures : np.ndarray
        Signatures of shape (n_paths, signature_length).

    Returns
    -------
    np.ndarray
        The averaged signature, shape (signature_length,).
    """
    # axis=0 averages over the paths, leaving one number per signature term.
    return signatures.mean(axis=0)


def compute_signatures_for_levels(processes: dict, levels: list[int]) -> dict:
    """Compute signatures and expected signatures for every process and level.

    Parameters
    ----------
    processes : dict
        Maps process name -> paths ndarray of shape
        (n_paths, path_length, dimension).
    levels : list[int]
        Truncation levels to test, e.g. [1, 2, 3, 4, 5].

    Returns
    -------
    dict
        Nested dictionary:
            results[process_name][level] = {
                "signatures": (n_paths, signature_length) array,
                "expected":   (signature_length,) array,
            }
    """
    results: dict = {}

    for name, paths in processes.items():
        results[name] = {}
        for level in levels:
            # All signatures for this process at this truncation level.
            signatures = compute_signatures(paths, degree=level)
            # The single averaged "fingerprint" for this process/level.
            expected = compute_expected_signature(signatures)

            results[name][level] = {
                "signatures": signatures,
                "expected": expected,
            }

    return results
