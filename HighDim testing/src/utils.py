"""
utils.py
========

Small, dependency-light helpers that the whole high-dimensional benchmark relies
on. Three groups of helpers live here:

1. **Signature geometry** -- how many coordinates a truncated signature has, and
   where each level lives inside the flat feature vector. This is the single most
   important piece of bookkeeping in a *multi-dimensional* signature experiment:
   in dimension ``d`` the level-``k`` block occupies exactly ``d**k`` coordinates,
   and pysiglib lays the levels out back-to-back (level 1, then level 2, ...),
   WITHOUT the leading constant "1" term (``scalar_term=False`` by default).

2. **Memory estimation** -- because we deliberately never store all signatures
   for all paths at once, we estimate up front how big one batch of raw paths and
   one batch of signatures will be, and warn (or shrink the batch) if a batch
   would be dangerously large.

3. **Logging** -- a compact banner printed at the start of every configuration so
   the run is self-documenting in the terminal.

None of this touches pysiglib; it is pure arithmetic and printing.
"""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# Signature geometry
# ---------------------------------------------------------------------------
def level_coord_count(dimension: int, level: int) -> int:
    """Number of signature coordinates at a single level.

    In dimension ``d`` the level-``k`` part of the signature is a rank-``k``
    tensor, so it has ``d ** k`` coordinates. (Example: d=20, level 2 -> 400
    coordinates; d=10, level 5 -> 100000 coordinates.)
    """
    return dimension ** level


def signature_length(dimension: int, depth: int, scalar_term: bool = False) -> int:
    """Total length of the truncated signature feature vector.

    With ``scalar_term=False`` (our default, matching pysiglib) the length is

        d + d**2 + ... + d**depth

    With ``scalar_term=True`` we add 1 for the leading empty-word "1" term.

    Examples
    --------
    d=20, depth=3 -> 20 + 400 + 8000                 = 8420   (+1 = 8421 with scalar)
    d=10, depth=5 -> 10 + 100 + 1000 + 10000 + 100000 = 111110 (+1 = 111111 with scalar)
    """
    total = sum(level_coord_count(dimension, k) for k in range(1, depth + 1))
    if scalar_term:
        total += 1
    return total


def level_slices(dimension: int, depth: int) -> dict[int, slice]:
    """Map each level 1..depth to the slice it occupies in the flat signature.

    Assumes the pysiglib layout with NO leading scalar term: the vector is the
    concatenation ``[level 1 | level 2 | ... | level depth]`` where level ``k``
    contributes ``d**k`` consecutive coordinates.

    Returns
    -------
    dict
        ``{level: slice(start, stop)}`` so callers can do ``sig[slices[k]]`` to
        pull out only the level-``k`` coordinates.
    """
    slices: dict[int, slice] = {}
    start = 0
    for level in range(1, depth + 1):
        count = level_coord_count(dimension, level)
        slices[level] = slice(start, start + count)
        start += count
    return slices


# ---------------------------------------------------------------------------
# Memory estimation / human-readable sizes
# ---------------------------------------------------------------------------
def format_bytes(num_bytes: float) -> str:
    """Turn a raw byte count into a short human-readable string (e.g. '44.4 MB')."""
    step = 1024.0
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < step:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= step
    return f"{num_bytes:.1f} PB"


def estimate_batch_path_bytes(
    batch_size: int, time_steps: int, dimension: int, dtype_bytes: int = 8
) -> int:
    """Bytes needed to hold ONE batch of raw paths, shape (batch, time, dim)."""
    return batch_size * time_steps * dimension * dtype_bytes


def estimate_batch_signature_bytes(
    batch_size: int, dimension: int, depth: int, dtype_bytes: int = 8
) -> int:
    """Bytes needed to hold ONE batch of signatures, shape (batch, sig_len)."""
    return batch_size * signature_length(dimension, depth) * dtype_bytes


def choose_safe_batch_size(
    requested_batch_size: int,
    dimension: int,
    depth: int,
    max_batch_signature_bytes: float,
    dtype_bytes: int = 8,
) -> tuple[int, bool]:
    """Shrink the batch size if one batch of signatures would exceed a budget.

    We never store every path's signature at once, but a *single* batch of
    signatures still has to fit in memory. If the requested batch would need more
    than ``max_batch_signature_bytes``, we reduce it to the largest value that
    fits (at least 1).

    Returns
    -------
    (safe_batch_size, was_reduced)
        ``safe_batch_size`` is what should actually be used; ``was_reduced`` is
        True if we had to shrink it (so the caller can print a warning).
    """
    per_path_bytes = signature_length(dimension, depth) * dtype_bytes
    max_paths = int(max_batch_signature_bytes // per_path_bytes)
    if max_paths >= requested_batch_size:
        return requested_batch_size, False
    return max(1, max_paths), True


def iter_batch_sizes(num_paths: int, batch_size: int):
    """Yield successive batch sizes that sum to ``num_paths``.

    The final batch may be smaller than ``batch_size`` when ``num_paths`` is not
    an exact multiple. This lets the caller keep ``num_paths`` fully configurable
    without forcing it to be divisible by the batch size.
    """
    remaining = num_paths
    while remaining > 0:
        this_batch = min(batch_size, remaining)
        yield this_batch
        remaining -= this_batch


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log_config_banner(
    config_name: str,
    dimension: int,
    depth: int,
    num_paths: int,
    time_steps: int,
    batch_size: int,
    time_aug: bool,
) -> None:
    """Print the required start-of-configuration banner.

    Includes the expected signature dimension and per-batch memory estimates so
    the run is transparent about how heavy it is before doing any work.
    """
    effective_dim = dimension + 1 if time_aug else dimension
    sig_len = signature_length(effective_dim, depth)
    path_bytes = estimate_batch_path_bytes(batch_size, time_steps, dimension)
    sig_bytes = estimate_batch_signature_bytes(batch_size, effective_dim, depth)

    print("=" * 70)
    print(f"CONFIGURATION: {config_name}")
    print("=" * 70)
    print(f"  path dimension d              : {dimension}")
    if time_aug:
        print(f"  time augmentation             : ON  -> effective d = {effective_dim}")
    else:
        print(f"  time augmentation             : OFF -> effective d = {effective_dim}")
    print(f"  signature depth               : {depth}")
    print(f"  expected signature dimension  : {sig_len:,} coordinates")
    print(f"  num_paths                     : {num_paths:,}")
    print(f"  time_steps                    : {time_steps}")
    print(f"  batch_size                    : {batch_size}")
    print(f"  est. raw path memory / batch  : {format_bytes(path_bytes)}")
    print(f"  est. signature memory / batch : {format_bytes(sig_bytes)}")
    print("-" * 70)


def ensure_dirs(*paths: Path) -> None:
    """Create each directory (and parents) if it does not already exist."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
