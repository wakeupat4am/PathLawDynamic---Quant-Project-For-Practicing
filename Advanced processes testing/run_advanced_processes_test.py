"""
run_advanced_processes_test.py
==============================

Stage 3 -- signatures on ADVANCED financial processes.

The earlier stages showed signatures recover the moments (Stage 1) and the
multi-channel geometry (Stage 2) of *Gaussian* paths. This stage pushes into the
processes real markets actually show, which break the Gaussian/Markov
assumptions:

    BS    Black-Scholes GBM ............ baseline (constant vol, i.i.d. Gaussian).
    HEST  Heston ....................... stochastic volatility.
    MERT  Merton ....................... jumps.
    BATES Bates ........................ jump-diffusion (stochastic vol + jumps).
    FBM   fractional Brownian motion ... non-Markov / long memory.
    RVOL  rough Bergomi ................ rough (autocorrelated) volatility.

For every process it computes, in the SAME batched/streaming way as Stage 2:

    * classical statistical summaries (terminal & increment mean/variance), and
    * the expected truncated signature,

then measures how far each process sits from the Black-Scholes baseline under
each lens and writes tables + figures into
``Advanced processes testing/results/``.

These are 1-D log-price paths, so we turn ON **time augmentation** by default:
the signature of a 1-D path is nearly blind to the ORDER of the returns, but the
whole point of the non-Markov (FBM) and rough-vol (RVOL) processes is that the
order/memory matters. Appending a time channel (effective dimension 2) makes the
signature sensitive to that ordering.

Run it from the project root (mind the space in the folder name):

    python "Advanced processes testing/run_advanced_processes_test.py"

Because it reuses the streaming signature/statistics/plotting machinery from
``HighDim testing/src``, that folder must sit alongside this one (it does in this
repo).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Make BOTH this stage's ``src`` and the shared Stage-2 ``src`` importable. The
# signature accumulation, statistical estimators, distance tables, plotting and
# geometry helpers are identical to Stage 2, so we reuse them rather than copy.
HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / "HighDim testing" / "src"
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(SHARED))

from simulate_advanced import build_advanced_specs, simulate_batch  # noqa: E402

from signature_highdim import RunningSignatureStats, compute_batch_signatures  # noqa: E402
from statistical_estimators import ProcessStatistics  # noqa: E402
from distance_metrics import (  # noqa: E402
    build_levelwise_distance_table,
    build_signature_distance_table,
    build_statistical_distance_table,
)
from plotting import (  # noqa: E402
    plot_levelwise_signature_distance,
    plot_sample_paths,
    plot_signature_distance_by_depth,
    plot_statistical_distances,
)
from utils import (  # noqa: E402
    choose_safe_batch_size,
    ensure_dirs,
    format_bytes,
    log_config_banner,
    signature_length,
)


# All processes are 1-D log-price paths compared at a single configuration. With
# time augmentation ON the effective signature dimension is 2, so depth 5 is
# cheap (2+4+8+16+32 = 62 coordinates) yet deep enough to expose higher-order
# path structure.
DIMENSION = 1
DEPTH = 5
DEFAULT_BATCH_SIZE = 500

# Comparisons: every advanced process against the Black-Scholes baseline, plus
# two cross-comparisons that isolate a single added feature.
BASE_COMPARISONS = [
    ("BS_vs_HEST", "BS", "HEST"),
    ("BS_vs_MERT", "BS", "MERT"),
    ("BS_vs_BATES", "BS", "BATES"),
    ("BS_vs_FBM", "BS", "FBM"),
    ("MERT_vs_BATES", "MERT", "BATES"),  # does adding stoch-vol on top of jumps show?
]
ROUGH_COMPARISONS = [
    ("BS_vs_RVOL", "BS", "RVOL"),
    ("FBM_vs_RVOL", "FBM", "RVOL"),  # two kinds of memory: long (H>.5) vs rough (H<.5)
]

# Per-batch signature memory budget (tiny here, but kept for symmetry/safety).
MAX_BATCH_SIGNATURE_BYTES = 1_000_000_000


def run_experiment(
    num_paths: int,
    time_steps: int,
    horizon: float,
    sigma: float,
    seed: int,
    time_aug: bool,
    include_rough: bool,
    batch_size_override: int | None,
) -> dict:
    """Run the whole Stage-3 experiment and return all tables + sample paths."""
    effective_dim = DIMENSION + 1 if time_aug else DIMENSION
    dt = horizon / time_steps  # years per step; annualised params assume this

    # -- Decide a safe batch size ------------------------------------------
    requested = batch_size_override or DEFAULT_BATCH_SIZE
    batch_size, was_reduced = choose_safe_batch_size(
        requested, effective_dim, DEPTH, MAX_BATCH_SIGNATURE_BYTES
    )
    if was_reduced:
        print(
            f"[warning] Requested batch_size={requested} exceeds the "
            f"{format_bytes(MAX_BATCH_SIGNATURE_BYTES)} per-batch budget; "
            f"reduced to {batch_size}."
        )

    log_config_banner(
        "advanced_1d_depth5", DIMENSION, DEPTH, num_paths, time_steps, batch_size, time_aug
    )
    print(
        f"  horizon T (years)             : {horizon}\n"
        f"  dt (years/step)               : {dt:.5f}\n"
        f"  baseline volatility sigma     : {sigma}"
    )
    if time_aug:
        print(
            f"[note] time augmentation is ON: effective signature dimension is "
            f"{effective_dim} (path value + time), so the signature sees the "
            f"ORDER of the returns."
        )
    print("-" * 70)

    # -- Build processes and per-process accumulators ----------------------
    specs = build_advanced_specs(sigma=sigma, include_rough=include_rough)
    sig_len = signature_length(effective_dim, DEPTH)

    sig_stats = {name: RunningSignatureStats(sig_len) for name in specs}
    stat_stats = {name: ProcessStatistics(DIMENSION) for name in specs}
    sample_paths: dict[str, np.ndarray] = {}

    # -- Stream over processes and batches ---------------------------------
    proc_offsets = {name: i for i, name in enumerate(specs)}
    for name, spec in specs.items():
        # Deterministic per-process generator: reproducible, mutually independent.
        proc_seed = seed + 101 * proc_offsets[name]
        rng = np.random.default_rng(proc_seed)

        t0 = time.time()
        remaining = num_paths
        while remaining > 0:
            this_batch = min(batch_size, remaining)

            paths, increments = simulate_batch(
                spec, this_batch, time_steps, dt, rng,
                time_aug=time_aug, end_time=horizon,
            )
            stat_stats[name].update(paths, increments)
            batch_sigs = compute_batch_signatures(paths, DEPTH)
            sig_stats[name].update(batch_sigs)

            # Keep a few example log-price paths (value channel only) for the figure.
            if name not in sample_paths:
                sample_paths[name] = paths[: min(3, this_batch), :, :1].copy()

            remaining -= this_batch

        elapsed = time.time() - t0
        print(f"  process {name:5s}: {num_paths:,} paths in {elapsed:5.1f}s "
              f"({spec.family})")

    # -- Collect final summaries -------------------------------------------
    summaries = {name: stat_stats[name].summary() for name in specs}
    expected_sigs = {name: sig_stats[name].expected_signature() for name in specs}
    coord_std_bs = sig_stats["BS"].coordinate_std()  # baseline std for standardising

    comparisons = [c for c in BASE_COMPARISONS]
    if include_rough:
        comparisons += ROUGH_COMPARISONS

    # -- Build tables ------------------------------------------------------
    config_rows = []
    for name, spec in specs.items():
        s = summaries[name]
        config_rows.append({
            "process": name,
            "family": spec.family,
            "dimension": DIMENSION,
            "signature_depth": DEPTH,
            "time_augmented": time_aug,
            "time_steps": time_steps,
            "horizon_years": horizon,
            "num_paths": num_paths,
            "terminal_log_return_mean": s["terminal_mean"][0],
            "terminal_log_return_variance": s["avg_terminal_variance"],
            "description": spec.description,
        })
    config_table = pd.DataFrame(config_rows)

    stat_summary_rows = []
    for name in specs:
        s = summaries[name]
        stat_summary_rows.append({
            "process": name,
            "terminal_mean_norm": s["terminal_mean_norm"],
            "terminal_variance": s["avg_terminal_variance"],
            "increment_variance": s["avg_increment_variance"],
            "notes": specs[name].description,
        })
    stat_summary_table = pd.DataFrame(stat_summary_rows)

    config_name = "advanced_1d_depth5"
    stat_distance_table = build_statistical_distance_table(
        config_name, summaries, comparisons
    )
    signature_distance_table = build_signature_distance_table(
        config_name, DEPTH, effective_dim, expected_sigs, comparisons
    )
    levelwise_table = build_levelwise_distance_table(
        config_name, DEPTH, effective_dim, expected_sigs, comparisons,
        coordinate_std=coord_std_bs,
    )

    return {
        "config_table": config_table,
        "stat_summary_table": stat_summary_table,
        "stat_distance_table": stat_distance_table,
        "signature_distance_table": signature_distance_table,
        "levelwise_table": levelwise_table,
        "sample_paths": sample_paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-paths", type=int, default=5_000,
                        help="Paths per process (scale up for smoother tails).")
    parser.add_argument("--time-steps", type=int, default=100)
    parser.add_argument("--horizon", type=float, default=1.0,
                        help="Path horizon in years (annualised params assume this).")
    parser.add_argument("--sigma", type=float, default=0.2,
                        help="Baseline annualised volatility shared by all processes.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-time-aug", action="store_true",
                        help="Turn OFF time augmentation (not recommended for 1-D paths).")
    parser.add_argument("--no-rough", action="store_true",
                        help="Skip the rough Bergomi process (RVOL).")
    parser.add_argument("--results-dir", type=str, default=None)
    args = parser.parse_args()

    results_root = Path(args.results_dir) if args.results_dir else HERE / "results"
    tables_dir = results_root / "tables"
    figures_dir = results_root / "figures"
    ensure_dirs(tables_dir, figures_dir)

    result = run_experiment(
        num_paths=args.num_paths,
        time_steps=args.time_steps,
        horizon=args.horizon,
        sigma=args.sigma,
        seed=args.seed,
        time_aug=not args.no_time_aug,
        include_rough=not args.no_rough,
        batch_size_override=args.batch_size,
    )

    # -- Write tables ------------------------------------------------------
    result["config_table"].to_csv(tables_dir / "advanced_process_config.csv", index=False)
    result["stat_summary_table"].to_csv(
        tables_dir / "advanced_statistical_summary.csv", index=False
    )
    result["stat_distance_table"].to_csv(
        tables_dir / "advanced_statistical_distances.csv", index=False
    )
    result["signature_distance_table"].to_csv(
        tables_dir / "advanced_signature_distances.csv", index=False
    )
    result["levelwise_table"].to_csv(
        tables_dir / "advanced_signature_levelwise_distances.csv", index=False
    )

    # -- Figures -----------------------------------------------------------
    plot_signature_distance_by_depth(
        result["levelwise_table"], figures_dir / "advanced_signature_distance_by_depth.png"
    )
    plot_levelwise_signature_distance(
        result["levelwise_table"], figures_dir / "advanced_levelwise_signature_distance.png"
    )
    plot_statistical_distances(
        result["stat_distance_table"], figures_dir / "advanced_statistical_distances.png"
    )
    plot_sample_paths(
        result["sample_paths"], figures_dir / "advanced_sample_paths.png"
    )

    print("=" * 70)
    print("DONE. Tables in results/tables/ and figures in results/figures/.")
    print("=" * 70)


if __name__ == "__main__":
    main()
