"""
run_highdim_gaussian_benchmark.py
=================================

High-dimensional Gaussian signature benchmark -- the next stage after the 1D
"Gaussian testing" experiments.

What this script does
---------------------
It runs TWO configurations back to back:

    Config 1:  d = 20, signature depth = 3   (higher-dimensional geometry,
                                               cross-channel interactions)
    Config 2:  d = 10, signature depth = 5   (deeper signature levels while the
                                               feature dimension stays manageable)

For each configuration it simulates several multivariate Gaussian processes
(A baseline, B mean-shift, C correlation-block, and optionally D common-shock),
and for every process it computes BOTH:

    * classical statistical summaries (terminal & increment mean/covariance), and
    * signature-based summaries (the expected truncated signature),

entirely in a BATCHED, STREAMING way -- it never stores all signatures for all
paths at once. It then measures how far apart the processes look under each lens
and writes tables + figures into ``HighDim testing/results/``.

Run it from the project root (mind the space in the folder name):

    python "HighDim testing/run_highdim_gaussian_benchmark.py"

Optional flags let you scale up (``--num-paths 100000``), pick a single config,
change batch sizes, raise the per-batch signature memory budget, control
``pysiglib`` CPU threads, or turn on time augmentation.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Make the sibling ``src`` package importable regardless of the current directory.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

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
from signature_highdim import RunningSignatureStats, compute_batch_signatures  # noqa: E402
from simulate_highdim_gaussian import build_process_specs, simulate_batch  # noqa: E402
from statistical_estimators import ProcessStatistics  # noqa: E402
from utils import (  # noqa: E402
    choose_safe_batch_size,
    ensure_dirs,
    format_bytes,
    log_config_banner,
    signature_length,
)


# The two configurations from the experiment design.
CONFIGS = [
    {
        "config_name": "d20_depth3",
        "dimension": 20,
        "depth": 3,
        "default_batch_size": 250,
    },
    {
        "config_name": "d10_depth5",
        "dimension": 10,
        "depth": 5,
        "default_batch_size": 100,
    },
]

# Which process pairs to compare. D is included only if the common-shock process
# is present (added dynamically below).
BASE_COMPARISONS = [
    ("A_vs_B", "A", "B"),
    ("A_vs_C", "A", "C"),
    ("B_vs_C", "B", "C"),
]

# Safety budget: the largest a SINGLE batch of signatures is allowed to be before
# we automatically shrink the batch size. The default is laptop-safe, but the CLI
# exposes a larger server setting when more RAM is available.
DEFAULT_MAX_BATCH_SIGNATURE_GB = 1.0


def run_configuration(
    config: dict,
    num_paths: int,
    time_steps: int,
    dt: float,
    seed: int,
    time_aug: bool,
    include_common_shock: bool,
    batch_size_override: int | None,
    max_batch_signature_bytes: int,
    n_jobs: int,
) -> dict:
    """Run one configuration end to end and return all its tables + sample paths."""
    config_name = config["config_name"]
    dimension = config["dimension"]
    depth = config["depth"]
    effective_dim = dimension + 1 if time_aug else dimension

    # -- Decide a safe batch size ------------------------------------------
    requested = batch_size_override or config["default_batch_size"]
    batch_size, was_reduced = choose_safe_batch_size(
        requested, effective_dim, depth, max_batch_signature_bytes
    )
    if was_reduced:
        print(
            f"[warning] Requested batch_size={requested} would exceed the "
            f"{format_bytes(max_batch_signature_bytes)} per-batch signature budget "
            f"for {config_name}; automatically reduced to {batch_size}."
        )

    # -- Start-of-config logging banner ------------------------------------
    log_config_banner(
        config_name,
        dimension,
        depth,
        num_paths,
        time_steps,
        batch_size,
        time_aug,
        max_batch_signature_bytes,
        n_jobs,
    )
    if time_aug:
        print(
            f"[note] time augmentation is ON: effective signature dimension is "
            f"{effective_dim} (d+1), not {dimension}."
        )

    # -- Build processes and per-process accumulators ----------------------
    specs = build_process_specs(dimension, include_common_shock=include_common_shock)
    sig_len = signature_length(effective_dim, depth)

    sig_stats = {name: RunningSignatureStats(sig_len) for name in specs}
    stat_stats = {name: ProcessStatistics(dimension) for name in specs}
    sample_paths: dict[str, np.ndarray] = {}

    # -- Stream over processes and batches ---------------------------------
    # Deterministic per-process seed offsets so the run is fully reproducible
    # (Python's built-in hash() is randomised per process, so we must NOT use it).
    proc_offsets = {name: i for i, name in enumerate(specs)}
    for name, spec in specs.items():
        # A dedicated, deterministic generator per (config, process) keeps the run
        # reproducible while making the processes independent of one another.
        # We fold the depth in too, so the two configs do not share increments.
        proc_seed = seed + 1000 * depth + 37 * proc_offsets[name]
        rng = np.random.default_rng(proc_seed)

        t0 = time.time()
        n_done = 0
        remaining = num_paths
        while remaining > 0:
            this_batch = min(batch_size, remaining)

            # 1. simulate a batch of paths (+ their increments)
            paths, increments = simulate_batch(
                spec, this_batch, time_steps, dt, rng, time_aug=time_aug
            )

            # 2. update the classical statistical estimators
            stat_stats[name].update(paths, increments)

            # 3. compute signatures for the batch and fold into the running mean
            batch_sigs = compute_batch_signatures(paths, depth, n_jobs=n_jobs)
            sig_stats[name].update(batch_sigs)

            # keep a few example paths from the very first batch for the figure
            if name not in sample_paths:
                sample_paths[name] = paths[: min(3, this_batch)].copy()

            # 4. discard the batch (paths, increments, batch_sigs go out of scope)
            n_done += this_batch
            remaining -= this_batch

        elapsed = time.time() - t0
        print(
            f"  process {name}: {n_done:,} paths done in {elapsed:5.1f}s "
            f"({specs[name].description})"
        )

    # -- Collect final summaries -------------------------------------------
    summaries = {name: stat_stats[name].summary() for name in specs}
    expected_sigs = {name: sig_stats[name].expected_signature() for name in specs}
    # Per-coordinate std from Process A, used for the standardised distances.
    coord_std_A = sig_stats["A"].coordinate_std()

    # Comparisons: add A_vs_D if the common-shock process exists.
    comparisons = list(BASE_COMPARISONS)
    if "D" in specs:
        comparisons.append(("A_vs_D", "A", "D"))

    # -- Build tables ------------------------------------------------------
    config_rows = []
    for name, spec in specs.items():
        config_rows.append(
            {
                "config_name": config_name,
                "process": name,
                "dimension": dimension,
                "signature_depth": depth,
                "time_steps": time_steps,
                "num_paths": num_paths,
                "batch_size": batch_size,
                "covariance_structure": spec.cov_structure,
                "mean_structure": spec.mean_structure,
            }
        )
    config_table = pd.DataFrame(config_rows)

    stat_summary_rows = []
    for name in specs:
        s = summaries[name]
        stat_summary_rows.append(
            {
                "config_name": config_name,
                "process": name,
                "terminal_mean_norm": s["terminal_mean_norm"],
                "avg_terminal_variance": s["avg_terminal_variance"],
                "avg_increment_variance": s["avg_increment_variance"],
                "notes": specs[name].description,
            }
        )
    stat_summary_table = pd.DataFrame(stat_summary_rows)

    stat_distance_table = build_statistical_distance_table(
        config_name, summaries, comparisons
    )
    signature_distance_table = build_signature_distance_table(
        config_name, depth, effective_dim, expected_sigs, comparisons
    )
    levelwise_table = build_levelwise_distance_table(
        config_name, depth, effective_dim, expected_sigs, comparisons,
        coordinate_std=coord_std_A,
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
    parser.add_argument("--num-paths", type=int, default=10_000,
                        help="Paths per process (configurable; scale toward 100000).")
    parser.add_argument("--time-steps", type=int, default=300)
    parser.add_argument("--dt", type=float, default=1.0,
                        help="Time increment; 1.0 keeps mu/Sigma as per-step values.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override the per-config default batch size.")
    parser.add_argument(
        "--max-batch-signature-gb",
        type=float,
        default=DEFAULT_MAX_BATCH_SIGNATURE_GB,
        help="Largest signature batch to allow in memory before auto-shrinking "
             "the batch size.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="CPU threads passed to pysiglib (-1 means all available cores).",
    )
    parser.add_argument("--time-aug", action="store_true",
                        help="Append a time channel (effective dimension becomes d+1).")
    parser.add_argument("--no-common-shock", action="store_true",
                        help="Skip optional Process D (common shock).")
    parser.add_argument("--config", choices=[c["config_name"] for c in CONFIGS],
                        default=None, help="Run only one configuration.")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Write tables/figures under this dir instead of the "
                             "default results/ (useful for scaling tests so the "
                             "canonical results are not overwritten).")
    args = parser.parse_args()
    max_batch_signature_bytes = int(args.max_batch_signature_gb * (1024 ** 3))

    results_root = Path(args.results_dir) if args.results_dir else HERE / "results"
    tables_dir = results_root / "tables"
    figures_dir = results_root / "figures"
    ensure_dirs(tables_dir, figures_dir)

    selected = [c for c in CONFIGS if args.config is None or c["config_name"] == args.config]

    # Accumulate each config's tables so we can write combined CSVs at the end.
    all_results = []
    for config in selected:
        result = run_configuration(
            config,
            num_paths=args.num_paths,
            time_steps=args.time_steps,
            dt=args.dt,
            seed=args.seed,
            time_aug=args.time_aug,
            include_common_shock=not args.no_common_shock,
            batch_size_override=args.batch_size,
            max_batch_signature_bytes=max_batch_signature_bytes,
            n_jobs=args.n_jobs,
        )
        all_results.append(result)
        # Per-config sample-path figure (first config's processes are enough, but
        # we save one figure using whichever config ran first below).

    # -- Concatenate tables across configs and write the CSVs --------------
    def _concat(key):
        return pd.concat([r[key] for r in all_results], ignore_index=True)

    _concat("config_table").to_csv(tables_dir / "highdim_process_config.csv", index=False)
    _concat("stat_summary_table").to_csv(
        tables_dir / "highdim_statistical_summary.csv", index=False
    )
    _concat("stat_distance_table").to_csv(
        tables_dir / "highdim_statistical_distances.csv", index=False
    )
    signature_distance_all = _concat("signature_distance_table")
    signature_distance_all.to_csv(
        tables_dir / "highdim_signature_distances.csv", index=False
    )
    levelwise_all = _concat("levelwise_table")
    levelwise_all.to_csv(
        tables_dir / "highdim_signature_levelwise_distances.csv", index=False
    )

    # -- Figures -----------------------------------------------------------
    plot_signature_distance_by_depth(
        levelwise_all, figures_dir / "highdim_signature_distance_by_depth.png"
    )
    plot_levelwise_signature_distance(
        levelwise_all, figures_dir / "highdim_levelwise_signature_distance.png"
    )
    plot_statistical_distances(
        _concat("stat_distance_table"),
        figures_dir / "highdim_statistical_distances.png",
    )
    # Sample-paths figure uses the FIRST selected config's example paths.
    plot_sample_paths(
        all_results[0]["sample_paths"],
        figures_dir / "sample_paths_first_5_dimensions.png",
    )

    print("=" * 70)
    print("DONE. Tables written to results/tables/ and figures to results/figures/.")
    print("=" * 70)


if __name__ == "__main__":
    main()
