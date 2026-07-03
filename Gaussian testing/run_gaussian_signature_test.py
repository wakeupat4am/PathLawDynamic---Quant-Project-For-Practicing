"""
run_gaussian_signature_test.py
==============================

MAIN SCRIPT for the first synthetic experiment.

What this script does, step by step:
    1. Create the output folders (if they do not exist yet).
    2. Simulate three Gaussian processes A, B, C with known mean/volatility.
    3. Measure their empirical moments (mean/variance) as a sanity check.
    4. Compute truncated signatures at levels [1, 2, 3, 4, 5].
    5. Average them into "expected signatures" (one fingerprint per process).
    6. Compare the fingerprints (L2 distance) for A_vs_B, A_vs_C, B_vs_C.
    7. Save two CSV tables and two figures inside this folder's results/.
    8. Print a beginner-friendly summary of what happened.

Run it from the project root (note the quotes -- the folder name has a space):

    python "Gaussian testing/run_gaussian_signature_test.py"
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Make the local `src/` package importable no matter where we run this from.
# We resolve paths relative to THIS file, so the space in "Gaussian testing"
# and the current working directory never cause problems.
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent          # .../Gaussian testing
SRC_DIR = HERE / "src"
sys.path.insert(0, str(SRC_DIR))

from simulate_gaussian import create_gaussian_test_processes  # noqa: E402
from signature_features import compute_signatures_for_levels  # noqa: E402
from analysis import (  # noqa: E402
    compute_signature_distances,
    compute_sample_moments,
)
from plotting import (  # noqa: E402
    plot_signature_distance_vs_level,
    plot_sample_paths,
)

# ---------------------------------------------------------------------------
# Default experiment settings (change these to explore different setups).
# ---------------------------------------------------------------------------
N_PATHS = 1000        # number of example paths per process
PATH_LENGTH = 20      # time steps per path
DIMENSION = 1         # single channel (one time series)
LEVELS = [1, 2, 3, 4, 5]  # signature truncation levels to test
SEED = 42             # master random seed => fully reproducible run

# Output locations, all INSIDE this folder (never touching the parent project).
TABLES_DIR = HERE / "results" / "tables"
FIGURES_DIR = HERE / "results" / "figures"


def main() -> None:
    # --- Step 1: make sure the output folders exist -------------------------
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GAUSSIAN SIGNATURE TESTING")
    print("=" * 70)
    print(
        f"Settings: n_paths={N_PATHS}, path_length={PATH_LENGTH}, "
        f"dimension={DIMENSION}, levels={LEVELS}, seed={SEED}\n"
    )

    # --- Step 2: simulate the three processes -------------------------------
    print("[1/6] Simulating Gaussian processes A, B, C ...")
    processes = create_gaussian_test_processes(
        n_paths=N_PATHS,
        path_length=PATH_LENGTH,
        dimension=DIMENSION,
        seed=SEED,
    )
    for name, info in processes.items():
        print(f"      Process {name}: {info['description']}")

    # --- Step 3: empirical sample moments (ground truth) --------------------
    print("\n[2/6] Measuring empirical moments of the increments ...")
    moment_rows = []
    for name, info in processes.items():
        moments = compute_sample_moments(info["paths"])
        # For dimension = 1 these arrays hold a single number; take element 0
        # for a clean scalar in the table.
        mean_val = float(moments["empirical_mean_of_increments"][0])
        var_val = float(moments["empirical_variance_of_increments"][0])
        moment_rows.append(
            {
                "process": name,
                "true_mean": info["mean"],
                "true_volatility": info["volatility"],
                "true_variance": info["volatility"] ** 2,
                "empirical_mean_of_increments": mean_val,
                "empirical_variance_of_increments": var_val,
            }
        )
        print(
            f"      Process {name}: empirical mean={mean_val:+.4f} "
            f"(true {info['mean']:+.4f}), "
            f"empirical variance={var_val:.4f} "
            f"(true {info['volatility'] ** 2:.4f})"
        )

    moments_df = pd.DataFrame(moment_rows)

    # --- Step 4 & 5: signatures and expected signatures ---------------------
    print("\n[3/6] Computing truncated signatures for each level ...")
    # compute_signatures_for_levels wants {name: paths_array}.
    process_paths = {name: info["paths"] for name, info in processes.items()}
    signature_results = compute_signatures_for_levels(process_paths, LEVELS)
    print("      Done (signatures + expected signatures stored for every level).")

    # --- Step 6: distances between expected signatures ----------------------
    print("\n[4/6] Comparing expected signatures (L2 distances) ...")
    distance_df = compute_signature_distances(signature_results)

    # --- Save the CSV tables ------------------------------------------------
    print("\n[5/6] Saving result tables ...")
    distance_csv = TABLES_DIR / "signature_distance_summary.csv"
    moments_csv = TABLES_DIR / "empirical_moments_summary.csv"
    distance_df.to_csv(distance_csv, index=False)
    moments_df.to_csv(moments_csv, index=False)
    print(f"      Saved: {distance_csv.relative_to(HERE)}")
    print(f"      Saved: {moments_csv.relative_to(HERE)}")

    # --- Generate the figures ----------------------------------------------
    print("\n[6/6] Generating figures ...")
    distance_fig = FIGURES_DIR / "signature_distance_vs_level.png"
    paths_fig = FIGURES_DIR / "sample_paths_A_B_C.png"
    plot_signature_distance_vs_level(distance_df, distance_fig)
    plot_sample_paths(processes, paths_fig)
    print(f"      Saved: {distance_fig.relative_to(HERE)}")
    print(f"      Saved: {paths_fig.relative_to(HERE)}")

    # --- Final beginner-friendly summary ------------------------------------
    print("\n" + "=" * 70)
    print("RUN SUMMARY")
    print("=" * 70)
    print(
        "We compared three Gaussian processes using truncated path signatures.\n"
        "Read the distance table like this:\n"
    )

    # Show the distance table grouped by comparison so the trend is obvious.
    for comparison in ["A_vs_B", "A_vs_C", "B_vs_C"]:
        subset = distance_df[distance_df["comparison"] == comparison].sort_values("level")
        distances = ", ".join(
            f"L{int(r.level)}={r.l2_distance_between_expected_signatures:.3f}"
            for r in subset.itertuples()
        )
        print(f"  {comparison}:  {distances}")

    print(
        "\nWhat to look for:\n"
        "  * A_vs_B (mean difference)     -> should be non-zero already at level 1.\n"
        "  * A_vs_C (variance difference) -> should grow noticeably once level 2 is added.\n"
        "  * From level 3 onward, extra distance should be small for Gaussian data,\n"
        "    because a Gaussian is fully characterised by its first two moments.\n"
        "\nSee results/tables/ for the CSVs and results/figures/ for the plots.\n"
        "The README.md in this folder explains everything in detail."
    )


if __name__ == "__main__":
    main()
