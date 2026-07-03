"""
run_higher_moment_test.py
=========================

STAGE 2 main script: do signatures recover HIGHER moments (skewness, kurtosis)?

Stage 1 (`run_gaussian_signature_test.py`) showed signatures recover the mean
(level 1) and variance (level 2) of Gaussian processes. But a Gaussian has no
independent information beyond its first two moments. This stage adds two
NON-Gaussian processes, each matched to the Gaussian in mean AND variance, so
that any difference the signatures detect must come from a higher moment:

    G : Gaussian     (baseline)
    S : Skew-normal  (adds a THIRD moment  -> skewness)   -> expect a jump at level 3
    T : Student-t    (adds a FOURTH moment -> kurtosis)   -> expect a jump at level 4

Signatures are computed with pysiglib (the same engine as stage 1).

Steps:
    1. Create output folders.
    2. Simulate G, S, T (more paths than stage 1: higher moments are noisier).
    3. Measure empirical moments up to kurtosis (a sanity check).
    4. Compute truncated signatures at levels [1..5] with pysiglib.
    5. Compare expected signatures (G_vs_S, G_vs_T, S_vs_T) at every level.
    6. Save CSV tables and figures inside this folder's results/.
    7. Print a beginner-friendly summary.

Run from the project root (mind the quotes -- the folder name has a space):

    python "Gaussian testing/run_higher_moment_test.py"
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Resolve paths relative to THIS file so the space in the folder name and the
# current working directory never matter.
HERE = Path(__file__).resolve().parent
SRC_DIR = HERE / "src"
sys.path.insert(0, str(SRC_DIR))

from simulate_nongaussian import create_higher_moment_processes  # noqa: E402
from signature_features import compute_signatures_for_levels  # noqa: E402
from analysis import (  # noqa: E402
    compute_signature_distances,
    compute_sample_moments,
    interpretation_higher_moment,
)
from plotting import (  # noqa: E402
    plot_signature_distance_vs_level,
    plot_sample_paths,
)

# --- Default experiment settings ------------------------------------------
N_PATHS = 4000        # more than stage 1: higher-moment signals are weaker
PATH_LENGTH = 20      # same window length as the finance data (20 days)
DIMENSION = 1         # single channel
LEVELS = [1, 2, 3, 4, 5]
SEED = 42
SKEW_ALPHA = 8.0      # skew-normal shape (larger => more skew)
T_DF = 5.0            # Student-t degrees of freedom (smaller => heavier tails)

TABLES_DIR = HERE / "results" / "tables"
FIGURES_DIR = HERE / "results" / "figures"

# The three comparisons for this stage, as (label, first_process, second_process).
COMPARISONS = [
    ("G_vs_S", "G", "S"),  # isolates skewness  (third moment)
    ("G_vs_T", "G", "T"),  # isolates kurtosis  (fourth moment)
    ("S_vs_T", "S", "T"),  # combined higher-moment difference
]


def main() -> None:
    # --- Step 1: output folders --------------------------------------------
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("HIGHER-MOMENT SIGNATURE TESTING (skewness & kurtosis)")
    print("=" * 70)
    print(
        f"Settings: n_paths={N_PATHS}, path_length={PATH_LENGTH}, "
        f"dimension={DIMENSION}, levels={LEVELS}, seed={SEED}, "
        f"skew_alpha={SKEW_ALPHA}, t_df={T_DF}\n"
    )

    # --- Step 2: simulate G, S, T ------------------------------------------
    print("[1/6] Simulating processes G (Gaussian), S (skew), T (heavy-tailed) ...")
    processes = create_higher_moment_processes(
        n_paths=N_PATHS,
        path_length=PATH_LENGTH,
        dimension=DIMENSION,
        seed=SEED,
        skew_alpha=SKEW_ALPHA,
        t_df=T_DF,
    )
    for name, info in processes.items():
        print(f"      Process {name}: {info['description']}")

    # --- Step 3: empirical moments up to kurtosis (sanity check) -----------
    print("\n[2/6] Measuring empirical moments (mean, variance, skew, kurtosis) ...")
    moment_rows = []
    for name, info in processes.items():
        m = compute_sample_moments(info["paths"])
        mean_val = float(m["empirical_mean_of_increments"][0])
        var_val = float(m["empirical_variance_of_increments"][0])
        skew_val = float(m["empirical_skewness_of_increments"][0])
        kurt_val = float(m["empirical_excess_kurtosis_of_increments"][0])
        moment_rows.append(
            {
                "process": name,
                "empirical_mean": mean_val,
                "empirical_variance": var_val,
                "empirical_skewness": skew_val,
                "empirical_excess_kurtosis": kurt_val,
            }
        )
        print(
            f"      {name}: mean={mean_val:+.3f}  var={var_val:.3f}  "
            f"skew={skew_val:+.3f}  excess_kurtosis={kurt_val:+.3f}"
        )
    moments_df = pd.DataFrame(moment_rows)
    print(
        "      (Note: mean~0 and variance~1 are MATCHED across G/S/T on purpose,\n"
        "       so only skew and kurtosis differ.)"
    )

    # --- Step 4: signatures with pysiglib ----------------------------------
    print("\n[3/6] Computing truncated signatures with pysiglib ...")
    process_paths = {name: info["paths"] for name, info in processes.items()}
    signature_results = compute_signatures_for_levels(process_paths, LEVELS)
    print("      Done (signatures + expected signatures stored for every level).")

    # --- Step 5: distances between expected signatures ---------------------
    print("\n[4/6] Comparing expected signatures (L2 distances) ...")
    distance_df = compute_signature_distances(
        signature_results,
        comparisons=COMPARISONS,
        interpretation_fn=interpretation_higher_moment,
    )

    # --- Save CSV tables ---------------------------------------------------
    print("\n[5/6] Saving result tables ...")
    distance_csv = TABLES_DIR / "higher_moment_distance_summary.csv"
    moments_csv = TABLES_DIR / "higher_moment_empirical_moments.csv"
    distance_df.to_csv(distance_csv, index=False)
    moments_df.to_csv(moments_csv, index=False)
    print(f"      Saved: {distance_csv.relative_to(HERE)}")
    print(f"      Saved: {moments_csv.relative_to(HERE)}")

    # --- Generate figures --------------------------------------------------
    print("\n[6/6] Generating figures ...")
    distance_fig = FIGURES_DIR / "higher_moment_distance_vs_level.png"
    paths_fig = FIGURES_DIR / "sample_paths_G_S_T.png"
    plot_signature_distance_vs_level(distance_df, distance_fig)
    plot_sample_paths(processes, paths_fig)
    print(f"      Saved: {distance_fig.relative_to(HERE)}")
    print(f"      Saved: {paths_fig.relative_to(HERE)}")

    # --- Final beginner-friendly summary -----------------------------------
    print("\n" + "=" * 70)
    print("RUN SUMMARY")
    print("=" * 70)
    print(
        "We compared three matched-variance processes with truncated signatures.\n"
        "Because mean and variance are identical, any difference must be a\n"
        "HIGHER moment. Distances by level:\n"
    )
    for label, _, _ in COMPARISONS:
        subset = distance_df[distance_df["comparison"] == label].sort_values("level")
        distances = ", ".join(
            f"L{int(r.level)}={r.l2_distance_between_expected_signatures:.3f}"
            for r in subset.itertuples()
        )
        print(f"  {label}:  {distances}")

    print(
        "\nWhat to look for:\n"
        "  * G_vs_S (skewness)  -> small at levels 1-2, JUMPS at LEVEL 3 (third moment).\n"
        "  * G_vs_T (kurtosis)  -> small at levels 1-3 (t is symmetric),\n"
        "                          JUMPS at LEVEL 4 (fourth moment / heavy tails).\n"
        "\nThis extends stage 1: signatures recover not only mean (L1) and\n"
        "variance (L2), but also skewness (L3) and kurtosis (L4) -- you just have\n"
        "to truncate at a high enough level to 'see' each moment.\n"
        "\nSee results/tables/ and results/figures/. The README explains more."
    )


if __name__ == "__main__":
    main()
