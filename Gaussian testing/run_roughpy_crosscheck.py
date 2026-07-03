"""
run_roughpy_crosscheck.py
=========================

Validation script: recomputes the A/B/C expected signatures with BOTH pysiglib
and roughpy and checks they agree to numerical precision.

If this passes, we know our signature pipeline is trustworthy and we can rely on
the pysiglib results in `run_gaussian_signature_test.py`.

Run from the project root (mind the quotes -- the folder name has a space):

    python "Gaussian testing/run_roughpy_crosscheck.py"
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Resolve paths relative to THIS file so the space in the folder name and the
# working directory never matter.
HERE = Path(__file__).resolve().parent
SRC_DIR = HERE / "src"
sys.path.insert(0, str(SRC_DIR))

from simulate_gaussian import create_gaussian_test_processes  # noqa: E402
from roughpy_crosscheck import compare_libraries  # noqa: E402

# Same settings as the main experiment so we validate exactly what we report.
# (A smaller n_paths would still be a valid check; we keep it identical here.)
N_PATHS = 1000
PATH_LENGTH = 20
DIMENSION = 1
LEVELS = [1, 2, 3, 4, 5]
SEED = 42

# How close is "close enough"? Both libraries use double precision, so genuine
# agreement should be at the level of tiny floating-point rounding.
TOLERANCE = 1e-9

TABLES_DIR = HERE / "results" / "tables"


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ROUGHPY vs PYSIGLIB CROSS-CHECK")
    print("=" * 70)
    print("Recomputing the same expected signatures with two independent")
    print("libraries and comparing them.\n")

    processes = create_gaussian_test_processes(
        n_paths=N_PATHS,
        path_length=PATH_LENGTH,
        dimension=DIMENSION,
        seed=SEED,
    )

    rows = []
    all_ok = True
    for name, info in processes.items():
        for level in LEVELS:
            result = compare_libraries(info["paths"], degree=level)
            difference = result["max_abs_difference"]
            passed = difference < TOLERANCE
            all_ok = all_ok and passed

            rows.append(
                {
                    "process": name,
                    "level": level,
                    "max_abs_difference": difference,
                    "agrees_within_tolerance": passed,
                }
            )
            status = "OK" if passed else "MISMATCH"
            print(f"  Process {name}, level {level}: max abs diff = {difference:.2e}  [{status}]")

    # Save the comparison as a small CSV so the evidence is on disk.
    report = pd.DataFrame(rows)
    out_csv = TABLES_DIR / "roughpy_pysiglib_agreement.csv"
    report.to_csv(out_csv, index=False)

    print("\n" + "=" * 70)
    if all_ok:
        print("RESULT: PASS - roughpy and pysiglib agree to numerical precision.")
        print("Our signature pipeline is validated by two independent libraries.")
    else:
        print("RESULT: FAIL - the two libraries disagree. Investigate before")
        print("trusting the main results!")
    print("=" * 70)
    print(f"Saved comparison table to: {out_csv.relative_to(HERE)}")

    # A non-zero exit code makes this usable as an automated test if desired.
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
