"""
plotting.py
===========

Small plotting helpers for the experiment. We use matplotlib with the "Agg"
backend so the script can run and save PNG files even on machines with no
display (e.g. a server or CI). Figures are saved, not shown interactively.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Use a non-interactive backend BEFORE importing pyplot. This must come first.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (import after backend selection)
import numpy as np
import pandas as pd


def plot_signature_distance_vs_level(distance_df: pd.DataFrame, output_path: Path) -> None:
    """Plot L2 distance between expected signatures against truncation level.

    One line per comparison (A_vs_B, A_vs_C, B_vs_C). This is the main figure:
    it shows at which level each type of difference "switches on".

    Parameters
    ----------
    distance_df : pd.DataFrame
        Output of `compute_signature_distances`.
    output_path : pathlib.Path
        Where to save the PNG (parent folders should already exist).
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Draw one line per comparison so the reader can compare their behaviour.
    for comparison, group in distance_df.groupby("comparison"):
        group = group.sort_values("level")
        ax.plot(
            group["level"],
            group["l2_distance_between_expected_signatures"],
            marker="o",
            label=comparison,
        )

    ax.set_xlabel("Signature truncation level")
    ax.set_ylabel("L2 distance between expected signatures")
    ax.set_title("How signature distance grows with truncation level")
    # Force integer ticks on the x-axis (levels are whole numbers).
    ax.set_xticks(sorted(distance_df["level"].unique()))
    ax.legend(title="Comparison")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_sample_paths(processes: dict, output_path: Path, n_examples: int = 5) -> None:
    """Plot a few example paths from each process, side by side.

    This is a sanity-check figure: process C (double volatility) should visibly
    wander further than A, and process B should drift gently upward.

    Parameters
    ----------
    processes : dict
        Maps process name -> info dict containing key "paths" of shape
        (n_paths, path_length, dimension). Only channel 0 is plotted.
    output_path : pathlib.Path
        Where to save the PNG.
    n_examples : int
        How many example paths to draw per process.
    """
    names = list(processes.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4), sharey=True)

    # Guard against the single-process edge case (axes would not be a list).
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        info = processes[name]
        paths = info["paths"]
        # Plot the first `n_examples` paths, channel 0 only.
        for i in range(min(n_examples, paths.shape[0])):
            ax.plot(paths[i, :, 0], alpha=0.8)

        # Include a self-explaining subtitle. Non-Gaussian processes carry a
        # 'label' describing what makes them special (skew / heavy tails);
        # fall back to mean/volatility for the plain Gaussian processes.
        subtitle = info.get("label", f"mean={info['mean']}, vol={info['volatility']}")
        ax.set_title(f"Process {name}\n{subtitle}")
        ax.set_xlabel("Time step")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Cumulative value")
    fig.suptitle("Example simulated Gaussian paths")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
