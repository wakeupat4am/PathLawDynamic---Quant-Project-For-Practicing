"""
plotting.py
===========

Figures for the high-dimensional benchmark. As in stage 1 we use the non-
interactive "Agg" backend so PNGs are written even with no display attached.

Four figures are produced:

    1. highdim_signature_distance_by_depth.png   -- whole-signature distance vs
       depth, one line per comparison, one subplot per config.
    2. highdim_levelwise_signature_distance.png  -- per-level distance vs level,
       one line per comparison, one subplot per config (normalised scale).
    3. highdim_statistical_distances.png         -- grouped bars of mean- and
       covariance-distance per comparison.
    4. sample_paths_first_5_dimensions.png       -- a few example paths, channels
       0-4, one subplot per process.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def plot_signature_distance_by_depth(
    levelwise_df: pd.DataFrame, output_path: Path
) -> None:
    """Whole-signature distance as depth grows, per config.

    We build a *cumulative* whole-signature distance across levels: at "depth k"
    we combine levels 1..k. Because levels occupy disjoint coordinate blocks, the
    L2 distance over levels 1..k is the sqrt of the sum of squared per-level raw
    distances -- exactly the whole-signature distance truncated at depth k. This
    turns the level table into a clean distance-vs-depth curve without recomputing
    signatures.
    """
    configs = sorted(levelwise_df["config_name"].unique())
    fig, axes = plt.subplots(
        1, len(configs), figsize=(7 * len(configs), 5), squeeze=False
    )

    for ax, config in zip(axes[0], configs):
        sub = levelwise_df[levelwise_df["config_name"] == config]
        for comparison, group in sub.groupby("comparison"):
            group = group.sort_values("level")
            # Cumulative L2 across levels = sqrt(cumsum(raw_level^2)).
            cum = np.sqrt(np.cumsum(group["raw_level_l2_distance"].values ** 2))
            ax.plot(group["level"].values, cum, marker="o", label=comparison)
        ax.set_title(f"{config}")
        ax.set_xlabel("Signature depth (levels 1..k combined)")
        ax.set_ylabel("Raw whole-signature L2 distance")
        ax.set_xticks(sorted(sub["level"].unique()))
        ax.legend(title="Comparison")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Signature distance vs truncation depth")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_levelwise_signature_distance(
    levelwise_df: pd.DataFrame, output_path: Path
) -> None:
    """Per-level signature distance, per config -- the key "switch-on" figure.

    We plot the STANDARDISED distance (each coordinate divided by Process A's own
    signature-coordinate std). This is the only one of the three views that
    removes BOTH the "more coordinates" and the "bigger numbers" scale effects, so
    the genuine structure becomes visible: the mean-shift comparison (A_vs_B)
    peaks at level 1, while the covariance-geometry comparisons (A_vs_C, A_vs_D)
    peak at level 2. The raw / level-normalised columns (which just grow with
    level) are still in the CSV for reference.
    """
    metric = "standardized_level_distance"
    has_std = metric in levelwise_df.columns
    if not has_std:  # fall back gracefully if standardisation was disabled
        metric = "normalized_level_distance"

    configs = sorted(levelwise_df["config_name"].unique())
    fig, axes = plt.subplots(
        1, len(configs), figsize=(7 * len(configs), 5), squeeze=False
    )

    for ax, config in zip(axes[0], configs):
        sub = levelwise_df[levelwise_df["config_name"] == config]
        for comparison, group in sub.groupby("comparison"):
            group = group.sort_values("level")
            ax.plot(
                group["level"].values,
                group[metric].values,
                marker="o",
                label=comparison,
            )
        ax.set_title(f"{config}")
        ax.set_xlabel("Signature level")
        ylabel = (
            "Standardised distance (÷ Process-A coord std)"
            if has_std
            else "Level-normalised distance (per-coordinate RMS)"
        )
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(sub["level"].unique()))
        ax.legend(title="Comparison")
        ax.grid(True, alpha=0.3)

    title = (
        "Level-wise signature distance (standardised — reveals switch-on structure)"
        if has_std
        else "Level-wise signature distance (normalised)"
    )
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_statistical_distances(stat_df: pd.DataFrame, output_path: Path) -> None:
    """Grouped bar chart of statistical mean- and covariance-distances.

    One group of bars per (config, comparison); two bars per group for the
    terminal mean L2 distance and the terminal covariance Frobenius distance.
    """
    stat_df = stat_df.copy()
    stat_df["group"] = stat_df["config_name"] + "\n" + stat_df["comparison"]
    groups = stat_df["group"].tolist()
    x = np.arange(len(groups))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(groups)), 5))
    ax.bar(
        x - width / 2,
        stat_df["mean_vector_l2_distance"].values,
        width,
        label="Terminal mean L2 distance",
    )
    ax.bar(
        x + width / 2,
        stat_df["covariance_frobenius_distance"].values,
        width,
        label="Terminal covariance Frobenius distance",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=8)
    ax.set_ylabel("Distance")
    ax.set_title("Classical statistical distances between processes")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_sample_paths(
    sample_paths: dict[str, np.ndarray], output_path: Path, n_examples: int = 3
) -> None:
    """Plot a few example paths for the first five channels, one subplot per process.

    Parameters
    ----------
    sample_paths : dict
        ``{process_name: paths}`` where paths has shape (n, time_steps, dim). Only
        channels 0..4 of the first ``n_examples`` paths are drawn.
    """
    names = list(sample_paths.keys())
    fig, axes = plt.subplots(
        1, len(names), figsize=(4.5 * len(names), 4), sharey=True, squeeze=False
    )

    for ax, name in zip(axes[0], names):
        paths = sample_paths[name]
        n_dims = min(5, paths.shape[2])
        for i in range(min(n_examples, paths.shape[0])):
            for c in range(n_dims):
                # Colour by channel, vary alpha lightly per example.
                ax.plot(paths[i, :, c], color=f"C{c}", alpha=0.8, linewidth=1.0)
        ax.set_title(f"Process {name}")
        ax.set_xlabel("Time step")
        ax.grid(True, alpha=0.3)

    axes[0][0].set_ylabel("Cumulative value")
    fig.suptitle("Example paths, channels 0-4 (colour = channel)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
