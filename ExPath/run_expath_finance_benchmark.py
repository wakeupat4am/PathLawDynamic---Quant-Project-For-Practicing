from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("ExPath/.matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("ExPath/.cache").resolve()))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[1]
EXPATH_DIR = ROOT_DIR / "ExPath"
RESULTS_DIR = EXPATH_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

sys.path.insert(0, str(ROOT_DIR / "Application"))
from run_realdata_signature_benchmark import (
    KNOWN_STRESS_PERIODS,
    build_score_table,
    compute_consecutive_distances,
    compute_trailing_distance_scores,
    find_top_episodes,
    load_realdata_inputs,
    returns_to_cumulative_paths,
    save_top_dates_table,
    standardize_features,
    summarize_known_periods,
)


WINDOW_SIZE = 20
ANOMALY_LOOKBACK = 60
ALPHA_GRID = [0.0, 0.15, 0.35, 0.70, 1.20]
CLOCK_TYPES = ["linear", "movement"]
TOP_EPISODES_TO_PRINT = 10
SIGNATURE_SCORE_FILE = (
    ROOT_DIR / "Application" / "results" / "tables" / "realdata_anomaly_scores.csv"
)


def ensure_directories() -> None:
    """Create ExPath result folders and writable matplotlib caches."""
    EXPATH_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)


def build_clock_channel(paths: np.ndarray, clock_type: str) -> np.ndarray:
    """Construct the clock channel used in the paper's augmented path."""
    n_paths, n_steps, _ = paths.shape
    if clock_type == "linear":
        base_clock = np.linspace(0.0, 1.0, n_steps, dtype=np.float64)
        return np.broadcast_to(base_clock, (n_paths, n_steps))

    if clock_type == "movement":
        step_lengths = np.linalg.norm(np.diff(paths, axis=1), axis=2)
        cumulative = np.cumsum(step_lengths, axis=1)
        total = cumulative[:, -1]
        safe_total = np.where(total > 1e-12, total, 1.0)
        movement_clock = np.zeros((n_paths, n_steps), dtype=np.float64)
        movement_clock[:, 1:] = cumulative / safe_total[:, None]
        return movement_clock

    raise ValueError(f"Unknown clock_type: {clock_type}")


def augment_paths(paths: np.ndarray, clock_type: str) -> np.ndarray:
    """Append the chosen clock channel to each path."""
    clock = build_clock_channel(paths, clock_type)
    return np.concatenate([paths, clock[..., None]], axis=2)


def compute_weighted_increments(augmented_paths: np.ndarray, alpha: float) -> np.ndarray:
    """Apply the paper's exponentially decayed increment weighting."""
    increments = np.diff(augmented_paths, axis=1)
    clock = augmented_paths[:, :, -1]
    ages = clock[:, -1][:, None] - clock[:, :-1]
    weights = np.exp(-alpha * ages)
    return increments * weights[:, :, None]


def compute_signed_area_features(weighted_increments: np.ndarray) -> np.ndarray:
    """Compute antisymmetric second-order signed-area features."""
    n_paths, _, dimension = weighted_increments.shape
    area = np.zeros((n_paths, dimension, dimension), dtype=np.float64)
    prefix = np.zeros((n_paths, dimension), dtype=np.float64)

    for step in range(weighted_increments.shape[1]):
        current = weighted_increments[:, step, :]
        area += prefix[:, :, None] * current[:, None, :]
        area -= current[:, :, None] * prefix[:, None, :]
        prefix += current

    area *= 0.5
    upper_i, upper_j = np.triu_indices(dimension, k=1)
    return area[:, upper_i, upper_j]


def compute_path_statistics(
    weighted_increments: np.ndarray, displacement: np.ndarray
) -> np.ndarray:
    """Compute the scalar path statistics from the paper in a finance-friendly form."""
    step_lengths = np.linalg.norm(weighted_increments, axis=2)
    total_length = step_lengths.sum(axis=1)
    mean_length = step_lengths.mean(axis=1)
    std_length = step_lengths.std(axis=1)
    min_length = step_lengths.min(axis=1)
    max_length = step_lengths.max(axis=1)
    displacement_norm = np.linalg.norm(displacement, axis=1)
    safe_total_length = np.where(total_length > 1e-12, total_length, 1.0)
    straightness = displacement_norm / safe_total_length
    mean_squared_length = np.mean(np.square(step_lengths), axis=1)

    if step_lengths.shape[1] > 1:
        # The paper denotes the final scalar by Δl but the extracted text does not
        # define it explicitly. Here we use the mean absolute change in step length,
        # which is the closest stable finance adaptation of a step-length trend term.
        delta_length = np.mean(np.abs(np.diff(step_lengths, axis=1)), axis=1)
    else:
        delta_length = np.zeros(step_lengths.shape[0], dtype=np.float64)

    return np.column_stack(
        [
            total_length,
            mean_length,
            std_length,
            min_length,
            max_length,
            displacement_norm,
            straightness,
            mean_squared_length,
            delta_length,
        ]
    )


def compute_expath_features(paths: np.ndarray, alpha: float, clock_type: str) -> np.ndarray:
    """Compute the finance-adapted ExpPath feature vector for each window."""
    augmented_paths = augment_paths(paths, clock_type=clock_type)
    weighted_increments = compute_weighted_increments(augmented_paths, alpha=alpha)

    displacement = weighted_increments.sum(axis=1)
    signed_area = compute_signed_area_features(weighted_increments)
    stats = compute_path_statistics(weighted_increments, displacement)

    return np.concatenate([displacement, signed_area, stats], axis=1)


def evaluate_configuration(
    paths: np.ndarray,
    window_end_dates: pd.DatetimeIndex,
    alpha: float,
    clock_type: str,
) -> dict[str, object]:
    """Run anomaly scoring for one ExPath configuration."""
    raw_features = compute_expath_features(paths, alpha=alpha, clock_type=clock_type)
    standardized_features, _, _ = standardize_features(raw_features)
    trailing_score = compute_trailing_distance_scores(
        standardized_features, lookback=ANOMALY_LOOKBACK
    )
    consecutive_score = compute_consecutive_distances(standardized_features)
    stress_summary = summarize_known_periods(window_end_dates, trailing_score)

    global_median = float(np.nanmedian(trailing_score))
    mean_stress_ratio = float(stress_summary["mean_vs_global_median_ratio"].mean())
    covid_row = stress_summary.loc[stress_summary["period"] == "2020 COVID crash"]
    covid_ratio = float(covid_row["mean_vs_global_median_ratio"].iloc[0])
    top_episodes = find_top_episodes(window_end_dates, trailing_score)

    return {
        "alpha": alpha,
        "clock_type": clock_type,
        "feature_dim": raw_features.shape[1],
        "global_median_score": global_median,
        "mean_stress_ratio": mean_stress_ratio,
        "covid_ratio": covid_ratio,
        "objective_score": mean_stress_ratio,
        "raw_features": raw_features,
        "standardized_features": standardized_features,
        "trailing_score": trailing_score,
        "consecutive_score": consecutive_score,
        "stress_summary": stress_summary,
        "top_episodes": top_episodes,
    }


def build_configuration_table(results: list[dict[str, object]]) -> pd.DataFrame:
    """Summarize the grid search over alpha and clock type."""
    rows = []
    for result in results:
        rows.append(
            {
                "alpha": result["alpha"],
                "clock_type": result["clock_type"],
                "feature_dim": result["feature_dim"],
                "global_median_score": result["global_median_score"],
                "mean_stress_ratio": result["mean_stress_ratio"],
                "covid_ratio": result["covid_ratio"],
                "objective_score": result["objective_score"],
                "top_episode_peak_date": result["top_episodes"]["peak_date"].iloc[0],
                "top_episode_peak_score": result["top_episodes"]["peak_score"].iloc[0],
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["objective_score", "covid_ratio"], ascending=False
    ).reset_index(drop=True)


def build_feature_table(
    window_end_dates: pd.DatetimeIndex,
    raw_features: np.ndarray,
    standardized_features: np.ndarray,
) -> pd.DataFrame:
    """Save the best ExPath features for inspection."""
    raw_columns = [f"expath_raw_{i:03d}" for i in range(raw_features.shape[1])]
    z_columns = [f"expath_z_{i:03d}" for i in range(standardized_features.shape[1])]
    table = pd.DataFrame({"window_end_date": window_end_dates.astype(str)})
    for idx, column in enumerate(raw_columns):
        table[column] = raw_features[:, idx]
    for idx, column in enumerate(z_columns):
        table[column] = standardized_features[:, idx]
    return table


def load_signature_scores() -> pd.DataFrame:
    """Load the previously computed signature benchmark scores for comparison."""
    if not SIGNATURE_SCORE_FILE.exists():
        raise FileNotFoundError(
            "Signature benchmark results are missing. Run "
            "'Application/run_realdata_signature_benchmark.py' first."
        )
    return pd.read_csv(SIGNATURE_SCORE_FILE, parse_dates=["window_end_date"])


def build_comparison_table(
    window_end_dates: pd.DatetimeIndex,
    best_result: dict[str, object],
    best_path2: dict[str, object],
    signature_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine the best ExPath run with PATH2 and signature scores."""
    comparison = pd.DataFrame({"window_end_date": window_end_dates})
    comparison["expath_trailing_score"] = best_result["trailing_score"]
    comparison["expath_consecutive_distance"] = best_result["consecutive_score"]
    comparison["path2_trailing_score"] = best_path2["trailing_score"]
    comparison["path2_consecutive_distance"] = best_path2["consecutive_score"]
    comparison["signature_trailing_score"] = signature_scores["signature_trailing_score"]
    comparison["signature_consecutive_distance"] = signature_scores[
        "signature_consecutive_distance"
    ]

    columns = [
        "expath_trailing_score",
        "path2_trailing_score",
        "signature_trailing_score",
    ]
    correlation = comparison[columns].dropna().corr()
    correlation.index.name = "metric"
    return comparison, correlation


def plot_configuration_search(configuration_table: pd.DataFrame) -> None:
    """Visualize how the ExPath objective moves across alpha and clock choices."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for clock_type in CLOCK_TYPES:
        subset = configuration_table.loc[configuration_table["clock_type"] == clock_type]
        ax.plot(
            subset["alpha"],
            subset["objective_score"],
            marker="o",
            linewidth=1.2,
            label=f"{clock_type} clock",
        )

    ax.set_title("ExPath finance adaptation: stress-separation objective by alpha")
    ax.set_xlabel("Decay alpha")
    ax.set_ylabel("Mean stress / global median ratio")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "configuration_search.png", dpi=150)
    plt.close(fig)


def plot_best_scores(comparison_table: pd.DataFrame) -> None:
    """Plot best ExPath, PATH2, and signature anomaly scores together."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(
        comparison_table["window_end_date"],
        comparison_table["expath_trailing_score"],
        label="ExPath",
        linewidth=1.1,
    )
    axes[0].plot(
        comparison_table["window_end_date"],
        comparison_table["path2_trailing_score"],
        label="PATH2",
        linewidth=1.1,
    )
    axes[0].plot(
        comparison_table["window_end_date"],
        comparison_table["signature_trailing_score"],
        label="Signature",
        linewidth=1.1,
    )
    for _, start, end in KNOWN_STRESS_PERIODS:
        axes[0].axvspan(pd.Timestamp(start), pd.Timestamp(end), color="grey", alpha=0.08)
    axes[0].set_title("ExPath vs PATH2 vs signature anomaly scores")
    axes[0].set_ylabel("Distance")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    for column, label in [
        ("expath_consecutive_distance", "ExPath"),
        ("path2_consecutive_distance", "PATH2"),
        ("signature_consecutive_distance", "Signature"),
    ]:
        values = comparison_table[column].to_numpy(dtype=np.float64)
        valid = ~np.isnan(values)
        normalized = np.full_like(values, np.nan)
        normalized[valid] = (values[valid] - values[valid].mean()) / values[valid].std()
        axes[1].plot(comparison_table["window_end_date"], normalized, linewidth=1.0, label=label)

    axes[1].set_title("Normalized consecutive window distances")
    axes[1].set_xlabel("Window end date")
    axes[1].set_ylabel("Z-scored distance")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "expath_vs_path2_vs_signature.png", dpi=150)
    plt.close(fig)


def plot_best_expath_scores(comparison_table: pd.DataFrame) -> None:
    """Plot the best ExPath anomaly score alone for readability."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        comparison_table["window_end_date"],
        comparison_table["expath_trailing_score"],
        linewidth=1.1,
        color="tab:green",
    )
    threshold = np.nanquantile(comparison_table["expath_trailing_score"], 0.95)
    ax.axhline(threshold, color="tab:red", linestyle="--", linewidth=0.9)

    for label, start, end in KNOWN_STRESS_PERIODS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="grey", alpha=0.08)

    ax.set_title("Best ExPath finance anomaly score")
    ax.set_xlabel("Window end date")
    ax.set_ylabel("Distance")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "best_expath_anomaly_score.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ensure_directories()

    windows, window_end_dates = load_realdata_inputs()
    cumulative_paths = returns_to_cumulative_paths(windows)
    signature_scores = load_signature_scores()

    results = []
    for clock_type in CLOCK_TYPES:
        for alpha in ALPHA_GRID:
            results.append(
                evaluate_configuration(
                    cumulative_paths, window_end_dates=window_end_dates, alpha=alpha, clock_type=clock_type
                )
            )

    configuration_table = build_configuration_table(results)
    best_result = next(
        result
        for result in results
        if result["alpha"] == configuration_table.iloc[0]["alpha"]
        and result["clock_type"] == configuration_table.iloc[0]["clock_type"]
    )
    path2_table = configuration_table.loc[configuration_table["alpha"] == 0.0]
    best_path2 = next(
        result
        for result in results
        if result["alpha"] == path2_table.iloc[0]["alpha"]
        and result["clock_type"] == path2_table.iloc[0]["clock_type"]
    )

    comparison_table, correlation_table = build_comparison_table(
        window_end_dates=window_end_dates,
        best_result=best_result,
        best_path2=best_path2,
        signature_scores=signature_scores,
    )

    score_table = build_score_table(
        window_end_dates,
        {
            "expath_trailing_score": best_result["trailing_score"],
            "expath_consecutive_distance": best_result["consecutive_score"],
            "path2_trailing_score": best_path2["trailing_score"],
            "signature_trailing_score": signature_scores["signature_trailing_score"].to_numpy(),
        },
    )
    top_dates_table = save_top_dates_table(score_table)

    configuration_table.to_csv(TABLES_DIR / "configuration_search.csv", index=False)
    build_feature_table(
        window_end_dates=window_end_dates,
        raw_features=best_result["raw_features"],
        standardized_features=best_result["standardized_features"],
    ).to_csv(TABLES_DIR / "best_expath_features.csv", index=False)
    comparison_table.to_csv(TABLES_DIR / "expath_vs_path2_vs_signature_scores.csv", index=False)
    correlation_table.to_csv(TABLES_DIR / "comparison_correlations.csv")
    best_result["stress_summary"].to_csv(TABLES_DIR / "best_expath_known_stress_periods.csv", index=False)
    best_result["top_episodes"].to_csv(TABLES_DIR / "best_expath_top_episodes.csv", index=False)
    top_dates_table.to_csv(TABLES_DIR / "top_dates_by_metric.csv", index=False)

    plot_configuration_search(configuration_table)
    plot_best_scores(comparison_table)
    plot_best_expath_scores(comparison_table)

    print("ExPath finance benchmark complete")
    print(f"Cumulative path tensor shape: {cumulative_paths.shape}")
    print(f"Grid searched configs: {len(results)}")
    print()
    print("Best ExpPath configuration:")
    print(configuration_table.head(1).to_string(index=False))
    print()
    print("Best PATH2 configuration:")
    print(path2_table.head(1).to_string(index=False))
    print()
    print("Best ExPath top episodes:")
    print(best_result["top_episodes"].head(TOP_EPISODES_TO_PRINT).to_string(index=False))
    print()
    print("Best ExPath known stress summary:")
    print(best_result["stress_summary"].to_string(index=False))
    print()
    print("Comparison correlations:")
    print(correlation_table.to_string())


if __name__ == "__main__":
    main()
