from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("Application/.matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("Application/.cache").resolve()))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
APPLICATION_DIR = ROOT_DIR / "Application"
RESULTS_DIR = APPLICATION_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

ROLLING_WINDOWS_FILE = DATA_DIR / "rolling_windows_20d_spy_qqq_tlt.npy"
LOG_RETURNS_FILE = DATA_DIR / "log_returns_spy_qqq_tlt.csv"

SIGNATURE_DEPTH = 3
ROLLING_WINDOW_SIZE = 20
ANOMALY_LOOKBACK = 60
TOP_EPISODES_TO_PRINT = 10

KNOWN_STRESS_PERIODS = [
    ("2011 US downgrade", "2011-08-01", "2011-08-31"),
    ("2015 China / growth scare", "2015-08-01", "2015-09-15"),
    ("2018 Q4 selloff", "2018-10-01", "2018-12-31"),
    ("2020 COVID crash", "2020-02-15", "2020-04-15"),
    ("2022 inflation / rates shock", "2022-04-01", "2022-10-31"),
]

sys.path.insert(0, str(ROOT_DIR / "HighDim testing" / "src"))
from signature_highdim import compute_batch_signatures


def ensure_directories() -> None:
    """Create Application output directories and local matplotlib caches."""
    APPLICATION_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)


def load_realdata_inputs() -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Load rolling windows and recover the matching window end dates."""
    windows = np.load(ROLLING_WINDOWS_FILE)
    returns = pd.read_csv(LOG_RETURNS_FILE, parse_dates=["Date"])

    if windows.ndim != 3 or windows.shape[1] != ROLLING_WINDOW_SIZE or windows.shape[2] != 3:
        raise ValueError(
            "Expected rolling windows with shape (num_windows, 20, 3); "
            f"got {windows.shape}."
        )

    expected_window_count = len(returns) - ROLLING_WINDOW_SIZE + 1
    if windows.shape[0] != expected_window_count:
        raise ValueError(
            "Window count does not match the returns table. "
            f"Expected {expected_window_count}, got {windows.shape[0]}."
        )

    window_end_dates = pd.DatetimeIndex(returns["Date"].iloc[ROLLING_WINDOW_SIZE - 1 :])
    return windows.astype(np.float64), window_end_dates


def returns_to_cumulative_paths(windows: np.ndarray) -> np.ndarray:
    """Turn 20 daily return vectors into cumulative paths with a zero starting point."""
    cumulative = np.cumsum(windows, axis=1)
    zero_start = np.zeros((windows.shape[0], 1, windows.shape[2]), dtype=np.float64)
    return np.concatenate([zero_start, cumulative], axis=1)


def standardize_features(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score features columnwise, protecting against zero-variance coordinates."""
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    safe_std = np.where(std > 1e-12, std, 1.0)
    standardized = (features - mean) / safe_std
    return standardized, mean, safe_std


def compute_trailing_distance_scores(features: np.ndarray, lookback: int) -> np.ndarray:
    """Distance from each feature vector to the mean of the trailing lookback window."""
    if len(features) <= lookback:
        raise ValueError(
            f"Need more than {lookback} windows to compute trailing anomaly scores."
        )

    scores = np.full(len(features), np.nan, dtype=np.float64)
    for idx in range(lookback, len(features)):
        reference_mean = features[idx - lookback : idx].mean(axis=0)
        scores[idx] = np.linalg.norm(features[idx] - reference_mean)
    return scores


def compute_consecutive_distances(features: np.ndarray) -> np.ndarray:
    """Distance between adjacent windows in feature space."""
    consecutive = np.full(len(features), np.nan, dtype=np.float64)
    diffs = np.diff(features, axis=0)
    consecutive[1:] = np.linalg.norm(diffs, axis=1)
    return consecutive


def flatten_return_windows(windows: np.ndarray) -> np.ndarray:
    """Flatten each 20x3 window into a single vector baseline."""
    return windows.reshape(windows.shape[0], -1)


def covariance_features(windows: np.ndarray) -> np.ndarray:
    """Use the upper triangle of the within-window covariance matrix as a baseline."""
    feature_rows = []
    upper_indices = np.triu_indices(windows.shape[2])
    for window in windows:
        cov = np.cov(window, rowvar=False)
        feature_rows.append(cov[upper_indices])
    return np.asarray(feature_rows, dtype=np.float64)


def volatility_features(windows: np.ndarray) -> np.ndarray:
    """Use within-window annualized volatility as a simple 3D baseline."""
    return np.std(windows, axis=1, ddof=1) * np.sqrt(252.0)


def build_feature_table(
    window_end_dates: pd.DatetimeIndex,
    feature_name: str,
    raw_features: np.ndarray,
    standardized_features: np.ndarray,
) -> pd.DataFrame:
    """Save feature vectors with readable column names for later inspection."""
    raw_columns = [f"{feature_name}_raw_{i:03d}" for i in range(raw_features.shape[1])]
    z_columns = [f"{feature_name}_z_{i:03d}" for i in range(standardized_features.shape[1])]
    return pd.DataFrame(
        np.column_stack([window_end_dates.astype(str), raw_features, standardized_features]),
        columns=["window_end_date", *raw_columns, *z_columns],
    )


def build_score_table(
    window_end_dates: pd.DatetimeIndex,
    score_map: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Combine per-window scores into one table."""
    score_table = pd.DataFrame({"window_end_date": window_end_dates})
    for score_name, values in score_map.items():
        score_table[score_name] = values
    return score_table


def find_top_episodes(
    dates: pd.DatetimeIndex,
    scores: np.ndarray,
    quantile: float = 0.95,
) -> pd.DataFrame:
    """Group contiguous high-score windows into stress episodes."""
    valid_scores = scores[~np.isnan(scores)]
    threshold = float(np.quantile(valid_scores, quantile))
    high_mask = np.asarray(scores >= threshold, dtype=bool)

    episodes: list[dict[str, object]] = []
    start_idx: int | None = None

    for idx, is_high in enumerate(high_mask):
        if is_high and start_idx is None:
            start_idx = idx
        if start_idx is not None and (not is_high or idx == len(high_mask) - 1):
            end_idx = idx if is_high and idx == len(high_mask) - 1 else idx - 1
            episode_scores = scores[start_idx : end_idx + 1]
            peak_relative = int(np.nanargmax(episode_scores))
            peak_idx = start_idx + peak_relative
            episodes.append(
                {
                    "episode_start": dates[start_idx],
                    "episode_end": dates[end_idx],
                    "peak_date": dates[peak_idx],
                    "peak_score": float(scores[peak_idx]),
                    "duration_windows": end_idx - start_idx + 1,
                    "threshold": threshold,
                }
            )
            start_idx = None

    episode_table = pd.DataFrame(episodes)
    if episode_table.empty:
        return episode_table

    return episode_table.sort_values("peak_score", ascending=False).reset_index(drop=True)


def summarize_known_periods(
    dates: pd.DatetimeIndex,
    scores: np.ndarray,
) -> pd.DataFrame:
    """Summarize average and peak scores during known market stress windows."""
    rows = []
    global_median = float(np.nanmedian(scores))

    for label, start, end in KNOWN_STRESS_PERIODS:
        mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
        period_scores = scores[mask]
        if len(period_scores) == 0:
            continue

        rows.append(
            {
                "period": label,
                "start_date": pd.Timestamp(start),
                "end_date": pd.Timestamp(end),
                "window_count": int(mask.sum()),
                "mean_score": float(np.nanmean(period_scores)),
                "max_score": float(np.nanmax(period_scores)),
                "median_score": float(np.nanmedian(period_scores)),
                "global_median_score": global_median,
                "mean_vs_global_median_ratio": float(np.nanmean(period_scores) / global_median),
            }
        )

    return pd.DataFrame(rows).sort_values("mean_score", ascending=False)


def save_top_dates_table(score_table: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Save the top anomaly dates for each score column."""
    rows = []
    for column in score_table.columns:
        if column == "window_end_date":
            continue
        top_rows = (
            score_table[["window_end_date", column]]
            .dropna()
            .sort_values(column, ascending=False)
            .head(top_n)
        )
        for rank, (_, row) in enumerate(top_rows.iterrows(), start=1):
            rows.append(
                {
                    "metric": column,
                    "rank": rank,
                    "window_end_date": row["window_end_date"],
                    "score": float(row[column]),
                }
            )
    return pd.DataFrame(rows)


def plot_anomaly_scores(score_table: pd.DataFrame) -> None:
    """Plot trailing anomaly scores for signature and baseline feature sets."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    plot_columns = [
        ("signature_trailing_score", "Signature trailing distance"),
        ("returns_trailing_score", "Flattened return trailing distance"),
        ("covariance_trailing_score", "Covariance trailing distance"),
        ("volatility_trailing_score", "Volatility trailing distance"),
    ]

    for ax, (column, title) in zip(axes, plot_columns):
        ax.plot(score_table["window_end_date"], score_table[column], linewidth=1.0)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        threshold = np.nanquantile(score_table[column], 0.95)
        ax.axhline(threshold, color="tab:red", linestyle="--", linewidth=0.9)
        for label, start, end in KNOWN_STRESS_PERIODS:
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="grey", alpha=0.08)
        ax.set_ylabel("Distance")

    axes[-1].set_xlabel("Window end date")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "realdata_anomaly_scores.png", dpi=150)
    plt.close(fig)


def plot_signature_vs_baselines(score_table: pd.DataFrame) -> None:
    """Overlay z-scored anomaly scores to compare the time-series shape across metrics."""
    fig, ax = plt.subplots(figsize=(14, 6))
    for column, label in [
        ("signature_trailing_score", "Signature"),
        ("returns_trailing_score", "Returns"),
        ("covariance_trailing_score", "Covariance"),
        ("volatility_trailing_score", "Volatility"),
    ]:
        values = score_table[column].to_numpy(dtype=np.float64)
        valid = ~np.isnan(values)
        normalized = np.full_like(values, np.nan)
        normalized[valid] = (values[valid] - values[valid].mean()) / values[valid].std()
        ax.plot(score_table["window_end_date"], normalized, linewidth=1.0, label=label)

    for label, start, end in KNOWN_STRESS_PERIODS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="grey", alpha=0.08)

    ax.set_title("Normalized anomaly scores: signatures vs baselines")
    ax.set_xlabel("Window end date")
    ax.set_ylabel("Z-scored anomaly score")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "signature_vs_baselines.png", dpi=150)
    plt.close(fig)


def plot_consecutive_signature_distance(
    score_table: pd.DataFrame,
    episode_table: pd.DataFrame,
) -> None:
    """Plot how sharply adjacent signature windows move through time."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(
        score_table["window_end_date"],
        score_table["signature_consecutive_distance"],
        linewidth=1.0,
        color="tab:blue",
    )

    for _, row in episode_table.head(5).iterrows():
        ax.axvspan(row["episode_start"], row["episode_end"], color="tab:red", alpha=0.12)

    ax.set_title("Consecutive signature distance between adjacent 20-day windows")
    ax.set_xlabel("Window end date")
    ax.set_ylabel("Distance")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "signature_consecutive_distance.png", dpi=150)
    plt.close(fig)


def compute_metric_correlations(score_table: pd.DataFrame) -> pd.DataFrame:
    """Correlate the main anomaly scores after dropping NaNs."""
    columns = [
        "signature_trailing_score",
        "returns_trailing_score",
        "covariance_trailing_score",
        "volatility_trailing_score",
    ]
    correlation = score_table[columns].dropna().corr()
    correlation.index.name = "metric"
    return correlation


def main() -> None:
    ensure_directories()

    windows, window_end_dates = load_realdata_inputs()
    cumulative_paths = returns_to_cumulative_paths(windows)

    signature_features = compute_batch_signatures(cumulative_paths, depth=SIGNATURE_DEPTH)
    signature_z, _, _ = standardize_features(signature_features)

    return_features = flatten_return_windows(windows)
    return_z, _, _ = standardize_features(return_features)

    cov_features = covariance_features(windows)
    cov_z, _, _ = standardize_features(cov_features)

    vol_features = volatility_features(windows)
    vol_z, _, _ = standardize_features(vol_features)

    score_table = build_score_table(
        window_end_dates=window_end_dates,
        score_map={
            "signature_trailing_score": compute_trailing_distance_scores(
                signature_z, ANOMALY_LOOKBACK
            ),
            "signature_consecutive_distance": compute_consecutive_distances(signature_z),
            "returns_trailing_score": compute_trailing_distance_scores(
                return_z, ANOMALY_LOOKBACK
            ),
            "covariance_trailing_score": compute_trailing_distance_scores(
                cov_z, ANOMALY_LOOKBACK
            ),
            "volatility_trailing_score": compute_trailing_distance_scores(
                vol_z, ANOMALY_LOOKBACK
            ),
        },
    )

    signature_episode_table = find_top_episodes(
        dates=window_end_dates,
        scores=score_table["signature_trailing_score"].to_numpy(dtype=np.float64),
    )
    known_period_table = summarize_known_periods(
        dates=window_end_dates,
        scores=score_table["signature_trailing_score"].to_numpy(dtype=np.float64),
    )
    correlation_table = compute_metric_correlations(score_table)
    top_dates_table = save_top_dates_table(score_table)

    build_feature_table(
        window_end_dates, "signature", signature_features, signature_z
    ).to_csv(TABLES_DIR / "signature_features_depth3.csv", index=False)
    score_table.to_csv(TABLES_DIR / "realdata_anomaly_scores.csv", index=False)
    signature_episode_table.to_csv(TABLES_DIR / "signature_top_episodes.csv", index=False)
    known_period_table.to_csv(TABLES_DIR / "signature_known_stress_periods.csv", index=False)
    correlation_table.to_csv(TABLES_DIR / "metric_correlations.csv")
    top_dates_table.to_csv(TABLES_DIR / "top_anomaly_dates_by_metric.csv", index=False)

    plot_anomaly_scores(score_table)
    plot_signature_vs_baselines(score_table)
    plot_consecutive_signature_distance(score_table, signature_episode_table)

    print("Real-data signature benchmark complete")
    print(f"Window tensor shape: {windows.shape}")
    print(f"Cumulative path tensor shape: {cumulative_paths.shape}")
    print(f"Signature depth: {SIGNATURE_DEPTH}")
    print(f"Signature feature shape: {signature_features.shape}")
    print(f"Anomaly lookback: {ANOMALY_LOOKBACK}")
    print()
    print("Top signature episodes:")
    print(signature_episode_table.head(TOP_EPISODES_TO_PRINT).to_string(index=False))
    print()
    print("Known stress period summary:")
    print(known_period_table.to_string(index=False))
    print()
    print("Metric correlations:")
    print(correlation_table.to_string())


if __name__ == "__main__":
    main()
