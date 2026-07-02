from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("figures/.matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("figures/.cache").resolve()))

import matplotlib
import numpy as np
import pandas as pd
import yfinance as yf

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TICKERS = ["SPY", "QQQ", "TLT"]
START_DATE = "2010-01-01"
WINDOW_SIZE = 20
DATA_DIR = Path("data")
FIGURES_DIR = Path("figures")


def ensure_directories() -> None:
    """Create output directories if they do not already exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)


def download_ohlcv_data(tickers: list[str], start_date: str) -> pd.DataFrame:
    """Download Yahoo Finance OHLCV data for the requested tickers."""
    data = yf.download(
        tickers=tickers,
        start=start_date,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
    )
    if data.empty:
        raise ValueError("No data was downloaded from Yahoo Finance.")
    return data


def extract_and_align_ticker_frames(raw_data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Keep OHLCV columns for each ticker and inner-join them on shared dates."""
    per_ticker_frames: list[pd.DataFrame] = []

    for ticker in tickers:
        if ticker not in raw_data.columns.get_level_values(0):
            raise KeyError(f"Ticker {ticker} was not present in the downloaded data.")

        ticker_frame = raw_data[ticker].copy()
        ticker_frame.columns = [
            f"{ticker}_{column.lower().replace(' ', '_')}" for column in ticker_frame.columns
        ]
        per_ticker_frames.append(ticker_frame)

    aligned = pd.concat(per_ticker_frames, axis=1, join="inner")
    aligned.index = pd.to_datetime(aligned.index)
    aligned.index.name = "Date"
    return aligned


def clean_prices(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Sort by date, drop rows with missing adjusted close data, and guard against duplicates."""
    adjusted_close_columns = [f"{ticker}_adj_close" for ticker in tickers]

    cleaned = prices.sort_index().copy()
    cleaned = cleaned.dropna(subset=adjusted_close_columns)

    duplicate_count = int(cleaned.index.duplicated().sum())
    if duplicate_count > 0:
        raise ValueError(f"Found {duplicate_count} duplicated dates in the aligned dataset.")

    return cleaned


def print_basic_information(prices: pd.DataFrame) -> None:
    """Print date coverage, shape, and missing values for the cleaned price table."""
    print("Basic information for cleaned aligned prices")
    print(f"Date range: {prices.index.min().date()} to {prices.index.max().date()}")
    print(f"Number of rows: {len(prices)}")
    print("Missing values per column:")
    print(prices.isna().sum().to_string())
    print()


def compute_log_returns(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Compute daily log returns from adjusted close prices."""
    adjusted_close = prices[[f"{ticker}_adj_close" for ticker in tickers]].copy()
    adjusted_close.columns = tickers

    log_returns = np.log(adjusted_close / adjusted_close.shift(1)).dropna()
    log_returns.columns = [f"{ticker}_log_return" for ticker in tickers]
    log_returns.index.name = "Date"
    return log_returns


def create_rolling_windows(log_returns: pd.DataFrame, window_size: int) -> np.ndarray:
    """Create rolling windows with shape (num_windows, window_size, num_tickers)."""
    values = log_returns.to_numpy(dtype=float)
    if len(values) < window_size:
        raise ValueError(
            f"Need at least {window_size} return rows to create rolling windows; got {len(values)}."
        )

    windows = np.array(
        [values[start : start + window_size] for start in range(len(values) - window_size + 1)]
    )
    return windows


def save_outputs(prices: pd.DataFrame, log_returns: pd.DataFrame, windows: np.ndarray) -> None:
    """Persist cleaned prices, returns, and rolling windows."""
    prices.to_csv(DATA_DIR / "raw_prices_spy_qqq_tlt.csv")
    log_returns.to_csv(DATA_DIR / "log_returns_spy_qqq_tlt.csv")
    np.save(DATA_DIR / "rolling_windows_20d_spy_qqq_tlt.npy", windows)


def plot_adjusted_close_prices(prices: pd.DataFrame, tickers: list[str]) -> None:
    """Plot adjusted close prices for all tickers."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for ticker in tickers:
        ax.plot(prices.index, prices[f"{ticker}_adj_close"], label=ticker, linewidth=1.2)

    ax.set_title("Adjusted Close Prices")
    ax.set_xlabel("Date")
    ax.set_ylabel("Adjusted Close")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "adjusted_close_prices.png", dpi=150)
    plt.close(fig)


def plot_log_returns(log_returns: pd.DataFrame, tickers: list[str]) -> None:
    """Plot daily log returns for all tickers."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for ticker in tickers:
        ax.plot(log_returns.index, log_returns[f"{ticker}_log_return"], label=ticker, linewidth=0.8)

    ax.set_title("Daily Log Returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log Return")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "log_returns.png", dpi=150)
    plt.close(fig)


def plot_rolling_volatility(log_returns: pd.DataFrame, tickers: list[str], window_size: int) -> None:
    """Plot rolling annualized volatility using the specified window size."""
    rolling_volatility = log_returns.rolling(window=window_size).std() * np.sqrt(252)

    fig, ax = plt.subplots(figsize=(12, 6))
    for ticker in tickers:
        ax.plot(
            rolling_volatility.index,
            rolling_volatility[f"{ticker}_log_return"],
            label=ticker,
            linewidth=1.0,
        )

    ax.set_title(f"Rolling {window_size}-Day Volatility")
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized Volatility")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "rolling_volatility_20d.png", dpi=150)
    plt.close(fig)


def print_final_summary(
    tickers: list[str], prices: pd.DataFrame, windows: np.ndarray, window_size: int
) -> None:
    """Print the final summary requested for the prepared dataset."""
    print("Final summary")
    print(f"Downloaded tickers: {', '.join(tickers)}")
    print(f"Final date range: {prices.index.min().date()} to {prices.index.max().date()}")
    print(f"Number of aligned trading days: {len(prices)}")
    print(f"Number of rolling windows: {len(windows)}")
    print(f"Shape of the rolling-window array: {windows.shape}")
    print(f"Window size used: {window_size}")


def main() -> None:
    ensure_directories()

    raw_data = download_ohlcv_data(TICKERS, START_DATE)
    aligned_prices = extract_and_align_ticker_frames(raw_data, TICKERS)
    cleaned_prices = clean_prices(aligned_prices, TICKERS)

    print_basic_information(cleaned_prices)

    log_returns = compute_log_returns(cleaned_prices, TICKERS)
    rolling_windows = create_rolling_windows(log_returns, WINDOW_SIZE)

    save_outputs(cleaned_prices, log_returns, rolling_windows)
    plot_adjusted_close_prices(cleaned_prices, TICKERS)
    plot_log_returns(log_returns, TICKERS)
    plot_rolling_volatility(log_returns, TICKERS, WINDOW_SIZE)

    print_final_summary(TICKERS, cleaned_prices, rolling_windows, WINDOW_SIZE)


if __name__ == "__main__":
    main()
