# PathLaw

This project currently contains a clean Yahoo Finance data-preparation pipeline for a beginner quant-finance workflow using `SPY`, `QQQ`, and `TLT`.

The pipeline does the following:

- downloads daily OHLCV data from Yahoo Finance with `yfinance`,
- uses `Adj Close` as the main price series,
- aligns the three tickers by shared trading dates with an inner join,
- removes rows with missing adjusted close data,
- sorts by date and checks for duplicated dates,
- computes daily log returns,
- creates rolling 20-trading-day windows with 3 dimensions per path,
- saves output datasets and exploratory figures.

## Current Project Content

```text
project/
├── data/
│   ├── raw_prices_spy_qqq_tlt.csv
│   ├── log_returns_spy_qqq_tlt.csv
│   └── rolling_windows_20d_spy_qqq_tlt.npy
├── figures/
│   ├── adjusted_close_prices.png
│   ├── log_returns.png
│   └── rolling_volatility_20d.png
├── prepare_yahoo_data.py
└── README.md
```

## Current Results

Data source:
- Yahoo Finance via `yfinance`

Tickers:
- `SPY`
- `QQQ`
- `TLT`

Date coverage after alignment:
- `2010-01-04` to `2026-07-01`

Dataset sizes:
- Aligned price rows: `4148`
- Log return rows: `4147`
- Rolling windows: `4128`

Rolling window configuration:
- Window size: `20` trading days
- Path dimension per step: `3`
- Saved array shape: `(4128, 20, 3)`

## Output Files

### 1. Raw aligned prices

File:
- `data/raw_prices_spy_qqq_tlt.csv`

Shape:
- `(4148, 19)`

Columns:
- `Date`
- `SPY_open`, `SPY_high`, `SPY_low`, `SPY_close`, `SPY_adj_close`, `SPY_volume`
- `QQQ_open`, `QQQ_high`, `QQQ_low`, `QQQ_close`, `QQQ_adj_close`, `QQQ_volume`
- `TLT_open`, `TLT_high`, `TLT_low`, `TLT_close`, `TLT_adj_close`, `TLT_volume`

### 2. Log returns

File:
- `data/log_returns_spy_qqq_tlt.csv`

Shape:
- `(4147, 4)`

Columns:
- `Date`
- `SPY_log_return`
- `QQQ_log_return`
- `TLT_log_return`

Return coverage:
- `2010-01-05` to `2026-07-01`

### 3. Rolling windows

File:
- `data/rolling_windows_20d_spy_qqq_tlt.npy`

Shape:
- `(4128, 20, 3)`

Interpretation:
- axis 0 = rolling window index
- axis 1 = time steps within each 20-day window
- axis 2 = ticker return channels in the order `SPY`, `QQQ`, `TLT`

## Figures

Saved plots:
- `figures/adjusted_close_prices.png`
- `figures/log_returns.png`
- `figures/rolling_volatility_20d.png`

These figures show:
- adjusted close prices over time,
- daily log returns over time,
- rolling 20-day volatility for each ticker.

## Main Script

File:
- `prepare_yahoo_data.py`

Main responsibilities:
- create `data/` and `figures/` automatically,
- download Yahoo Finance data,
- align and clean the price table,
- compute log returns,
- build rolling windows,
- save CSV, NumPy, and plot outputs,
- print a final run summary.

## Run

From the project root:

```bash
python3 prepare_yahoo_data.py
```

Required Python packages:

```bash
pip install yfinance pandas numpy matplotlib
```

## Current Status

Implemented:
- Yahoo Finance data download
- cleaning and date alignment
- adjusted close handling
- log return computation
- rolling-window export
- exploratory plots

Not implemented yet:
- path signatures
- signature kernels
- signature-based anomaly detection
- signature-based regime detection
