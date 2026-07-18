# PathLaw

This project investigates how **path signatures characterise stochastic
processes** by recovering their moments and geometry, working toward applying
signatures to real financial paths. It has two parts:

1. A **Yahoo Finance data-preparation pipeline** (`SPY`, `QQQ`, `TLT`) — the real
   data the signature methods will eventually be applied to.
2. A staged set of **synthetic signature experiments** that validate the
   signature machinery on data whose true moments/geometry are known:
   - **`Gaussian testing/`** — 1D experiments: signatures recover the **mean**
     (level 1), **variance** (level 2), **skewness** (level 3) and **kurtosis**
     (level 4). Validated against `roughpy`.
   - **`HighDim testing/`** — high-dimensional experiments (`d=20`/depth 3 and
     `d=10`/depth 5): signatures recover **multi-channel geometry** — cross-asset
     **covariance and co-movement** (level 2+) — and are compared head-to-head
     with classical statistical moment estimators. See that folder's README for
     the full write-up and results.
   - **`Advanced processes testing/`** — advanced financial processes that break
     the Gaussian/Markov assumptions: **stochastic volatility (Heston)**, **jumps
     (Merton)**, **jump-diffusion (Bates)**, **non-Markov fractional Brownian
     motion**, and **rough (autocorrelated) volatility (rough Bergomi)**.
     Signatures tell them apart and localise *where* each feature — vol
     clustering, jumps, long memory, roughness — shows up by level. See that
     folder's README for the full write-up and results.

## Data pipeline

This part is a clean Yahoo Finance data-preparation pipeline for a beginner
quant-finance workflow using `SPY`, `QQQ`, and `TLT`.

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
├── data/                               # real SPY/QQQ/TLT data (pipeline output)
│   ├── raw_prices_spy_qqq_tlt.csv
│   ├── log_returns_spy_qqq_tlt.csv
│   └── rolling_windows_20d_spy_qqq_tlt.npy
├── figures/
│   ├── adjusted_close_prices.png
│   ├── log_returns.png
│   └── rolling_volatility_20d.png
├── prepare_yahoo_data.py               # the data pipeline
├── Gaussian testing/                   # STAGE 1: 1D signature experiments
│   └── README.md                       #   mean/variance/skew/kurtosis by level
├── HighDim testing/                    # STAGE 2: high-dim signature experiments
│   └── README.md                       #   multi-channel geometry vs stats estimators
├── Advanced processes testing/         # STAGE 3: advanced (non-Gaussian) processes
│   └── README.md                       #   Heston, Merton, Bates, fBM, rough vol
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

## Signature experiments

The synthetic signature work lives in its own folders, each with a detailed
README. Run them from the project root (mind the spaces in the folder names):

```bash
# Stage 1 — 1D: mean / variance / skewness / kurtosis by signature level
python "Gaussian testing/run_gaussian_signature_test.py"
python "Gaussian testing/run_higher_moment_test.py"

# Stage 2 — high-dimensional geometry vs classical statistical estimators
python "HighDim testing/run_highdim_gaussian_benchmark.py"

# Stage 3 — advanced processes: stochastic vol, jumps, fBM, rough volatility
python "Advanced processes testing/run_advanced_processes_test.py"
```

**Stage 2 headline results** (`d=20`/depth 3 and `d=10`/depth 5, 10,000 paths):

- **Mean shift (A vs B)** → detected by the mean estimator and by signature
  **level 1**.
- **Correlation blocks / common shock (A vs C, A vs D)** → detected by the
  covariance estimator and by signature **level 2+** cross-terms `Sⁱʲ` — the
  multi-channel *geometry* that 1D paths cannot express.
- Raw high-level signature distances are dominated by scale, so **normalised /
  standardised** distances are reported to reveal the true structure.
- Fully **streaming/batched**: verified flat at **≈0.46 GB** peak for a 100,000-
  path run. See `HighDim testing/README.md` for tables, figures and full detail.

## Current Status

Implemented:
- Yahoo Finance data download, cleaning, date alignment, log returns, rolling
  windows, exploratory plots
- **Path signatures** on synthetic data (`pysiglib`, cross-checked with `roughpy`)
- **1D moment recovery** — mean, variance, skewness, kurtosis by signature level
- **High-dimensional geometry recovery** — cross-channel covariance/co-movement
  vs classical statistical estimators, at scale
- **Advanced non-Gaussian processes** — stochastic volatility (Heston), jumps
  (Merton), jump-diffusion (Bates), non-Markov fractional Brownian motion, and
  rough (autocorrelated) volatility (rough Bergomi); signatures separate them and
  localise each feature by level

Not implemented yet:
- signatures applied to the real SPY/QQQ/TLT rolling windows
- signature kernels, signature-based anomaly / regime detection
