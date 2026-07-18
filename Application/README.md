# Application

This folder contains the first real-data application of the project’s
signature pipeline to the prepared `SPY` / `QQQ` / `TLT` rolling windows.

## What this experiment does

Script:
- [run_realdata_signature_benchmark.py](/Users/dienmayhaituyet/Documents/PathLaw/Application/run_realdata_signature_benchmark.py)

Inputs:
- [data/rolling_windows_20d_spy_qqq_tlt.npy](/Users/dienmayhaituyet/Documents/PathLaw/data/rolling_windows_20d_spy_qqq_tlt.npy)
- [data/log_returns_spy_qqq_tlt.csv](/Users/dienmayhaituyet/Documents/PathLaw/data/log_returns_spy_qqq_tlt.csv)

Method:
- converts each 20-day return window into a 21-step cumulative return path with a zero start,
- computes truncated signatures at depth `3`,
- standardizes the signature coordinates across windows,
- computes a trailing anomaly score using distance to the mean of the previous `60` windows,
- compares signatures against three baselines:
  - flattened return windows,
  - window covariance features,
  - window volatility features.

## Current Output Files

Figures:
- [results/figures/realdata_anomaly_scores.png](/Users/dienmayhaituyet/Documents/PathLaw/Application/results/figures/realdata_anomaly_scores.png)
- [results/figures/signature_vs_baselines.png](/Users/dienmayhaituyet/Documents/PathLaw/Application/results/figures/signature_vs_baselines.png)
- [results/figures/signature_consecutive_distance.png](/Users/dienmayhaituyet/Documents/PathLaw/Application/results/figures/signature_consecutive_distance.png)

Tables:
- [results/tables/realdata_anomaly_scores.csv](/Users/dienmayhaituyet/Documents/PathLaw/Application/results/tables/realdata_anomaly_scores.csv)
- [results/tables/signature_features_depth3.csv](/Users/dienmayhaituyet/Documents/PathLaw/Application/results/tables/signature_features_depth3.csv)
- [results/tables/signature_top_episodes.csv](/Users/dienmayhaituyet/Documents/PathLaw/Application/results/tables/signature_top_episodes.csv)
- [results/tables/signature_known_stress_periods.csv](/Users/dienmayhaituyet/Documents/PathLaw/Application/results/tables/signature_known_stress_periods.csv)
- [results/tables/metric_correlations.csv](/Users/dienmayhaituyet/Documents/PathLaw/Application/results/tables/metric_correlations.csv)
- [results/tables/top_anomaly_dates_by_metric.csv](/Users/dienmayhaituyet/Documents/PathLaw/Application/results/tables/top_anomaly_dates_by_metric.csv)

## Current Results

Configuration:
- rolling-window tensor shape: `(4128, 20, 3)`
- cumulative signature-path tensor shape: `(4128, 21, 3)`
- signature depth: `3`
- signature feature shape: `(4128, 39)`
- anomaly lookback: `60` windows

Headline findings:
- The strongest signature anomaly episode is the `2020-03-05` to `2020-06-30` COVID regime, peaking on `2020-03-18` with score `117.8171`.
- The next strongest cluster is the `2011-08` US downgrade / stress period, with a peak on `2011-08-22` at `38.5618`.
- The `2022` inflation / rates shock also stands out repeatedly, though less explosively than `2020`.
- Signature scores are related to simpler baselines, but not identical:
  - signature vs returns correlation: `0.7362`
  - signature vs covariance correlation: `0.7902`
  - signature vs volatility correlation: `0.7367`

Known stress period summary from [signature_known_stress_periods.csv](/Users/dienmayhaituyet/Documents/PathLaw/Application/results/tables/signature_known_stress_periods.csv):

- `2020 COVID crash`: mean score `36.0587`, max `117.8171`, about `18.67x` the global median score.
- `2011 US downgrade`: mean score `13.2774`, max `38.5618`, about `6.87x` the global median score.
- `2022 inflation / rates shock`: mean score `6.4244`, max `15.9021`, about `3.33x` the global median score.
- `2015 China / growth scare`: mean score `4.5103`, max `8.9328`, about `2.34x` the global median score.
- `2018 Q4 selloff`: mean score `2.9490`, max `10.6082`, about `1.53x` the global median score.

Top signature episodes from [signature_top_episodes.csv](/Users/dienmayhaituyet/Documents/PathLaw/Application/results/tables/signature_top_episodes.csv):

- `2020-03-05` to `2020-06-30`, peak `2020-03-18`, score `117.8171`
- `2011-08-16` to `2011-08-25`, peak `2011-08-22`, score `38.5618`
- `2022-12-01` to `2022-12-08`, peak `2022-12-08`, score `22.9248`
- `2011-08-04` to `2011-08-10`, peak `2011-08-10`, score `21.3183`
- `2011-09-19` to `2011-09-23`, peak `2011-09-22`, score `17.3220`

## Interpretation

The real-data bridge is now working. The signature-based anomaly score clearly
reacts to major cross-asset stress episodes, especially the 2020 COVID crash,
and it highlights a mix of:
- large shocks in returns,
- shifts in covariance structure,
- changes in multi-asset path geometry across the rolling window.

This is enough to justify the next step:
- compare multiple signature depths,
- test alternative path transforms such as lead-lag or time augmentation,
- move from anomaly scoring into clustering or regime labeling.
