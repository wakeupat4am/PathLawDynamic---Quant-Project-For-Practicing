# ExPath

This folder implements a finance-adapted version of the paper
`ExpPath: Semantic Trajectory Modelling in Discourse with Exponentially Weighted Path Representations`
on the project’s real `SPY` / `QQQ` / `TLT` rolling windows.

This is an adaptation, not a literal NLP reproduction. The paper is for
sentence-embedding trajectories; here the same compact path idea is applied to
multi-asset return paths.

## What is implemented

Script:
- [run_expath_finance_benchmark.py](/Users/dienmayhaituyet/Documents/PathLaw/ExPath/run_expath_finance_benchmark.py)

Inputs reused from the project:
- [data/rolling_windows_20d_spy_qqq_tlt.npy](/Users/dienmayhaituyet/Documents/PathLaw/data/rolling_windows_20d_spy_qqq_tlt.npy)
- [data/log_returns_spy_qqq_tlt.csv](/Users/dienmayhaituyet/Documents/PathLaw/data/log_returns_spy_qqq_tlt.csv)
- [Application/results/tables/realdata_anomaly_scores.csv](/Users/dienmayhaituyet/Documents/PathLaw/Application/results/tables/realdata_anomaly_scores.csv)

## Finance adaptation

For each 20-day rolling window:
- convert returns into a cumulative 3D path with a zero start,
- augment the path with a clock channel,
- compute exponentially weighted increments,
- build compact path features:
  - total displacement,
  - antisymmetric signed-area terms,
  - scalar path statistics.

Two clock types are tested:
- `linear`
- `movement`:
  cumulative realized path movement, analogous to the paper’s semantic clock

Decay grid:
- `alpha in {0.00, 0.15, 0.35, 0.70, 1.20}`

Interpretation:
- `alpha = 0` is the paper’s `PATH2` baseline
- `alpha > 0` is `ExpPath`

## Optimization strategy

Because this finance version is unsupervised, there is no train/test classifier
objective like in the paper. Instead, configurations are ranked by how strongly
their anomaly scores separate known stress periods from the global background.

Objective used:
- average ratio of period mean score to global median score across:
  - 2011 US downgrade
  - 2015 China / growth scare
  - 2018 Q4 selloff
  - 2020 COVID crash
  - 2022 inflation / rates shock

## Notes

The paper’s final scalar statistic includes `Δl`, but the extracted PDF text
does not define it explicitly. In this implementation, `Δl` is adapted as the
mean absolute change in step length across consecutive increments. This keeps
the feature stable and close in spirit to a path-roughness / local-change term.
