# Ubuntu Server Guideline

This file is the shortest path to running the large high-dimensional benchmark on
an Ubuntu server.

## Assumptions

- Ubuntu 22.04 or 24.04
- Git installed
- Python 3.11+ available as `python3`
- You already have access to the repository over GitHub
- This benchmark currently uses CPU threads through `pysiglib`; the GPUs are not
  used unless the code is rewritten

## 1. Clone the repo

```bash
git clone https://github.com/wakeupat4am/PathLawDynamic---Quant-Project-For-Practicing.git
cd PathLawDynamic---Quant-Project-For-Practicing
```

## 2. Install system packages

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip build-essential git
```

## 3. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
```

## 4. Install Python dependencies

```bash
pip install -r "HighDim testing/requirements.txt"
```

If `pysiglib` fails to build or install, stop there and fix that first. The
benchmark depends on it.

## 5. Quick smoke test

Run a small check before launching a long job:

```bash
python "HighDim testing/run_highdim_gaussian_benchmark.py" \
  --config d20_depth5 \
  --num-paths 1000 \
  --batch-size 8 \
  --n-jobs 16 \
  --max-batch-signature-gb 4 \
  --results-dir "HighDim testing/results_smoke_d20_depth5"
```

The smoke-test outputs will be saved under:

- `HighDim testing/results_smoke_d20_depth5/tables/`
- `HighDim testing/results_smoke_d20_depth5/figures/`

## 6. Required server run order

Your server spec:

- 63 CPU
- 230 GB RAM
- 6 GPU, currently unused by this benchmark

Run the large jobs in this order:

### Run 1: `d=20`, `depth=5`

```bash
python "HighDim testing/run_highdim_gaussian_benchmark.py" \
  --config d20_depth5 \
  --num-paths 200000 \
  --batch-size 256 \
  --n-jobs 48 \
  --max-batch-signature-gb 32 \
  --results-dir "HighDim testing/results_d20_depth5_200k"
```

Results will be saved in:

- `HighDim testing/results_d20_depth5_200k/tables/`
- `HighDim testing/results_d20_depth5_200k/figures/`

### Run 2: `d=10`, `depth=6`

```bash
python "HighDim testing/run_highdim_gaussian_benchmark.py" \
  --config d10_depth6 \
  --num-paths 200000 \
  --batch-size 2000 \
  --n-jobs 48 \
  --max-batch-signature-gb 32 \
  --results-dir "HighDim testing/results_d10_depth6_200k"
```

Results will be saved in:

- `HighDim testing/results_d10_depth6_200k/tables/`
- `HighDim testing/results_d10_depth6_200k/figures/`

### Run 3: `d=30`, `depth=5`

Only do this if the first two runs are stable.

```bash
python "HighDim testing/run_highdim_gaussian_benchmark.py" \
  --config d30_depth5 \
  --num-paths 100000 \
  --batch-size 128 \
  --n-jobs 48 \
  --max-batch-signature-gb 48 \
  --results-dir "HighDim testing/results_d30_depth5_100k"
```

Results will be saved in:

- `HighDim testing/results_d30_depth5_100k/tables/`
- `HighDim testing/results_d30_depth5_100k/figures/`

## 7. How to run correctly

Always run from the project root:

```bash
cd PathLawDynamic---Quant-Project-For-Practicing
source .venv/bin/activate
```

Then launch exactly one server configuration at a time with:

- `--config` set to one of `d20_depth5`, `d10_depth6`, `d30_depth5`
- `--results-dir` set to a fresh output folder for that run
- `--batch-size`, `--n-jobs`, and `--max-batch-signature-gb` set explicitly

Do not omit `--config` on the server. If you omit it, the script will run every
preset in `CONFIGS`, including the large ones.

Each run writes:

- CSV tables to `<results-dir>/tables/`
- PNG figures to `<results-dir>/figures/`

The main output files are:

- `<results-dir>/tables/highdim_process_config.csv`
- `<results-dir>/tables/highdim_statistical_summary.csv`
- `<results-dir>/tables/highdim_statistical_distances.csv`
- `<results-dir>/tables/highdim_signature_distances.csv`
- `<results-dir>/tables/highdim_signature_levelwise_distances.csv`
- `<results-dir>/figures/highdim_signature_distance_by_depth.png`
- `<results-dir>/figures/highdim_levelwise_signature_distance.png`
- `<results-dir>/figures/highdim_statistical_distances.png`
- `<results-dir>/figures/sample_paths_first_5_dimensions.png`

## 8. More aggressive runs

If the smoke test and the first large run are stable, increase only one of these
at a time:

- `--num-paths`
- `--batch-size`
- `--max-batch-signature-gb`
- `--n-jobs`

Good next step:

```bash
python "HighDim testing/run_highdim_gaussian_benchmark.py" \
  --config d10_depth6 \
  --num-paths 1000000 \
  --batch-size 7000 \
  --n-jobs 56 \
  --max-batch-signature-gb 64 \
  --results-dir "HighDim testing/results_d10_depth6_1m"
```

## 9. What the new flags do

- `--n-jobs`: passes the thread count into `pysiglib`
- `--max-batch-signature-gb`: increases the allowed memory for one signature batch
- `--batch-size`: requested batch size; the script will automatically reduce it if
  it would exceed the memory budget
- `--results-dir`: keeps large-run outputs separate from the canonical checked-in
  results

## 10. Monitoring

Useful commands while the benchmark is running:

```bash
htop
free -h
nproc
df -h
```

If the machine starts swapping or the process becomes unstable, lower:

- `--batch-size`
- `--max-batch-signature-gb`
- `--n-jobs`

## 11. Important caveat

This benchmark is still CPU-oriented. The 6 GPUs do not help yet. To use those,
the signature computation path would need a GPU-capable implementation instead of
the current `pysiglib` CPU call.
