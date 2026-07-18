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
  --config d20_depth3 \
  --num-paths 1000 \
  --batch-size 250 \
  --n-jobs 16 \
  --max-batch-signature-gb 4 \
  --results-dir "HighDim testing/results_smoke"
```

## 6. Recommended large runs on your server

Your server spec:

- 63 CPU
- 230 GB RAM
- 6 GPU, currently unused by this benchmark

Start with one of these CPU runs:

### Option A: current deep config, scaled up

```bash
python "HighDim testing/run_highdim_gaussian_benchmark.py" \
  --config d10_depth5 \
  --num-paths 200000 \
  --batch-size 4000 \
  --n-jobs 48 \
  --max-batch-signature-gb 32 \
  --results-dir "HighDim testing/results_d10_depth5_200k"
```

### Option B: current wide config, scaled up

```bash
python "HighDim testing/run_highdim_gaussian_benchmark.py" \
  --config d20_depth3 \
  --num-paths 500000 \
  --batch-size 20000 \
  --n-jobs 48 \
  --max-batch-signature-gb 32 \
  --results-dir "HighDim testing/results_d20_depth3_500k"
```

### Option C: both default configs together, larger path count

```bash
python "HighDim testing/run_highdim_gaussian_benchmark.py" \
  --num-paths 200000 \
  --n-jobs 48 \
  --max-batch-signature-gb 32 \
  --results-dir "HighDim testing/results_200k_both"
```

## 7. More aggressive runs

If the smoke test and the first large run are stable, increase only one of these
at a time:

- `--num-paths`
- `--batch-size`
- `--max-batch-signature-gb`
- `--n-jobs`

Good next step:

```bash
python "HighDim testing/run_highdim_gaussian_benchmark.py" \
  --config d10_depth5 \
  --num-paths 1000000 \
  --batch-size 7000 \
  --n-jobs 56 \
  --max-batch-signature-gb 64 \
  --results-dir "HighDim testing/results_d10_depth5_1m"
```

## 8. What the new flags do

- `--n-jobs`: passes the thread count into `pysiglib`
- `--max-batch-signature-gb`: increases the allowed memory for one signature batch
- `--batch-size`: requested batch size; the script will automatically reduce it if
  it would exceed the memory budget
- `--results-dir`: keeps large-run outputs separate from the canonical checked-in
  results

## 9. Monitoring

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

## 10. Important caveat

This benchmark is still CPU-oriented. The 6 GPUs do not help yet. To use those,
the signature computation path would need a GPU-capable implementation instead of
the current `pysiglib` CPU call.
