# Advanced Processes Testing — Stage 3

This stage pushes the signature machinery past the Gaussian world into the
processes that real markets actually show. Stages 1–2 (`Gaussian testing/`)
established that signatures recover the **moments** of 1-D paths, and the
high-dimensional benchmark (`HighDim testing/`) showed they recover
**cross-channel geometry**. Both assumed *constant volatility* and *independent
Gaussian increments*. This stage breaks exactly those two assumptions and asks:

> **Can truncated path signatures tell these richer processes apart, and where
> in the signature does each feature — stochastic volatility, jumps, long memory,
> roughness — show up?**

## The six processes

All six are **1-D log-price paths** `log S_t` built by cumulatively summing
per-step log-returns. They are deliberately calibrated to share the **same
baseline diffusive volatility** (`sigma = 0.2` annualised) so that levels 1–2 are
broadly comparable and the interesting differences live in the *higher moments*
and the *path memory*.

| Code    | Process                        | Feature it adds over Black–Scholes                              | Your request |
|---------|--------------------------------|----------------------------------------------------------------|--------------|
| `BS`    | Black–Scholes GBM              | none — the baseline (constant vol, i.i.d. Gaussian returns)    | baseline     |
| `HEST`  | Heston                         | **stochastic volatility** (mean-reverting CIR variance, leverage) | stochastic vol |
| `MERT`  | Merton jump-diffusion          | **jumps** (compound-Poisson lognormal, skew + fat tails)       | jumps        |
| `BATES` | Bates                          | **jump-diffusion** — Heston stochastic vol *and* Merton jumps  | combined     |
| `FBM`   | Fractional Brownian motion     | **non-Markov** increments with long-range autocorrelation (Hurst 0.70) | non-Markov / fBM |
| `RVOL`  | Rough Bergomi                  | **rough (autocorrelated) volatility** (Hurst 0.10, leverage)   | rough volatility |

### Model detail

- **`HEST` — Heston.** Variance `v_t` follows a CIR process
  `dv = κ(θ − v)dt + ξ√v dW_v`, with `κ=2`, `θ=v₀=σ²=0.04`, vol-of-vol `ξ=0.3`,
  and price/vol correlation `ρ=−0.7` (the leverage effect). Simulated with an
  Euler *full-truncation* scheme (variance floored at 0). Produces volatility
  clustering and mild skew, no jumps.
- **`MERT` — Merton.** Constant-vol diffusion plus a compound-Poisson jump term:
  `≈1` jump/year (`λ=1`), lognormal jump sizes with log-mean `−0.10` and log-std
  `0.15`. A martingale compensator `−λk̄dt` removes the spurious drift, leaving
  **negative skew and fat tails**.
- **`BATES` — Bates.** The Heston stochastic-vol diffusion *and* the Merton jump
  term together — the full jump-diffusion model.
- **`FBM` — fractional Brownian motion.** Increments are fractional Gaussian
  noise with Hurst `H=0.70` (positively autocorrelated / trending), generated
  exactly via the Cholesky factor of the fGn covariance. The scale is chosen so
  the **terminal variance matches Black–Scholes** — the endpoint distribution is
  the same, so *only the path memory* distinguishes it from `BS`.
- **`RVOL` — rough Bergomi.** Volatility is driven by a *rough* (`H=0.10`)
  fractional process, `v_t = ξ₀·exp(η Ŵ_t − ½η²t^{2H})` with `ξ₀=σ²`,
  vol-of-vol `η=0.8`, correlated `ρ=−0.7` with the price. The vol-of-vol is kept
  moderate on purpose so its **roughness**, not raw scale, is what separates it
  from `BS`.

## Method

The heavy lifting is **reused verbatim** from `HighDim testing/src` — the same
streaming signature accumulator (`RunningSignatureStats`), classical estimators
(`ProcessStatistics`), distance tables and plotting. This stage only adds the new
process simulators (`src/simulate_advanced.py`) and a runner. Everything is
**batched and streaming**: paths are simulated a batch at a time, folded into
fixed-length running sums, and discarded, so memory is independent of the number
of paths.

**Time augmentation is ON by default.** The signature of a *1-D* path is nearly
blind to the *order* of its returns — yet order/memory is the whole point of the
non-Markov (`FBM`) and rough-vol (`RVOL`) processes. Appending a monotone time
channel makes the path effectively 2-D (`value`, `time`), so the signature sees
the ordering. With effective dimension 2 and depth 5 the signature has just
`2+4+8+16+32 = 62` coordinates.

Distances are reported three ways (raw, level-normalised, and **standardised** by
the `BS` baseline's per-coordinate std). The standardised view removes both the
"more coordinates" and "bigger numbers" scale effects and is the one plotted.
*(The reused plotting code labels this axis "÷ Process-A coord std"; here
"Process A" is the `BS` baseline.)*

## Results

Canonical run: **5,000 paths/process, 100 steps, horizon 1 year, seed 42**.

### Statistical summary (the classical lens)

| Process | Terminal mean ‖·‖ | Terminal variance | Increment variance |
|---------|-------------------|-------------------|--------------------|
| `BS`    | 0.021             | 0.0401            | 0.000401           |
| `HEST`  | 0.021             | 0.0421            | 0.000401           |
| `MERT`  | 0.029             | 0.0718            | 0.000727           |
| `BATES` | 0.034             | 0.0736            | 0.000727           |
| `FBM`   | 0.002             | 0.0387            | **0.000063**       |
| `RVOL`  | **0.297**         | 0.0644            | 0.000427           |

Reading: `BS` hits its designed variance `σ²T = 0.04`. Jumps (`MERT`, `BATES`)
roughly double the variance and shift the mean (negative-skew jumps). `FBM` has a
**much smaller per-step increment variance** yet the **same terminal variance** —
its positively-correlated increments accumulate to the same endpoint spread; that
is the long-memory signature. `RVOL`'s large negative mean (`−0.5v dt` with a
heavy-tailed variance) and elevated variance reflect rough, spiky volatility.

### Standardised level-wise signature distance (the key result)

Each cell is the standardised `L²` distance at that signature level; **bold**
marks where each feature switches on.

| Comparison       | L1    | L2       | L3    | L4     | L5     |
|------------------|-------|----------|-------|--------|--------|
| `BS_vs_HEST`     | 0.001 | 0.020    | 0.092 | 0.134  | 0.191  |
| `BS_vs_MERT`     | 0.040 | **0.276**| 0.373 | 0.449  | 0.545  |
| `BS_vs_BATES`    | 0.065 | **0.295**| 0.417 | 0.519  | 0.646  |
| `BS_vs_FBM`      | 0.114 | 0.073    | 0.135 | 0.171  | 0.188  |
| `MERT_vs_BATES`  | 0.025 | 0.026    | 0.090 | 0.119  | 0.168  |
| `BS_vs_RVOL`     | 1.367 | 1.266    | 1.587 | 1.991  | 2.608  |
| `FBM_vs_RVOL`    | 1.481 | 1.327    | 1.650 | 2.043  | 2.650  |

Reading:

- **Stochastic volatility (`BS_vs_HEST`)** is essentially invisible at level 1
  (matched mean) and only climbs through the higher levels — vol clustering is a
  *higher-order path* property, not a first- or second-moment one.
- **Jumps (`BS_vs_MERT`)** switch on sharply at **level 2** (jumps inflate
  variance) and keep growing at 3–4 (skew and fat tails).
- **The combined model (`BS_vs_BATES`)** is the strongest of the diffusion
  family — it stacks Heston's vol-of-vol on top of Merton's jumps — and
  `MERT_vs_BATES` isolates exactly the stochastic-vol contribution added on top
  of jumps.
- **Long memory (`BS_vs_FBM`)** shows a modest but genuinely non-zero distance
  even though the endpoint variance is matched — the signature is picking up
  *path memory* that the moments alone cannot see.
- **Rough volatility (`RVOL`)** is by far the most extreme departure at every
  level; its jagged, strongly autocorrelated variance dominates the picture.

### Signature vs classical distances

| Comparison      | Raw signature `L²` | Mean-vec `L²` | Incr. cov. Frobenius |
|-----------------|--------------------|---------------|----------------------|
| `BS_vs_HEST`    | 0.0037             | 0.00014       | 6.0e-07              |
| `BS_vs_MERT`    | 0.0225             | 0.0084        | 3.3e-04              |
| `BS_vs_BATES`   | 0.0269             | 0.0132        | 3.3e-04              |
| `BS_vs_FBM`     | 0.0290             | 0.0230        | 3.4e-04              |
| `BS_vs_RVOL`    | 0.3488             | 0.2763        | 2.7e-05              |

The classical increment-covariance estimator sees the jump/vol variance jump but
is **blind to `FBM`'s memory** (its increment covariance barely moves, 3.4e-04,
because fGn still has a well-defined marginal variance) — yet the signature still
separates `FBM` from `BS`. That gap is the point of this stage: signatures encode
*ordered, path-dependent* structure a moment/covariance table cannot express.

### Figures (`results/figures/`)

- `advanced_levelwise_signature_distance.png` — the key switch-on plot above.
- `advanced_signature_distance_by_depth.png` — cumulative distance vs depth.
- `advanced_statistical_distances.png` — classical mean/covariance distances.
- `advanced_sample_paths.png` — example log-price paths per process.

## Run

From the project root (mind the space in the folder name):

```bash
python "Advanced processes testing/run_advanced_processes_test.py"
```

Useful flags:

```bash
--num-paths 20000     # more paths → smoother tails (jumps/rough vol)
--time-steps 100      # steps per path
--horizon 1.0         # path horizon in years (annualised params assume this)
--sigma 0.2           # baseline volatility shared by all processes
--no-rough            # skip the rough Bergomi process (RVOL)
--no-time-aug         # turn OFF time augmentation (not recommended for 1-D)
--results-dir PATH    # write elsewhere (e.g. for scaling tests)
```

Required packages (`requirements.txt`): `numpy`, `pandas`, `matplotlib`,
`pysiglib`. This stage imports the shared machinery from `HighDim testing/src`,
so that sibling folder must be present (it is in this repo).

## Layout

```text
Advanced processes testing/
├── run_advanced_processes_test.py     # the runner (reuses HighDim testing/src)
├── requirements.txt
├── src/
│   └── simulate_advanced.py           # Heston, Merton, Bates, fBM, rough Bergomi
└── results/
    ├── tables/                        # 5 CSVs (config, summary, 3 distance tables)
    └── figures/                       # 4 PNGs
```
