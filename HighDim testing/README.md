# High-Dimensional Gaussian Signature Benchmark

This folder is the **next stage** of the path-signature research project, moving
from the 1-dimensional experiments in `../Gaussian testing/` to **genuinely
high-dimensional paths**. The goal is to show how signatures behave when a path
has real multi-dimensional **geometry** (cross-channel structure), and to compare
signature-based summaries against classical statistical moment estimators.

```bash
# from the project root (mind the space in the folder name):
pip install -r "HighDim testing/requirements.txt"
python "HighDim testing/run_highdim_gaussian_benchmark.py"

# scale up later:
python "HighDim testing/run_highdim_gaussian_benchmark.py" --num-paths 100000
# or run a single configuration:
python "HighDim testing/run_highdim_gaussian_benchmark.py" --config d20_depth3
# or let a larger server use more RAM / CPU:
python "HighDim testing/run_highdim_gaussian_benchmark.py" \
  --config d10_depth5 \
  --num-paths 200000 \
  --batch-size 4000 \
  --max-batch-signature-gb 32 \
  --n-jobs 48
```

---

## 1. Why the previous experiment was only 1D

The `Gaussian testing` stage used paths of **dimension 1** — a single time
series (one random walk). That was the right way to *start*: with one channel we
could cleanly demonstrate that **mean → signature level 1** and **variance →
signature level 2**, and later that **skewness → level 3**, **kurtosis → level 4**.

But a 1D path has a hidden limitation. In one dimension the signature essentially
only "sees" the path's **endpoint**, and the level-`k` term is basically

```
(total displacement)^k / k!
```

So higher signature levels in 1D behave like **powers of a single number**. They
carry almost no *new* geometric information — which is exactly why the 1D README
had to warn that the growth of raw distance at levels 4–5 was "a scale effect,
not new information." **1D paths simply have no geometry to test.**

## 2. Why higher-dimensional paths introduce geometry

Once a path has `d > 1` channels, the signature stops being about a single
endpoint. From **level 2 onward it contains genuine cross-channel terms**:

- `S^{i,j}` (level 2) — the signed area / iterated integral **between channel i
  and channel j**. This is where **covariance and co-movement** live.
- `S^{i,j,k}` (level 3) and higher — **ordered** interactions among three or more
  channels (the order matters: `S^{i,j} ≠ S^{j,i}`).

This is *real geometry*: correlation blocks, market-factor co-movement, and the
ordered relationships between assets are all encoded in level 2+ cross terms.
None of this exists in 1D. Testing it is the whole point of this stage.

## 3. What each path dimension represents

Each path is multi-dimensional:

```
X_t = ( r_t^1 , r_t^2 , ... , r_t^d )
```

where `r_t^i` is the simulated **return/increment of synthetic asset i** at time
`t`. So **the path dimension `d` is the number of synthetic asset / factor return
streams evolving together**:

- `d = 20` → 20 synthetic asset return streams evolving together.
- `d = 10` → 10 synthetic asset return streams evolving together.

A path is built as a **multivariate Gaussian random walk**: draw a Gaussian
increment vector each step and take the cumulative sum.

## 4. Path dimension vs signature level (do not confuse these)

These are two completely different axes:

| Concept | Meaning | In this experiment |
|---|---|---|
| **Path dimension `d`** | number of channels / assets / factors | 20 or 10 |
| **Signature depth / level** | order of the iterated-integral features | up to 3 or 5 |

- **Path dimension** decides how many coordinates the path has *at each time step*.
- **Signature level** decides how *deep* (how high-order) the iterated integrals go.

They interact through the feature count: at depth `N` in dimension `d`, level `k`
contributes exactly **`d^k`** coordinates.

## 5. Why `d=20 / depth=3` and `d=10 / depth=5`

Two complementary configurations, because feature count explodes as `d^k`:

- **Config 1 — `d=20`, depth `3`:** tests **higher-dimensional geometry and
  cross-channel interactions** (many assets, wide level-2 block of 400 cross
  terms), while keeping the depth shallow enough that the feature vector stays
  moderate.
- **Config 2 — `d=10`, depth `5`:** tests **deeper signature levels** (up to
  5th-order iterated integrals / interactions) while keeping the per-step channel
  count — and therefore the feature dimension — **more manageable**.

## 6. Signature feature dimensions

With no leading constant term (pysiglib default), the truncated signature length
is `d + d² + … + d^depth`:

**Config 1 — d=20, depth=3:**
```
20 + 20² + 20³ = 20 + 400 + 8000 = 8420 coordinates
(8421 if the constant "1" term is included)
```

**Config 2 — d=10, depth=5:**
```
10 + 10² + 10³ + 10⁴ + 10⁵ = 10 + 100 + 1000 + 10000 + 100000 = 111110 coordinates
(111111 with the constant term)
```

So `d=20/depth=3` is well-suited to **geometry testing** (wide but shallow), and
`d=10/depth=5` to **deeper-level interaction testing** (narrow but deep).

## 7. Why batching is necessary

At `d=10 / depth=5` each path's signature has **111,110** float64 numbers ≈ 0.9 MB.
Storing all signatures for **10,000** paths would need ≈ **8.9 GB** — and toward
100,000 paths, ≈ 89 GB. That does not fit in memory.

So we **never store all signatures for all paths at once**. For every process we:

1. simulate a **batch** of paths,
2. compute signatures for the batch,
3. fold the batch into **running sums** (sum, sum-of-squares, count),
4. **discard** the batch.

From the running sums we recover the **expected signature** (sum ÷ count) and the
per-coordinate **standard deviation** — using only three vectors of fixed length,
*independent of `num_paths`*. The same streaming trick is used for the classical
estimators (running `Σx` and `Σxxᵀ`). Memory per batch is estimated and printed
before each run, and the batch size is **automatically reduced** if a single
batch of signatures would exceed a 1 GB budget (it never does at these settings).

> **Scaling confirmed.** A full `--num-paths 100000` run of both configs peaked at
> **≈ 0.46 GB RSS** — flat regardless of path count, exactly as the streaming
> design intends — and took ≈ 2 min for config 1 and ≈ 26 min for config 2. The
> extra paths only *sharpen* the results (e.g. the A_vs_C level-1 noise floor
> drops from 0.013 at 10k paths to 0.004 at 100k, making the level-2 covariance
> signal stand out even more clearly).
>
> The script now also exposes `--max-batch-signature-gb` and `--n-jobs`, so the
> same streaming logic can use much larger RAM and CPU budgets on a remote Linux
> server without changing the core experiment code.

## 8. What Processes A, B, C (and D) represent

All four are multivariate Gaussian random walks with increments
`dX ~ N(μ·dt, Σ·dt)` (`dt = 1` by default, so `μ`, `Σ` are the per-step values).

| Process | Mean structure | Covariance structure | Interpretation |
|---|---|---|---|
| **A** baseline | 0 | identity | `d` independent assets, no drift, unit vol |
| **B** mean-shift | first 25% of channels = **0.05**, rest 0 | identity | a subset of assets has a directional **trend** |
| **C** correlation-block | 0 | identity **+ correlated blocks** | assets have **sector / block co-movement** |
| **D** common-shock *(optional)* | 0 | market factor + idiosyncratic | a **market-wide factor** drives co-movement |

Block structure for **C** (`ρ` = pairwise correlation):
- `d=20`: channels 0–4 have `ρ=0.5`, channels 5–9 have `ρ=0.3`, channels 10–19
  stay independent.
- `d=10`: channels 0–4 have `ρ=0.5`, channels 5–9 have `ρ=0.3`.

Because only **one** property changes per process, each comparison isolates one
effect: **A vs B → mean**, **A vs C → covariance geometry**, **A vs D → common
factor**, **B vs C → mixed**.

## 9. Classical statistical estimators

Computed by streaming over batches, for each process:

1. terminal mean vector `E[X_T]`,
2. terminal covariance matrix `Cov[X_T]`,
3. increment mean vector `E[dX]`,
4. increment covariance matrix `Cov[dX]`,
5. average marginal variance (mean of the covariance diagonal).

And between processes: the **L2 distance** between mean vectors and the
**Frobenius distance** between covariance matrices (both for terminal and
increment moments).

> **Terminal vs increment covariance.** The *terminal* covariance grows with the
> number of time steps (≈ `300 × Σ` here), which inflates its sampling-noise
> floor. The *increment* covariance stays `O(1)` and is therefore the **cleaner**
> covariance detector — see the results below.

## 10. Signature-based summaries

For each configuration and process we compute the **expected truncated
signature**, then between processes we report:

- the **raw L2 distance** between expected signatures,
- **level-wise** distances (using only level-1 coordinates, only level-2, …),

each under three normalisations, because raw high-level coordinates have huge
scale:

1. **Raw** distance — straight L2, no rescaling.
2. **Level-normalised** — divide a level's distance by `√(number of coordinates
   in that level)` → a per-coordinate RMS (removes the "more coordinates" effect).
3. **Standardised** *(optional)* — divide each coordinate by **Process A's**
   signature-coordinate std before measuring distance (removes the raw-magnitude
   scale effect). **This is the fairest view and the one plotted.**

## 11. How to interpret the expected results

- **A vs B (mean shift)** should appear strongly in the **mean estimator** and in
  **signature level 1**.
- **A vs C (correlation blocks)** and **A vs D (common shock)** should appear
  strongly in the **covariance estimator** and in **signature level 2+**
  (the cross terms `S^{i,j}`).
- Higher levels **amplify scale**, so **raw and level-normalised distances keep
  growing with level regardless of content** — only the **standardised** distances
  reveal the true "switch-on" structure. This is why normalised distances must
  always be reported alongside the raw ones.

---

## Actual results (`seed = 42`, `num_paths = 10,000`, `time_steps = 300`)

### Classical statistical distances

| Config | Comparison | mean-vec L2 | **incr. cov Frobenius** | terminal cov Frobenius |
|---|---|---|---|---|
| d20_depth3 | A_vs_B (mean) | **33.35** | 0.017 | 82.5 |
| d20_depth3 | A_vs_C (corr) | 1.00 | **2.610** | 791.6 |
| d20_depth3 | A_vs_D (shock)| 1.09 | **1.798** | 532.7 |
| d10_depth5 | A_vs_B (mean) | **21.33** | 0.009 | 42.9 |
| d10_depth5 | A_vs_C (corr) | 1.02 | **2.612** | 773.8 |
| d10_depth5 | A_vs_D (shock)| 0.88 | **0.901** | 280.5 |

**Reading:** the mean-shift (B) lights up the **mean** estimator (33.35 vs a ~1.0
noise floor) but *not* the increment covariance (0.017 ≈ 0). The correlation
block (C) and common shock (D) light up the **covariance** estimator
(increment-cov 2.6 / 1.8 vs 0.017) but *not* the mean. Exactly the intended
separation. (The terminal-cov column has a large noise floor because it scales
with `time_steps`; the increment-cov column is the clean one.)

### Level-wise signature distances (standardised — the switch-on structure)

Standardised distance = per-coordinate distance in units of Process A's own
signature-coordinate variability. **Bigger = more different.**

**Config 1 — d20_depth3:**

| Comparison | Level 1 | Level 2 | Level 3 |
|---|---|---|---|
| A_vs_B (mean) | **0.433** | 0.135 | 0.084 |
| A_vs_C (corr) | 0.013 | **0.094** | 0.014 |
| A_vs_D (shock)| 0.014 | **0.064** | 0.015 |

**Config 2 — d10_depth5:**

| Comparison | Level 1 | Level 2 | Level 3 | Level 4 | Level 5 |
|---|---|---|---|---|---|
| A_vs_B (mean) | **0.388** | 0.109 | 0.098 | 0.033 | 0.028 |
| A_vs_C (corr) | 0.018 | **0.183** | 0.016 | 0.074 | 0.016 |
| A_vs_D (shock)| 0.016 | **0.068** | 0.015 | 0.028 | 0.016 |

**Reading:**

- **A_vs_B peaks at level 1** (0.433 / 0.388) and decays — the **mean shift is a
  first-order (level-1) property**, recovered immediately, matching the mean
  estimator.
- **A_vs_C and A_vs_D are near-zero at level 1 and jump at level 2** (C: 0.094 /
  0.183; D: 0.064 / 0.068) — the **covariance geometry lives in the level-2
  cross terms `S^{i,j}`**, exactly where the covariance estimator also detects it.
  (In `d10_depth5`, A_vs_C also shows a secondary bump at level 4, an even-order
  echo of the same second-moment structure.)

This is the headline of the high-dimensional stage: **path geometry that a 1D
experiment could never show — cross-asset covariance and co-movement — is
recovered by signature level 2+, in agreement with the classical covariance
estimator, while the mean stays a level-1 effect.**

### The scale caveat, made concrete

The **raw** and **level-normalised** distances (in the CSVs) do *not* show this
structure — they grow monotonically with level for *every* comparison, because
level-`k` coordinates are both more numerous (`d^k`) and numerically larger. For
example, raw A_vs_C level-3 distance in `d20_depth3` is ≈ 2748, dwarfing the
level-2 value of ≈ 396 — purely a magnitude artifact. Only after standardising by
Process A's coordinate std does the genuine "level-2 covariance" signal emerge.
**Always report the normalised/standardised distances, not just the raw ones.**

---

## Outputs

**Tables** (`results/tables/`):

| File | Contents |
|---|---|
| `highdim_process_config.csv` | per-process config: `d`, depth, steps, paths, batch size, mean & covariance structure |
| `highdim_statistical_summary.csv` | terminal-mean norm, avg terminal & increment variance, notes |
| `highdim_statistical_distances.csv` | mean L2 + covariance Frobenius distances (terminal & increment) |
| `highdim_signature_distances.csv` | whole-signature raw & level-normalised distances |
| `highdim_signature_levelwise_distances.csv` | per-level raw / normalised / standardised distances + interpretation |

**Figures** (`results/figures/`):

| File | Shows |
|---|---|
| `highdim_signature_distance_by_depth.png` | raw whole-signature distance vs depth (levels 1..k combined), per config |
| `highdim_levelwise_signature_distance.png` | **the key figure** — standardised distance per level, revealing the switch-on structure |
| `highdim_statistical_distances.png` | grouped bars: terminal mean vs covariance distances |
| `sample_paths_first_5_dimensions.png` | example paths, channels 0–4, per process |

---

## Configuration & scaling flags

```
--num-paths N      paths per process (default 10000; scale toward 100000)
--time-steps T     steps per path (default 300)
--dt DT            time increment (default 1.0; keeps mu/Sigma as per-step values)
--batch-size B     override the per-config default (250 for d20/depth3, 100 for d10/depth5)
--seed S           RNG seed (default 42, fully reproducible)
--time-aug         append a monotone time channel -> effective dimension becomes d+1
                   (default OFF, so d stays exactly 20 or 10; the run clearly
                    reports the changed effective signature dimension when ON)
--no-common-shock  skip optional Process D
--config NAME      run only d20_depth3 or d10_depth5
```

**Time augmentation** is **off by default** so the path dimension stays exactly
20 or 10. When turned on, the effective signature dimension becomes `d+1` and the
runner prints an explicit note to that effect.

---

## File guide

```
HighDim testing/
├── run_highdim_gaussian_benchmark.py   <- orchestrates both configs, streaming
├── requirements.txt
├── src/
│   ├── simulate_highdim_gaussian.py    <- multivariate processes A/B/C/D, batched
│   ├── signature_highdim.py            <- pysiglib wrapper + running signature stats
│   ├── statistical_estimators.py       <- streaming mean/covariance estimators
│   ├── distance_metrics.py             <- L2 / Frobenius + the table builders
│   ├── plotting.py                     <- the four figures
│   └── utils.py                        <- signature geometry, memory estimation, logging
└── results/
    ├── tables/                         <- CSV outputs
    └── figures/                        <- PNG outputs
```

## What this stage does *not* do (yet)

No Heston, no Merton jumps, no fractional Brownian motion, no rough volatility.
This stage is deliberately limited to the **high-dimensional Gaussian** benchmark
and the **statistical-vs-signature** comparison. Those richer processes are the
natural next steps once this geometry baseline is understood.
