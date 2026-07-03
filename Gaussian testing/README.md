# Gaussian Signature Testing

This folder holds the **synthetic experiments** of the signature research
project. They are small, self-contained, and do **not** use the real
SPY/QQQ/TLT finance data — that comes later. There are two stages:

- **Stage 1 (Gaussian):** can signatures recover the **mean** and **variance**?
  → `run_gaussian_signature_test.py`
- **Stage 2 (non-Gaussian):** can signatures recover **higher moments** —
  **skewness** (3rd) and **kurtosis** (4th)? → `run_higher_moment_test.py`

Signatures are computed with **pysiglib** in both stages (roughpy is used only
as an independent validation cross-check).

---

## Why this folder exists

My professor asked me to investigate how **path signatures characterise
stochastic processes by recovering their moments** (mean, variance, and higher
moments). The recommended way to start is with the simplest possible process —
a **Gaussian process** — and then gradually make the processes more complex
(adding higher moments) while testing different **signature truncation levels**.

This folder implements that very first step: a controlled Gaussian experiment
where we already know the true mean and variance, so we can check whether
truncated signatures recover that information.

---

## Professor's research question

> *"We want to check the way signatures characterise processes by checking how
> it recovers their moments. Note, you have to truncate the level of the
> signatures. Start with a Gaussian process (only 2 moments) and complexify the
> processes (higher moments) and test the truncation levels."*

In plain language:

> **Can truncated path signatures characterise a stochastic process by
> recovering its moment information (mean, variance, …)?**

---

## Why start with Gaussian processes?

- A Gaussian (normal) distribution is the simplest well-understood random model.
- **A Gaussian is *completely* described by only its first two moments:** its
  **mean** and its **variance/covariance**. It has no independent higher-moment
  information (its skewness is 0 and its kurtosis is fixed).
- Therefore, if signatures work as expected, the **mean** and **variance**
  differences between Gaussian processes should show up at **low truncation
  levels**, and higher levels should not reveal any genuinely *new* information.

That makes Gaussian processes the perfect "known answer" to calibrate against
before moving on to harder, non-Gaussian processes.

---

## What are path signatures?

A **path** is just a line that evolves over time (think of a price chart).

The **path signature** is a list of numbers that summarises the *shape* of that
path. You can think of it as a hierarchy of increasingly detailed descriptors:

- **Level 1** captures *first-order movement* — roughly, how far and in which
  direction the path travelled overall (linked to the **mean / drift**).
- **Level 2** captures *second-order information* — areas and squared movement
  (linked to **variance** and, in higher dimensions, to **covariance** between
  channels).
- **Level 3, 4, 5, …** capture increasingly fine, higher-order detail about the
  path's shape (linked to higher moments like **skewness** and **kurtosis**).

A tiny bit of intuition for a **1-dimensional** path (one time series), which is
what we use here: the level-`k` signature term is essentially

```
(total change of the path) ^ k  /  k!
```

So the **average** (expected) level-`k` term is related to the `k`-th moment of
the path's total displacement. That is exactly why signatures and moments are
connected — and why we test whether comparing signatures recovers moment
differences.

---

## What does truncation level mean?

- The **full signature is infinite** — it has terms at every level 1, 2, 3, ….
- A computer cannot store infinitely many numbers, so we **truncate**: we keep
  only levels 1 up to some finite **degree** `N`.
- This experiment tests **degrees 1, 2, 3, 4 and 5**.
- Higher truncation levels give **more information** but also produce a **much
  larger feature vector**, and (as we will see) the higher-level numbers are
  much **larger in magnitude**.

---

## Experiment design

We simulate **three Gaussian processes**, each with **1000 paths** of **20 time
steps**, in **1 dimension**, using a fixed random seed (`42`) so results are
reproducible.

| Process | Mean | Volatility | What it is for |
|---------|------|------------|----------------|
| **A** | 0.00 | 1.0 | baseline |
| **B** | 0.05 | 1.0 | only the **mean** changed |
| **C** | 0.00 | 2.0 | only the **volatility** changed |

Because only **one** property is changed at a time, each comparison isolates one
kind of difference:

- **A vs B** → tests **mean** recovery (first moment).
- **A vs C** → tests **variance / volatility** recovery (second moment).
- **B vs C** → a **combined** mean + variance difference.

### How we compare two processes

1. Compute the truncated signature of **every** path.
2. Average them to get the **expected signature** (the process's "fingerprint").
3. Measure the **L2 (straight-line) distance** between two processes' expected
   signatures. A bigger distance = the signature sees the two processes as more
   different.

---

## How to run

From the **project root** (mind the quotes — the folder name has a space):

```bash
pip install -r "Gaussian testing/requirements.txt"

# Stage 1 - mean & variance (Gaussian, pysiglib):
python "Gaussian testing/run_gaussian_signature_test.py"

# Stage 2 - skewness & kurtosis (non-Gaussian, pysiglib):
python "Gaussian testing/run_higher_moment_test.py"

# Validation cross-check (pysiglib vs roughpy):
python "Gaussian testing/run_roughpy_crosscheck.py"
```

On macOS you may need `python3` instead of `python`.

If `pysiglib` (or `roughpy` for the cross-check) is missing, the code stops with
a clear message telling you exactly what to `pip install`.

---

## Outputs

Everything is written **inside this folder only**:

- `results/tables/signature_distance_summary.csv` — L2 distance between expected
  signatures for every level and every comparison, with a plain-English
  interpretation column.
- `results/tables/empirical_moments_summary.csv` — the measured mean and
  variance of each process, next to their true values (a sanity check).
- `results/figures/signature_distance_vs_level.png` — the main plot: distance
  vs truncation level, one line per comparison.
- `results/figures/sample_paths_A_B_C.png` — a few example paths from each
  process (C should visibly wander further; B should drift upward).
- `results/tables/roughpy_pysiglib_agreement.csv` — produced by the cross-check
  script; shows pysiglib and roughpy agree to numerical precision.

**Stage 2 (higher moments)** adds:

- `results/tables/higher_moment_distance_summary.csv` — L2 distances for
  G_vs_S / G_vs_T / S_vs_T at every level, with interpretations.
- `results/tables/higher_moment_empirical_moments.csv` — mean, variance,
  skewness and excess kurtosis of each process (the matched-moments sanity check).
- `results/figures/higher_moment_distance_vs_level.png` — distance vs level; the
  skew line switches on at level 3, the heavy-tail line at level 4.
- `results/figures/sample_paths_G_S_T.png` — example paths of G, S, T.

---

## Actual results (from `seed = 42`)

**Empirical moments** — the simulator behaves exactly as intended:

| Process | true mean | empirical mean | true variance | empirical variance |
|---------|-----------|----------------|---------------|--------------------|
| A | 0.00 | +0.005 | 1.0 | 1.01 |
| B | 0.05 | +0.045 | 1.0 | 1.01 |
| C | 0.00 | +0.008 | 4.0 | 4.02 |

**Signature distances** (L2 distance between expected signatures):

| Comparison | L1 | L2 | L3 | L4 | L5 |
|------------|------|------|------|-------|-------|
| A_vs_B (mean)     | 0.808 | 0.808 | 7.29 | 9.16 | 34.7 |
| A_vs_C (variance) | 0.054 | 31.25 | 31.36 | 860.3 | 917.8 |
| B_vs_C (combined) | 0.862 | 31.25 | 32.79 | 854.8 | 901.5 |

---

## How to interpret the results

Read the table by looking at **where each difference "switches on"**:

- **A_vs_B (mean difference):** the distance is already **non-zero at level 1**
  (0.808). This confirms that **the mean is a first-order (level-1) property** —
  signatures recover it immediately.

- **A_vs_C (variance difference):** the distance is **almost zero at level 1**
  (0.054, because both processes have the same mean) and then **jumps sharply at
  level 2** (31.25). This is the key result: **variance is a second-order
  (level-2) property**, so it only appears once level 2 is included.

- **B_vs_C:** a mix of both effects, non-zero at level 1 (mean gap) and jumping
  again at level 2 (variance gap).

This is exactly the behaviour the professor's question predicts: **the first two
moments of a Gaussian are recovered by signature levels 1 and 2.**

### An important caveat about the higher levels

You will notice the raw distances **keep growing** at levels 4 and 5 (e.g.
A_vs_C reaches 860 and 917). **This is not new information — it is a scale
effect.** The level-`k` signature term contains the `k`-th power of the path's
displacement, so higher-level coordinates are simply **numerically much larger**.
When we double the volatility (A vs C), those large level-4/5 numbers differ by a
lot in absolute terms, which inflates the raw L2 distance — even though a
Gaussian carries no genuinely independent information beyond its first two
moments.

So the honest reading is:

- The **qualitative switch-on structure** is the real signal:
  **mean → level 1, variance → level 2.**
- The **growing raw magnitude** at higher levels is mostly a **numerical scaling
  artifact**, not evidence that higher moments matter for a Gaussian.
- To measure higher-level *information* fairly, a good next step is to
  **normalise** the signature levels (for example, standardise each coordinate,
  or divide level-`k` terms by a scale factor) so that different levels are put
  on a comparable footing. That refinement is left for a later experiment.

### Reading the plot

In `signature_distance_vs_level.png`:

- The **A_vs_C** line starts near zero and **jumps up at level 2** → level 2 is
  capturing the variance information.
- The **A_vs_B** line starts above zero already at level 1 → level 1 captures the
  mean.

---

## How this answers the professor's question

This experiment checks whether truncated signatures recover moment information by
comparing the **expected signatures** of Gaussian processes with **known** mean
and variance differences.

Because a Gaussian process is fully determined by its first two moments, the
experiment tests whether **level 1 recovers the mean** and **level 2 recovers the
variance** — and it does:

- The **mean** difference (A vs B) is visible from **level 1**.
- The **variance** difference (A vs C) switches on at **level 2**.

This directly demonstrates the mechanism the professor described: signatures
characterise a process through its moments, and **truncating at level 2 is
already enough to capture everything that defines a Gaussian.** Any further
growth at higher levels is numerical scale, not new distributional information —
which is precisely what we would expect for a purely Gaussian process, and sets
up the natural next question: *what happens with non-Gaussian processes that DO
have higher moments?*

---

## Stage 2: higher moments (skewness & kurtosis)

Stage 1 proved signatures recover the first two moments. Stage 2 answers the
follow-up: **can signatures recover the THIRD and FOURTH moments too?** This is
run by `run_higher_moment_test.py` (again using **pysiglib**).

### The idea: match the low moments to isolate the high ones

We build three processes that all share the **same mean (0) and the same
variance (1)**, so **levels 1 and 2 cannot tell them apart**. They differ only
higher up:

| Process | Distribution | Skewness (3rd) | Excess kurtosis (4th) |
|---------|--------------|----------------|------------------------|
| **G** | Gaussian | 0 | 0 |
| **S** | Skew-normal (`alpha=8`) | **large** | small |
| **T** | Student-t (`df=5`) | ~0 | **large** (heavy tails) |

Matching the variance is the whole trick: because the low moments are identical,
any difference the signature detects **must** come from a higher moment.

- **G vs S** isolates **skewness** → expect a jump at **level 3**.
- **G vs T** isolates **kurtosis / tail risk** → expect a jump at **level 4**.
- **S vs T** is a combined higher-moment difference.

### One subtlety: the Central Limit Theorem dilutes higher moments

A path here is a random walk (a running sum of 20 independent increments). In one
dimension, a signature only "sees" the path's **endpoint**, and summing many
increments makes that endpoint look **more Gaussian** (the CLT), which shrinks
the higher-moment signal. That is why stage 2 uses **more paths (4000 vs 1000)** —
the skew/kurtosis signal is weaker and needs more samples to rise above noise.

### Actual results (from `seed = 42`)

**Empirical moments** — the processes are exactly as designed (variance matched,
only skew/kurtosis differ):

| Process | mean | variance | skewness | excess kurtosis |
|---------|------|----------|----------|-----------------|
| G | -0.00 | 1.01 | +0.01 | +0.03 |
| S | -0.00 | 0.99 | **+0.92** | +0.76 |
| T | -0.00 | 1.00 | +0.05 | **+4.44** |

**Signature distances** (L2 between expected signatures):

| Comparison | L1 | L2 | L3 | L4 | L5 |
|------------|------|------|------|------|------|
| G_vs_S (skewness) | 0.011 | 0.119 | **1.781** | 6.089 | 21.83 |
| G_vs_T (kurtosis) | 0.033 | 0.033 | 0.749 | **6.006** | 9.914 |
| S_vs_T (combined) | 0.022 | 0.127 | 1.036 | 1.045 | 13.12 |

### How to read this

- **G_vs_S (skewness):** tiny at levels 1–2 (mean and variance matched), then
  **jumps at level 3** (0.12 → 1.78). **Skewness is a third-moment / level-3
  effect**, and the signature only "sees" it once level 3 is included. ✅
- **G_vs_T (kurtosis):** tiny at levels 1–2 **and still small at level 3** (0.75),
  because the Student-t is **symmetric** and has almost no third moment — then it
  **jumps at level 4** (0.75 → 6.01). **Kurtosis / tail risk is a fourth-moment /
  level-4 effect.** ✅
- The `higher_moment_distance_vs_level.png` figure shows this as two lines that
  stay flat and then "switch on" at their own level (S at 3, T at 4).

This is the headline result of stage 2: **each moment becomes visible at its own
signature level** — mean at 1, variance at 2, skewness at 3, kurtosis at 4. To
"see" the `k`-th moment, you must **truncate the signature at level ≥ k**. That
is a direct, concrete demonstration of the professor's point that you *have to
truncate the level of the signatures*, and that richer processes need higher
truncation levels.

### Why this matters for finance (motivation for later stages)

Skewness and heavy tails (kurtosis) are exactly the "tail-risk" features that
matter for returns: crashes are large negative moves (skew) and fat tails
(kurtosis). Stage 2 shows that a signature truncated at level 4 already carries
that tail-risk information — useful once we move to the real SPY/QQQ/TLT windows.

---

## The two signature libraries: pysiglib vs roughpy

Both **pysiglib** and **roughpy** compute the *same* mathematical object — the
truncated path signature — but they play different roles here:

| | **pysiglib** | **roughpy** |
|---|---|---|
| Role | **Workhorse** — computes all the main results | **Reference** — independent cross-check |
| Style | One batched call: `pysiglib.sig(paths, degree)` | Stream/increment model: build a stream, take its signature |
| Speed | Fast, processes all 1000 paths at once | Slower, one path at a time (fine for validation) |
| Used in | `signature_features.py` (the experiment) | `roughpy_crosscheck.py` (the validation) |

**Why bother with a second library?** Signatures are easy to get subtly wrong
(truncation conventions, the leading constant "1" term, how the path is
parametrised). Computing the same signature two independent ways and checking
they match is the simplest way to *trust* the numbers before scaling up to
harder processes or real financial data.

### The cross-check

`run_roughpy_crosscheck.py` recomputes the A/B/C expected signatures at every
level with **both** libraries and compares them. To make roughpy match pysiglib
we difference each path into increments, build a `LieIncrementStream`, take its
signature, and drop the leading "1" term (see `src/roughpy_crosscheck.py` for the
fully commented details).

**Result: the two libraries agree to numerical precision** — the largest
disagreement across all processes and levels is about `1e-13`, which is just
floating-point rounding. This confirms our pipeline is correct. The evidence is
saved to `results/tables/roughpy_pysiglib_agreement.csv`.

---

## Next research steps

1. ~~**Skew-normal processes**~~ — **DONE (stage 2).** The third moment
   (skewness) switches on at **level 3**. See `run_higher_moment_test.py`.
2. ~~**Student-t processes**~~ — **DONE (stage 2).** Heavy tails / kurtosis
   (fourth moment) switch on at **level 4**. See `run_higher_moment_test.py`.
3. **Mixture-of-Gaussians processes** — multi-modal behaviour that a single
   Gaussian cannot describe.
4. **Jump processes** — sudden discontinuous moves.
5. ~~**Compare `pysiglib` with `roughpy`**~~ — **DONE.** Both libraries agree to
   numerical precision (see `run_roughpy_crosscheck.py` and the "two signature
   libraries" section above). Comparing speed/extra features is still open.
6. **Normalise the signature levels** so higher-level comparisons measure
   information rather than raw magnitude.
7. **Apply the same pipeline to the real SPY/QQQ/TLT rolling windows** already
   prepared by the main project, once the synthetic tests are understood.

---

## File guide (for future me)

```
Gaussian testing/
├── README.md                          <- this file
├── requirements.txt                   <- dependencies
├── run_gaussian_signature_test.py     <- STAGE 1: mean & variance
├── run_higher_moment_test.py          <- STAGE 2: skewness & kurtosis
├── run_roughpy_crosscheck.py          <- validation: pysiglib vs roughpy
├── src/
│   ├── simulate_gaussian.py           <- Gaussian processes A/B/C (stage 1)
│   ├── simulate_nongaussian.py        <- skew-normal & Student-t (stage 2)
│   ├── signature_features.py          <- computes signatures with pysiglib
│   ├── analysis.py                    <- distances + empirical moments
│   ├── plotting.py                    <- the figures
│   └── roughpy_crosscheck.py          <- same signatures via roughpy
└── results/
    ├── tables/                        <- CSV outputs
    └── figures/                       <- PNG outputs
```

Every source file is heavily commented to explain the *reasoning*, not just the
syntax — start with `run_gaussian_signature_test.py` and follow the imports.
