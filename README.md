# CALIPER

**Calibrated measurement instruments for language-model internals, adapted from systems
neuroscience.**

Interpretability reads the internals of language models without an agreed measurement
standard. Methods are proposed, adopted, and used in deployment audits before anyone
establishes what they measure, how reliably, or how often they report structure that is not
there. Systems neuroscience answered the same problem with a discipline of measurement —
response characterisation, explicit null models, and calibration against known ground truth.
This repository carries that discipline across.

Specification: [`docs/specification.md`](docs/specification.md) · Plan of record:
[`docs/plan.md`](docs/plan.md)

---

## The core idea

An MLP neuron's pre-activation is exactly `w · s`, where `s` is its own layer's
post-LayerNorm residual stream and `w` is that neuron's input weight column. With respect to
*that* stimulus space its dependence is exactly one-dimensional and the true direction is
known — verified here to `3.3e-06`, correlation `1.00000000`.

That is **free, exact ground truth inside a real network**, which makes it the strictest
available test of a subspace estimator. Only once an instrument passes there does the
stimulus move to earlier layers, where real computation intervenes and the answer is no
longer known.

---

## Status — Phase 0 (calibration)

| Experiment | Establishes | Result |
|---|---|---|
| **E0.1** Analytic control | Estimator recovers a known direction | **Cause found, fix built** — gate re-run pending |
| **E0.2** Random-direction null | What "found something" means | **PASSED** — detection threshold R² = 0.044 |
| **E0.3a** Recovery ceiling | Whether failures are information limits | **DONE** — 0.976–0.998 at every sparsity; they are not |
| **E0.3b** Operating point | Cheapest configuration that still recovers | **DONE** — 8k tokens × 1600 steps × 2 restarts |
| **E0.5** Classical baselines | Whether a simpler classical method suffices | **PASSED** — it does not |
| **E0.3** Required-N table | Sample complexity, indexed by events | **DONE** — K=1 needs ~200 events; **K≥2 not reached at any N** (joint estimator degenerates to K=1) |

**Headline: the calibration phase did its job.** The first instrument we scored was silently
wrong on ~20% of its inputs. Nothing downstream would have revealed it.

---

## What Phase 0 found

### 1. Rank-1 subspace search fails on real units, silently

On ~20% of GPT-2 layer-6 neurons the estimator converges to a **confidently wrong**
direction. Restart agreement is 0.85–0.99 and gives no warning at all.

It is not a local minimum, not the loss, and not a property of the neurons. E0.1g settled it:

```
failing neurons:  R² at the TRUE direction    = 1.000
                  R² the optimiser reached    = 0.809
                  |w inside the k=2 subspace| = 0.997
```

A perfect solution exists, and the two-dimensional fit **already contains it**. The answer is
found and then lost in the k=1 parameterisation.

**Fix — `fit_cascade`:** fit k+1 dimensions where the landscape is benign, then search inside
that subspace for the best k-dimensional one. A k-dim subspace of a (k+1)-dim space is fixed
by the single direction it discards, so for k=1 the search is one angle. Result: n2977
0.452 → 0.999, n230 0.866 → 0.999.

### 2. Held-out R² is not a reliable proxy for direction recovery

Neither method dominates, so the estimator fits both and selects by held-out R². Measured
over 8 neurons against the oracle choice:

```
above 0.95:  direct 5/8   cascade 6/8   selected 7/8   oracle 8/8
regret (oracle − selected): median 0.0000, max 0.0427
```

Selection beats either method alone and is usually exact. But it has a clean counterexample:

```
n1989   direct  0.964   R² 0.973531
        cascade 0.922   R² 0.988305   ← selected
```

Not a degenerate tie — a decisive 0.015 margin **in favour of the less accurate direction**.
The fitted nonlinearity absorbs a slightly wrong subspace, so the objective measures fit
quality, not direction accuracy, and the two come apart.

This matters beyond this estimator: **in Phase A there is no ground truth, so held-out R² is
the criterion anyone would reach for, and it is not sufficient.** The protocol is therefore to
report method disagreement as a per-unit reliability flag rather than a false point estimate.

### 3. Classical closed-form estimators do not suffice

Bussgang's theorem says that for `a = f(w·s)` with *Gaussian* `s`, ridge regression recovers
`w` up to scale for any `f`, with no optimisation. If that held, the fitted estimator would be
ornamental. Measured on 30 neurons:

| method | median | fraction > 0.95 |
|---|---|---|
| spike-triggered average | 0.287 | 0.00 |
| decorrelated STA (Bussgang) | 0.528 | 0.03 |
| STC top eigenvector | 0.322 | 0.00 |
| **fitted bottleneck** | **0.995** | **0.87** |

One neuron in thirty by any closed form, against twenty-six by the fitted estimator. Two
assumptions fail together: the residual stream is strongly non-Gaussian, and **GELU is
non-monotone over the range these units occupy** — 90–99% of positions sit below zero, 55–90%
past GELU's minimum — which drives the Bussgang constant `E[f'(z)]` toward zero and empties
the spike-triggered average of signal.

### 4. The detection threshold, and a warning that does not apply here

The source literature warns that with correlated stimuli at D ~ 10³, even a random projection
explains much of a response. Two mandatory gates in the specification were built around it.
Measured over 2,000 random directions × 12 neurons:

| quantity | value |
|---|---|
| **detection threshold** (max null p99) | **R² = 0.0443** |
| true-direction R² | 0.700 – 0.985 |
| alignment null over 24,000 draws (median / p99 / max) | 0.024 / 0.091 / 0.144 |

**The warning does not apply to this substrate** — random directions explain at most 4.4%
where the true direction explains 70–98%. Every future claim now has a number to clear.

### 5. Failures are not information limits

Synthetic units with known directions, built on the *real* residual stream, swept across
firing rates from 0.5% to 80%:

| N_eff/D | 0.96 | 2.59 | 7.45 | 18.2 | 24.8 |
|---|---|---|---|---|---|
| ceiling | 0.996 | 0.978 | 0.996 | 0.997 | 0.998 |

Flat at 0.98–1.00 even with fewer informative positions than stimulus dimensions. So a neuron
failing at 0.46 has real headroom, not a data limit.

### 6a. Joint estimation at K ≥ 2 degenerates to K = 1

The required-N sweep, run with an additive construction so every planted dimension is visible at
every position:

| K | N = 2k | 4k | 8k |
|---|---|---|---|
| 2 | 0.502 | 0.509 | 0.521 |
| 3 | 0.356 | 0.362 | 0.377 |

Those are ≈ 1/2 and ≈ 1/3 — the exact signature of **recovering one direction perfectly and
missing the rest**, and it does not move with data. The joint estimator at K > 1 behaves as a K = 1
estimator. Any multi-dimensional readout needs the cascade generalised (find one, project out,
find next) or must proceed one direction at a time. The E0.1 gate showed the k=2 fit *contains*
the true direction at 0.997, so the information is there; the joint parameterisation loses it.

### 6. Optimisation, not data, is the binding constraint

The operating-point sweep (synthetic, sparse-firing, d = 768):

| tokens | steps | restarts | median | **min** |
|---|---|---|---|---|
| 8,000 | 1600 | 2 | 0.9995 | **0.9621** |
| 16,000 | 800 | 1 | 0.9994 | 0.0077 |
| 16,000 | 1600 | 2 | 0.9998 | 0.8482 |

**Doubling the data makes worst-case recovery worse** at a fixed optimisation budget. 8k
tokens with more steps and restarts beats 16k with fewer, at half the cost. Operating point:
**8,000 × 1,600 × 2**.

Note the criterion is the **minimum**, not the median. Selecting on median is the error that
started this whole investigation — E0.1's median was 0.99 while a fifth of units were broken.

---

## Protocol corrections forced by measurement

- **PCA truncation is out.** MLP read-directions retain 0.352 of their norm in the top-64 PCA
  subspace against 0.267 for random directions. Retaining 80% needs D ≈ 512, which is no
  reduction. The specification's "reduce to D ≈ 50–100" would have destroyed the signal.
- **Whitening is out.** Round-trips at 1.000 on synthetic data, returns 0.02 on the real
  residual stream — the covariance is near-singular at d = 768 and the transform amplifies
  noise. Raw space is the protocol.
- **Sample complexity is indexed by events, not positions.** These units sit below GELU's zero
  90–99% of the time, so a position count overstates available information by an order of
  magnitude.
- **A response transform is not the answer.** Rank-gaussianising the response looked decisive
  on three probe neurons and, run at scale, *halved* the pass rate (0.75 → 0.50) while
  inflating apparent k=2 gain. Reverted.
- **Batching gives 4.5×, not the 20–40× first estimated.** Measured, plateauing by batch 32.

---

## Layout

```
caliper/activations.py   stimulus/response extraction via forward hooks
caliper/estimator.py     rank-K bottleneck estimator; fit_cascade; nulls
caliper/batched.py       batched fitting over the shared stimulus (4.5x)
caliper/runtime.py       device selection; resumable checkpointing
experiments/             one script per experiment, results as JSON
tests/                   synthetic recovery, invariance, batching, resumability
results/                 experiment outputs, committed
```

## Running

```
pip install torch transformers numpy scipy pytest
set PYTHONPATH=.
python -m pytest tests/ -q
python experiments/e01_analytic_control.py --neurons 20 --tokens 8000
```

Runs on CPU. `pick_device("auto")` takes CUDA when present; runs are resumable, so a killed
session continues rather than restarting.

## References

Sharpee, Rust & Bialek, *Maximally informative dimensions*, Neural Computation 16(2), 2004 ·
Williamson, Sahani & Pillow, *The equivalence of information-theoretic and likelihood-based
methods for neural dimensionality reduction*, PLoS Comput Biol 11(4), 2015 · Rowekamp &
Sharpee, Network 22, 2011 · Paninski, Network 14, 2003 · Semedo et al., *Cortical areas
interact through a communication subspace*, Neuron 102(1), 2019.
