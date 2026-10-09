# F-0 technical note: which checks can see which error

**9 October 2026.** Working note for the flagship (`docs/flagship-plan.md` §5, F-0), not paper
prose. Simulation: `experiments/f0_simulation.py`, results `results/f0_simulation.json`.

## 1. Setup

A unit reads one direction w of a stimulus s with covariance C = Q diag(λ) Q'. An estimator
maps a sample D = {(s_t, y_t)} and a seed ξ to a direction v(D, ξ). Error is measured against w
(|cos| < 0.95 fails, as in Paper 1).

Split the error of a fit into a part that changes when the data are redrawn and a part that
does not:

    v(D, ξ) - w  =  (E_D[v] - w)          bias: the same for every sample
                  + (v - E_D[v])          variance: moves with the sample

## 2. Three kinds of check

1. **Seed checks** (restart agreement, seed replicates) compare v(D, ξ1) with v(D, ξ2) on the
   *same* D. If the estimator's minimiser is unique given D, they agree whatever the error;
   with several basins they measure basin mass. Either way, the sample's contribution to the
   error is constant across the comparison, so it is invisible.
2. **Resampling checks** compare v(D1) with v(D2) on independent samples:

       E|v1 - v2|^2 = 2 tr Σ,        E|v - w|^2 = tr Σ + |bias|^2

   They estimate the variance part of the error, twice over, and are blind to bias. The ratio
   of mean disagreement to mean error is 2 when error is pure variance and 0 when it is pure
   bias.
3. **Geometric checks** compare the answer with the geometry of the data: the share of v in
   the directions carrying the bottom 1% of stimulus variance. For least squares,
   Σ = σ²(X'X)⁻¹ ≈ σ² C⁻¹ / n, so variance error concentrates in the low-λ directions, and an
   answer with weight there is an answer the data do not pin down. A geometric check can also
   see a bias that moves the answer into such directions; it cannot see a bias that lives
   where the data are rich.

**Regularisation moves error between the classes.** A ridge penalty, or early stopping, shrinks
the poorly sampled directions towards zero. Their variance becomes a consistent shortfall, a
bias, so resampling checks lose sight of it while the fit still misses w there.

## 3. Part A: least squares on a linear unit (500 units, d = 128, n = 4,000)

Per-unit noise varies (SNR log-uniform), so units differ in difficulty.

| regime | failures | disagreement / error | AUC seed | AUC data replicate | AUC held-out R² | AUC low-variance share |
|---|---|---|---|---|---|---|
| well (2 decades) | 2 | **2.01** | 0.50 | 1.00 | 0.99 | 0.70 |
| ill (4 decades) | 219 | **1.77** | 0.50 | 1.00 | 1.00 | 0.89 |
| ill + ridge 3e-3 | 240 | **1.04** | 0.50 | 0.93 | 0.93 | 0.76 |
| bias (fixed distortion A = I + 5uv') | 151 | **0.00** | 0.50 | 0.39 | 0.46 | **0.92** |

- Seed checks are exactly uninformative (agreement 1 for every unit, AUC 0.50) in every regime.
- With unregularised least squares, the disagreement-to-error ratio is 2 (well) and 1.8 (ill),
  as the decomposition predicts, and 98% of the error energy sits in the bottom-1% directions in
  the ill-conditioned regime.
- A ridge penalty halves the ratio (1.04): part of the error has become bias.
- A pure bias is invisible to resampling and to held-out fit. Here the geometric check still
  catches it (0.92), because this distortion moves the estimand into directions the observed
  data barely occupy. That is a property of this distortion, not a guarantee.
- Held-out R² is strong in Part A because failures here are driven by noise, which also lowers
  R². Paper 1's hard case (noiseless, fits well, wrong) needs the nonlinear estimator: Part B.

## 4. What this predicts for the real data

- **X-1's failed primary.** On real units, cross-sample agreement beat held-out R² by only 0.065
  on converged-wrong fits, while the low-variance share beat it by 0.218. Paper 1's estimator is
  implicitly regularised (ridge warm start, early stopping on held-out R²). The note predicts
  that much of a converged-wrong fit's error is then a consistent shortfall that resampling
  cannot see, while the spurious weight that remains in low-variance directions is still
  visible to geometry. Part B and F-1 test this.
- **Dead steering vectors** are a bias-type failure (the read position, not the sample), which
  is why second-template stability is weak on them (S-2: 0.53-0.61).
- **F-13**: classify each failure type by its disagreement-to-error ratio under resampling; the
  table follows from this note.

## 5. Status

Part A: done, deterministic (seed 20261009). Part B (Paper 1's estimator, noiseless GELU units,
24 units, d = 64): running.
