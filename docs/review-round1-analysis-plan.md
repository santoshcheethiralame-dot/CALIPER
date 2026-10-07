# Review round 1: analyses run in response to the panel (filed 7 Oct 2026, before running)

The simulated panel of 7 Oct asked for these analyses. All are **exploratory**. They change no
pre-registered endpoint, and the paper reports them as exploratory. Script:
`experiments/analyse_review_round1.py`. Output: `results/review_round1.json`. Every one uses
archived rows and saved directions only. Where an analysis needs the stimulus, it rebuilds it
with the run's own corpus seed.

## R1. Route-matched comparison (panel B1)

The AUC of restart agreement and of held-out R2 in two settings. Both are per arm.
- **(a)** Units where the direct route was selected.
- **(b)** All units, against a failure label taken from the direct route's own alignment
  (`align_direct` < 0.95).

## R2. Function-space restart agreement (B2)

**Data.** B-15a saved its five direct-route restarts.

**Variants.**
- **Covariance cosine.** The median pairwise cosine between restarts under the stimulus
  covariance metric.
- **Random restarts only.** Restart agreement recomputed from restarts 1-4, which leaves out
  the ridge warm start.

**Scoring.** Each variant's AUC is taken against two labels:
- the Euclidean label;
- the fresh-text functional label.

## R3. All-arms table (B3)

**Arms.** Every arm on disk:
- the six pooled arms;
- Pythia-70m and Pythia-410m, plus 410m at 3,200 steps;
- B-15a/b/c;
- B-17.

**Columns.** For each arm:
- n and the number of failures;
- the AUC of each check;
- the difference, with a stratified bootstrap 95% CI (2,000 resamples, failures and passes
  resampled separately);
- the estimator version, device and precision, where recorded.

## R4. Pooling (B4)

**HKSJ pooling.** Random effects with Hartung-Knapp-Sidik-Jonkman CIs and a 95% prediction
interval, over the six filed arms and again over the four corrected arms.

**Paired permutation test.** Per arm, on rank-standardised scores, swapping the two checks
within each unit, with 10,000 permutations. The paper's existing "label-permutation" result
is relabelled as the negative control it was filed as.

## R5. Incremental value with response-only descriptors (B5)

**Baseline.** Descriptors computable from the response alone:
- active fraction (y > 0);
- response mean;
- response standard deviation;
- response kurtosis.

**Model and scoring.**
- The model is logistic regression.
- The score is cross-validated AUC, 50 repeats of 5-fold stratified CV.
- The increment from adding each check has a bootstrap CI over units, taken over the whole
  CV pipeline, 200 resamples.

**Arms.** B-14, B-8b and B-2c.

## R6. Within-run pass@k (B6)

**The curve.** B-15a's direct route, pass@1 to pass@5, with Clopper-Pearson CIs.
- pass@k is the best-of-k restarts selected by held-out R2, as the estimator selects them.
- The curve is given with and without the warm-started restart 0.

**Model fits.** Three models are fitted by binomial likelihood and compared by AIC, as a
descriptive comparison:
- one-class geometric;
- two-class;
- beta-geometric.

## R7. Error budget (B8)

**Per replicate.**
- Cohen's kappa against B-14.
- Aggregate failure counts.

**ICC.** Computed on arccos-transformed alignment.

## R8. Error spectrum (B9)

**Arms.** B-8b, B-15a, B-2c and B-17.

**Stimulus spectrum.** Per arm:
- eigenvalues;
- condition number;
- effective rank.

**Error decomposition.** For every failing unit, the error is the fitted unit direction minus
the unit-normalised w, sign-aligned. Its energy is split across stimulus eigen-directions,
reported as the share of error energy in the directions that together carry the bottom 1%
of stimulus variance. This is given separately for converged-wrong fits, under-fitted fits
and B-17's fits.

## Reporting rule

Every number above enters the paper labelled exploratory, in a table that separates
confirmatory, pre-specified secondary and exploratory results.
