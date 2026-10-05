# B-14 Addendum 1 — secondary analyses fixed before unblinding

**Filed 5 October 2026, while B-14 is running.** At filing, only the row count of
`results/b14_primary_gpt2_indep.jsonl` has been looked at (30 of 300). No alignment, signal
or AUC from B-14 has been viewed. The primary endpoint and decision table in
`preregistration-b14-primary-rerun.md` are unchanged. Everything below is secondary or
descriptive and is added for rigour, following the 5 Oct literature pass.

## Procedure: blind analysis

The analysis script is written and tested on **B-1b's rows**, which are already unblinded,
and on B-1b with its failure labels permuted. It is committed before it is run on B-14 and
is then run on B-14 once. Any change after that run is reported as a deviation, with the
original output kept.

## Added secondary analyses

1. **Identifiable-direction label.** The stimulus is the `ln_2` output s = gamma * z + beta,
   with z zero-mean across coordinates, so s . (1/gamma) is constant on every token. The
   component of w along 1/gamma is not identifiable from data. A second label scores
   alignment against w with that component projected out (w_perp), and for the fitted
   direction too. GPT-2 L6 has no unit whose share along 1/gamma limits Euclidean cosine
   below 0.95; the primary label is unchanged and this label is reported beside it.
2. **Functional label.** The correlation of the two projections on held-out tokens,
   corr(v . s, w . s), which equals the covariance-weighted cosine. Reported as a second
   continuous outcome.
3. **Threshold sweep.** AUC of each signal against the failure label at bars 0.90, 0.92,
   0.95, 0.97 and 0.98, with label prevalence at each. Descriptive, no test.
4. **Threshold-free association.** Spearman rho between each signal and continuous
   alignment.
5. **PR-AUC** for each signal, beside ROC-AUC, because failures are a minority.
6. **Nuisance baselines and incremental value.** AUC of each pre-fit or trivial predictor
   alone: the neuron's active fraction, its mean pre-activation z_mean, the response
   kurtosis, and the GELU first-order coefficient mean(GELU'(w . s)). Then the 5-fold
   cross-validated AUC of logistic(baselines) against logistic(baselines + held-out R2) and
   against logistic(baselines + restart agreement). This answers whether R2 adds anything
   beyond unit difficulty.
7. **Calibration.** A reliability diagram with equal-mass bins and the Brier score of a
   Platt-scaled R2 rule. Isotonic scaling is not used, since there are too few failures.
8. **Controls.** Negative: the primary DeLong comparison rerun with failure labels
   permuted (1,000 permutations), expected to be null. Positive: a "cheating" signal equal
   to alignment itself, expected to give AUC 1.

## What will not change

The primary endpoint, the decision table, the unit draw, and the 0.95 primary bar. If any
secondary analysis contradicts the primary, both are reported and the primary governs the
recommendation.
