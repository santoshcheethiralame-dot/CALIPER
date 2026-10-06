# Pre-registration: S-4, Study 3 reanalysis (equivalence tests and exact intervals)

**Filed 7 October 2026, before `experiments/analyse_s3_reanalysis.py` is run.** No new data.
The archived Study 3 runs are already unblinded, so what is fixed here is the *test and its
margin*, before the equivalence results are computed.

## Why

Several Study 3 claims read "indistinguishable" off a large p-value (C18 real vs random at
alpha 6, p = 0.33; C20 introspective vs neutral, p = 0.44-0.75; C24 at alpha 32768, p = 0.19).
A large p-value is not evidence of equivalence. A reviewer will ask for an equivalence test,
and the 2/30 positive control needs an exact interval.

## Fixed now

- **Equivalence test:** paired TOST on per-concept P(YES) differences (n = 30 concepts), using
  two one-sided t-tests. **Margin +/-0.10** in P(YES), absolute: the smallest difference we
  would call a real effect at the readout's scale (Macar et al.'s reported detection rate is
  about 0.11). Equivalence is declared if both one-sided p < 0.05, i.e. the 90% CI of the mean
  difference lies inside +/-0.10. Bootstrap 90% CIs are reported beside the t-based ones.
- **Comparisons tested**, introspective framing unless stated, all paired by concept:
  1. C18: real vs norm-matched random at alpha 6 (unnormalised).
  2. C19: shuffle vs random at alpha 2, 4, 6.
  3. C20: introspective vs neutral-matched, real vectors, alpha 2, 4, 6.
  4. C24: real vs random at alpha 32768 (unit-norm).
  5. C50-C52: real vs random and real vs span at alpha-frac 0.30, 0.40, 0.50, 0.60.
- **No correction across the family.** TOST is reported per comparison, as an equivalence
  claim is per comparison.
- **Exact intervals:** Clopper-Pearson 95% CIs for the positive-control reproduction (2/30 under
  the pre-registered scorer, C40) and for detection rates by alpha in `s3_generation.jsonl`.
- **Dose-response:** mean P(YES) with a 95% bootstrap CI (concepts resampled) per run, framing,
  control and alpha, written to `results/s3_reanalysis.json` for Paper 2's figures.
- **Vector status table:** each run's vector read position (template tail or concept token)
  and its steering positive-control result where one exists (C31, C32, C48), as recorded in
  the notebook.

## How it is read

- A comparison that fails TOST and whose ordinary test is also non-significant is reported as
  **inconclusive**, not "indistinguishable".
- Any Study 3 sentence currently saying "indistinguishable" is rewritten to whichever of
  "equivalent within +/-0.10", "inconclusive" or "different" the test supports.
