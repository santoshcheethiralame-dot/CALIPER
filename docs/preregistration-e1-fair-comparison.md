# Pre-registration: E-1, the checks compared on one route, with independent restarts and an untouched test set

**Filed 15 October 2026, before any E-1 data.** Paper 1, to become its main experiment. Scripts,
committed with this filing: `experiments/e1_units.py` (unit draw), `experiments/e1_fit.py` (fits),
`experiments/analyse_e1.py` (scorer, run once when both arms are complete).

## Why

An internal review of Paper 1 (15 Oct) raised four design points. (1) Held-out $R^2$ is a validation
score: the same held-out tokens stop training, choose between routes and are reported. (2) Restart
agreement and held-out $R^2$ describe different fits on the units where the cascade route is selected,
and the first restart is warm-started from ridge, so the restarts are not independent. (3) The
Euclidean label calls functionally equivalent fits wrong along the layer norm's null direction. (4)
The proposed remedies (null-direction removal, the low-variance diagnostic, the practitioner rule)
were found on the data they explain. E-1 removes all four from one comparison.

## Design

**Units.** 200 GPT-2 small layer-6 and 100 GPT-Neo-125m layer-10 MLP units never fitted in any earlier
run, drawn once by `e1_units.py` (seeds 20261015 and 20261016) from the units absent from every
results row, so the two arms share no id. Written to `results/e1_units.json` before any fit.

**Data, three ways.** Training and validation tokens: the corpus draw of every earlier run (300
documents, corpus seed 0), 8,000 tokens shuffled at the token level; the first 20% are the validation
split, the rest train. Test tokens: 4,000 tokens from corpus-seed-1 documents that are not in the
training draw. The test set never touches the fit: not early stopping, not restart selection.

**Estimator.** The least-squares single-index estimator of the paper, direct route only ($k = 1$), five
restarts all initialised at random (no ridge warm start), per-unit seeds from the unit id, per-unit
early stopping on the validation split, 1,600 steps, float32. The selected fit is the restart with the
best validation $R^2$. Every restart's direction is saved.

**Checks, all ground-truth-free.**
- *Restart agreement*: the median pairwise $|\cos|$ over the five restarts.
- *Test $R^2$*: the selected direction's $R^2$ on the test documents, through a binned link (60
  equal-count bins) fitted on the training rows only.
- *Validation $R^2$*: the estimator's own score, as the paper reports it now; descriptive here.
- *Restart $R^2$ spread*: max minus min validation $R^2$; descriptive.

**Label (primary).** The identifiable label: a fit fails when $|\cos(P\hat{v}, P\mathbf{w}_j)| < 0.95$,
with $P = I - \mathbf{n}\mathbf{n}^\top$ and $\mathbf{n}$ the normalised $1/\gamma$ of the layer norm
feeding the MLP. The Euclidean $|\cos(\hat{v}, \mathbf{w}_j)| < 0.95$ is the secondary label.

## Predictions

**P1 (primary, GPT-2 layer 6, identifiable label).** $\Delta = $ AUC(restart agreement) $-$ AUC(test
$R^2$) at predicting failure is negative, with the upper bound of the stratified bootstrap 95% CI
(2,000 resamples, failures and passes resampled separately) below zero. DeLong $p$ and the paired
permutation $p$ are reported beside it. Readings: *holds* (CI below zero); *not replicated* (CI
includes zero); *reversed* (CI above zero); *underpowered* (fewer than 10 failures). Basis: D-1
gave $-0.147$ $[-0.215, -0.078]$ on this label with the old design; B-14r's route-matched comparison
gave $-0.136$. Five independent restarts may make restart agreement more informative, so the
prediction can fail.

**P2 (GPT-Neo-125m layer 10, same test).** Same direction and readings, as a replication. Basis: X-1a
gave $-0.035$ $[-0.116, +0.049]$, so a smaller gap is expected and *not replicated* is plausible.

**P3 (the practitioner rule, prospective, both arms pooled, identifiable label).** Accepting fits
with validation $R^2 \geq 0.995$ (the paper's rule, as stated) certifies more correct directions than
accepting restart agreement $\geq 0.97$, and its share of wrong accepted fits is not higher (point
estimate of the difference $\leq 0$; bootstrap 95% CI over units reported). The same rule with test
$R^2$ in place of validation $R^2$ is reported beside it. Thresholds are the paper's, fixed before
this run.

**P4 (the low-variance mechanism, prospective).** Among fits with test $R^2 > 0.99$, the share of
the error in the directions carrying the bottom 1% of stimulus variance ($\mathbf{n}$ removed; the
paper's `err_low_ident`) separates failing from passing fits with AUC $\geq 0.85$. Scored only with at
least 5 such failures and 5 passes; the diagnostic uses $\mathbf{w}_j$, so it tests the mechanism, not
a practitioner check. Basis: 0.965 in the paper's arms and 0.929 in the X-1 replication.

## Secondary (descriptive)

The Euclidean label; AUC(validation $R^2$) $-$ AUC(test $R^2$) (does reusing the validation split
change the check?); restart agreement against validation $R^2$; the failure classes (under-fitted:
test $R^2 \leq 0.99$; converged-wrong: above); verdicts changed by removing $\mathbf{n}$; median test
minus validation $R^2$.

## Exclusions and stopping

All drawn units; a unit whose response SD is at most $10^{-4}$ is reported and not fitted (as in
every earlier run). Rows resume by unit. Scored once, when both arms have every row; nothing is
re-scored after the outputs are seen. The analysis code was smoke-tested on two GPT-2 units fitted
for 150 steps with three restarts (output not kept).

## Cost

Local CPU queue 16, after queue 15. Expected 8 to 12 hours for both arms (to be reported from the
run's `seconds_per_unit`).

## Amendment 1 (15 October 2026, before any E-1 row): a code-text arm

**Arm.** 100 of the 200 GPT-2 layer-6 units (`default_rng(20261017).choice` from that list, written to
`results/e1_units.json` as `gpt2_l6_code`), fitted exactly as above on Python source code instead of
prose: the non-test `.py` files of the installed Python 3.12.10 standard library (676 files, PSF
licence, no download), in a seeded order whose first 300 files are the training draw and the rest the
test pool, so training and test never share a file. The run's summary records the file order.

**P5 (secondary).** On this arm, $\Delta$ = AUC(restart agreement) $-$ AUC(test $R^2$) is negative,
with the same readings as P1. Because the units are shared with the prose arm, the domain is the only
thing that changes; their verdicts on prose and on code are reported side by side (descriptive). The
code arm is not pooled into P3.

**Order.** The code arm runs after both prose arms in queue 16.

## Not in E-1

A second estimator family, proposed in the same review, is left for a later filing.
