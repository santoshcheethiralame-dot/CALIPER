# Pre-registration: E-2, a second estimator family (maximally informative dimensions) on E-1's units

**Filed 15 October 2026, before any E-2 data and before any E-1 data.** Paper 1. Scripts committed with
this filing: `experiments/e2_mid.py` (fits) and `experiments/analyse_e2.py` (scorer, which calls E-1's
scorer unchanged).

## Why

An internal review of Paper 1 (15 Oct) asked whether the comparison of restart agreement and held-out
fit depends on the estimator. Every Paper 1 arm uses one family: a least-squares single-index model
with a small neural link, related to maximally informative dimensions (MID) only through a
likelihood equivalence. E-2 fits MID itself, a different objective (mutual information, no link
function, no network) on exactly E-1's units, data, splits, test set and label, so the estimator is
the only thing that changes.

## Design

**Units.** E-1's 200 GPT-2 small layer-6 units (primary) and 100 GPT-Neo-125m layer-10 units
(secondary), from `results/e1_units.json`.

**Data and checks.** As E-1 (`docs/preregistration-e1-fair-comparison.md`): 8,000 corpus-seed-0
tokens with the first 20% as validation, 4,000 test tokens from unseen corpus-seed-1 documents,
per-coordinate standardisation, and test $R^2$ through a 60-bin link fitted on the training rows.
Restart agreement is the median pairwise $|\cos|$ over five restarts.

**Estimator.** MID: the direction maximising the plug-in mutual information between the standardised
projection (soft-binned onto 32 Gaussian kernels on $[-4, 4]$) and the response (16 equal-count bins).
Adam (learning rate 0.02, up to 1,600 steps), five restarts from random directions seeded by unit id,
per-unit early stopping on validation information (patience 200 steps); the selected restart has
the highest validation information. Smoke test before filing: 8 E-1 GPT-2 units at 800 steps and three
restarts, alignment 0.98 to 1.00 (output not kept).

**Label.** The identifiable label (primary) and the Euclidean label, as in E-1.

## Predictions

**P1 (GPT-2, identifiable label).** On MID fits, AUC(restart agreement) $-$ AUC(test $R^2$) is negative
with the bootstrap CI below zero. Readings as E-1's P1, including *underpowered* below 10 failures;
MID may fail rarely, in which case P1 is reported as underpowered and the failure rate is the result.

**P2 (cross-family agreement).** On MID fits, the $|\cos|$ between the MID fit and E-1's fit of the same
unit predicts MID failure better than MID's own restart agreement: AUC(restart agreement) $-$
AUC(cross-family agreement) is negative with the bootstrap CI below zero. Basis: restarts of one
estimator share its biases; a second family's errors need not coincide. If both families fail the
same way (for example along a direction the stimulus barely explores), P2 fails, which is itself
informative.

## Secondary (descriptive)

Per-unit verdicts of the two families on the same units (McNemar); the same tests on GPT-Neo layer 10;
the Euclidean label; failure classes; cross-family agreement against test $R^2$.

## Exclusions, stopping, cost

All units; dead units reported and not fitted. Scored once, when both E-2 arms and E-1's matching arms
are complete. Local CPU queue 17, after E-1 (queue 16) and a short test-$R^2$ job on earlier arms;
about 40 to 80 seconds per unit on this laptop, so roughly 3 to 7 hours.
