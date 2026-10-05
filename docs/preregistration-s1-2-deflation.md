# Pre-registration — S1-2, can deflation recover K=2? The Study 2 gate

**Filed 5 October 2026, before the run.** S1-2 had a criterion in its script and in the
notebook's §7.8, but no filed document. On 4 Oct the code started judging it on the best of
all deflation arms, a change made after `resp_only` was measured higher in development.
This filing restores the criterion as it was originally written and labels everything added
since then as exploratory.

## Question

Joint rank-K estimation recovers one direction and misses the rest: mean alignment 0.5213
at K=2 and 0.3768 at K=3 (`results/e03_required_n.json`, additive planted units). That is
the arithmetic of one direction found plus chance. Does finding one direction, removing
it and finding the next do better?

## What I have already seen, disclosed

Development probes on synthetic planted units, not the run:

- additive: projected deflation 0.945, response-only 0.995
- multiplicative: projected 0.506, response-only 0.930
- a one-unit probe at the real settings: joint 0.512, plain 0.746, cascade 0.740

These are why the multiplicative coupling and the response-only arm exist. They are also
why neither can carry the primary.

## Primary endpoint and criterion

**Median subspace alignment of the `cascade` arm at K=2, N=8000, additive coupling, over
24 planted units, must exceed 0.80.** The arm, coupling, cell and bar are those in the
script docstring and §7.8 as first written. The comparison baseline is the archived joint
0.5213, re-measured in the same session by the `joint` arm.

| cascade median at K=2, N=8000, additive | reading (unchanged from §7.8) |
|---|---|
| > 0.80 | Study 2 green-lit for sem 6 |
| 0.60 - 0.80 | one planted direction per trait; P4's multitrait matrix goes one direction at a time |
| above joint but <= 0.60 | estimator limit; restrict multi-dimensional claims to K=1 |
| <= joint | cut Study 2 from the paper; report the K>=2 degeneracy as an estimator-family limit |

## Exploratory, reported but not judged

- The `plain` and `resp_only` arms, and the best-of-arms figure the script prints.
- The multiplicative coupling (no archived joint baseline exists for it).
- K=3, and N=2000.
- Per-step alignment traces.
- **Minimum principal-angle cosine** beside the mean. A mean of 0.52 hides "one of two
  found", and the minimum shows it directly.
- Arms that may be added before the run, each exploratory if added:
  (a) whitening the stimulus with a shrinkage covariance, plus a warm start from first-order
  and second-order moments (pHd / SIR) or from the average gradient outer product of an
  unconstrained fit;
  (b) response deflation that subtracts the fitted g1(v1 . x), not the unit-coefficient
  linear term the current code subtracts, followed by backfitting.

If an exploratory arm clears 0.80 and the primary does not, that is reported as a lead for
a new filed run, not as the gate passing.

## Fixed now

The planted generator (`experiments/planted_units.py`, additive form pinned to
`e03_required_n.py` by `tests/test_planted_units.py`), 24 units, 1,600 steps, 2 restarts,
the alignment metric (mean cosine of principal angles), and the table above.
