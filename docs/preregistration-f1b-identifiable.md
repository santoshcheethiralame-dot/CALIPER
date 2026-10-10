# Pre-registration: F-1b, does interventional data repair fits that are wrong on the identifiable label?

**Filed 10 October 2026, before any F-1b fit.** Flagship (`docs/flagship-plan.md` §13). Replaces
F-1 as the evidence for Gate F-A. Units: `results/f1b_units.json`, drawn by
`experiments/f1b_units.py` and committed with this filing. Queue: `experiments/rerun_queue9.sh`,
on the laptop CPU after queue 8. Scorer: `experiments/analyse_f1b.py`, committed with this filing.

## Why F-1 does not answer the question

F-1's filed primary held (targeted augmentation repaired 26/26 converged-wrong units), but:
- 19 of the 26 were wrong only along the layer-norm null direction 1/gamma, which real layer-norm
  outputs never move along. Any synthetic sample pins it: random augmentation repaired 25/26.
- On the 7 units wrong on the identifiable label, targeted repaired 7 and natural 5 (p 0.25).
- Every arm is a fresh fit, and a re-fit alone moves about one verdict in five (B-15, B-14r).
  F-1 had no arm that measured that, and its units were selected for being wrong, so some repairs
  are regression to the mean.

## Units (fixed now; `results/f1b_units.json`)

From every GPT-2 and GPT-Neo run with saved fitted directions, scored on the identifiable label:
- **A** (primary): all 17 units whose archived fit is converged-wrong on that label (identifiable
  alignment < 0.95, held-out R2 > 0.99);
- **B**: 30 of the 69 identifiable under-fitted units, drawn with `default_rng(0)`;
- **C** (harm): 10 of B-8b's 79 passing units, drawn the same way.

A unit found in several runs takes the settings of the first in the order B-8b, X-1a, X-1b, B-15a,
B-15b, B-15c, B-17. Pythia is left out: its arm was loaded in float16 at the time. 57 units in 6
setting groups.

## Arms (each re-fits every unit with its source run's settings)

1. **targeted**: 2,000 synthetic samples in the bottom-1% variance subspace, as in F-1;
2. **random**: 2,000 isotropic synthetic samples, as in F-1;
3. **natural**: 2,000 extra natural tokens from disjoint documents, as in F-1 (count-matched);
4. **reseed** (new): no augmentation, fit seed changed (base + 100). It measures how often a
   verdict moves through re-fitting and regression to the mean alone.

All verdicts are on the **identifiable label**: 1/gamma of the layer norm feeding the MLP removed
from fit and reference, alignment >= 0.95 to pass.

## Primary endpoint and prediction

Population A, targeted against reseed, exact one-sided McNemar on the 17 paired verdicts.
**Prediction: targeted passes more, p < 0.05.**

With 17 units the test needs a large difference: at least six discordant pairs, all in favour of
targeted, to reach p < 0.05. A smaller true effect will read as a null.

## Secondary (no predictions)

1. Targeted against natural, two-sided McNemar, with both counts. This is the interventional
   question proper; F-1 suggests the difference is small.
2. Random against targeted, two-sided: does the low-variance targeting matter once 1/gamma is out
   of the label?
3. Natural against reseed: do 2,000 more real tokens beat re-fitting alone?
4. Pass counts per arm in populations B and C; harm = population C units that fail under each arm.
5. Each arm's median share on 1/gamma.

## What the outcomes mean for Gate F-A

- **Primary holds and targeted beats natural:** interventional data repairs practical
  non-identifiability, and the remedy claim stands.
- **Primary holds but targeted is close to natural:** more data repairs it, and the
  interventional claim is dropped.
- **Primary fails:** re-fitting alone explains the repairs at this sample size. The flagship
  leads with detection (the 1/gamma share and the low-variance share), as F-A's failure branch says.

## Not done

No change to arms, sizes, the unit lists or the label after data arrive. No unit is dropped; a fit
that fails to run is reported as missing. B-8b and the other pre-6-Oct runs had an unseeded cascade
route, so no arm reproduces their base fits; the arms are compared with each other, never with
the base fit.
