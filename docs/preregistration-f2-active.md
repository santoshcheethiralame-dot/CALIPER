# Pre-registration: F-2, do top-responding natural tokens repair under-fitted units?

**Filed 16 October 2026, before any F-2 fit.** Flagship (`docs/flagship-plan.md`, F-2). Units and
per-group settings: F-1b's (`results/f1b_units.json`), populations B and C. Fitter:
`experiments/e01_gate.py --augment active` (added with this filing). Scorer:
`experiments/analyse_f2.py`, committed with this filing. Queue: `experiments/rerun_queue19.sh`,
after queue 18.

## Why

Under-fitted units are sparse (notebook Q6: median firing fraction 0.02-0.05 against 0.12-0.21 for
passing units), so a uniform sample spends most of its tokens where the unit is silent. F-1b
measured three ways of adding 2,000 rows on its population B (30 identifiable under-fitted units):

| arm | passes / 30 |
|---|---|
| reseed (no new data) | 13 |
| natural (2,000 uniform fresh tokens) | 15 |
| targeted (2,000 synthetic, low-variance subspace) | 26 |
| random (2,000 synthetic, isotropic) | 27 |

Synthetic rows repair almost every unit, but a practitioner fitting a feature in a real model
cannot always query it off the data manifold. The question F-2 asks is whether real tokens, chosen
where the unit responds, close most of that gap. This is what interpretability practice already
does when it reads a unit's top-activating examples.

## The new arm

**active**: from the same fresh documents as F-1b's natural arm (corpus seed + 1, documents in the
fitting corpus removed), collect 20,000 tokens; keep the 2,000 with the largest response of the
unit; add them to the fitting rows. One unit per run, because the selection is per unit. The
held-out rows are unchanged, as in F-1b. The fitted row count equals the natural arm's; only the
forward passes used for selection differ (20,000 against 2,000), and forward passes are cheap next
to fits.

No importance weights. The units are noiseless and the estimator's link is fitted to the unit's
exact response, so selecting inputs by the response changes the input distribution but not the
function being fitted; a correctly specified fit is not biased by that shift. Whether the shift
hurts in practice is what population C's harm count measures.

The pool size was set at 20,000 before any fit because the fresh documents yield about 25,400
collectable tokens (checked on corpus seed 2).

## Primary endpoint and prediction

Population B (30 units), active against F-1b's natural arm, exact one-sided McNemar on paired
identifiable-label verdicts (alignment >= 0.95). **Prediction: active passes more, p < 0.05.**

## Secondary (no predictions)

1. Active against reseed and against random (two-sided McNemar with both counts).
2. Harm: population C (10 passing units) failures under active, against F-1b's arms.
3. Median identifiable alignment per arm.

## What the outcomes mean

- **Primary holds:** the under-fitted class has a practitioner's remedy on the data manifold:
  sample where the unit responds. The flagship reports two failure classes with two remedies.
- **Primary fails and active is close to natural:** where the unit responds is not the missing
  information; the synthetic arms repair through directions real tokens do not cover, and the
  remedy for sparse units stays off-manifold.
- **Harm on population C above natural's (1 of 10):** the selection shift costs passing units; reported
  as the price of the remedy.

## Not done

No change to the pool, the count, the units, the label or the bar after data arrive. A unit whose run
fails is reported as missing. The F-1b arms are not re-run: active is compared with them as filed,
on the same units and settings (`--independent-units` makes each unit's fit independent of which
other units share its run).
