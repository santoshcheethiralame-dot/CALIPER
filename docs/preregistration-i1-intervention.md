# Pre-registration: I-1, what a collapsed direction does when used to intervene

**Filed 17 October 2026, before any I-1 run.** Paper 1, answering the review's first point
(predictively equivalent or wrong?). Script: `experiments/i1_intervention.py`, which runs and scores
once. No new fit: it uses saved directions.

## Why

A collapsed fit predicts the unit almost perfectly on the stimulus (stimulus-weighted alignment
0.9998 at GPT-Neo L6), so "wrong" needs a definition. Three notions are exact on this substrate:
predictive equivalence (projections agree on the data), weight recovery (the identifiable label) and
interventional efficacy. Adding u to the MLP input moves unit j's pre-activation by exactly
alpha * (w_j . u), so efficacy per unit norm is the Euclidean cosine, the null direction included,
because an intervention leaves the data's manifold. I-1 measures efficacy and selectivity (the share of
the whole layer's pre-activation change that lands on the target) and the model-level effect.

## Design

Runs (first 20 units of each, in saved order): least squares at GPT-Neo L6 (B-17), GPT-2 L3 (B-17c),
OPT-125m L5 (F-3a); MID at GPT-Neo L6 and GPT-2 L3 (E-2b); controls, least squares and MID at GPT-2 L6
(E-1, E-2). Directions per unit: w_j; the fitted direction; the fitted direction with the layer's
near-constant coordinates zeroed; a random direction (`default_rng(0)`). Each is added at 0.25 x the
median MLP-input norm to the output of the norm feeding the MLP, at every position of about 2,000
tokens from unseen documents.

## Predictions

- **P1.** At every flagged run, the median selectivity of the fitted directions is below a tenth of
  the true direction's.
- **P2.** Where zeroing restores alignment (least squares GPT-Neo L6, MID GPT-Neo L6, MID GPT-2 L3), the
  zeroed directions' median selectivity is at least half the true direction's.
- **P3.** At both controls, the fitted directions' median selectivity is at least half the true one's.
- **P4.** The measured change in the target unit matches alpha * (w_j . u) within 1% (median relative
  error), in every run.

Next-token KL is reported for every direction, with no prediction.

## Not done

No dose, unit, run or threshold is changed after seeing output.
