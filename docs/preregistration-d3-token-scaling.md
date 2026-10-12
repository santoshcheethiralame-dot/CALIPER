# Pre-registration: D-3, does held-out R²'s lead fade with the token budget on the corrected estimator?

**Filed 11 October 2026, before any D-3 run.** Paper 1, internal review round 3, item D3.

## Why

Paper 1 reports that held-out R²'s lead over restart agreement is largest where fits are
data-limited and fades where they are not. The evidence is B-10, a token-budget sweep on 50 GPT-2
layer-6 units: Δ −0.062, −0.138, −0.201 and −0.052 at 2k, 4k, 8k and 16k tokens (the 16k
interval [−0.218, +0.107]). B-10 used the coupled estimator, which the paper replaced (shared
early stopping, batch-position seeding), scored the Euclidean label only, and saved no fitted
directions. D-3 repeats the sweep on the corrected estimator, on fresh units, with directions
saved, so that the identifiable label can be scored.

## Design

**Units.** 150 GPT-2 layer-6 MLP units never fitted before and not in D-1's draw:
`numpy.random.default_rng(20261012)` over the layer's 3,072 units minus every unit id in any
earlier result file (858) and minus D-1's 200 (pool 2,200). `experiments/d3_units.py` →
`results/d3_units.txt`, committed with this filing.

**Fits.** The same 150 units at four budgets, 2,000, 4,000, 8,000 and 16,000 tokens, each with
`--restarts 2 --steps 1600 --batch 32 --layer 6 --independent-units`, fit seed 0, corpus seed 0,
float32, directions saved, as B-10 except the estimator. Outputs
`results/d3_gpt2_l6_t<budget>.jsonl`. Budgets run in the order 8k, 16k, 4k, 2k, so the primary's
two cells come first.

**Queue.** Local CPU queue 11 (`experiments/rerun_queue11.sh`), after queue 10 (D-1). About
33 hours of CPU.

## Definitions

As D-1. Identifiable label: |cos| between fit and w, both with u = 1/γ of `ln_2` removed, below
0.95. Euclidean label: the filed one. Checks: restart agreement and held-out R² of the selected
fit, as stored. Δ(T) = AUC(restart agreement) − AUC(held-out R²) over all failures vs passes at
budget T.

## Prediction

**P1 (primary).** On the identifiable label, Δ(16k) − Δ(8k) is above 0: the lead shrinks between
8k and 16k. Paired bootstrap over units (2,000 resamples; the same resampled units at both
budgets); the reading follows the 95% interval.

**Readings.**
- Fewer than 10 identifiable failures at 8k or at 16k: underpowered, and Paper 1 keeps "fades" as
  exploratory, citing B-10 and D-3 descriptively.
- **Lead shrinks** (interval above 0): the fading becomes a confirmatory result on the corrected
  estimator.
- **No change detected** (interval spans 0): Paper 1 drops "fades" and reports the lead as stable
  across 8k-16k within the interval.
- **Lead grows** (interval below 0): Paper 1 states that B-10's fading does not replicate on the
  corrected estimator.

No direction is assumed beyond the one P1 tests.

## Secondary, reported as stated

1. Δ(T) at every budget with its bootstrap interval, both labels.
2. P1 on the Euclidean label (B-10's label).
3. Class counts per budget: failures, converged-wrong (R² > 0.99), under-fitted, both labels.
4. B-10's coupled-estimator numbers beside D-3's, descriptively.

## Analysis

`experiments/analyse_d3.py`, committed with this filing. Developed with B-15a standing in for 8k
and B-15b for 16k (`results/d3_dev.json`): it reproduces their identifiable Δ (−0.175, −0.159) and
Euclidean Δ (−0.169, −0.084). Run once, when all four budgets have 150 rows.

## Not done

- No unit, budget, threshold, label or reading is changed after seeing output.
- If the queue stops early, completed budgets are reported and the rest marked not run; P1 is
  computed only if both 8k and 16k are complete.

## Addendum 1 (12 October 2026, committed 10:22, before the 16k cell's first row exists)

**Why.** An exploratory analysis of the 15 noiseless arms with saved directions (1,580 fits) found
that, at thresholds practitioners use, held-out R² certifies about twice as many correct directions
as restart agreement (rule A, held-out R² ≥ 0.995: 1,162 correct certified; rule B, restart agreement
≥ 0.97: 520), with false-certification rates within a few points of each other on the identifiable
label. The rate advantage comes from two 20-unit arms; on GPT-2 layer 6, D-3's condition, the two
rules tie. The pairing and thresholds were chosen with those data in view, so the result is
exploratory. This addendum tests it once on data no one has seen.

**Population.** D-3's 16k cell, all 150 units, identifiable label as D-3 defines it. The 8k cell's
rows already exist; they were not read and are excluded from this addendum.

**Rules (frozen; no other pairing is scored).** A: held-out R² of the selected fit ≥ 0.995.
B: stored restart agreement ≥ 0.97.

**Endpoints.** E1, yield: Y_A − Y_B, the share of identifiable-correct fits each rule accepts.
E2, false certification: FCR_A − FCR_B, the share of each rule's accepted fits that are wrong.

**Inference.** Paired unit bootstrap, 2,000 resamples, `numpy.random.default_rng(20261012)`,
percentile 95% intervals. A resample in which a rule accepts nothing is dropped from E2, and the
number dropped is reported.

**Readings.** E1 holds if its lower bound is above 0. E2, non-inferiority margin +0.03: non-inferior
if the upper bound is below +0.03; R² worse if the lower bound is above 0; inconclusive otherwise.
With fewer than 5 accepted-wrong fits across both rules, E2 is underpowered and reported
descriptively.

**Predictions.** E1 holds (expected; low information). E2 has no directional prediction: on GPT-2
layer 6 the in-sample estimate is a tie, and at 16k converged-wrong fits should raise R²'s rate.

**Paper reading.** E2 non-inferior: "comparable false certification at twice the yield" is confirmed
on fresh units at 16k. R² worse: the operating-point paragraph states that held-out R² over-certifies
once failures are converged-wrong, consistent with the fading lead.

**Scorer.** `certification()` in `experiments/analyse_d3.py`, added with this addendum and run once
with the rest of D-3. Development run on B-15a/B-15b standing in for 8k/16k
(`results/d3_dev.json`): E1 +0.49 [+0.37, +0.60], E2 underpowered (1 accepted-wrong fit).
