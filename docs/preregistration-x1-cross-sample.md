# Pre-registration: X-1, does agreement across token samples catch what restart agreement misses?

**Filed 8 October 2026, before any X-1 run.** Paper 1, hardening plan items 1 and 2
(`docs/paper1-hardening-plan.md`).

## Why

**The lead.** An exploratory comparison on data already seen (notebook §4, 8 Oct;
`results/cross_sample_agreement.json`) suggested a new check:
- Agreement between two fits of the same unit on different token samples predicts failure
  at AUC 0.922.
- On converged-wrong fits it scores 0.911, where held-out R2 scores 0.713 and restart
  agreement 0.685.

**Weaknesses of that lead.**
- Only 6 converged-wrong units.
- The two fits differed in restart count as well as token sample.
- It was found after the data were seen.

**The mechanism to test** (the 7 Oct identifiability analysis). Converged-wrong fits differ
from w in directions the stimulus barely excites. Those directions should move when the token
sample changes, and not when only the initialisation changes.

X-1 tests the check on fresh units with a clean design. It also replicates the
identifiability findings confirmatorily.

## Design

**Model and units.**
- GPT-Neo-125m, layer 10, the arm with the most converged-wrong fits (20 of 100 in B-8b).
- 100 units never fitted before: drawn with `numpy.random.default_rng(20261008)` from the
  layer's 3,072 units minus B-8b's 100.
- The list is in `results/x1_units.txt`, committed with this filing.

**Two fits per unit, identical except for the token sample:**
- corrected estimator (`--independent-units`);
- 2 restarts, fit seed 0;
- 8,000 tokens, 1,600 steps, float32;
- saved directions.

| fit | stimulus |
|---|---|
| X-1a | corpus seed 0 (as every earlier run) |
| X-1b | corpus seed 1 with every corpus-seed-0 document removed (`--exclude-corpus-seed 0`): document-disjoint from X-1a |

**Fresh-text diagnostics.** `diagnose_units.py` on X-1a with `--corpus-seed 2
--exclude-seed 0`.

## Definitions

- **Labels from X-1a.** Failure means |cos(v-hat, w)| < 0.95. A failure is converged-wrong if
  its held-out R2 > 0.99, and under-fitted otherwise.
- **Cross-sample agreement.** |cos| between the unit's X-1a and X-1b selected directions.
  It is ground-truth-free: computing it needs only two fits.

## Predictions

**P1 (primary).** On converged-wrong vs passing units, AUC(cross-sample agreement) minus
AUC(held-out R2) is above 0. The stratified bootstrap 95% CI (2,000 resamples) excludes 0.
- With fewer than 8 converged-wrong units, P1 is reported as underpowered and not extended.

**P2.** At least 80% of converged-wrong fits reach a stimulus-weighted alignment of at least
0.99 on fresh text.

**P3.** In the bottom-1%-variance directions of the stimulus covariance:
- converged-wrong fits have a median error-energy share of at least 0.70;
- under-fitted fits have a median share below 0.50.

**Readings.**
- **P1 holds:** "data-resampling agreement detects what initialisation agreement and held-out
  fit cannot" becomes a confirmatory result in Paper 1.
- **P1 fails:** the exploratory lead is reported as not replicated.
- **P2 and P3 hold:** the identifiability section becomes confirmatory, not exploratory.

## Secondary, reported as stated

1. All failures: cross-sample agreement vs held-out R2 (DeLong).
2. Restart agreement in every comparison.
3. The same comparisons with labels taken from X-1b.
4. The under-fitted class.

## Analysis

`experiments/analyse_x1.py`, developed on B-15a vs B-15b (GPT-2 layer 6) before any X-1 row
existed (`results/x1_dev.json`). There it reported P1 as underpowered (6 converged-wrong
units), P2 at 5 of 6, and P3 at 0.957 vs 0.306. Run once on X-1.

## Not done

- No unit, seed, threshold or class boundary is changed after seeing output.
- The cross-sample check is not tuned. It is the single |cos| between two fits.

## Addendum 1 (8 October 2026, before any X-1 run)

**Added prediction P4, from an exploratory pass on earlier arms.** The pass is in
`experiments/explore_patterns.py`, Q1/Q2, and notebook §4 of 8 Oct.

**What the earlier arms showed.** The quantity is the share of the fitted direction's squared
norm lying in the stimulus eigendirections that carry the bottom 1% of variance. It is
ground-truth-free: it needs one fit and the stimulus covariance. It separated converged-wrong
from passing units at these AUCs:
- 0.953 on GPT-Neo layer 10;
- 0.941, 0.961 and 0.947 on three GPT-2 layer-6 re-fits;
- 0.972 on Pythia-160m layer 6.

Held-out R2 scored 0.55 to 0.72 on the same comparisons. The same share computed for the true
w did not predict the class (AUC 0.17 to 0.50).

**P4.** On X-1a, comparing converged-wrong with passing units:
- AUC(fitted-direction low-variance share) is at least 0.85;
- AUC(fitted-direction low-variance share) minus AUC(held-out R2) has a stratified bootstrap
  95% CI above 0.

**Definition.** The bottom-1%-variance eigendirections of X-1a's own stimulus covariance, as
for P3. With fewer than 8 converged-wrong units, P4 is reported as underpowered.

**Code.** Added to `experiments/analyse_x1.py` before any X-1 row exists. On the development
pair (B-15a/b) it reports P4 as underpowered (6 units).

**Status.** P1 to P3 are unchanged. P4 is confirmatory for this new check: the units are fresh
and the threshold and definition are fixed here.
