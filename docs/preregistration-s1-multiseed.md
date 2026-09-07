# Pre-registration - S1, multi-seed selection by held-out R2

**Filed 7 September 2026, before the run. Nothing below changes after seeing results.**

## Where this comes from, stated plainly

C13's gate found 77/100 units recovered, failing its own >=0.90 bar. C41 and C42 then
established that the failures are not unrecoverable: their objective is rugged, and the
recovered direction varies enormously across nominally equivalent configurations
(median spread 0.161, max 0.634) while passing units do not vary at all (0.003).

C43 piloted a repair on **six units chosen as the worst failures**: fit at several seeds,
keep the fit with the best held-out R2. On those six the median went from 0.5175 to
0.9669, test_r2 correlated with true alignment at +0.953, and the median regret against
an oracle was 0.0000.

**This design was chosen after seeing that pilot.** The protocol below - five seeds, one
restart, 800 steps - was sized on the same six units, where k=5 saturated. That is
disclosed here because it is the kind of choice a pre-registration exists to constrain,
and it will be disclosed in any write-up.

## Protocol (frozen)

| element | value |
|---|---|
| units | the same 100 as C13, by neuron id from `results/e01_gate.jsonl` |
| model / layer / corpus | GPT-2 small, layer 6, `sample_corpus(n_docs=300, seed=0)`, 20,000 tokens |
| fits per unit | 5, seeds 0-4, `n_restarts=1`, `steps=800`, `k=1` |
| selection | the fit with the highest `test_r2`. No use of ground truth anywhere in selection |
| scoring | alignment to the true weight column; pass = alignment > 0.95 **and** k2_gain < 0.01, identical to C13 |

## Primary endpoint

Pass rate under multi-seed selection, with a 95% Wilson interval.

**Criterion, fixed now: the lower bound of the interval exceeds 0.90** - the same bar
C13 was scored against and failed at 77/100 [0.679, 0.842].

- **PASS.** A ground-truth-free selection rule repairs the instrument, and the paper
  reports both numbers: 77% for a single fit and the new figure for five.
- **FAIL but improved.** Report the improvement with its paired interval and describe the
  instrument as partially repaired, with the residual failure rate as the headline.
- **NO IMPROVEMENT.** The pilot was a six-unit artifact. Report that plainly; it would
  mean the lottery account does not generalise beyond the units it was found on.

## Secondary endpoints

1. **Paired per-unit change** against C13's `align_direct`, with a bootstrap interval
   over units. This is the comparison the design is actually built for, since both arms
   see identical data.
2. **Correlation of `test_r2` with true alignment** across all 500 fits, and the median
   regret of test_r2 selection against the per-unit oracle. C43's pilot gave +0.953 and
   0.0000 on six units; whether that holds on 100 is the mechanism check.
3. **Cost of the rule for units that never needed it** - the change on units C13 already
   passed. A repair that damages healthy units is not a repair.

## Committed in advance

- Both the single-fit and multi-seed pass rates are reported, whichever direction the
  result falls, and the 77% is not retired.
- No change to the seed count, the step count, the selection statistic or the pass bar
  after seeing any part of the output. If the run dies partway it resumes on the same
  frozen settings.
- The disclosure above - that the protocol was sized on a favourable six-unit pilot -
  appears in the results section, not only here.
- If the primary passes, the claim is "selection by held-out R2 repairs the instrument on
  this model, layer and unit sample", not "the estimator works".
