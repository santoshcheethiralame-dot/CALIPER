# Paper A - revised outline, Study 1 leading

**Drafted 8 September 2026. Supersedes `paper/PLAN.md` section 0, which positioned this
as a Study 3 paper.** The reason for the change is in the lab notebook under
"Literature scout, 8 September 2026".

---

## The one-sentence version

> A language model gives you something no brain does: for one class of unit, the correct
> answer is written in the weights. We score the field's own direction-finding estimator
> against it, find it silently wrong on 23% of units, and show the same failure of
> calibration recurs when the readout is the model's own self-report.

## Why this order

Study 3's space is contested - binary injection-detection has already been shown
artifactual (arXiv 2512.12411, control r=0.999) and content-agnostic (Lederman &
Mahowald 2603.05414), steering vectors already shown non-identifiable (Venkatesh &
Kurapath 2602.06801), and a reality-check paper is published at COLM 2026. Our A2 result
was never ours.

Study 1's space is empty, and its ground truth is **free** - an MLP neuron's input weight
column *is* its direction, so no planting, no proxy, no assumption. That is a better
position than anything Study 3 can offer us.

---

## Title candidates

1. *Free Ground Truth: Calibrating Direction-Finding in Language Models*
2. *The Estimator Is Confidently Wrong 23% of the Time, and You Cannot Tell*
3. *Checking the Instrument: What a Neuron's Weights Reveal About How We Read Models*

## Abstract (draft)

> Interpretability methods that recover a direction from activations are validated by
> whether their output looks reasonable, because the correct direction is unknown. For
> one class of unit it is not unknown: an MLP neuron's pre-activation is exactly the
> projection of its own layer's residual stream onto its input weight column, so that
> column is the answer, available for free in any open model. We score a standard
> subspace estimator against it on 100 GPT-2 units under a pre-registered criterion. It
> recovers the known direction for 77 units [95% CI 68-84] and fails on the rest, and the
> failures are invisible: median alignment across all units is 0.993, and on the failures
> the estimator's own restart agreement is 0.85-0.99. A perfect solution exists for every
> failure - held-out R2 at the true direction is 1.000 - so these are optimisation
> failures, not information limits. We characterise them: the objective is rugged, a
> single unit's recovered direction ranges from 0.066 to 0.983 across seeds alone, and
> the failures concentrate where the unit sits deep in GELU's non-monotone region. A
> disagreement statistic between two estimation routes predicts failure at AUC 0.915
> without using ground truth, and we cross-validate its operating points. A five-seed
> selection rule that repaired the failures in a six-unit pilot did not survive a
> pre-registered test at n=100 - 76/100 against 77 - and we report that. We then show the
> same calibration gap at a second readout, the model's own report of an injected state,
> where the reported detection rate moves by 3-5x under choices no paper reports: the
> readout method, the scoring rule, the extraction position, the prompt framing, and the
> normalisation of injection strength.

---

## Structure

### 1. Introduction
The validation problem: every readout in interpretability is checked against plausibility
because truth is unavailable. One place it *is* available. What that buys.

### 2. Free ground truth
The identity: pre-activation = w . s, so w is the direction, exactly, verifiable to
3.3e-06. Why this is stronger than planted ground truth - no injection, no assumption
that the plant behaves like a natural feature.

### 3. Scoring the estimator (Study 1) - **the core**

| result | number | run |
|---|---|---|
| recovery under a pre-registered bar | **77/100**, Wilson [0.679, 0.842], FAIL | C13 |
| median alignment across all units | 0.993 - failures invisible in aggregate | C13 |
| restart agreement on failures | 0.85-0.99 - confidently wrong | C13 |
| oracle check | R2 = 1.000 at the true direction | C9 |
| seed range within one unit | **0.066 to 0.983** | C43 |
| classical baselines | STA 0/30, decorrelated STA 1/30, STC 0/30 vs 26/30 | C3 |
| random-direction null | R2 <= 0.044 | C1 |
| required N | ~200 informative events, K=1 | C14 |
| K >= 2 | degenerates to K=1 at every N | C14 |

### 4. A detector that needs no ground truth
Disagreement between two estimation routes, AUC 0.915, cross-validated operating points.
**This is the deliverable** - it runs where truth does not exist, which is everywhere
else. Report the honest ROC, not a tuned threshold.

### 5. A repair that failed
Five-seed selection by held-out R2 recovered the worst units in a six-unit pilot
(0.5175 -> 0.9669, test_r2 correlating with true alignment at +0.953). Pre-registered at
n=100 it gave **76/100 against 77** - repairs 8, breaks 9, McNemar p=1.000. Reported
because it was filed. Includes the confound we built in: 5 cheap draws vs 1 expensive
draw, not matched budget.

### 6. The same gap at a second readout (Study 3)
Not a competing empirical claim. A demonstration that the calibration problem is not
specific to unit-level estimation. Five knobs, each moving the reported detection rate
with the model untouched:

| knob | movement |
|---|---|
| readout method | 50% -> 7% (first token vs generated text) |
| scoring rule | 6.7% -> 33.3%, and FPR 0% -> 17% |
| extraction position | three mutually contradictory conclusions |
| strength normalisation | Gemma and Qwen 138x apart, uncomparable |
| prompt framing | 8x inflation of the response to content-free vectors |

Cite 2512.12411, 2603.05414, 2602.06801 as agreeing on the underlying instability. **Do
not re-litigate whether detection is content-free** - that is settled and it is theirs.

### 7. Limitations
One model family for Study 1 (GPT-2 small, layer 6). Rank-1 only; K>=2 unresolved. Study
3 is one model, one layer, 30 concepts, and its significant cells sit where steering is
weakly validated (C48). The extraction-position finding is narrower than it first
appeared - position is already known to matter (2602.00333).

### 8. Reproducibility
Every pre-registration with its filing date, every raw JSONL, the lab notebook including
the three reversals and the failed repair.

---

## What still needs doing

| item | cost | blocking? |
|---|---|---|
| Adopt prompt-token averaging for extraction, re-check | 1 Kaggle session | no - Study 3 is now a chapter |
| A-14 validated-steering window | 1 session, sheet written | **no longer blocking** - settles a chapter claim |
| Study 1 on a second model (Pythia / GPT-2 medium) | CPU, ~1 day | **YES - "one model" is the obvious reviewer objection and Study 1 is now the paper** |
| Matched-budget re-run of the seed rule | CPU, ~4 h | no - improves section 5 |

**The priority changed.** A second model for **Study 1** now matters more than anything
in Study 3, because Study 1 is the contribution and one model family is its weakest point.

---

## Venue

Unchanged in tier, better fitted. An interpretability workshop - BlackboxNLP, ICML
MechInterp, NeurIPS ATTRIB. This is a calibration-and-controls paper with free ground
truth, which is exactly that genre. arXiv first, as planned.
