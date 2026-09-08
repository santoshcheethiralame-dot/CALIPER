# Paper A — outline v3, Study 1 leads and Study 2 is chapter two

**Drafted 8 September 2026. Supersedes OUTLINE_v2.md, which had Study 3 as chapter two.**
Reason in the lab notebook under "Literature scout 2, 8 September 2026".

---

## The through-line

> Every direction we read out of a language model is validated by whether steering with
> it works. That validation is invalid — orthogonal directions steer near-equivalently —
> and we can prove it, because there are two places where the true direction is knowable.
> At the unit level it is **free**, written in the weights. At the trait level it can be
> **planted**. The same estimator is silently wrong in both, and the field's own check
> cannot see it.

Three levels, one argument, in descending order of how solid our ground truth is.

---

## Title candidates

1. *Steering Is Not Validation: Calibrating Direction-Finding Where the Answer Is Known*
2. *Free Ground Truth: What a Neuron's Weights Reveal About How We Read Models*
3. *The Instrument Is Confidently Wrong, and the Standard Check Cannot Tell*

## Abstract (draft)

> Interpretability methods that recover a direction from activations — receptive-field
> estimators, persona vectors, concept vectors — are validated by whether intervening
> along the recovered direction changes behaviour. That check is known to be weak:
> orthogonal perturbations achieve near-equivalent steering efficacy, so behavioural
> equivalence classes are large. We ask what happens when the true direction is knowable.
> At the unit level it is free: an MLP neuron's pre-activation is exactly the projection
> of its layer's residual stream onto its input weight column, verifiable to 3.3e-06. We
> score a standard subspace estimator against 100 GPT-2 units under a pre-registered
> criterion. It recovers the known direction for 77 [95% CI 68–84] and fails on the rest,
> invisibly: median alignment across all units is 0.993 and the estimator's own restart
> agreement on the failures is 0.85–0.99. The failures are optimisation, not information —
> held-out R² at the true direction is 1.000, and one unit's recovered direction ranges
> from 0.066 to 0.983 across random seeds alone. A disagreement statistic between two
> estimation routes predicts failure at AUC 0.915 using no ground truth. A five-seed
> selection rule that repaired the failures in a pilot did not survive a pre-registered
> test at n=100 (76 against 77) and we report it. At the trait level we plant a known
> direction and run the persona-vector extraction pipeline on it, measuring recovery at
> the plant layer and against computational depth. We close with a shorter demonstration
> at the self-report level, where the reported rate moves 3–5× under choices no paper
> states.

---

## Structure

### 1. Introduction
Every readout is validated by steering. Cite 2602.06801: steering does not identify a
direction. So what is the field's validation actually worth? Two places let us find out.

### 2. Two kinds of knowable truth
**Free** — pre-activation = w·s, so w is the direction, exactly. **Planted** — inject a
known direction and ask the pipeline to find it. Free is stronger (no assumption that a
plant behaves like a natural feature); planted is more general (works where no weight
column exists). We use both.

### 3. Chapter one — the unit level, free ground truth *(complete)*

| result | number | run |
|---|---|---|
| recovery, pre-registered bar | **77/100**, Wilson [0.679, 0.842], FAIL | C13 |
| median alignment, all units | 0.993 — failures invisible | C13 |
| restart agreement on failures | 0.85–0.99 — confidently wrong | C13 |
| oracle | R² = 1.000 at the truth | C9 |
| seed range, one unit | **0.066 to 0.983** | C43 |
| classical baselines | STA 0/30, dSTA 1/30, STC 0/30 vs 26/30 | C3 |
| random-direction null | R² ≤ 0.044 | C1 |
| required N | ~200 informative events at K=1 | C14 |
| K ≥ 2 | degenerates to K=1 at every N | C14 |

### 4. A check that needs no ground truth
Disagreement between two estimation routes, AUC 0.915, cross-validated operating points.
**The deliverable** — it runs where truth does not exist. Honest ROC, not a tuned point.

### 5. A repair that failed
Five-seed selection by held-out R²: pilot 0.5175 → 0.9669 with test_r2 correlating to
true alignment at +0.953; pre-registered at n=100 it gave **76/100 against 77**, repairing
8 and breaking 9, McNemar p=1.000. Reported because it was filed. Includes the confound
we built in — five cheap draws against one expensive draw, not matched budget.

### 6. Chapter two — the trait level, planted ground truth *(to run)*

**Why it matters now:** persona vectors are in production monitoring (2607.13162 audits
open-weight models with them; 2605.13329 traces them through pretraining) and are
validated by steering effect and finetuning correlation (r = 0.76–0.97, 2507.21509).
2602.06801 shows steering cannot identify a direction. **So the deployed tool rests on a
check that is known not to work, and nobody has run the one that would.**

| run | question | needs |
|---|---|---|
| **P1** | Recovery at the plant layer. **Positive control — if this fails, stop** | K=1 |
| **P2** | Recovery vs computational depth. **The headline figure** | K=1 |
| P3 | Strength × N sweep — required-N for trait extraction | K=1 |
| P4 | Multitrait–multimethod matrix | **blocked by K≥2 (C14)** — restrict to one direction at a time, or cut |

P5 (AIPsy-Affect stimulus set) kills the "you designed the stimuli" objection for free.

### 7. Chapter three — the self-report level *(short)*
Not a competing empirical claim; the space is settled and not ours (2512.12411 shows
binary detection artifactual with an r=0.999 control; 2603.05414 content-agnostic;
2605.26242 at COLM 2026). We contribute only what they explicitly do not do: five knobs
moving the reported rate with the model untouched — readout method (50% → 7%), scoring
rule (6.7% → 33.3%, FPR 0% → 17%), extraction position, strength normalisation (138×
across models), prompt framing (8× on content-free vectors).

### 8. Limitations
Study 1 is one model family, layer 6, rank-1. Study 2 is one model, one plant layer.
Study 3 is one model, 30 concepts, and its significant cells sit where steering is only
weakly validated (C48). Extraction-position sensitivity is already documented (2602.00333).

### 9. Reproducibility
Every pre-registration with its filing date, every raw JSONL, the notebook including
three reversals and a failed repair.

---

## Priority queue

| item | cost | why |
|---|---|---|
| **Study 1, second model** (Pythia / GPT-2 medium) | CPU, ~1 day | "One model" is the obvious objection and Study 1 is the paper |
| **Study 2 P1** | 1 Kaggle session | Positive control. Gates the whole chapter |
| **Study 2 P2** | 1 Kaggle session | The headline figure of chapter two |
| A-14 window | 1 session, sheet written | Settles a chapter-three claim only |
| Matched-budget seed re-run | CPU ~4 h | Improves section 5 |

**Budget two sessions for P1 and treat the first as likely lost** — this week cost four
Kaggle sessions to loader and protocol bugs.

## Venue
Unchanged: interpretability workshop (BlackboxNLP, ICML MechInterp, NeurIPS ATTRIB),
arXiv first. Now a better fit — calibration and controls with two kinds of knowable truth.
