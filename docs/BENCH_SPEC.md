# CALIPER bench — specification v1

**Filed 8 September 2026.** What we are building, why it is not already built, and
the runs that would ship it.

---

## 1. The one-sentence claim

Interpretability runs on reliability heuristics — restart agreement, held-out fit,
method disagreement, representational stability — that decide whether a result is
trusted. **None of them has been calibrated against a known-correct answer**, because
calibrating one requires exactly the ground truth whose absence motivated it. We have
found the one place that ground truth is free and exact, and we calibrate them there.

The deliverable is not a benchmark of methods. It is a **calibration bench for the
checks**. Given a direction-finding method and a proposed reliability signal, it
returns that signal's operating characteristic against ground truth.

The project is named after a measuring instrument. The deliverable should be one.

---

## 2. Why this is open — the scout, 8 September 2026

Three literatures converge on the same hole from three sides.

### 2a. Reliability signals exist and are validated against proxies, never ground truth

| work | signal | validated against |
|---|---|---|
| [Geometric Canary (2604.17698)](https://arxiv.org/html/2604.17698v2) | representational stability (Shesha), unsupervised | **behavioural** — accuracy drop under steering, task degradation under drift. AUC 0.990 on LoRA drift. 69 embedding models |
| [Unstable Features, Reproducible Subspaces (2606.12138)](https://arxiv.org/abs/2606.12138) | cross-seed feature stability | **reconstruction and downstream prediction** |
| [Building Fast, Evaluating Slow (2607.19386)](https://arxiv.org/html/2607.19386) | — | finds methodological variance exceeds architectural variance across autointerp metrics |
| [Analysing Generalisation and Reliability of Steering Vectors (2407.12404)](https://arxiv.org/html/2407.12404v1), [(Un)Reliability (2505.22637)](https://arxiv.org/abs/2505.22637) | steerability spread | **behavioural** — steering effect size, which is often negative |

Every one asks *does the signal track behaviour or reconstruction*. None asks
*does the signal track whether the recovered direction is the right direction*.
That question needs a right answer to exist.

### 2b. Ground truth for direction recovery is always synthetic

Feature Recovery Rate, the Linear Representation Bench, InterpBench, Tracr,
CLEVR-XAI — all **plant or compile** the answer, then measure recovery. That is
sound and it is also a different object: a planted feature in a toy model is not a
feature a real model learned.

Weight-space work — [bilinear MLPs (2410.08417)](https://arxiv.org/pdf/2410.08417),
ROTATE, SVD-of-weights — treats weights as an *analysis target*. Using the weight
column as a **scoring key for an activation-based estimator** does not appear in
this literature.

### 2c. Evaluation harnesses do not carry a ground-truth-free detector

[ObserverBench (2609.03026)](https://arxiv.org/abs/2609.03026) scores observers by
downstream action quality on held-out cases and states plainly that it has no
ground-truth-free failure detector. [MIB](https://benchmarking-interpretability.csail.mit.edu/)
and AxBench rank methods head to head.

### 2d. Consequence

The gap is a rectangle with three closed sides. Signals without ground truth;
ground truth without realism; harnesses without signals. **Free exact ground truth on
a real model, used to calibrate the signals, closes it.**

---

## 3. The substrate — why the ground truth is free

An MLP neuron's pre-activation is exactly `s . w`, where `s` is the post-layernorm
residual read at that layer and `w` the neuron's input weight column. The direction
is not estimated, planted, or simulated. It is read off the weight matrix.

Verified: residual 3.3e-06 on GPT-2 small; correlation 1.0000000000 on Pythia-160m.

This gives, for free, on a real trained model:

- an exact target direction per unit,
- an unlimited supply of units (3,072 per layer, 12 layers, two families already
  wired),
- a natural difficulty spread — the hard units are hard for reasons the model chose,
  not reasons we injected.

**The honest limit, stated up front.** It holds for a unit reading its own layer.
It does not extend to residual-stream features, SAE latents, or persona directions.
The bench measures signals on the one substrate where measurement is possible, and
the transfer of those calibrations elsewhere is an assumption the bench cannot test.
That limitation goes in the abstract, not the appendix.

---

## 4. What the bench is, concretely

Three registries and one report.

### 4.1 Method adapter

A method maps `(stimulus, response) -> direction`. Already implemented and scored
(E0.5, n=30, GPT-2 layer 6, pass = alignment > 0.95):

| method | passing | median alignment |
|---|---|---|
| spike-triggered average | 0/30 | 0.2866 |
| decorrelated STA | 1/30 | 0.5280 |
| STC, top eigenvector | 0/30 | 0.3221 |
| **fitted rank-1 bottleneck** | **26/30** | **0.9950** |

Four methods, one ground truth, already run. The bench's method half exists.

### 4.2 Signal registry

A signal maps whatever the method exposes to one scalar, computed **without the
ground truth**. Already computed at n=100 (GPT-2) and n=100 (Pythia-160m):

| signal | GPT-2 AUC | Pythia AUC | cost |
|---|---|---|---|
| held-out R2 (negated) | 0.906 | 0.995 | free, the fit computes it |
| two-route disagreement | 0.802 | 0.975 | one extra fit |
| k2 gain | 0.367 | — | one extra fit, and it does not work |
| **restart agreement** | **not written to disk** | **not written to disk** | free, already computed |

Restart agreement is the field's default check and it is the one number the gate
run computes and discards. `caliper/estimator.py:29` exposes it as
`Fit.stability`; `caliper/batched.py:143` populates it on every fit;
`experiments/e01_gate.py` never records it.

### 4.3 Substrate registry

GPT-2 small (77/100 pass) and Pythia-160m (93/100 pass), same protocol, same bar.
The failure rate is model-dependent — 7% to 23% — and the intervals do not overlap,
so the bench reports a range, never a single headline rate.

### 4.4 The report — a spec sheet

One page per (method, signal, substrate):

```
method     fitted rank-1 bottleneck, 1600 steps, 2 restarts, 8000 tokens
substrate  GPT-2 small, layer 6, 100 units drawn at random
recovery   77/100   Wilson 95% [0.679, 0.842]     bar: alignment > 0.95
floor      random-direction null, R2 <= 0.044
required-N 8000 tokens; the binding quantity is events, not positions

signal              AUC     TPR@FPR=0.10    discarded-good@73%-catch
held-out R2        0.906         ...                5
disagreement       0.802         ...               21
restart agreement    ???         ...              ???
```

The right-hand columns are the point. An AUC is a summary; a practitioner needs to
know what a threshold costs them.

---

## 5. What is genuinely new, stated so a reviewer can attack it

1. **The weight column as a scoring key.** Exact ground truth for direction finding
   on a real model, no planting. Not found in the scouted literature.
2. **A calibrated ground-truth-free reliability signal.** The idea of using
   self-consistency is old and patented; a *measured operating characteristic*
   against a known-correct direction is not in the scouted literature.
3. **The negative that follows.** Aggregate quality hides the failures: Pythia's
   median alignment is 0.9994 while containing a unit recovered at 0.0761. Fewer
   failures, more severe, median further from the truth.

Not new, and cited as such: that reliability matters; that seeds are unstable; that
disagreement correlates with error; that steering vectors are non-identifiable.

---

## 6. Risks

**Novelty rests on one asset.** If a reviewer says "MLP neurons reading their own
layer is a special case", the answer is that it is the only non-special case anyone
has — and it is still narrow. Mitigation: state it in the abstract; add a second
family (done); add a third if cheap.

**A signal may win that we did not want to win.** If restart agreement matches
held-out R2, the finding becomes a cost-benefit result rather than a correction.
That is still publishable and it is pre-registered as an outcome, not a failure.

**Proximity to ObserverBench.** Different question — directions, not actions — and
it has no ground-truth-free detector. Position against it explicitly in related
work rather than hoping nobody notices.

---

## 7. What ships in v1

Nothing here requires a GPU or a new idea. In order:

- B-1 restart-agreement calibration, GPT-2, n=100 — the headline
- B-2 the same on Pythia-160m — transfer
- B-3 signal combination, offline
- B-4 required-N per signal, offline
- B-5 spec-sheet generator, one script

Detail, criteria and failure branches: lab notebook section 7.8.
Pre-registration for B-1/B-2: `preregistration-b1-stability-calibration.md`.
