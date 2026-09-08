# Proposal III — what is already built, and what remains

**For mentor review · 8 September 2026**
Department of Computer Science and Engineering, PES University

Team: Santosh Cheethirala (PES1UG24CS127) · Nivas Reddy Dandu (PES1UG24AM075) ·
C. Sree Krishna Koushik (PES1UG24CS128) · Marreddy Rushi Eswar Reddy (PES1UG24AM158)

---

## 1. Why this document exists

Proposal III was selected in August: measure the units of a language model the way
systems neuroscience measures the response properties of a neuron. The August research
report specified the instrument, named the estimators, and left one row of its own
capability table marked **Open**:

> Ground-truth validation of unit characterisation — **Open**

That row is now closed, on a real trained model, with an exact answer rather than a
simulated one. This document reports what was built between August and today, states
plainly what it does and does not establish, and lists what remains.

The work was done under the working title CALIPER. It is Proposal III's instrument
paper, not a separate project.

---

## 2. The measurement problem, stated in the proposal's own terms

A neuroscientist recording from a cell asks: *what feature of the stimulus does this
cell respond to?* The classical answers are reverse correlation (the spike-triggered
average), spike-triggered covariance, and — the method the August report identified as
the crown jewel — **Maximally Informative Dimensions** (Sharpee, Rust & Bialek, 2004).

The same question can be asked of a unit in a language model, and the report found that
essentially nobody has: five arXiv papers use MID, all in neuroscience, none on a modern
deep network.

The obstacle is the same one neuroscience has: **you cannot check the answer.** When an
estimator reports that a unit responds to some direction in the input space, there is
nothing to compare it against. Every characterisation in this literature is reported
without a correctness check.

---

## 3. What changed: the answer is available for free

For one class of unit, the answer is not hidden. **The input to a multi-layer-perceptron
unit is exactly the dot product of the post-normalisation residual vector with that
unit's input weight column.** The direction the unit responds to is not something to be
estimated — it is a column of the weight matrix, and it can simply be read off.

Verified numerically:

| model | check | result |
|---|---|---|
| GPT-2 small | reconstruction residual | 3.3e-06 |
| Pythia-160m | correlation with the true pre-activation | 1.0000000000 |

This gives an exact target direction, per unit, on a real trained model, at no compute
cost, in unlimited supply (3,072 units per layer, twelve layers, two model families
already wired).

**This is a stronger asset than the plan the August report proposed.** Phase D there
suggested calibrating against InterpBench — circuits compiled to have a known answer.
That is sound, but a compiled circuit is not a feature a model learned. The weight column
is exact *and* natural: the units are hard for reasons the model chose during training,
not reasons we introduced.

**The honest limit, stated first rather than buried.** This holds for a unit reading its
own layer. It does not extend to residual-stream features, sparse-autoencoder latents,
or behavioural directions. The instrument is calibrated where calibration is possible;
whether those calibrations transfer elsewhere is an assumption, and the work cannot test
it. This belongs in the abstract of any paper, not the appendix.

---

## 4. Result 1 — the neuroscience estimators, scored against the truth

The August report predicted spike-triggered covariance would fail on language-model
activations, and proposed using it as "the baseline that fails". That prediction was
tested directly. Thirty units, GPT-2 small, layer 6; a unit counts as recovered if the
estimated direction aligns with the true weight column above 0.95.

| estimator | recovered | median alignment |
|---|---|---|
| spike-triggered average | 0 / 30 | 0.2866 |
| decorrelated (whitened) STA | 1 / 30 | 0.5280 |
| spike-triggered covariance | 0 / 30 | 0.3221 |
| **rank-K bottleneck (MID)** | **26 / 30** | **0.9950** |

The prediction holds, and the mechanism is identifiable rather than merely observed. The
classical estimators rest on Bussgang's theorem, which requires an approximately Gaussian
stimulus and a monotone nonlinearity. A language model's residual stream is strongly
non-Gaussian, and the GELU activation is non-monotone over the range the units actually
occupy. Both conditions fail, driving Bussgang's constant toward zero — which is exactly
the regime MID was invented to handle.

This is a self-contained methods result: the neuroscience toolkit transfers to language
models, but only the modern member of it, and the reason is a property of the
architecture that can be stated precisely.

---

## 5. Result 2 — the instrument fails silently, and the failure rate is model-dependent

Scaling the working estimator to 100 randomly chosen units per model, with the pass bar
fixed in advance at alignment > 0.95:

| model | recovered | Wilson 95% interval | median alignment | worst unit |
|---|---|---|---|---|
| GPT-2 small | 77 / 100 | [0.679, 0.842] | 0.9933 | 0.1445 |
| Pythia-160m | 93 / 100 | [0.863, 0.966] | 0.9994 | 0.0761 |

Three things follow, and the third is the one that matters.

1. **The instrument is not reliable at the unit level.** Both models fail a
   pre-registered 0.90 bar. This was reported as a failure against our own filed
   criterion rather than rewritten afterwards.
2. **The rate is a property of the model, not a constant.** The intervals do not
   overlap, so the honest statement is a range — roughly 7% to 23% — not a single
   headline number.
3. **Aggregate quality conceals it, and conceals it *better* as the model gets
   cleaner.** Pythia's median alignment is 0.9994 — excellent by any summary statistic
   anyone reports — while that same population contains a unit recovered at 0.0761. A
   population-level analysis built on these estimates would inherit the errors invisibly.

Point 3 is why this is the necessary first phase of Proposal III rather than a detour
from it. Proposal III's later phases fit tuning curves and pairwise maximum-entropy
models **across a population of units**. In systems neuroscience, the error
characteristics of the single-unit estimate are established before population structure
is inferred from it, for exactly this reason. Doing it in the other order means
population structure and estimation error are not separable.

---

## 6. Result 3 — which reliability check actually works

If the instrument fails silently on some units, the practical question is whether a
practitioner without ground truth can tell which ones. Several candidate signals are in
routine use across interpretability. Each was scored as a detector against the known
answer:

| signal | GPT-2 (AUC) | Pythia (AUC) | cost |
|---|---|---|---|
| held-out R², negated | 0.906 | 0.995 | free — the fit already computes it |
| two-route disagreement | 0.802 | 0.975 | one extra fit |
| rank-2 improvement | 0.367 | — | one extra fit, and it does not work |
| **restart agreement** | **measurement in progress** | queued | free |

Two findings stand already. The signal that works best is the cheapest one, and it is
not the one we originally expected. And a signal that sounds principled — checking
whether a second dimension improves the fit — is close to useless as a detector at 0.367.

**Restart agreement is the notable gap.** Re-running an optimisation from several random
starts and checking that the answers agree is the most widely used sanity check in this
area. It has, as far as our literature search found, never been scored against a known
correct answer, because doing so requires the ground truth whose absence motivates the
check. An eight-unit pilot already on disk suggests why this is worth measuring
carefully: correctly-recovered units agree across restarts to within 0.0003, but **two of
four failing units also agree to within 0.05 while being wrong.** The check appears to
catch severe failures and miss subtle ones. A 100-unit measurement is running now and
will settle it.

The criteria were filed before that run started, including the outcome in which restart
agreement wins and our current recommendation has to be amended.

---

## 7. What this adds up to

A calibration bench for unit-characterisation methods, on real models, with exact ground
truth. It reports, for any estimator and any reliability signal: recovery rate with a
confidence interval, the signal's operating characteristic, the data required, and the
random-direction floor.

Positioned against existing work: evaluation harnesses in this area rank methods against
each other or against downstream behaviour; ground truth for direction recovery is
otherwise always planted or compiled; and reliability signals are validated against
reconstruction quality or behavioural outcome, never against a known-correct direction.
The combination — exact ground truth on a real model, used to calibrate the checks
themselves — is the contribution.

---

## 8. What remains

**Near term, no GPU required, this month.**

- Finish the restart-agreement measurement on both models (running).
- Test whether the signals combine — do three weak detectors beat the best single one?
- Establish how much data each signal needs before its operating characteristic stabilises.
- Ship the bench as a script that produces the report page for a supplied estimator.

**The bridge to the rest of Proposal III.**

- Extend from single-unit directions to **tuning curves** over a controlled stimulus
  battery — the proposal's descriptive layer, now with a calibrated instrument beneath it.
- Then the **pairwise maximum-entropy** population layer, which the August research pass
  verified has no existing language-model application.

**Open questions we cannot yet answer.**

- Does the calibration transfer to units that are not reading their own layer? Unknown,
  and not testable with this ground truth.
- Joint estimation of two or more dimensions currently collapses to one dimension at every
  sample size. This must be fixed before any multi-dimensional tuning analysis.

---

## 9. Two things we would like guidance on

1. **Authorship and division of work across four people.** The instrument work is
   currently concentrated; the tuning-curve battery and the population layer split
   cleanly across the team, but the split should be agreed before it is built rather than
   after.
2. **Whether to date-stamp the earlier introspection result.** The previous phase
   produced a control experiment that a full-text check confirms is not reported in any
   of the three current papers in that debate. It requires no further compute. It is
   preliminary work for a proposal that was not selected, so the question is whether it
   is worth a short preprint now or should simply be set aside.
