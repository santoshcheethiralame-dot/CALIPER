# Pivot Proposal — Planted Personas

**What it would take to make this the paper people forward · 2 September 2026**

---

## 0. Where the research landed

Six targeted checks over two days. Here is what is taken and what is open, so the proposal below
is built only on the open part.

| Angle | Status | Evidence |
|---|---|---|
| Efficient coding tested against a counted prior | **Taken** | Cacioli 2603.20642 / 2604.04469; Benjamin et al. Nat Comms 2022 |
| Refusal is not one direction | **Taken and already revised** | Cone geometry 2502.17420; multiple directions 2602.02132 |
| Construct validity applied to LLM evaluation | **Taken** | 12 papers in 2026, incl. an MTMM framework 2605.08522 |
| Internal directions fail validity checks | **Taken** | 2606.30449 — but *without* ground truth |
| Keyword-free stimulus batteries for mechanistic emotion work | **Taken and released** | AIPsy-Affect 2604.23719, MIT, 480 items |
| Direction-finding fails silently where truth is known | **Ours** | Phase 0, 77% [68–84%], n = 100 |
| **Injecting a known direction and testing whether persona/steering extraction recovers it** | **Open** | 40 papers searched, zero do it |
| **The same silent failure in neuroscience's own method, with a detector** | **Open** | Neuroscience knows MID errors are large on real V1 but cannot say which fits are wrong |

The two open rows are where this proposal lives. Everything else has been claimed by someone in
the last eighteen months, and I have been wrong about novelty four times this week by not checking
first. These two were checked.

---

## 1. The story in three parts

The strongest paper available is not a single result. It is one measurement discipline applied
in three directions, and each part makes the next one land harder.

### Part 1 — A standard method fails silently *(have it)*

A rank-one subspace estimator — the workhorse for finding "the direction for X" — applied where
the true direction is known exactly, recovers it for **77% of units [95% CI 68–84%]**. The other
23% fail with no warning: median alignment across all units is 0.993, restart agreement on the
failures is 0.85–0.99. A perfect solution exists for every failing unit and is provably reachable.
Disagreement between two estimation strategies flags the failures at p < 0.001, using no ground
truth at all.

*Phase 0. Measured, pre-registered, in the repository.*

### Part 2 — A high-profile safety tool depends on that method *(build it)*

**Persona vectors** (Anthropic, 2507.21509) are directions for character traits — sycophancy,
malice, hallucination-proneness — extracted by difference of means between trait-expressing and
trait-suppressing responses. They are used to monitor character drift, steer it, and filter
training data. Every validation in that literature is by *effect*: does steering along the vector
produce the trait? It does — but steering along **any** direction correlated with the trait will
produce the trait. Effect validation cannot distinguish the right direction from a correlated
wrong one.

Nobody has checked correctness, because no true persona direction exists to check against.

**We create one.** Using the injection machinery the team already built and validated in
APERTURE, plant a known direction `v` at layer L. The model's behaviour shifts. Run the
persona-vector extraction pipeline — the public code — on injected-versus-baseline generations.
Measure `|extracted · v|`. Ground truth by construction, and the extraction procedure is theirs,
not ours.

Then sweep what matters: extraction layer relative to injection layer, injection strength, sample
count, trait type. The recovery curve against *layers of computation between plant and read* is
exactly the depth-sweep idea from the original specification, applied to traits instead of
neurons.

### Part 3 — Neuroscience's own method shares the flaw, and never knew *(test it)*

Our estimator is, by the Williamson–Sahani–Pillow equivalence, **maximally informative
dimensions** — the standard neuroscience tool for receptive-field estimation under natural
stimuli, in use since 2004. Neuroscience already suspects it: re-analyses of V1 data report MID
errors "large, comparable to linear regression." But no one can say *which* fits are wrong,
because a brain has no ground truth.

Our Phase 0 finding is precisely a silent-failure result about this estimator, plus a detector
that needs no ground truth. If the failure transfers to the neuroscience regime — Poisson-spiking
model neurons driven by natural stimuli, the standard simulation — then we hand neuroscience a
tool for auditing twenty years of published receptive fields.

That is the reverse direction. Everyone imports neuroscience into AI. This exports a finding back.

---

## 2. Why this drops jaws, honestly assessed

**The sentence:**

> A method used to monitor the character of frontier AI systems, and the same method used to map
> receptive fields in the brain, returns a confidently wrong answer roughly a quarter of the
> time — and we can prove it only because we found a place where the right answer is known.

Three properties make it land, and one property limits it.

- **A belief overturned, at a named institution.** "Persona vectors work" is currently accepted
  because steering works. Showing steering-validated directions can be wrong directions is a
  specific claim about a tool in production use.
- **Genuine reciprocity.** Neuroscience → LLM (the method), psychology → mechanism (the trait
  construct), LLM → neuroscience (the audit). This is exactly the traffic the mentor responded
  to, and each leg is a measurement rather than an analogy.
- **Every branch produces a paper.** If persona extraction recovers planted directions at 0.99
  everywhere, that is the *first* correctness validation of a deployed safety tool — a positive
  result the field needs. If MID transfers cleanly to Poisson neurons, the flaw is estimator-
  intrinsic and the neuroscience claim strengthens; if it doesn't, we have characterised exactly
  which assumption protects biology, which is itself the reciprocal finding.

**The limit:** it is still a methods paper. It does not discover what a persona *is*. What it
does is establish, for the first time, whether the instruments people use to find personas are
pointing at the right thing — which is the question that has to be answered before any
persona-science can be trusted. That is a strong position but it should not be oversold as
psychology.

---

## 3. The experiments

Numbered to continue Phase 0. All use the calibrated estimator, the disagreement detector, and
the resumable batched pipeline already in the repository.

### P1 · Injection recovery at the plant layer
Plant `v` at layer L; extract at layer L by difference of means over N generations; measure
`|extracted · v|`. **This should be near-perfect** — it is a linear read of a linear plant — and
serves as the positive control that the pipeline works. If it fails here, stop.

### P2 · The depth curve
Extract at layers L+1 … L+k. The planted direction propagates nonlinearly; the extracted
"persona vector" at a downstream layer is its image, not the vector itself. **How fast does
recovery degrade with computational distance, and does the disagreement detector flag the
degraded extractions?** This is the headline figure of Part 2.

### P3 · Strength and sample-size sweep
Persona vectors are extracted from finite samples of generations. Sweep injection strength × N.
Gives a required-N table for trait extraction — the Part 2 analogue of E0.3.

### P4 · Multitrait–multimethod matrix
Plant several distinct directions. Extract each by difference of means, by a linear probe, by
our subspace estimator. Convergent validity: do methods agree on the same planted trait?
Discriminant validity: do they separate different planted traits? And — the column psychology
never gets — score every cell against the planted truth.

### P5 · AIPsy-Affect as the stimulus set
Use the released 480-item keyword-free battery for the trait-eliciting prompts rather than
designing our own. Kills the "you built the stimuli to get the result" objection for free and
puts a validated instrument at the front of the pipeline.

### P6 · Poisson transfer *(Part 3)*
Simulate linear–nonlinear–Poisson neurons driven by natural stimuli — the canonical neuroscience
model — with known filters. Run the same estimator and detector. **Does the 23% silent failure
appear, and does disagreement flag it?** This is the single experiment that decides whether Part
3 is a headline or a paragraph.

### P7 · Disagreement detector on real neural data *(Part 3, stretch)*
Public natural-stimulus recordings exist (Allen Brain Observatory, CRCNS). Run the detector; report
what fraction of units it flags. Cannot be validated — that is the point — but it is the first
audit of its kind and the number is reportable either way.

---

## 4. What carries over, and what changes

**Carries over unchanged:** the estimator, the cascade, the disagreement detector, the random-
direction null, the required-N result, the batched resumable pipeline, every test, and the entire
calibration discipline. Phase 0 is not discarded; it becomes Part 1 of the paper and the
validation layer under Parts 2 and 3.

**Changes:**

| | Before | After |
|---|---|---|
| Primary target | Dimensionality of MLP neurons | Correctness of persona/trait direction extraction |
| Ground truth | Free — neuron weights | Manufactured — injected direction |
| Model | GPT-2 small | Needs an instruct model with personas (~7B) → **Kaggle GPU required** |
| Reciprocity | Method import only | Import + export (Part 3) |
| Phase A (depth sweep on neurons) | Core | Demoted; the depth idea survives as P2 |

The neuron-dimensionality work is not lost — it is what the depth curve *is*, with a better-
motivated target. But the mentor should know it is being de-emphasised.

---

## 5. What would kill it, and what we would do

| Risk | If it happens | Response |
|---|---|---|
| P1 fails — extraction cannot recover a planted direction even at the plant layer | Pipeline or injection is broken | Stop, fix, do not proceed to P2 |
| P2 recovers at 0.99 at every depth | Persona vectors are validated | Publish that — a first, and a positive result the field needs; Part 2 becomes a validation rather than a critique |
| P6 shows no silent failure on Poisson neurons | The flaw is specific to LLM-style units | Part 3 becomes "here is the assumption that protects biology," which is the reciprocal finding in a different form |
| An instruct model shows no clean persona behaviour at ~7B | Signal too weak to extract | Fall back to the trait-like directions APERTURE already injects (concepts), which are documented to shift behaviour |
| Someone publishes injection-based validation first | Scooped on Part 2 | Part 1 and Part 3 stand alone; date-stamp Part 1 on arXiv now |

The one that worries me most is the second row, and it is worth saying plainly: **if Anthropic's
method turns out to be correct, that is a good result and we publish it as such.** The project is
not invested in persona vectors failing. It is invested in someone finally checking.

---

## 6. Compute and timeline

Part 2 needs an instruct model large enough to exhibit personas. That means Kaggle — the 30 h/week
on a single account covers it comfortably; the 12-hour cap is handled by the checkpointing already
built. Part 3's simulations are CPU work. Part 1 is done.

| | |
|---|---|
| **Sep 2026** | Close Phase 0 (additive K≥2 table running). Date-stamp Part 1 on arXiv. Port pipeline to Kaggle. |
| **Oct 2026** | P1, P2 — the two decisive experiments. Go/no-go on Part 2 by end of month. |
| **Nov–Dec 2026** | P3, P4, P5. P6 in parallel on CPU. |
| **Jan–Feb 2027** | P7 stretch. Write Paper 1 as the three-part story. |
| **Feb–Mar 2027** | Submit — ICML 2027 main or ACL 2027; fallback BlackboxNLP / ICML MechInterp. |

That is aggressive. The alternative — Paper 1 as Phase 0 alone — is safer and less interesting.
This proposal spends the risk on P2 and P6, both of which are decidable within a month.

---

## 7. What to say to the mentor

1. Phase 0 found something real: a standard method fails silently a quarter of the time, and we
   can detect it without knowing the answer.
2. That method is used, right now, to monitor the character of frontier models — and nobody has
   ever checked whether it returns the right direction, because nothing existed to check against.
   We can build the thing to check against.
3. The same method is neuroscience's standard tool for receptive fields, and neuroscience already
   suspects it but has no way to know which fits are wrong. Our detector may be that way.
4. So the project becomes the reciprocity she wanted, in both directions — but as measurement,
   not analogy.
5. Two experiments in October decide whether this is the paper or Phase 0 alone is.
6. We need Kaggle GPU time for one of them, and her view on whether the persona framing is one
   she wants the capstone to carry.
