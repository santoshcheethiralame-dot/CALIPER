# CALIPER — Complete Project Reference

## Everything, for the first review

**Department of Computer Science and Engineering, PES University**

Santosh Cheethirala (PES1UG24CS127) · Nivas Reddy Dandu (PES1UG24AM075)
C. Sree Krishna Koushik (PES1UG24CS128) · Marreddy Rushi Eswar Reddy (PES1UG24AM158)

*Version of September 2026. Read Part I and Part IX; the rest is reference.*

---

# PART I — WHAT THIS PROJECT IS

## 1. The problem, in one paragraph

Modern language models are made of numbers, and nobody knows what most of those numbers mean.
The field that tries to find out is called **interpretability**. Almost every method it uses
works the same way: it extracts a **direction** — a particular combination of the model's
internal numbers that supposedly stands for some concept, like "this text is about France" or
"the model is being sycophantic."

Here is the problem. **Nobody checks whether the direction that comes out is the right one.**
There is no way to check, because in a real model nobody knows the true answer to compare
against. The field has been extracting directions and trusting them for years, with no
measurement of how often the procedure is correct.

## 2. What we did about it

We found a place inside a real language model where the true answer *is* known, exactly and for
free. We used it to measure how often direction-finding actually works.

**It works 77 times out of 100.** The other 23 times it produces a confidently wrong answer,
with no warning that anything went wrong.

We then established why, built a partial fix, and — most usefully — found a way to detect the
failures *without* knowing the answer, which means the finding is usable in the ordinary case
where no ground truth exists.

## 3. Why it matters

Interpretability results are now used in real safety reviews at major AI companies. Directions
extracted this way are used to monitor models, to steer their behaviour, and to decide whether
a model is safe to release. If the extraction procedure is silently wrong a quarter of the
time, that matters — and until now there was no way to know.

The comparison that lands: in 2026, researchers showed that sparse autoencoders — the field's
most popular interpretability tool — barely beat a random baseline. That result was influential
precisely because it was a **negative result about a tool people were already deploying**. Ours
is the same shape.

---

# PART II — HOW WE GOT HERE

Worth knowing because the review will ask, and because the path shows judgement.

## 4. The proposal stage

In August 2026 we submitted seven capstone proposals. The mentor selected two:

- **Proposal III** — measuring language model units the way neuroscientists measure brain
  cells
- **Proposal V** — estimating how many model vulnerabilities remain undiscovered

We researched both in depth and produced a full report. Two of our own novelty claims turned
out to be wrong and we corrected them: tuning curves *have* been used in vision models, and
capture–recapture *has* thirty years of use in software engineering. Correcting them made both
proposals stronger, because inheriting a mature method is a better position than claiming
virgin ground.

The mentor preferred Proposal III.

## 5. The audit, and the first pivot

We wrote an ambitious version of Proposal III built around testing the **efficient coding
hypothesis** — a sixty-year-old theory in neuroscience — in language models, on the grounds
that a model's "environment" is its training corpus and can therefore be counted exactly,
which is impossible for any animal.

We then ran a six-lens adversarial audit of our own proposal. It found:

- The framing was **already published** (arXiv:2603.20642, which has "Efficient Coding" in its
  title and counts over six million integer mentions from a corpus).
- The obvious objection — *"a model trained to minimise loss under a distribution will allocate
  capacity by that distribution automatically"* — is not a hunch but a **published theorem**
  (Benjamin et al., *Nature Communications*, 2022).
- A companion result showed number tuning curves appearing in **randomly initialised** networks
  (Kim & Paik, *Science Advances*, 2021), so the effect does not even require training.

Consensus estimate: 8–15% chance of a strong publication as written.

**We killed the idea.** This is the single best decision in the project so far. Two days of
work saved a year of building on a claim that would not have survived review.

## 6. The second framing, which is the current one

Instead of chasing one theory, we asked what neuroscience actually has that interpretability
lacks. The answer is not a technique — it is a **discipline**:

> Before you trust an instrument, test it on something where you already know the answer.

Neuroscience built that habit over forty years because brain recording is expensive and
error-prone. Interpretability never did, because it grew out of machine learning, where you
evaluate on benchmarks rather than calibrate instruments.

That single imported habit is what produced everything in this report.

---

# PART III — BACKGROUND YOU NEED

## 7. How a language model works, minimally

A language model predicts the next chunk of text. Text is split into **tokens** (words or
word-parts). Each token becomes a list of numbers. Those numbers pass through a stack of
**layers** — GPT-2 has 12 — and each layer reads the numbers, computes something, and writes
its result back.

The shared channel that carries numbers between layers is the **residual stream**. In GPT-2 it
carries **768 numbers** at every position in the text.

Inside each layer there is a section containing **3,072 neurons**. Each neuron takes all 768
numbers, multiplies each by a fixed number stored in the model (its **weights**), adds them up,
and applies a curve called **GELU** to produce a single output.

```
neuron output  =  GELU( w₁·x₁ + w₂·x₂ + ... + w₇₆₈·x₇₆₈ )
```

The weights `w` never change after training. **You can read them straight out of the model
file.** This is the key to everything.

## 8. What interpretability is trying to do

Given a concept — "France", "sycophancy", "refusal" — find the direction in those 768 numbers
that represents it. The main approaches:

- **Probes** — train a simple classifier on the internals to detect the concept.
- **Sparse autoencoders (SAEs)** — decompose a layer's activity into many interpretable pieces.
- **Steering vectors** — a direction you add to the internals to change behaviour.
- **Persona vectors** — steering vectors for personality traits (Anthropic, 2025), extracted by
  taking the difference in internal activity between responses showing the trait and responses
  not showing it.

All four produce a direction. None can be checked for correctness.

## 9. The neuroscience connection

Neuroscientists face the identical problem: they record a brain cell firing and want to know
what it responds to. They developed tools for exactly this — the **spike-triggered average**,
**spike-triggered covariance**, and later **maximally informative dimensions** (Sharpee, Rust &
Bialek, 2004).

Crucially, **none of those tools has ever been validated**, because no one knows what a real
neuron truly encodes. Neuroscience has flown on assumption for forty years, not from
carelessness but from necessity.

A language model is the first system that is complex enough to be interesting and transparent
enough to check.

## 10. Glossary

**Activation** — the number a neuron produces for a given input. Changes with every input.

**Alignment** — how close two directions are. 1.000 identical, 0.000 unrelated. Our main score.
Above 0.95 counts as correct.

**Batching** — fitting many neurons at once instead of one at a time. Gives us 4.5× speed.

**Cascade** — our fix: fit two directions first, then search inside that result for the best
single one.

**Confidence interval** — the range the true value plausibly lies in, given we tested a sample.

**Construct validity** — a psychology term: does your measurement measure what you claim? A
century of theory exists because human traits cannot be observed directly.

**Direction (vector)** — a list of 768 numbers saying how much each internal number matters.

**Dimension** — one independent thing a unit responds to.

**GELU** — the response curve GPT-2 neurons apply. **Not simply increasing**: for negative
inputs it dips down then comes back up, so two different inputs can give the same output.

**Ground truth** — the correct answer, known independently.

**Held-out data** — data set aside and not used for fitting, so you can check honestly.

**LayerNorm** — a rescaling applied before a layer reads the stream.

**Median** — the middle value. We use it instead of the average so extremes don't distort it.

**Null model / random baseline** — what pure guessing achieves. Needed to know if a result
means anything.

**Optimiser / gradient descent** — the search that hunts for the best direction, improving step
by step. Can get stuck. This was our whole problem.

**p-value** — the chance of seeing a pattern this strong by luck. p = 0.0008 is about one in a
thousand.

**Pre-registration** — writing down what counts as success *before* running the experiment, so
you cannot move the goalposts afterwards.

**Probe** — a simple classifier trained on internals. Produces a direction.

**R²** — how much of a neuron's behaviour a direction explains. 1.000 perfect, 0.000 nothing.

**Residual stream** — the shared 768-number channel flowing through the model.

**Restart** — running the search again from a different start. Standard practice assumes
agreeing restarts mean a right answer. **We showed that assumption is false.**

**Sparse autoencoder (SAE)** — popular tool that breaks a layer's activity into pieces.

**Spike-triggered average / covariance** — classical neuroscience direction-finding, ~40 years
old.

**Steering vector** — a direction added to internals to change behaviour.

**Subspace** — a set of directions considered together. A 2-dimensional subspace is a plane.

**Token** — a chunk of text, usually a word or word-part.

**Weights** — the fixed numbers stored in the model saying how a neuron mixes its inputs.

---

# PART IV — THE SCIENCE

## 11. The ground truth

A neuron's output is `GELU(weights · inputs)`. So with respect to **its own layer's input**,
that neuron depends on exactly one direction, and that direction is **its own weights**, which
sit in the model file.

We verified this rather than assuming it:

| check | result |
|---|---|
| recomputed value vs the model's actual value | agrees to 3.3 parts in a million |
| correlation | 1.00000000 |
| is the response a function of the projection alone? | yes, confirmed |

**This is not a discovery about models — it is true by construction. It is a test bed.** The
contribution is noticing it can be used as one.

> **Important limitation, state it before she does:** this gives ground truth for a *neuron's
> input direction*. It gives no ground truth for "the sycophancy direction," because no such
> truth exists in the model. That matters for the personality strand (Part VII).

## 12. The estimator

The method searches for the direction `V` and response curve `f` that best explain a unit:

```
minimise over V and f:    Σ ( actual response − f(V · input) )²
```

This is **maximally informative dimensions**, the neuroscience method, in the form proved
equivalent to it by Williamson, Sahani & Pillow (2015). Its first application to a language
model.

Design decisions, each with a reason:

- **All directions fitted together, never one at a time.** Greedy one-at-a-time search is
  biased when inputs are correlated — published benchmarks show it costing 0.875 → 0.60.
- **No compressing the data first.** See correction 1 below.
- **Held-out data for every score**, so we cannot fool ourselves.
- **Multiple restarts**, reported — though we then showed restart agreement is not
  trustworthy.

---

# PART V — EVERY EXPERIMENT AND RESULT

## 13. E0.1 — the main test

**What we did.** Picked 100 neurons at random from GPT-2 layer 6. Fed the model ordinary
English text and recorded what went into the layer and what each neuron did. Hid the weights.
Asked the method to work out the direction. Compared against the weights.

**The criterion was written down and filed before running:** the run passes if the lower bound
of the 95% confidence interval on the success rate exceeds 90%.

**Result:**

| | |
|---|---|
| neurons tested | 100 |
| found correct | **77** |
| 95% confidence interval | **68% to 84%** |
| median score across all 100 | **0.993** |
| worst score | 0.14 |
| restart agreement on failures | 0.85 to 0.99 |
| **verdict** | **FAIL** (needed 90%) |

**The finding is the gap between 0.993 and 77%.** The summary statistic looks excellent while a
quarter of individual answers are wrong. And the standard internal check — do repeated runs
agree? — says yes on the wrong answers.

## 14. E0.1b — is it just under-trained?

Re-ran the failures with **twenty times** more computation. Only **1 of 5** improved. So it is
not a matter of trying harder.

## 15. E0.1e — is it the response curve?

Held everything fixed and changed only GELU. Result: **ReLU, which is simply increasing, did
worse than GELU**. So the dip in GELU is not the cause. What matters is how much of the
response range survives where the neuron actually operates.

The revealing detail: with a plain straight-line response — where finding the direction is
ordinary regression and should be trivial — the method still scored **0.970, not 1.000**. That
pointed at the search itself.

## 16. E0.1g — does a right answer exist?

Forced the method to use the true direction and measured how well it explained the neuron.

| | |
|---|---|
| R² at the true direction | **1.000** |
| R² the search actually reached | 0.809 |
| how often the 2-direction fit contained the true direction | **99.7%** |

**A perfect answer exists, is reachable, and the 2-direction version already contains it.** The
method finds the answer and loses it. That is a search problem — not the data, not the neurons,
not the model.

## 17. E0.1h and E0.1i — the fix and the selection rule

**The fix (cascade):** ask for two directions, then search inside that result for the best
single one. Neuron 2977 went **0.45 → 0.996**.

It does not help everywhere, so we run both methods and keep whichever scores better on
held-out data.

| | above 0.95 |
|---|---|
| original method | 5 of 8 |
| cascade | 6 of 8 |
| **run both, keep better** | **7 of 8** |
| a perfect chooser would get | 8 of 8 |

Selecting by held-out score is near-optimal — the median regret against a perfect chooser is
**exactly zero**. But it is not perfect: on one neuron it confidently preferred the *less*
accurate direction, because a flexible response curve can compensate for a slightly wrong
direction. **Fit quality and direction accuracy come apart.**

## 18. The detector — the most useful result

| | success rate |
|---|---|
| the two methods agree (69% of neurons) | **87%** |
| the two methods disagree (31% of neurons) | **55%** |

p = 0.0008.

**This needs no ground truth.** It works in the ordinary case where nobody knows the answer. So
every future measurement can carry a confidence label — which is exactly what a measurement
discipline is supposed to provide.

## 19. E0.2 — what does "we found something" mean?

Tried **2,000 random directions** on each of 12 neurons.

| | |
|---|---|
| a random direction explains at most | **4.4%** of a neuron's behaviour |
| the true direction explains | 70% to 98% |
| best alignment from 24,000 random draws | 0.144 |

**Every future claim now has a number to beat.** The field does not currently have one.

This also overturned a warning we had built two mandatory steps around: the literature says
that with correlated inputs, random directions look deceptively good. **We measured it. It does
not apply here.**

## 20. E0.3a — are the failures an information limit?

Built artificial neurons on the *real* internals, with directions we chose, and varied how
often they fire.

| informative events ÷ 768 | 0.96 | 2.59 | 7.45 | 18.2 | 24.8 |
|---|---|---|---|---|---|
| best achievable score | 0.996 | 0.978 | 0.996 | 0.997 | 0.998 |

Flat at 0.98–1.00 everywhere. **So a neuron failing at 0.46 has real room to improve — it is
not starved of data.**

## 21. E0.3 — how much data is needed

| text | informative events | worst-case score |
|---|---|---|
| 1,000 tokens | 100 | 0.71 |
| **2,000 tokens** | **200** | **0.99** |
| 4,000 tokens | 400 | 0.99 |
| 16,000 tokens | 1,600 | 0.99 |

**200 informative events is enough; more adds nothing.** Genuinely surprising — the space has
768 dimensions and we expected to need more examples than dimensions, not four times fewer.

*Note: the two- and three-direction figures are being re-run. Our first attempt used a setup
that made those cases artificially hard, so we are not reporting them.*

## 22. E0.5 — do the classical neuroscience methods work here?

Tested the forty-year-old tools on 30 neurons where the answer was known.

| method | got it right |
|---|---|
| spike-triggered average | **0 of 30** |
| decorrelated spike-triggered average | 1 of 30 |
| spike-triggered covariance | **0 of 30** |
| **our method** | **26 of 30** |

**Why they fail — two assumptions break at once:**

1. They assume the input data is bell-shaped. Model internals are not.
2. They assume a neuron's response always rises as its input rises. It does not: these neurons
   sit **below zero 90–99% of the time**, and **past GELU's turning point 55–90% of the time**,
   where the response actually *falls*.

**This is our most neuroscience-facing result.** These tools have never been checkable in a
brain. We checked them where the answer is knowable, and explained the failure rather than just
reporting it.

## 23. All results in one table

| # | Experiment | Question | Answer |
|---|---|---|---|
| E0.1 | Analytic control | Does it find a known direction? | 77/100, [68–84%], silently |
| E0.1b | Budget | Is it under-trained? | No — 1 of 5 improved with 20× |
| E0.1e | Response curve | Is GELU to blame? | No — ReLU is worse |
| E0.1g | Reachability | Does a right answer exist? | Yes — R² 1.000, in the 2-D fit 99.7% |
| E0.1h/i | Fix + selection | Can we repair it? | Partly; 5/8 → 7/8 |
| — | Detector | Can we spot failures blind? | Yes — 87% vs 55%, p=0.0008 |
| E0.2 | Random baseline | What is chance? | 4.4%; threshold established |
| E0.3a | Ceiling | Is it a data limit? | No — ceiling flat at 0.98+ |
| E0.3 | Sample size | How much data? | 200 events; more adds nothing |
| E0.5 | Classical methods | Do the old tools transfer? | No — 1 of 30 |

---

# PART VI — WHAT WE GOT WRONG

This section is deliberate. Report it rather than hide it: it is the strongest evidence that
the calibration-first design works.

## 24. Corrections forced by measurement

Six decisions in our own plan were overturned by data, not argument.

1. **We planned to compress the data before analysing.** Measured: this destroys the signal. A
   neuron's true direction keeps only 0.352 of itself in the compressed space, against 0.267
   for a *random* direction — barely above chance. Removed.
2. **A related preprocessing step** proved numerically unstable on real data. Removed.
3. **The literature's warning about random directions does not apply here.** We had built two
   mandatory steps around it.
4. **Data should be counted in informative events, not text positions** — about ten times
   fewer.
5. **Our pass criterion could not work on 20 neurons.** One failure is 95%, two is 90% — the
   threshold was finer than the measurement. Restated and tested on 100.
6. **A proposed improvement made things worse.** Reshaping the response looked decisive on
   three test neurons; run on 20 it *halved* the success rate, from 75% to 50%. Reverted.

## 25. Mistakes we made during the work

- **Two conclusions drawn from a badly chosen quality measure.** A shortcut scoring method
  turned out to be unreliable for neurons with extreme values, and produced two apparent
  findings that were artefacts. Both retracted.
- **A speed estimate off by a factor of eight.** We predicted 20–40× from a code change and
  measured 4.5×.
- **A wrong diagnosis, corrected by better statistics.** We blamed a configuration change for a
  drop in success rate. Proper confidence intervals showed the two configurations were
  statistically indistinguishable — we had read noise as signal on a small sample.
- **Four novelty claims that did not survive a full literature read.** Each time a plausible gap
  survived a keyword search and died on reading the neighbouring papers.

**Every one was caught before it reached a conclusion**, because the ground-truth check exists.
That is the argument for the whole approach, demonstrated rather than asserted.

---

# PART VII — WHERE THIS GOES

## 26. The immediate next question

Does the same silent failure affect the methods the field uses daily — probes, SAE directions,
steering vectors? They all solve the same kind of problem. None has been checked.

## 27. The personality strand

Research on model personality has recently become mechanistic. Rather than giving a model a
questionnaire, researchers extract a **direction** for a trait — "sycophantic", "evil", "prone
to making things up" — from the difference in internal activity between responses that show the
trait and responses that do not. Anthropic's **persona vectors** (July 2025) is the prominent
example, now used to monitor and steer model character.

**That extraction is the same kind of procedure we just showed fails silently.** And it is
validated by checking that steering along the direction produces the trait — which is close to
circular, since the direction came from trait-expressing responses.

Recent work (arXiv:2606.30449) tested several such methods and found they fail generalisation
and specificity checks. **It could not say how wrong they are**, because there is no true
direction to compare against.

**What we could do:** we cannot use neuron weights as ground truth for a personality trait — no
such truth exists. But we can **plant** one: train a small model with a trait direction we
choose ourselves, then run the field's own extraction methods against a known answer.

*Honest status: this requires building trait ground truth first, which is real work, not a
reframing. It is a question for the mentor, not a decision made.*

## 28. Full timeline

| Period | Work | Milestone |
|---|---|---|
| **Sep–Oct 2026** | Finish data-requirement measurements; speed up the code; assign team tracks | Phase 0 closes |
| **Oct–Dec 2026** | Write up Phase 0 | **arXiv preprint, December** |
| **Jan–Mar 2027** | Submit to conference or workshop | First submission |
| **Nov 2026–Jun 2027** | **Phase A** — move to earlier layers where the answer is unknown. Central question: how many independent things does one unit respond to? | Main measurement |
| **Feb 2027** | **Hard gate:** removing a discovered direction must actually change the model's behaviour. If not, the numbers are curve-fitting and the phase stops. | Go/no-go |
| **Jul–Nov 2027** | Placement season — reduced availability, writing rather than experiments | **Main paper deadline, September** |
| **Jan–May 2028** | Revisions, final experiments, thesis, defence | Graduation |

## 29. Publication plan

**Paper 1 — Phase 0.** Six results, all measured, all negative or cautionary, all requiring the
answer to be known in advance. Preprint December 2026; venue early 2027.

**Paper 2 — Phase A.** The main measurement. Target September 2027 for a 2028 conference.

Paper 1 also strengthens Paper 2: the first thing any reviewer asks is whether the instrument
was checked, and we will be able to point at a published answer.

---

# PART VIII — PRACTICAL

## 30. The software

```
caliper/activations.py   pulls the internal numbers out of the model
caliper/estimator.py     the direction-finding method, plus the cascade fix
caliper/batched.py       fits many neurons at once (4.5x faster)
caliper/runtime.py       device selection; runs resume if interrupted
experiments/             one script per experiment; results saved as data files
tests/                   11 automated tests, all passing
docs/                    specification, plan, timeline, pre-registration
```

About 1,500 lines. Every result is reproducible from a saved file. Runs on an ordinary laptop
CPU — no special hardware has been needed.

Measured speed: 63.7 seconds per neuron for the full procedure.

## 31. Work split for four people

| Track | Owner | Content |
|---|---|---|
| **A** | | Extracting internal numbers; managing text data; making runs fast and resumable |
| **B** | | The estimator itself; the cascade fix; random baselines; stability checks |
| **C** | | Causal tests — does removing a direction actually change behaviour? |
| **D** | | Ground-truth construction; artificial networks with planted answers; statistics |

**This is the largest open risk in the project and it is not technical.** Each member needs an
identifiable contribution by review time. Ask the mentor to help allocate.

## 32. Resources

Nothing is currently blocked. The next phase is about 20 hours of computation on a laptop,
which is manageable. Free cloud GPU (Kaggle, 30 hours/week) covers the heavier runs.

**A GPU would let us measure 1,000 units instead of 300**, which would materially strengthen
the main result. Worth asking about department access.

## 33. Risks

| Risk | Severity | Response |
|---|---|---|
| **Only one team member active** | **Highest** | Assign tracks now; ask the mentor for help |
| Capacity — the work is about equal to the time available, with no slack | High | Already cut scope twice; one more cut held in reserve |
| Trusting shortcut measurements | High — has caught us repeatedly | Rule: a shortcut is a shortcut. Decide with the real measurement |
| Being scooped | Medium | The instrument is hard to scoop; preprint in December |
| Compute | Low now | Scoped to fit; GPU would improve rather than rescue |

---

# PART IX — DEFENDING THE WORK

## 34. Questions and answers

**"Isn't this just debugging your own code?"**
No. The method is not ours — it is a standard approach used across the field. What we
contributed is the **test**. Measuring that a diagnostic has a 23% false-negative rate is not
debugging the diagnostic; it is what tells you how much to trust every result it gives.

**"This seems basic for a capstone."**
The finding is not basic: "a widely used class of method is silently wrong a quarter of the
time, and here is a detector" is the shape of results that get attention. The machinery is
simple **on purpose** — the hard part was building a situation where the truth is knowable. And
we were wrong repeatedly during the phase; the ground-truth check caught every one. A more
complicated project without that check would have shipped those errors.

**"Where is the neuroscience?"**
Three places. The estimator is a neuroscience method, applied to a language model for the first
time. Section E0.5 is a neuroscience result — forty-year-old tools tested where the answer is
knowable for the first time. And the discipline itself, calibrating before trusting, is the
imported habit that produced everything here.

**"What is the novelty? Hasn't this been done?"**
Be precise, because parts *are* occupied. Testing whether extracted directions are reliable is
being done (arXiv:2606.30449). Applying validity theory to model evaluation is being done,
about twelve papers in 2026. **Measuring how wrong a direction is against a known answer is
not**, because until now there was no known answer.

**"77% — good or bad?"**
Bad, and that is the point — bad relative to what everyone assumes, which is roughly 100%.
Nobody publishes a caveat saying their direction might be wrong a quarter of the time, because
nobody knew.

**"Your test failed. Isn't that a problem?"**
We wrote the passing bar down before running, at 90%, and got 77%. We are reporting the failure
rather than moving the bar. That is the correct outcome for a calibration phase — it tells us
the instrument's true reliability, which is the number we needed.

**"Why GPT-2? It's old."**
The ground-truth trick needs readable weights; GPT-2 is small enough to run 100 experiments on
a laptop; and the finding is about the *method*, not the model. Next step is confirming it on
the Pythia family, which is modern and open.

**"How much did you do versus tools?"**
The ground-truth idea, the experimental design, the pre-registration discipline, and every
judgement about what results mean. Be ready to explain neuron 2977 end to end from memory.

## 35. Where not to overclaim

- **Do not say the neuroscience methods "don't work."** Say they do not transfer here, and name
  the two assumptions that break.
- **Do not say persona vectors fail 23% of the time.** We have not tested persona vectors. Same
  family, never checked.
- **Do not claim the validity framing is new.** About twelve 2026 papers use it.
- **Do not present the two- and three-direction data figures.** That run used a flawed setup and
  is being redone.
- **Do not oversell the fix.** It does not repair every neuron. It gives a 77% success rate
  *with a working failure detector*, which is more valuable but a different claim.
- **If you do not know, say so and write it down.** Reporting our own failures is the strongest
  thing about this phase; continue it in the room.

## 36. The one-page summary

```
THE STORY
  Interpretability extracts directions. Nobody can check them.
  We found a place where the true answer is known, exactly and for free.
  Correct 77 times in 100 [68-84%]. Median score 0.993 -> looks perfect.
  Restarts agree 0.85-0.99 on the wrong answers. No warning at all.

WHY
  A perfect answer exists (R2 = 1.000 at the true direction).
  The 2-direction fit contains the right answer 99.7% of the time.
  So it is a search failure, not data, not neurons, not the model.

THE FIX AND THE DETECTOR
  Narrow down from two directions -> neuron 2977: 0.45 to 0.996.
  Methods agree    (69%): 87% correct
  Methods disagree (31%): 55% correct     p = 0.0008
  Detector needs no ground truth -> works in the real case.

SUPPORTING
  Classical neuroscience methods: 1 of 30.
     (inputs not bell-shaped + GELU falls over the operating range)
  Random guess explains <= 4.4%; true direction 70-98%.
  200 informative events is enough. More data adds nothing.

WHAT WE GOT WRONG (say it)
  6 plan decisions overturned by measurement.
  A proposed fix that halved the success rate when tested at scale.
  Two artefact findings from a shortcut measure. All caught by ground truth.

ASK HER
  1. GPU access?
  2. Publish Phase 0 separately?
  3. How to split work across four people?     <- most important
  4. Add the personality strand?
```

---

# APPENDIX — Key references

**Methods we use.** Sharpee, Rust & Bialek, *Maximally informative dimensions*, Neural
Computation 16(2), 2004 · Williamson, Sahani & Pillow, *Equivalence of information-theoretic
and likelihood-based methods*, PLoS Comput Biol 11(4), 2015 · Rowekamp & Sharpee, Network 22,
2011 · Paninski, Network 14, 2003.

**Why the field needs this.** *Sanity Checks for Sparse Autoencoders* (arXiv:2602.14111) ·
*Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers*
(arXiv:2501.17727) · Huang et al., *Rigorously Assessing Natural Language Explanations of
Neurons*, BlackboxNLP 2023.

**The personality strand.** *Persona Vectors* (arXiv:2507.21509, Anthropic, 2025) ·
*Internal-State Probes Read the Situation* (arXiv:2606.30449) · *AIPsy-Affect* (arXiv:2604.23719,
a released keyword-free stimulus battery) · *Analyzing the Generalization and Reliability of
Steering Vectors* (arXiv:2407.12404).

**What killed our first framing.** arXiv:2603.20642 (efficient coding in transformers) ·
Benjamin et al., Nature Communications 13:7972, 2022 · Kim & Paik, Science Advances, 2021.

**Project documents.** `docs/specification.md` · `docs/plan.md` · `docs/timeline.md` ·
`docs/preregistration-e01-n100.md` · all results in `results/`.
