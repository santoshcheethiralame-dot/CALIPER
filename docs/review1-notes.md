# Review 1 — Complete Notes

**For the presenting team. Everything in plain words. Written 3 September 2026.**

How to use this: read the whole thing once. Then keep the glossary and the question bank
open during the review. The slide notes at the end are what you say; everything before them
is what you need to know to answer whatever comes after.

---

## 1. The problem, three ways

**In one sentence.** We have tools that claim to read what is going on inside a language
model, and nobody has ever checked whether they work on a case where the right answer was
already known.

**In one paragraph.** When people study a language model from the inside, they use tools
that read off a number or a direction and say "this neuron responds to X", "this vector is
the model's honesty trait", "the model reports it can sense changes in its own activity".
Every one of those is a *reading* — like a thermometer reading. A thermometer is trusted
because someone once put it in ice water and boiling water and checked that it said 0 and
100. Nobody has done that for the tools that read language models. We are building the
ice water and the boiling water — cases where we already know the answer — and using
them to find out how often each tool is right, how often it is wrong, and whether it can
tell the difference.

**In one minute (say this out loud when asked).** Think of a doctor with a new heart
monitor. Before trusting it on a patient, they test it on a machine that produces a known
heartbeat. If the monitor says 70 when the machine says 70, good. If it says 70 when the
machine says 90, the monitor is broken, and now you know. Neuroscience has done this for
a hundred years with simple animals whose nervous systems are small enough to check —
that is where every tool for reading brain activity was first tested. Nobody has done it for
the tools that read AI models. We build the "known heartbeat" for language models at three
levels — a single neuron, a personality trait, and the model's report about itself — and
we test the tools against it. What comes out is not "the model is honest" or "the model
can introspect". What comes out is "this tool is right 77% of the time, here is when it is
wrong, and here is a warning light that comes on when it is about to be wrong."

---

## 2. Glossary

Every term the panel might hear or read, in the order they are likely to come up. Plain
definitions first; the technical name in brackets where there is one.

**Language model.** A program that predicts the next word. GPT, Gemma, Qwen, Claude. Made
of many layers; each layer transforms the text a little further.

**Interpretability.** The field that tries to understand *why* a model gives the output it
gives, by looking inside rather than only at the answers. Our project is about whether the
tools of this field can be trusted.

**Neuron / unit.** One of the millions of tiny computing elements inside the model. Each
one takes in a number and puts out a number. We use "unit" to avoid implying it works like
a biological neuron.

**Activation.** The number a neuron outputs on a given input. "The activation of unit 2977
on the word 'river' was 4.3."

**Residual stream.** The main data highway running through the model. Every layer reads
from it and writes back to it. When people talk about "the model's internal state at layer
37", they mean the numbers on this highway at that point.

**Direction / vector.** Inside the model, ideas are represented as directions in a very
high-dimensional space (thousands of dimensions). "The model's concept of *elephant*" is a
direction. A tool that reads a neuron is trying to find which direction that neuron looks
along.

**Concept vector.** A direction built by taking the model's internal state when it thinks
about a concept, minus its state on average. Used to represent "elephant" or "grief" as a
single arrow.

**Injection / steering.** Adding a vector to the model's internal state on purpose, to push
it toward a concept. If you inject the *elephant* vector, the model starts talking about
elephants. Also called *activation steering*.

**Persona vector.** A direction that is claimed to encode a personality trait — sycophancy,
honesty, evil. Published work claims you can find these and turn them up or down.

**Introspection.** The model reporting on its own internal state. "Do you notice anything
unusual happening inside you right now?" A recent Anthropic paper claims models can
sometimes do this.

**Readout / reading tool / estimator.** Any method that takes activations and produces a
claim about what they mean. A probe, a steering vector, a tuning curve, an introspection
prompt — all readouts. Our whole project is about calibrating readouts.

**Ground truth.** The right answer, known independently of the tool. If you know a
neuron's true direction because you built the network, that is ground truth. Almost all
interpretability work has none.

**Preparation.** A neuroscience term. A simple organism or piece of tissue chosen because
it is small enough that you can know the answer and test your instrument on it. The squid
giant axon, the sea slug *Aplysia*, the worm *C. elegans*. Our project builds preparations
for language models: cases where we planted or already know the answer.

**Tuning curve.** A plot of how much a neuron fires as you vary the input. A neuron "tuned
to vertical lines" fires most for vertical, less for tilted. We ask the same of language
model units.

**Spike-triggered average (STA).** The classic neuroscience way to find what a neuron
responds to: average all the inputs that made it fire. We tested this on language model
units. It found the right answer for 1 unit in 30. Our method found 26.

**Calibration.** Checking an instrument against known cases before using it on unknown
ones. The thermometer in ice water.

**Null distribution.** What the tool reports when there is nothing to find. If a tool says
"this neuron responds to *justice* at strength 0.04" and random noise also produces 0.04,
the finding is nothing. We measure the null for every tool.

**False positive.** The tool says "found it" when there was nothing there.

**Silent failure.** The tool gives a confident wrong answer with no sign anything went
wrong. This is our central finding at the unit level: 23% of the time the tool is wrong and
looks exactly as sure as when it is right.

**Disagreement flag.** Our warning light. We run the reading two different ways; where they
disagree, the reading is unreliable. It does not need ground truth, so it works on real
models where we do not know the answer.

**Confidence interval (CI).** A range that says how sure we are. "77%, CI [68%, 84%]"
means: our best estimate is 77%, and we are 95% confident the true number is between 68
and 84.

**Pre-registration.** Writing down, before running an experiment, exactly what you will
measure and what each outcome will mean — then committing that document with a date. It
stops you from changing your mind after seeing the results. We have done this for every
experiment. One of our own predictions failed under it, and we reported that.

**Wilcoxon test / p-value.** A standard statistical test. "p = 0.001" means: if there were
truly no difference, you would see a gap this big about one time in a thousand. Small p,
real difference.

**Quantisation / 4-bit.** Shrinking a model so it fits in less memory by storing each
number with fewer digits. A 27-billion-parameter model normally needs ~54 GB; in 4-bit it
needs ~17 GB, which fits two free Kaggle GPUs. Some precision is lost; we checked that the
effects we study survive it.

**Kaggle.** A free platform that gives 30 hours a week of GPU time. Every large-model
experiment in this project runs there on a free account. Anyone on the panel could re-run
it.

**GPT-2.** A small, old, open language model (2019). We use it for the unit-level work
because it runs on a laptop CPU and, crucially, its neurons' true directions can be read
straight out of its weights — free ground truth.

**Gemma3-27B.** Google's open 27-billion-parameter model. Used because it is the exact
model the Anthropic introspection paper tested, so our results are directly comparable.

---

## 3. Why anyone should care — use cases

The panel will ask "what is this for?" These are the honest answers, most concrete first.

**AI safety audits.** Companies and regulators increasingly want to check whether a model
has a hidden trait — deception, sycophancy, bias — by reading its internals. Every such
audit uses a readout. If the readout is wrong 23% of the time and does not say so, the
audit is worth less than it looks. We measure that number.

**Trusting a model's self-report.** Models are now asked "are you being honest?", "do you
notice anything odd?", and the answers are treated as evidence. Our Study 3 shows that on
the exact model and prompt from a published paper, the model's "yes, I detect something"
is mostly a reaction to *being disturbed at all*, not to what was put in — a vector with
no content in it produces the same report. If you are going to trust a model's word about
itself, you need to know that.

**Model debugging.** When a model behaves badly, engineers look for the neuron or direction
responsible. A tool that points at the wrong neuron sends them down the wrong path. Our
disagreement flag tells them when the pointer is unreliable.

**Persona and character products.** Companies ship models with tuned personalities and
claim to control them with persona vectors. Whether a persona vector actually captures the
trait, or a mixture of unrelated things, has never been tested against a planted case. Study
2 (sem 6) plants one and checks.

**Reproducibility for the field.** Every experiment here runs on a free account in under
an hour. We publish the code, the data, and the pre-registration dates. Anyone can check.

**Foundational.** Neuroscience learned the hard way that reading tools need preparations.
Interpretability is a young field repeating neuroscience's early mistakes. The transfer of
that discipline is the intellectual contribution.

---

## 4. How we do it — the three levels

The same idea at each level: plant or know the answer, apply the tool, measure how often
it is right, and build a warning light for when it is wrong.

### Level 1 — the unit (a single neuron)

**What we know.** In a standard transformer, a neuron's input is a simple dot product of the
residual stream with a fixed weight vector. That weight vector *is* the neuron's true
direction. It is sitting in the model file. Free, exact ground truth for every neuron.

**What the tool does.** Given only the neuron's activations and the inputs, it tries to
recover that direction — the same problem neuroscientists face with a real neuron whose
wiring they cannot see.

**How we check.** Compare the recovered direction with the true one. A score of 1.0 means
perfect; we require above 0.95.

**What we found (Phase 0, done).** The tool recovers 77 of 100 units. The other 23 fail
*silently* — the tool is just as confident. But a second method disagrees with the first
exactly where it fails, so we have a warning light. And a better search recovers most of
the failures.

### Level 2 — the trait (a personality direction)

**What we plant.** A direction we choose, added to a model so that it now carries a known
"trait". We know the direction because we put it there.

**What the tool does.** The published persona-vector method tries to find it.

**How we check.** Same as Level 1: does the recovered direction match the planted one?
Then the multi-trait check from psychology (Campbell & Fiske, 1959): does the *honesty*
tool find honesty and not sycophancy?

**Status.** Sem 6. Blocked until the estimator can handle more than one direction at once.

### Level 3 — the self (the model's own report)

**What we plant.** A concept vector injected into a 27B model, exactly as Anthropic did.

**What the tool does.** Ask the model "do you detect an injected thought?" and score its
answer.

**How we check.** Against the case where nothing was injected (their control) *and*
against a vector of the same size with no content in it (our control). Also: ask the same
question without ever mentioning the model or its mind.

**What we found (Study 3, done).** The effect reproduces. But a content-free vector produces
most of it, and at the published operating point all of it. Asking about "yourself" adds
nothing. The model's report is mostly "something happened to me", not "here is what".

---

## 5. What we use, and where it comes from

The panel will ask "what dataset, what tools, what hardware". Have this table ready.

| what | source | why this one |
|---|---|---|
| **GPT-2 small** | HuggingFace, open, 2019 | Runs on a laptop CPU; its neuron directions are readable from the weights — free ground truth |
| **Gemma3-27B-it** | Google, open weights, via Kaggle Models | The model in the Anthropic introspection paper; results directly comparable |
| **Qwen2.5-32B-Instruct** | Alibaba, open weights, HuggingFace | Ungated; independent replication reported; second model for robustness |
| **Text corpus** | Project Gutenberg (public domain books) | Stimuli for driving neurons; no personal data, no licensing issues |
| **Concept lists** | Our own, 30 nouns and abstract terms; baseline of 30 unrelated nouns | Matches the published recipe |
| **PyTorch, HuggingFace Transformers** | Open source | Standard model loading and hooks |
| **bitsandbytes, accelerate** | Open source | 4-bit loading across two GPUs |
| **NumPy, SciPy** | Open source | The estimator and all statistics |
| **Laptop CPU / RTX 4070** | Team hardware | Every unit-level experiment |
| **Kaggle, 2× Tesla T4, free tier** | Kaggle | Every large-model experiment; 30 GPU-h/week |
| **APERTURE** | Our own prior repo | Earlier introspection pipeline; reused ideas, not code |

**Key papers (the ones to name if asked).**

| paper | what we take from it |
|---|---|
| Macar, Yang, Wang, Wallich, Ameisen, Lindsey (Anthropic, 2026) | The introspection claim we reproduce and control |
| Sharpee, Rust, Bialek (2004); Williamson, Sahani, Pillow (2015) | The neuroscience method for finding what a neuron responds to; we adapt it to LM units |
| Willmore & Tolhurst (2001) | Why "sparse" neurons are not the same as a sparse population — a caution we inherit |
| Campbell & Fiske (1959); Cronbach & Meehl (1955) | The psychology standard for validating a trait measure; Level 2 uses it |
| Nisbett & Wilson (1977) | People confabulate reasons for their own behaviour; the parallel for model self-reports |
| Distill *Circuits* (2020–21); Cacioli (2026) | Prior tuning-curve work on vision models and on number representation — where we differ |

**No personal data. No proprietary data. No paid compute.** Say this once; panels like it.

---

## 6. What "novel" means here

The panel may ask what is new. Three precise answers; do not overclaim.

1. **Free ground truth at the unit level.** The observation that an MLP neuron's true
   direction is its own weight column, so a real model can serve as its own preparation.
   Simple, and as far as our literature search found, unused for this purpose.
2. **The content-free control at the self level.** The published introspection work
   controls against *no injection*. We control against *an injection with no content*, and
   the two give different answers. That control had not been run.
3. **The transfer of the calibration discipline.** Null, required sample size,
   pre-registered interval, disagreement flag — as one package applied to interpretability
   readouts, borrowed from a field that learned it the hard way.

What is **not** new, and we say so: tuning curves have been done on vision models; the
neuroscience estimators are decades old; persona vectors and concept injection are
published methods. We are testing them, not inventing them.

---

## 7. Sem 5 timeline to state

| review | what the panel will see |
|---|---|
| **1 (today)** | Problem, team, literature, plan |
| **2** | Requirements (the four calibration rules) and system design |
| **3** | Prototype: the unit-level instrument, with its measured 77/100 and the warning light |
| **4** | Prototype on a real question: the introspection result, and the sem 6 plan |

Deliverables by end of sem 5: a calibrated estimator with a published error rate; one
complete experiment at the self level; a preprint; the repository with data and
pre-registrations.

---

## 8. Question bank

Short answers. If a question is not here, the honest answer is "we haven't measured that
yet" followed by when you will.

**"What is the application?"**
Anyone who audits a model by reading its internals — safety teams, regulators, debugging
engineers — is trusting a readout. We tell them how often it is wrong and when.

**"Is this a software project or a research project?"**
Both. The software is the calibrated estimator library and a reproducible Kaggle
pipeline. The research is what it measures.

**"What is your dataset?"**
Public-domain text from Project Gutenberg for stimuli, and open-weight models. No
collection, no personal data.

**"What hardware do you need?"**
A laptop for the unit level. Free Kaggle GPUs for the large models. We have both. A lab GPU
would speed things up; it is not required.

**"Why neuroscience? This is computer science."**
Because neuroscience is the only field with a century of experience reading activity from
units it did not design — and it learned that every reading tool needs a known case to be
tested on first. We are importing that lesson.

**"What have you actually done so far?"**
Built the unit-level instrument, tested it on 100 neurons with known answers, found it fails
silently on 23 and built a warning light that predicts which. Reproduced a published
introspection result on a free GPU and ran a control that had not been run. Both are
written up.

**"What will be different at the end of the semester?"**
A preprint on arXiv, an estimator with a published error rate, and a decision on whether
the trait level goes ahead.

**"Who does what?"**
Four tracks: estimator and integration; literature and framing; replication on a second
model; calibration. One owner each. [Name them.]

**"What is the biggest risk?"**
Technical: the estimator finds only one direction at a time, and real neurons may respond
to more. We have a fallback — restrict claims to one direction. Non-technical: keeping four
people working on separable tracks.

**"What if the tools turn out to work fine?"**
Then we will have shown that with numbers, and the field gets a certified instrument. A
calibration that passes is as publishable as one that fails. Ours partly failed, and that is
the more interesting result.

**"Can you demo it?"**
Yes. The estimator recovers a known neuron direction on GPT-2 in about a minute on a
laptop. [Have it ready.]

**"How is this different from existing interpretability benchmarks?"**
Existing benchmarks compare tools to each other. We compare each tool to the truth, on
cases where the truth is available. Different question.

**"Is 30 concepts / 100 neurons enough?"**
For a calibration, yes: the confidence intervals are reported and they are wide enough to
be honest. Sem 6 scales the unit level to 300.

---

## 9. Slide-by-slide speaker notes

**Slide 1 — Team and mentor.** Names, mentor, "problem selected 18 August from a set of
seven proposals; the mentor chose this one for its neuroscience framing." Ten seconds.

**Slide 2 — The problem.** Read the one-paragraph version from §1. Do not paraphrase. Then
the thermometer line: "Nobody has put these thermometers in ice water."

**Slide 3 — Three tools, three levels, zero checks.** One row per level: unit / trait /
self. For each: the published tool, the claim it makes, and "checked against a known
answer: never." Keep it to one table.

**Slide 4 — The neuroscience framing.** One image: a squid axon or *Aplysia*. "Every tool
for reading brain activity was first tested on something simple enough to check. That is
what a preparation is. We build preparations for language models." Then the free-ground-
truth point at the unit level in one sentence.

**Slide 5 — Literature.** ~40 papers, three strands. One sentence per strand on what it has
and what it lacks. Name the Anthropic paper by author; it is the one the panel may have
heard of.

**Slide 6 — Where we differ from prior work.** Three rows: prior tuning-curve work on
vision models — ours is on language models with ground truth; Cacioli on number
representation — ours plants the prior rather than correlating with it; Anthropic's control
is no-injection — ours is content-free injection. Do not claim more than the row says.

**Slide 7 — The semester.** The four-review table from §7 and the work division. End on:
"By Review 4 you will see a calibrated instrument with a published error rate and one
complete experiment. Both are already running."

Total: 10–12 minutes. Leave the rest for questions.

---

## 10. If you remember only five things

1. **Every reading tool is a thermometer nobody has tested in ice water.**
2. **We build the ice water — cases where the answer is known — at three levels: unit,
   trait, self.**
3. **At the unit level the tool is right 77% of the time and fails silently the rest; we
   built the warning light.**
4. **At the self level, a vector with no content produces the same "I detect something"
   as a real one. The published control cannot see that.**
5. **Everything is pre-registered, open, and runs on a free account.**
