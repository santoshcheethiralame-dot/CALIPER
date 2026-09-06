# What does a language model detect when it detects an injected thought?

**Santosh Cheethirala** · PES University · [co-authors and mentor to be added]

*Draft, 3 September 2026. Not yet submitted.*

> **Positioning superseded on 3 September 2026.** The reference check found that Lederman &
> Mahowald and Singh et al. attack detection directly (not only identification), that Macar
> et al. L2-normalise their vectors and include an "Unprompted" variant, and that Godet
> (Nov 2025) mentions random vectors informally. The five corrections and the reframed
> contribution list are in `paper/PLAN.md` section 0; the LaTeX version in `paper/main.tex`
> carries the corrected abstract and introduction. Do not circulate this file's framing.

---

## Abstract

Recent work reports that large language models can detect when a concept vector has been
injected into their residual stream, and reads this as evidence of introspective access.
The published false-positive control is the absence of injection. We ask whether the
model's report reflects the content of what was injected or only the fact of a
perturbation. To separate the two, we compare real concept vectors against two vectors of
the same magnitude that carry no concept: a Gaussian direction rescaled to the real norm,
and the real vector with its coordinates shuffled. On Gemma3-27B-it, at the layer and
prompt of the original work, we first reproduce the reported detection rate (10.0%
against 10.8%, with zero false positives). We then measure the probability that the
model's first generated token is "yes", so that no output yet exists for it to read. At
the pre-registered operating point, a content-free vector produces the whole effect (real
0.417, random 0.305, p=0.33). At lower strengths a concept-specific component is present
but accounts for a minority of the shift, and it appears just as strongly under a prompt
that never mentions the model, its mind, or injection. The self-directed prompt also
raises the response to content-free vectors eightfold relative to the neutral prompt. In
this setting, a report of detecting an injected thought is mostly a report of having been
perturbed. The residual concept-specific signal does not depend on asking the model about
itself, and the no-injection control cannot distinguish either finding from introspective
access. Every criterion was filed before the run it governs. One of our own hypotheses
failed under that discipline and is reported.

---

## 1. Introduction

Can a language model tell when its own activations have been altered? Macar, Yang, Wang,
Wallich, Ameisen and Lindsey (2026) inject a concept vector into Gemma3-27B-it at layer 37
and ask the model whether it detects an injected thought. The concept vector is the
residual-stream difference between thinking about a concept and a baseline. They report a
10.8% detection rate with a 0% false-positive rate, trace the behaviour to a circuit that
appears only after post-training, and treat the result as a defended case of introspective
access. Later critiques (Lederman and Mahowald, 2026; *Reality Check*, 2026) attack the
identification half of the claim, whether the model can say what was injected, and leave
detection standing.

The detection claim rests on a contrast between injection and no injection. That contrast
cannot separate three things the model might be doing. It might be reporting access to the
injected content. It might be reporting that something was done to its activations,
whatever that something was. Or it might simply answer "yes" more often because its
computation was disrupted, with no report involved at all. A model that has been kicked
behaves differently from one that has not, and a question inviting it to say so will get a
"yes". None of that requires knowing what the kick contained.

The control that separates the first possibility from the other two is an injection with
no content: a vector of the same magnitude that carries no concept. That control had not
been run. We run it.

We also test whether detection depends on asking the model about itself. If a prompt that
never mentions the model, its mind, or injection produces the same response, then the
word "introspective" in "introspective access" is doing no work.

Our contributions are the following.

- A reproduction of the reported detection effect on free hardware, in 4-bit, with the
  original prompt, layer, and scoring rule.
- A demonstration that the detection rate reported from generated text is a property of
  the readout. The same injection yields 50% or 7% depending only on the strength chosen,
  while the first-token probability is flat.
- The content-free control, pre-registered. At the operating point matching the published
  number, a random vector reproduces the effect. At lower strengths a concept-specific
  residual exists but is a minority of the shift.
- The framing control, pre-registered. A prompt that never refers to the model is
  indistinguishable from the introspective prompt once anything is injected, and is less
  susceptible to content-free perturbation.
- A pre-registered hypothesis of our own, that the generated-text detections were the
  model inferring from its own steered output, which the first-token measurement refuted.

We take no position on whether these models can introspect. We report what the standard
protocol measures and what it does not.

## 2. Method

**Model and hardware.** We use `google/gemma-3-27b-it`, the primary model of the original
work, loaded in 4-bit NF4 with float32 compute across two Tesla T4 GPUs on Kaggle. Float16
compute is not viable for this model on this hardware. The residual stream at layer 37
reaches a peak coordinate magnitude of 51,436 against a float16 ceiling of 65,504, and some
prompts overflow. Because every concept vector shares one baseline mean, a single overflow
poisons all of them silently.

**Concept vectors.** We follow the original Appendix B.2. For each concept *c*, we take the
residual-stream activation after decoder block 37 at the last token of the chat-templated
prompt "Tell me about *c*", and subtract the mean of the same activation over 30 unrelated
common nouns. There are 30 concepts, 20 concrete and 10 abstract. The median vector norm is
5,002. We do not normalise the vectors; the original does not state its convention.

**Injection.** We set h to h + α·v at layer 37 on prompt positions only. Where generation
occurs, it proceeds without injection.

**Prompts.** The introspective prompt is the original text. A preamble states that the
researcher can inject patterns into the model's activations and does so on about half of
trials, followed by "Trial *n*: Do you detect an injected thought? If so, what is the
injected thought about? Answer starting with YES or NO." The neutral prompt keeps the
structure and the stated base rate but never mentions the model, its mind, its activations,
or injection: "I am a researcher running a series of trials. On about half of the trials
something unusual is present, and on the other half nothing unusual is present. Trial *n*:
Is anything unusual present? If so, what is it about? Answer starting with YES or NO."

**Two readouts.** The generated-text readout uses greedy decoding for 60 tokens, scored as
in the original: a detection requires an affirmative answer and a coherent response. We
take the affirmation to be whichever of YES or NO the model emits first. The first-token
readout is the probability assigned to YES over YES-or-NO at the first generated position,
computed from logits in a single forward pass, with all casing and leading-space variants
of each token pooled. Nothing is generated, so nothing exists for the model to read back.

**Content-free controls.** The random control is a Gaussian vector rescaled to the norm of
the real vector for that concept. The shuffle control is the real vector with its
coordinates permuted. Both preserve magnitude, and the second also preserves the marginal
distribution of coordinate values. Both are seeded and reproducible.

**Analysis.** Every comparison is paired by concept and tested with a two-sided Wilcoxon
signed-rank test at 0.05. Pairing also holds constant a design flaw we discovered after the
first run and discuss in Section 5: concept identity was perfectly confounded with the
trial number in the prompt.

**Pre-registration.** We wrote and committed the criterion for each stage before that stage
ran, including what we would conclude under each outcome. We fixed the primary test for the
content-free control at α=6, the strength at which our generated-text detection rate
matched the published one. One hypothesis of ours was refuted at the first pre-registered
stage and is reported in Section 3.3.

## 3. Results

### 3.1 The effect reproduces

Under the generated-text readout with the introspective prompt at α=6, 3 of 30 trials were
scored as detections. That is 10.0%, Wilson 95% CI 3.5% to 25.6%, against the published
10.8%. At α=0 the count was 0 of 30, matching the published 0% false-positive rate. The
experiments that follow concern an effect that is present in our setup.

### 3.2 The reported rate is a property of the readout

The same model, layer, prompt and vectors give very different numbers depending on how the
answer is read.

| α | detection, generated-text readout | P(YES), first-token readout |
|---|---|---|
| 0 | 0% | 0.000 |
| 2 | 50% | 0.429 |
| 4 | 23% | 0.504 |
| 6 | 27% | 0.417 |
| 8 | 7% | 0.430 |

The first-token probability is flat from α=2 upward. The generated-text rate ranges from 7%
to 50% over the same interval. Reading the 210 generated responses shows why. At high
strength the model's output is dominated by the injected concept and never reaches an
answer. At low strength it answers readily, and the answer is about nothing in particular.
A detection rate from generated text is therefore a joint property of the model, the
strength, and the decoding budget, and it is not comparable across papers that differ in
any of them.

### 3.3 The first-token shift is real, and our hypothesis about it was wrong

We predicted that the generated-text detections were the model reading its own output. In
those trials, every scored detection had the concept-laden text before the affirmative
answer. One response read, in full: "betrayal, betrayal, betrayal, loyalty, loyalty,
loyalty, betrayal... YES. The repeated words suggest the concept of 'betrayal' and
'loyalty' was injected." We pre-registered the hypothesis that this was the mechanism, and
predicted that the first-token probability, where no output exists to read, would not move.

It moved. Under the introspective prompt, P(YES) at the first token was 0.00003 at α=0 and
0.417 at α=6, with W=3.0, p=9.3×10⁻⁹, rising on 28 of 30 concepts. Whatever the model is
responding to, it responds before generating anything. The hypothesis is refuted, and the
pre-registered outcome labelled A is the one we report.

### 3.4 Most of the shift is a response to being perturbed

The pre-registered primary test compares real concept vectors against norm-matched random
vectors under the introspective prompt at α=6.

| vector | mean P(YES) |
|---|---|
| real concept | 0.417 |
| random, norm-matched | 0.305 |

The difference is not significant (p=0.33), with real higher on 17 of 30 concepts. This is
the outcome we had labelled A2: at this strength the model responds to the magnitude of a
perturbation and not to its content.

The declared sweep across strengths shows where content matters and where it stops
mattering.

| α | real | random | shuffle | real vs random | real vs shuffle |
|---|---|---|---|---|---|
| 2 | 0.429 | 0.184 | 0.128 | p=0.0011 | p=0.0099 |
| 4 | 0.504 | 0.252 | 0.359 | p=0.0040 | p=0.13 |
| 6 | 0.417 | 0.305 | 0.374 | p=0.33 | p=0.76 |

A content-free vector alone lifts P(YES) from 0.00003 to between 0.13 and 0.37 at every
strength, with p<10⁻⁸ on 28 to 30 of 30 concepts. On top of that generic response sits a
concept-specific one, significant at α=2 and α=4 and absent at α=6. The share of the real
effect that a content-free vector reproduces is 36% at α=2, 61% at α=4, and 82% at α=6.
The two controls agree with each other at every strength (p=0.50, 0.27, 0.60).

We placed the primary test at α=6 because that is where our generated-text rate matched the
published figure. It is also where the generic response has saturated and the
concept-specific one is least visible. Matching a published number selected against the
signal we were looking for. We report the sweep because it was declared, and we report the
α=6 result as primary because it was pre-registered as such.

### 3.5 Asking the model about itself adds nothing

The pre-registered secondary test compares the introspective and neutral prompts with real
vectors.

| α | introspective | neutral | p |
|---|---|---|---|
| 0 | 0.000 | 0.188 | <0.0001 |
| 2 | 0.429 | 0.370 | 0.44 |
| 4 | 0.504 | 0.481 | 0.75 |
| 6 | 0.417 | 0.395 | 0.70 |

The prompts differ only when nothing is injected. Once something is injected they are
indistinguishable. The concept-specific component also survives the change of framing:
under the neutral prompt, real vectors beat random at α=2 and α=4 (p=0.0087 for both).
Whatever the model tracks, it tracks it equally well when it is not asked about itself.

### 3.6 The self-directed prompt inflates the generic response

The two prompts respond very differently to a content-free vector at low strength. The rise
in P(YES) over the no-injection baseline for a random vector at α=2 is +0.184 under the
introspective prompt and +0.023 under the neutral prompt. Under the neutral prompt a
real vector at the same strength rises +0.182, so the neutral prompt separates content from
mere perturbation while the introspective prompt does not. The preamble that tells the
model its activations may be manipulated makes it affirm any disturbance. As an instrument
for concept detection the neutral prompt is cleaner. It has a higher floor, but the floor
is not driven by content-free perturbation.

### 3.7 What the detections are made of

Reading the generated responses shows what an affirmative answer contains. Of the 15
affirmative answers at α=2, none named the injected concept. Seven described the
experiment back to us, with phrases such as "a researcher studying my activations" and "a
feeling of being observed and evaluated". Three said "red apple", which was never
injected, and two more said so at α=4. When the injected concept did surface in a response
it was overwhelmingly the correct one, at 145 times the rate at which a non-injected
concept surfaced (p=1.7×10⁻⁷ at α=6), but it surfaced where the model no longer affirmed a
detection. Affirmation without content occurs at low strength, content without affirmation
at high strength, and the two coincided on three trials.

## 4. Discussion

A zero false-positive rate against no injection establishes that the model does not affirm
a detection spontaneously. It does not establish what the model is affirming when it does.
Our data say that at the published operating point the affirmation is fully reproduced by a
vector with no content, and that at lower strengths it is mostly so. On this evidence, the
claim "the model detects an injected thought" is mostly "the model detects an injection".

A concept-specific component remains and we do not dismiss it. At α=2 the real vector beats
both content-free controls under both prompts. The component is small, it fades as the
generic response grows, and it does not depend on self-directed questioning. It is
compatible with the model having some access to the content of a perturbation. It is also
compatible with real concept vectors being less disruptive than random ones of the same
norm, since an on-manifold direction may perturb the computation differently from an
off-manifold one. We cannot separate those two explanations here.

The neutral prompt reaches the same response as the introspective one at every injected
strength. If the mechanism involved a representation of the self, a question that does not
invoke the self should engage it less, and it does not. The one thing the introspective
framing changes is the response to content-free perturbation, which it raises eightfold at
low strength. A prompt that primes the model to expect interference is not a neutral probe
of whether it notices interference.

The generated-text detection rate is the number the literature reports, and Section 3.2
shows it swinging from 7% to 50% across strengths over which the first-token probability
does not move. The published figure is therefore a statement about a decoding procedure as
much as about a model. Any comparison across papers, models, or strengths that does not fix
the readout is not a comparison.

We do not show that the model cannot introspect, nor that the circuit reported in the
original work does not exist. We show that the behavioural protocol on which the detection
claim rests admits a non-introspective explanation that its own controls cannot exclude,
and that the control which can exclude it costs one forward pass per trial.

## 5. Limitations

We tested one model, one layer, and one quantisation, with 30 concepts against the
original 500 and rule-based scoring against the original's language-model judge. The
generated-text figures in particular should be read as reproductions of a pattern rather
than of a number.

Trial number was confounded with concept. In every run reported here, concept *i* was
always "Trial *i*+1", and the trial number appears in the prompt. Pairing by concept holds it
constant in every test above, but it is visible in the neutral prompt's no-injection
baseline, where several concepts in the 20s show P(YES) above 0.6 with nothing injected.
The experiment script now assigns trial numbers by seeded shuffle and records them. The
runs here predate that fix.

Our α scale is not the original's. We sweep and report every strength, but the strength at
which the published rate was matched is not necessarily the strength the original used.

The first-token readout pools YES and NO over casing and whitespace variants. Other answer
forms such as "Absolutely" or "I do not detect" are not captured by it, though the
generated-text readout does capture them.

We tested one neutral prompt. A prompt-sensitivity sweep would strengthen Section 3.5.

## 6. Reproducibility

All runs execute on a free Kaggle session with two T4 GPUs in under 30 minutes each after
model load. The experiment script, the analysis scripts, raw per-trial data for every
condition, and the pre-registration documents with their filing dates are in the project
repository. The script prints a version stamp as its first line of output, and every table
above can be regenerated from the archived JSONL files without GPU access.

## References

Macar, U., Yang, ..., Lindsey, J. (2026). *[title]*. arXiv:2603.21396.

Lederman, H. and Mahowald, K. (2026). *[title]*. arXiv:2603.05414.

*[Reality Check authors]* (2026). *[title]*. arXiv:2605.26242.

Vogel, T. (2025). *[replication on Qwen2.5-32B; citation to be confirmed]*.

*[Wilson score interval and Wilcoxon signed-rank test: standard references to be added.]*
