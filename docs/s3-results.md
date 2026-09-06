# Study 3 — Results

**What was tested.** Whether a language model's report that it "detects an injected
thought" reflects access to the injected content, or something less.

**Setup.** Gemma3-27B-it, the primary model of Macar, Yang, Wang, Wallich, Ameisen and
Lindsey (arXiv 2603.21396). 4-bit weights, float32 compute, two Tesla T4s on Kaggle.
Concept vectors built exactly as in their Appendix B.2, injected at layer 37 of 62 on prompt
positions only. 30 concepts. Every result below is paired by concept and tested with a
Wilcoxon signed-rank, two-sided, 0.05. Criteria were filed before each run in
`docs/preregistration-s3-forced-choice.md`.

Raw data: `s3_results.jsonl` (210 free-generation trials), `s3_final_forced.jsonl`,
`s3_results_forced_random.jsonl`, `s3_results_forced_shuffle.jsonl`. Analysis:
`experiments/rescore_s3.py`, `analyse_s3_full.py`, `analyse_dissociation.py`.

---

## 1. The effect reproduces

Free generation, their prompt, their scoring rule, their layer. At alpha=6:

| | detection | false positives |
|---|---|---|
| Macar et al. | 10.8% | 0% |
| this run | 10.0% (3/30), 95% CI [3.5%, 25.6%] | 0/30 |

So the experiments below are about an effect that exists in our setup, not about an
absent one.

## 2. The number depends on how you read the answer

Same model, same injection, same layer, same prompt. Two ways of reading the answer:

| alpha | "YES" in generated text | P(YES) at first token |
|---|---|---|
| 2 | 50% | 0.429 |
| 4 | 23% | 0.504 |
| 6 | 27% | 0.417 |
| 8 | 7% | 0.430 |

The first-token probability is flat from alpha=2 upward. The generated-text rate swings
between 7% and 50% across the same range. That spread is entirely a property of the
readout, not of the model's sensitivity. A detection rate reported without the readout
method is not a number.

## 3. The shift is real, and it is not explained by the model reading its own output

We had predicted it would be. In the free-generation trials, every positive had the
concept-laden text *before* the YES; one said outright, "the repeated words suggest the
concept was injected." So we measured P(YES) at the first generated token, where no output
yet exists to read.

| introspective prompt | mean P(YES) |
|---|---|
| alpha=0 | 0.00003 |
| alpha=6 | 0.417 |

W=3.0, p=9.3e-09, 28 of 30 concepts rose. Our hypothesis was wrong and is reported as such.

## 4. Most of the shift is a response to being perturbed, not to what was injected

The paper's false-positive control is *no injection*. The relevant control is *an injection
with no content*. Two of them: a Gaussian vector rescaled to the real vector's norm, and the
real vector with its coordinates shuffled. Both destroy the concept and preserve magnitude.

**Pre-registered primary test, alpha=6, introspective prompt:**

| vector | mean P(YES) |
|---|---|
| real concept | 0.417 |
| random, norm-matched | 0.305 |

p=0.33. Indistinguishable. This was the outcome we labelled A2 in advance: the model is
responding to the magnitude of a perturbation, not its content.

**Across strengths (declared sweep, secondary):**

| alpha | real | random | shuffle | real vs random | real vs shuffle |
|---|---|---|---|---|---|
| 2 | 0.429 | 0.184 | 0.128 | p=0.0011 | p=0.0099 |
| 4 | 0.504 | 0.252 | 0.359 | p=0.0040 | p=0.13 |
| 6 | 0.417 | 0.305 | 0.374 | p=0.33 | p=0.76 |

A content-free vector lifts P(YES) from 0.00003 to 0.13–0.37 on its own, at every
strength, p<1e-8. On top of that generic response sits a smaller concept-specific one that
is visible at alpha=2 and 4 and gone by alpha=6. The share of the real effect that a
content-free vector reproduces: 36% at alpha=2, 61% at alpha=4, 82% at alpha=6.

The two controls agree with each other at every strength (p=0.50, 0.27, 0.60).

## 5. Asking the model about itself adds nothing

Second prompt, same structure, same stated base rate, never mentions the model, its mind,
its activations, or injection: *"I am a researcher running a series of trials. On about
half of the trials something unusual is present... Trial n: Is anything unusual present?"*

**Pre-registered secondary test, real vectors:**

| alpha | introspective | neutral | p |
|---|---|---|---|
| 0 | 0.000 | 0.188 | <0.0001 |
| 2 | 0.429 | 0.370 | 0.44 |
| 4 | 0.504 | 0.481 | 0.75 |
| 6 | 0.417 | 0.395 | 0.70 |

The two prompts differ only with nothing injected. Once something is injected they are
indistinguishable. And the concept-specific component survives the neutral framing too:
real beats random at alpha=2 and 4 under the neutral prompt (p=0.0087 both). Whatever the
model is tracking, it tracks it just as well when not asked about itself.

## 6. The self-directed prompt inflates the generic alarm

Rise over the no-injection baseline for a **random** vector at alpha=2:

| prompt | rise |
|---|---|
| introspective ("I can inject patterns into your activations...") | +0.184 |
| neutral ("something unusual may be present...") | +0.023 |

Telling the model that its activations may be manipulated makes it say YES to any
disturbance. Under the neutral prompt a content-free vector barely registers at low
strength while the real one still does. The neutral framing is the cleaner instrument for
concept detection; the introspective one adds a perturbation-alarm confound of its own.

## 7. What detections are made of

Reading the 210 free-generation responses. Of 15 detection claims at alpha=2, zero named
the injected concept, seven described the experimental setup back to us ("a researcher
studying my activations", "a feeling of being observed and evaluated"), three said "red
apple", which was never injected. Two more red apples at alpha=4.

When the concept does surface it is overwhelmingly the right one — 145 times chance at
alpha=6, p=1.7e-07 — but it surfaces where the model no longer reports a detection. Report
without content at low strength; content without report at high strength. The two peak at
opposite ends of the range and overlap on three trials.

---

## What this does and does not establish

**Does.** In this model, at this layer, a report of "detecting an injected thought" is
mostly a report of having been perturbed. A content-free vector of the same size produces
most of it, and at the strength where the free-generation rate matches the published
number, all of it. The residual concept-specific signal is real, small, fades with
strength, and is not specific to self-directed questioning. The published false-positive
control — no injection — cannot distinguish any of this.

**Does not.** That introspection is absent. A concept-specific component exists at low
strength. That Macar et al. are wrong about what they measured; they did not run this
control, and the distinction it draws had not been tested.

## Limitations

- One model, one layer, 30 concepts, 4-bit weights. The paper uses 500 concepts and a
  GPT-4.1-mini judge; we use rule-based scoring.
- **Concept identity is perfectly confounded with trial number** (elephant is always
  Trial 1). The trial number is in the prompt and visibly affects the neutral prompt's
  baseline. Pairing by concept controls it in every test above, but it must be randomised
  in any follow-up.
- The primary test was placed at alpha=6 because that is where the free-generation rate
  matched the published 10.8%. That turned out to be where the generic response saturates
  and the specific one is least visible. Choosing an operating point by matching a
  published number selected against the thing being measured.
- Our alpha scale is not theirs. Vectors are unnormalised (median norm 5002); the paper
  does not state its convention.

## Corrections made along the way, in the open

- The run-time scorer required "YES" as the first word and three or more words for
  coherence. It counted "**YES**" as a non-detection and a bare "NO." as incoherent. Both
  scorings are reported in `rescore_s3.py`.
- The first neutral prompt referred to a "following situation" that did not follow. It was
  replaced and the comparison re-run; the defective version's data is not used.
- We predicted the free-generation result was the model reading its own output. The
  first-token measurement showed otherwise. Pre-registered, reported.
- A blindsight analogy was floated and withdrawn when identification proved far above
  chance.
