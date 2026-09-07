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

## 1. The effect reproduces — weakly, and the number depends on the scoring rule

**Corrected 7 September 2026.** An audit of the archived generation data against the
pre-registered scorer found that the figure previously reported here, 10.0% (3/30), is
not produced by that scorer and is not reproducible from the data by any documented
rule. It has been replaced by the pre-registered number. See "Corrections made along the
way".

Free generation, their prompt, their layer, at alpha=6, n=30:

| scoring rule | detection | false positives at alpha=0 |
|---|---|---|
| **pre-registered scorer** (affirmative answer at the start of the response, plus a coherence check) | **6.7% (2/30)**, 95% CI [1.8%, 21.3%] | **0/30** |
| "YES" anywhere in the response | 33.3% (10/30) | 16.7% (5/30) |
| Macar et al., reported | 10.8% | 0% |

**The pre-registered rule is primary and gives 6.7%**, below the published 10.8% and
with an interval that includes it. The permissive rule gives 33.3% but also destroys the
false-positive control, since it counts five detections at alpha=0 where nothing was
injected. Neither rule reproduces 10.8% closely, and we do not claim a quantitative
replication.

The two trials the pre-registered scorer counts at alpha=6 both name the wrong concept:

> `YES. The injected thought is about a blue house.`   (umbrella injected)
> `Yes. The injected thought is about a "red apple" ...`   (bridge injected)

**Detection accompanied by a correct identification occurs 0 times in 30 at alpha=6, and
0 times in 210 across the whole sweep.** That is the finding to carry forward; the
detection rate itself is fragile to a scoring choice and the identification rate is not.

Note also what the permissive rule reveals about the false-positive control: a rule that
counts "YES" anywhere reports 16.7% detections on the no-injection condition. **A
detection rate is only interpretable next to the false-positive rate produced by the
same rule**, and the published 0% should be read with that in mind.

## 2. The number depends on how you read the answer

**Corrected 7 September 2026.** The generated-text column previously read 50 / 23 / 27 /
7 percent. The first two are reproducible from the archived data under a
"YES anywhere in the response" rule; **the 27% and 7% match no rule computable from it**
and have been replaced. The first-token column was checked against
`data/s3/s3_forced_real_sweep.jsonl` and is correct to the digit.

Same model, same injection, same layer, same prompt. Three ways of reading the answer:

| alpha | generated text, pre-registered rule | generated text, "YES" anywhere | P(YES) at first token |
|---|---|---|---|
| 2 | 43% (13/30) | 50% (15/30) | 0.429 |
| 4 | 17% (5/30) | 23% (7/30) | 0.504 |
| 6 | 7% (2/30) | 33% (10/30) | 0.417 |
| 8 | 0% (0/30) | 10% (3/30) | 0.430 |

**The finding is unchanged and if anything is sharper.** The first-token probability is
flat from alpha=2 upward, varying between 0.417 and 0.504 across the whole range. The
generated-text rate over the same range swings from 43% to 0% under the pre-registered
rule, and from 50% to 10% under the permissive one. Same model, same injection, same
layer: the spread is a property of the readout, not of the model's sensitivity.

Note that the two generated-text columns do not even agree with each other on the shape
— the pre-registered rule falls monotonically while the permissive rule rises again at
alpha=6. **A detection rate reported without both the readout method and the scoring rule
is not a number.**

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

**7 September 2026 — the reproduction figure and the readout table.** An audit of the
archived generation data against the pre-registered scorer found two numbers in this
document that the data does not produce.

*Section 1* reported the alpha=6 detection rate as 10.0% (3/30) beside the published
10.8%. The pre-registered scorer gives **6.7% (2/30)**; a permissive "YES anywhere" rule
gives 33.3% (10/30). The 3/30 corresponds to neither: it is the subset of trials in which
concept-laden text precedes an affirmative answer, identified by reading the transcripts
rather than by any filed rule. The pre-registration states that the scorer is final and
that no re-scoring follows the results, so **6.7% is the number this study is entitled
to report**, and it is the one now in section 1.

This matters beyond one figure. The value that was reported is the one closest to the
published 10.8%, and alpha=6 was then chosen as the operating point *because* it matched
— a choice section 4 already criticises on separate grounds. Selecting a scoring rule
that reproduces a target number is precisely what a pre-registered scorer exists to
prevent, and the safeguard was in place and simply not consulted.

*Section 2* reported the generated-text detection rate as 50 / 23 / 27 / 7 percent across
alpha 2, 4, 6, 8. The first two reproduce under the permissive rule; **27% and 7% match
no rule computable from the archived data.** Both columns have been recomputed under both
rules and are now stated with their counts. The first-token column was verified against
`s3_forced_real_sweep.jsonl` and was correct.

Neither correction touches the results in sections 3 to 7, which come from the
forced-choice readout and never call the generation scorer. The false-positive rate of
0/30 also holds under the pre-registered rule, though not under the permissive one,
where alpha=0 yields 5/30 — recorded in section 1 because a detection rate is only
interpretable beside the false-positive rate produced by the same rule.


- The run-time scorer required "YES" as the first word and three or more words for
  coherence. It counted "**YES**" as a non-detection and a bare "NO." as incoherent. Both
  scorings are reported in `rescore_s3.py`.
- The first neutral prompt referred to a "following situation" that did not follow. It was
  replaced and the comparison re-run; the defective version's data is not used.
- We predicted the free-generation result was the model reading its own output. The
  first-token measurement showed otherwise. Pre-registered, reported.
- A blindsight analogy was floated and withdrawn when identification proved far above
  chance.
