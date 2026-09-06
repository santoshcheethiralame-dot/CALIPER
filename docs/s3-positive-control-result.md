# Study 3, Stage 1 — positive control result

Run: Gemma3-27B-it, 4-bit NF4, float32 compute, 2x Tesla T4 (Kaggle).
Layer 37 of 62. 30 concepts x 4 injection strengths = 120 trials.
Raw data: `s3_results.jsonl`. Re-scoring: `experiments/rescore_s3.py`.

## The pre-registered gate passed. It was measuring the wrong thing.

The filed criterion was detection >= 5% at alpha=4 with false positives <= 5%.
The run returned 16.7% and 0.0%, so on its own terms the gate opened.

It should not have. The scorer counted any response beginning with "YES" as a
detection, without checking that the "YES" was about the injected concept. Once that
is checked, the result reverses.

## Re-scored from the same saved text

| alpha | n | old scorer | says YES | names the concept | **says YES and names it** |
|---|---|---|---|---|---|
| 0 | 30 | 0% | 0% | 0% | **0%** |
| 2 | 30 | 43% | 50% | 0% | **0%** |
| 4 | 30 | 17% | 23% | 3% | **0%** |
| 8 | 30 | 0% | 7% | 33% | **0%** |

The claim in Macar et al. is that the model reports a detection and that report is
about the injected concept. That happened **zero times in 120 trials.**

## What the model claims to detect, when it claims anything

| alpha | responses claiming detection | about the injected concept | about the experiment itself | "red apple" / "blue bird" |
|---|---|---|---|---|
| 2 | 15 | 0 | 7 | 3 |
| 4 | 7 | 0 | 6 | 2 |
| 8 | 2 | 0 | 0 | 2 |

"About the experiment itself" means the response is assembled out of the researcher
preamble: *"a researcher studying my activations"*, *"a feeling of being observed and
evaluated"*. The preamble tells the model an injection is happening, and the model
describes that instead of any concept.

"Red apple" and "blue bird" were never injected. They are what the model produces when
asked to name a concept it has no access to - a canonical example, generated rather
than detected. Categories overlap slightly, so rows do not sum exactly.

## The dissociation

At alpha=8 the injected concept reaches the output in **10 of 30** responses -
`eagles`, `Dolphins.`, `volcanoes and lava`, `rivers and the flow of water`, `engines.`,
`jealousy of others' happiness`. The injection is plainly working.

Of those 10 responses, **0 claim to detect an injected thought**, and all 10 fail to
emit YES or NO at all.

Where the injection is weak enough for the model to answer the question, it reports
detections that are about nothing. Where it is strong enough to visibly control the
output, the self-report channel goes silent.

## What this does and does not establish

Does not: that Macar et al. are wrong. Our alpha scale is not theirs. Our concept
vectors are unnormalised with median norm 5002, and the paper does not state its
convention, so their alpha=4 may sit anywhere in our range. We have not yet found a
strength where the concept influences the output *and* the model is still coherent
enough to be asked about it - that window is between our alpha=4 and alpha=8 and is
unswept. A null result from an unswept parameter is not a result.

Does: give a concrete, measured demonstration that a detection number can be produced
almost entirely by prompt echo and confabulation, and that the standard first-token
scoring rule cannot tell the difference. That is a methodological finding about how
this effect is measured, and it is independent of whether the effect is real.

## Known limitations of this run

- 30 concepts, not their 500.
- Rule-based scoring, not their GPT-4.1-mini judge.
- 4-bit weights. fp16 compute overflows this model outright (peak activation 51436
  against an fp16 ceiling of 65504), which is why compute is float32.
- **Concept identity is perfectly confounded with trial number** - elephant is always
  Trial 1, freedom always Trial 30. The trial index appears in the prompt. This must be
  randomised before any result is reported.

## Next

Sweep alpha = 3, 5, 6 at the same layer to locate the coherent-but-influenced window.
Then, and only then, the neutral-framing comparison at that strength.
