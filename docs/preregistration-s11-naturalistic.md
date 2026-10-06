# Pre-registration: S-11, re-running APERTURE's naturalistic arm (R9)

**Filed 7 October 2026, before the run.** Paper 2. Code: APERTURE's own, at mirror commit
`b5bb2fb` (`aperture.naturalistic.collect_naturalistic_hf`), unchanged. Run sheet:
`kaggle/NEXT_SESSION_S11.md`.

## Why

R9 (15 July) is the strongest available answer to "injected states are off-distribution
damage". It found that directions extracted with APERTURE's recipe decode the concept a
passage evokes without naming it: 11/16 = 0.688 [0.438, 0.875] against 0.062 chance, after
mean-centring. Its raw data are lost, so it cannot be cited as a CALIPER measurement
(`docs/APERTURE_INHERITANCE.md`). S-11 re-runs it as designed.

## Design, exactly R9's

| setting | value |
|---|---|
| model | gemma-2-2b-it, 8-bit |
| layer | 13 |
| direction extraction | `extract_hf`, 12 pairs |
| concepts and passages | the 16 dev concepts and `data/concepts/contexts.yaml`, whose passages never name their concept (enforced by `load_contexts`) |
| classification | last-token residual of the passage, mean-centred across the 16 passages, nearest direction by dot product |
| report | passage + distractor + report prompt, 12 new tokens, rules-graded |

The distractor and report strings are R9's:
- distractor: `"\n\nQ: What is the capital of France?\nA: Paris.\n\n"`;
- report: `"Q: Setting that aside, in one word, what concept was most on your mind just now?\nA:"`.

Library versions are recorded. A-F1 showed 8-bit greedy decoding drifting across versions, so
an exact match to R9's 11/16 is not expected and is not the criterion.

## Primary endpoint and criterion

- **Endpoint:** activation identifiability, k/16.
- **Criterion:** k ≥ 4, which is exact one-sided binomial p < 0.05 against chance 1/16
  (P(X ≥ 4) = 0.019).
- **If it holds:** S-11 replaces R9 as the citable naturalistic result, reported with its
  exact 95% interval.
- **If it fails:** R9 does not replicate, and Paper 2 drops the naturalistic argument
  rather than citing the lost run.

## Secondary

- **Uncentred classification of the same activations.** R9's notes predict this collapses
  to one concept. Computed offline from the saved activations.
- **Verbal report accuracy.** It is reported, but it is **not** interpreted as
  introspection: the passage stays in context, so the report is reading comprehension.
- **A second extraction seed** (pairs seed 1), reported beside the primary. Seed variance
  is the one source R9's interval left out.
