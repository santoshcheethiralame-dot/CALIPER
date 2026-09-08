# Pre-registration — Study 2 P1/P2, planted-direction recovery

**Filed 8 September 2026, before the run.**

## Why this exists

Persona vectors are deployed. Open-weight models are audited with them (arXiv 2607.13162)
and they are traced through pretraining (2605.13329). They are validated two ways: by
**steering effect**, and by **correlation with finetuning-induced shifts**
(r = 0.76–0.97, 2507.21509).

Venkatesh & Kurapath (2602.06801) showed that orthogonal perturbations achieve
near-equivalent steering efficacy, so behavioural equivalence classes are large and
**steering cannot identify a direction**. That is the primary validation the persona
literature relies on.

Nobody has planted a known direction and asked whether the extraction pipeline recovers
it. This run does that.

## What is planted, and why random

The planted direction `v` is a **random unit vector**, not a concept vector. A concept
vector would confound two questions — whether the pipeline recovers what was planted, and
whether the model has a natural representation of the trait. P1 asks only the first.

## Protocol (frozen)

| element | value |
|---|---|
| model / layer | Gemma-3-27B-it, 4-bit NF4, fp32 compute, plant layer 37 of 62 |
| planted directions | 8 random unit vectors, `--plant-seed 0` |
| strengths | `--alpha-frac 0.10 0.20 0.40` of the 36,245 concept-token residual norm |
| injection | `span="all"` — present throughout generation, not prompt-only |
| elicitation | 16 neutral prompts, deliberately bland, identical across conditions |
| generation | 40 new tokens, greedy |
| **extraction** | generate under injection, then **re-read the resulting text with NO injection** and take the mean activation over response tokens; difference against the shared no-injection baseline |
| score | `abs(cos(diff, v))` |
| null | `abs(cos(diff, v_other))` for an independent random unit vector |

**The extraction step is deliberately the harder version.** Persona vectors are extracted
from generated text, not from the steered forward pass, so we re-read the text clean. A
planted direction is only recovered here if it changed the text enough to be recoverable
afterwards. An extractor that succeeds only while the injection is still switched on has
not been tested.

## Primary endpoint — P1, recovery at the plant layer

Median `abs(cos(diff, v))` at extract layer 37, against the median null, across 8 planted
directions, at the strongest tested setting.

**Criterion, fixed now: recovery must exceed the null by a margin that does not overlap,
and must exceed 0.30 in absolute terms.**

- **PASS.** The pipeline recovers a planted cause. P2 proceeds and Study 2 is live.
- **FAIL, recovery near the null.** The extraction pipeline does not recover a planted
  direction even at the layer it was planted in. **Stop. Do not run P2.** Debug in this
  order: injection span, whether the plant changed the text at all, elicitation-prompt
  diversity, then the number of prompts.
- **FAIL, recovery above the null but below 0.30.** Partial recovery. Report the value,
  run P2 to see whether it decays with depth, and describe the pipeline as weakly
  recovering rather than working.

This is a positive control on a linear read of a linear plant, run through a realistic
extraction path. A failure here means nothing downstream in Study 2 is interpretable —
the same logic as the steering control that caught the C31 dead-vector bug.

## Secondary — P2, the depth curve

Recovery at extract layers above the plant layer, `--extract-layers 39 43 47 55`. The
question is how fast the extracted direction drifts from the true cause as computation
intervenes. **This is chapter two's headline figure if P1 passes.**

## Committed in advance

- The null is reported beside every recovery number, always. A recovery figure without
  its null is not a result.
- If P1 fails, that is reported as the finding for this study — an extraction pipeline in
  production use that cannot recover a planted cause — rather than quietly debugged until
  it passes. Any post-failure fix is a **new** run under a new filing, and both are
  reported.
- No change to plant count, prompt set, strengths, or the scoring rule after seeing
  output.
- Recovery is scored against the plant `v` only. No searching over which planted
  direction "worked".
