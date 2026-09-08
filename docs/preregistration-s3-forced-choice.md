# Pre-registration — Study 3, forced-choice detection

**Filed 2 September 2026, before the run. Nothing below changes after seeing results.**

## Where this comes from

The free-generation positive control (`docs/s3-positive-control-result.md`) reproduced the
detection rate reported by Macar et al. — 10.0% at alpha=6, 95% CI [3.5%, 25.6%], against
their 10.8% — with 0/30 false positives at alpha=0.

All three positive trials had the same structure: concept-laden text was generated
**first**, and the affirmative answer came after it. In one, the model stated the
inference explicitly:

> `betrayal, betrayal, betrayal, loyalty, loyalty, loyalty, ...`
> `YES. The repeated words suggest the concept of "betrayal" and "loyalty" was injected.`

That is inference from self-observed output. It is not access to the perturbation. But
n = 3, and three examples cannot carry the claim.

## Question

Does injection shift the model's answer **before any output exists to read**?

We measure P(YES) at the first generated token — one forward pass, nothing generated,
no self-observation possible. Genuine access to the perturbation should shift this
probability. Reading one's own output cannot.

## Protocol (frozen)

| Element | Value |
|---|---|
| Model | google/gemma-3-27b-it, 4-bit NF4, float32 compute, 2x T4 |
| Layer | 37 of 62 |
| Strengths | alpha = 0, 2, 4, 5, 6, 8 |
| Concepts | the same 30, same order, same vectors as the free-generation run |
| Framings | introspective (theirs) and neutral (never mentions the model) |
| Measure | sigmoid(logsumexp(YES logits) − logsumexp(NO logits)) at the last prompt position |
| Answer tokens | first token of YES / " YES" / Yes / " Yes" / yes / " yes", and the NO equivalents |
| Test | Wilcoxon signed-rank, paired by concept, two-sided, 0.05 |

Pairing by concept also neutralises a flaw in the free-generation run: concept identity is
perfectly confounded with trial number (elephant is always Trial 1), and the trial number
appears in the prompt. Comparing the same concept across strengths holds that constant.

## Criteria, fixed now

**Primary.** Mean P(YES) under the introspective framing, alpha=6 versus alpha=0.

- **A — significant increase.** First-token sensitivity to injection exists. The model has
  some access that does not depend on reading its own output. Our post-hoc-inference
  account of the free-generation result is then wrong about the mechanism, and we report
  that. The output-steering confound in the *generated* answers still stands separately.
- **B — no significant change, or a decrease.** The free-generation detection rate cannot
  be explained by access to the perturbation at the moment of answering. Combined with the
  three positive trials all showing concept-then-YES ordering, the available explanation is
  that the model reads its own steered output and infers backwards.
- **C — a decrease specifically.** Injection *suppresses* the affirmative answer. This
  would need explaining, not celebrating, and we would report it as an anomaly rather than
  as support for either account.

**Secondary.** Neutral versus introspective framing at matched alpha. If the two move
together, whatever is being measured is not specific to self-directed questioning.

## What outcome B would not establish

That Gemma cannot introspect, or that Macar et al. are wrong. It would establish that at
this layer, these strengths, and this quantisation, the detection number has an available
non-introspective explanation that their protocol does not rule out — and that the control
which rules it out is cheap and was not run.

## Committed in advance

- The scorer above is final. No re-scoring after seeing results.
- If the run fails on a bug, the fix is to the bug only, and the criteria stay as written.
- Both outcomes get written up. Outcome A is reported as prominently as outcome B.

---

# Addendum — norm-matched control and corrected neutral framing

**Filed 2 September 2026, after the forced-choice result, before the control run.**

## What the first run settled

Outcome **A**. Introspective framing, P(YES) at the first token: 0.00003 at alpha=0,
0.417 at alpha=6. Wilcoxon W=3.0, p=9.3e-09, 28/30 concepts rose. Injection shifts the
answer before any output exists to read, so the free-generation result is not explained by
the model reading its own steered text. That was our hypothesis and it is wrong.

## What it did not settle, and why this addendum exists

**The shift may be sensitivity to being perturbed rather than to the concept.** Nothing so
far compares a real concept vector against any other vector of the same size.

**The framing comparison is unusable as run.** The neutral prompt says "Read the following
situation" when no situation follows, and it drops the preamble, leaving a baseline of
0.671 against the introspective prompt's 0.00003. A null with no headroom is not evidence.

## New conditions

| Condition | Vector | Purpose |
|---|---|---|
| real | v_c, as before | reference |
| random | Gaussian, rescaled to the same norm as v_c | destroys content, preserves magnitude |
| shuffle | coordinates of v_c permuted, seed 0 | destroys direction, preserves magnitude and marginals |

`neutral_matched` replaces the broken neutral prompt: same preamble structure, same stated
base rate, never mentions the model, its mind, its activations, or injection.

## Criteria, fixed now

**Primary.** Introspective P(YES) at alpha=6: real versus random, Wilcoxon signed-rank
paired by concept, two-sided, 0.05.

- **A1 — real significantly above random.** The shift carries concept information. This is
  the strongest available support for the detection claim, and we report it as such.
- **A2 — real indistinguishable from random.** The model is responding to the magnitude of
  a perturbation, not its content. "Detection" is then a report that something was done to
  it, which is a materially weaker claim than the one the literature defends, and the
  distinction has not been tested by anyone.
- **A3 — random significantly above real.** Report as an anomaly. Do not interpret.

**Secondary.** Introspective versus neutral_matched, real vectors, at each alpha. Now that
both prompts carry a preamble and a stated base rate, a difference is interpretable.

## Committed in advance

- Whichever way A1/A2 falls, it is the headline of the write-up.
- The `shuffle` condition is reported alongside `random` whether or not it agrees with it.
- No further scorer or prompt changes after seeing these numbers.

---

# Addendum 2 — the unit-norm protocol

**Filed 3 September 2026, before the run.**

## Why

Macar et al. (2603.21396) L2-normalise the concept vector before scaling by alpha. Every run
above used unnormalised vectors (median norm 5,002), so our alpha is on a different scale
from theirs and the "10.0% vs 10.8%" match in the results document was obtained by
sweeping, not by following their protocol. This run follows their protocol.

On the unit-norm protocol, alpha=4 is a perturbation of 4/||h|| where ||h|| is the
residual-stream norm at the read position. That norm has not been measured for Gemma3-27B
at layer 37 and is the quantity that decides whether their alpha=4 is a 1% or a 0.001%
perturbation. The script now prints it.

## Protocol

| element | value |
|---|---|
| vectors | as before, then L2-normalised (`--normalise`); random control rescaled to norm 1 |
| strengths | log-spaced: 1, 2, 4, 8, 16, 32, 64, 128, 512, 2048 |
| prompts | introspective and neutral_matched |
| readout | first-token P(YES) |
| conditions | real and random (shuffle omitted; it agreed with random at every strength above) |
| test | Wilcoxon signed-rank, paired by concept, two-sided, 0.05 |

## Criteria, fixed now

**Onset.** The smallest alpha at which mean introspective P(YES) with real vectors exceeds
0.10. Call it alpha*. If no alpha in the sweep reaches 0.10, report that the unit-norm
protocol does not produce detection at any tested strength on this model and quantisation,
and stop.

**Primary.** Real versus random at alpha* and at the next strength above it.
- A1: real above random at either → concept-specific component present on their protocol.
- A2: indistinguishable at both → the shift on their protocol is content-free.

**Secondary.** Where does alpha=4 sit relative to alpha*? If alpha* > 4 by more than one
step, their headline strength produces no first-token shift in our setup, and the paper
says so rather than claiming a reproduction.

**Committed.** The log-spaced sweep is the whole design. No fine sweep is added after
seeing results unless the onset falls between two steps, in which case a single fine sweep
between those two steps is run and reported as such.

---

# Addendum 3 — extending the unit-norm sweep upward

**Filed 7 September 2026, after the C21 onset analysis, before the extension run.**

## Why the Addendum 2 stop clause does not apply

Addendum 2 said: if no alpha in the sweep reaches mean introspective P(YES) of 0.10,
report that the unit-norm protocol does not produce detection at any tested strength
and stop. No alpha reached it — the sweep runs 0.0000 flat to alpha=512 and hits
0.0003 at alpha=2048.

That clause assumed a grid that bracketed the onset. It did not, and two measurements
say so.

**The grid was 4.9x too low.** Unnormalised vectors have median norm 5002.36, so
`alpha_unit = 5002 x alpha_unnorm`. The top of the sweep, alpha_unit = 2048, is
alpha_unnorm = 0.409. The lowest unnormalised strength that produced any effect was
alpha_unnorm = 2, which is alpha_unit = 10,005.

**The signal is switching on at the top of the grid.** alpha=2048 against alpha=0,
introspective framing, Wilcoxon signed-rank paired by concept: W = 432,
p = 3.0e-06, rose on 27 of 30 concepts. Monotone from alpha=128 upward.

Reporting a null here would be a false negative from an unswept parameter. Study 3
already made that error once, at alpha=8 in the free-generation run, and the fix was
the same: sweep the range the effect actually occupies.

## What this addendum authorises, and what it does not

**Authorised:** one upward extension of the grid. It is an extension, not the "single
fine sweep between two steps" that Addendum 2 permits, and it will be labelled as an
extension wherever it is reported.

**Not authorised:** any change to the readout, the scorer, the prompts, the concept
set, the pairing, or the test. Those stay exactly as filed.

## Protocol

| element | value |
|---|---|
| vectors | as Addendum 2: L2-normalised; random control rescaled to norm 1 |
| strengths | 2048, 4096, 8192, 16384, 32768 (spanning alpha_unnorm 0.41 to 6.6) |
| conditions | real first; then random at alpha* and the next step up |
| everything else | unchanged from Addendum 2 |

The strengths are chosen to bracket the entire range over which the unnormalised runs
showed the effect appear (alpha_unnorm 2) and saturate (alpha_unnorm 6), so the grid
cannot fall short a second time.

## Criteria, fixed now

**Onset.** As Addendum 2: the smallest alpha at which mean introspective P(YES) with
real vectors exceeds 0.10. Call it alpha*.

- If alpha* is found, run the random arm at alpha* and one step above, and apply
  Addendum 2's A1/A2/A3 criteria unchanged. That is the primary test and it remains
  unrun.
- If no alpha up to 32,768 reaches 0.10 — which would contradict the scale arithmetic,
  since alpha_unit 32,768 is alpha_unnorm 6.6 and that strength demonstrably works —
  then the discrepancy is in the injection path under `--normalise`, not in the model.
  Debug the code; do not report a null.

**Secondary, and now the more consequential question.** Record the median
residual-stream norm at the read position, which version 2026-09-07a writes to a
`.config.json` sidecar rather than printing. Then:

- If their alpha=4 on a unit vector is a perturbation of order 1e-4 of that norm, the
  literal unit-norm reading of their protocol is untenable and we say so: the likely
  convention is scaling relative to activation scale, under which our unnormalised
  runs are the closer match to their protocol.
- If it is of order 1e-1 or larger, the literal reading stands and their effect is
  produced by a much smaller perturbation than ours.

**Committed in advance.** Whichever of those two the norm implies is reported, and
section 4.1 of the paper is written to match it rather than to preserve the current
draft's concession. No further extension of the grid without a further addendum.


---

# Addendum 4 - the validated-steering window

**Filed 8 September 2026, before the run.**

## Why this exists

C45/C46 found real below random significantly at 5%, 10% and 20% of the residual norm,
and **not** at 40% (neutral p=0.16, introspective p=0.58).

C48 then measured where the vectors actually steer. Semantic steering against a 3/30
baseline: **2/30 at 10%, 5/30 at 20%, 10/30 at 40%, 12/30 at 76%**.

So the significant cells sit where steering is at or barely above chance, and the one
cell where steering is properly demonstrated is null. That tension is the reason for this
run and it was not anticipated when Addendum 3 was written.

Two readings, and the existing data cannot separate them:

- **A.** The effect is real and the 40% cell is underpowered at n=30.
- **B.** The effect is an artifact of strengths too weak to do anything, and it correctly
  disappears once the perturbation is large enough to matter.

## Protocol (frozen)

| element | value |
|---|---|
| model / layer | Gemma-3-27B-it, 4-bit NF4, fp32 compute, layer 37 |
| vectors | `--vector-pos concept` (default since 2026-09-07e), L2-normalised |
| strengths | `--alpha-frac 0 0.30 0.40 0.50 0.60` of the 36,245 concept-token norm |
| conditions | real (`none`), `random`, `span` |
| framings | introspective and neutral_matched |
| readout | first-token P(YES) |
| test | Wilcoxon signed-rank, paired by concept, two-sided, 0.05 |

The window is 30-60%. The floor is set by C48's steering evidence; the ceiling because
coherence was already 27/30 at 40% and 14/30 at 76%, and a degraded model's logits are
not worth reading. **60% is the last strength reported regardless of what is seen.**

## Criteria, fixed now

**Primary.** Real versus random, neutral framing, **pooled across 30-60%** (the four
non-zero strengths, paired by concept within strength).

- **B1 - real significantly below random.** The suppression survives into the validated
  window. The C45/C46 result stands and the 40% null was power. This is then the
  headline and the 5-20% cells become supporting evidence rather than the claim.
- **B2 - no significant difference.** The effect does not survive where the instrument is
  demonstrably working. **The C45/C46 headline is then reported as confined to strengths
  at which the vectors do not measurably steer, which is a substantial weakening**, and
  Paper A's central claim becomes the readout-dependence and the framing results rather
  than A3.
- **B3 - real significantly above random.** Report as an anomaly. Do not interpret.

**Secondary.**

1. `span` versus `random` in this window, testing whether C49's refutation of the
   off-manifold account holds where steering is validated.
2. Real versus the alpha=0 baseline under the neutral prompt, testing whether the
   below-baseline suppression seen at 1-10% persists.
3. Per-strength tests, reported alongside the pooled primary and labelled secondary.

## Committed in advance

- The pooled neutral comparison is primary. Introspective is reported but not used for
  the verdict: its values in this range run 0.01-0.20 and the low end all means
  near-certain NO.
- Whichever of B1/B2/B3 lands is reported as the headline, and **B2 is reported as
  prominently as B1**. A result that weakens our own prior claim gets the same billing.
- No strengths added after seeing output. If the effect appears only at an untested
  strength, that is future work and is labelled as such.
- C48's steering numbers are quoted next to any claim from this run, so nobody reads a
  detection figure without knowing whether the vectors were doing anything there.
