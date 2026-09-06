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
