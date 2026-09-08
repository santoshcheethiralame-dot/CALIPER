# Pre-registration — Study 2 P1b, planted **concept** direction recovery

**Filed 8 September 2026, before the run. Supersedes `preregistration-s2-p1.md`, which
governed a run now recorded as void (C54/C55).**

## Why the first attempt was void, stated before anything new is run

P1 planted **random unit directions**. Recovery sat *below* the null at every strength
(medians 0.0064 / 0.0086 / 0.0117 against nulls of 0.0128 / 0.0117 / 0.0100), above null
on 8 of 24 rows, flat across a 4× strength range, and P2 showed the same flatness at
every depth from +0 to +18 with 16 of 40 above null — exactly chance.

The previous filing permits reporting that as a finding. It should not be, and this
filing records why: **a random direction has no natural representation in the model**, so
the text it produces carries no consistent signal for difference-of-means to recover. The
persona pipeline works precisely because a *trait* direction makes the model write
trait-consistent text. My earlier justification — that a random plant removes the
confound between "does the pipeline recover the plant" and "does the model represent the
trait" — removed the confound by removing the mechanism the method depends on.

Total absence of signal at the plant layer *and* at every depth downstream is what a
mis-specified test looks like, not a failing instrument.

## What changes

The plant is now a **real concept vector** — one of the 30 already built by the pipeline,
which C48 showed demonstrably steers Gemma at 40% of the residual norm (semantic steering
10/30 against a 3/30 baseline, coherence 27/30).

**This makes P1b a necessary-condition test, not a general validation, and the write-up
must say so.** A concept direction is the *easy* case: the model already represents it.
Passing therefore does not establish that the pipeline recovers arbitrary planted traits.
**Failing establishes that it recovers nothing at all**, which is the question worth
asking first.

## Protocol (frozen)

| element | value |
|---|---|
| model / layer | Gemma-3-27B-it, 4-bit NF4, fp32 compute, plant layer 37 of 62 |
| plants | the **first 8 concepts** (elephant, spider, eagle, dolphin, volcano, desert, library, harbor), L2-normalised, read at `--vector-pos concept` |
| strengths | `--alpha-frac 0.20 0.40 0.60` of the 36,245 concept-token norm; **0.40 is the primary**, chosen because C48 measured steering there |
| injection | `span="all"` |
| elicitation | the same 16 neutral prompts, identical across conditions |
| generation | 40 new tokens, greedy |
| extraction | generate under injection, **re-read the text with no injection**, mean over response tokens, difference against the shared no-injection baseline |
| recovery | `abs(cos(diff, v_c))` for the planted concept `c` |
| **null A (primary)** | `abs(cos(diff, v_c'))` for a *different* concept `c'` — conservative, since concept vectors share structure |
| null B | `abs(cos(diff, random unit))` — reported alongside, free |

## The manipulation check — new, and it gates everything

**The void run could not distinguish "extraction failed" from "the plant never reached
the text", because the text was never logged.** That is fixed: every generation is
written to the output file.

Before any recovery number is interpreted, report the **semantic steering rate** of the
injected generations — the fraction containing the concept word or a concept-specific
associate — against the α=0 baseline rate, using the same scorer as C48.

- **If steering at 40% is not clearly above baseline, the run is VOID again**, reported
  as such, and no recovery number is quoted. The instrument was not working and the
  question was not asked.
- Only if steering is above baseline does the primary endpoint mean anything.

This is the check whose absence made the first run uninterpretable. It is the single most
important addition here.

## Primary endpoint

Median `abs(cos(diff, v_c))` across the 8 planted concepts at **α = 40% of norm**,
against median null A.

**Criterion, fixed now: recovery must exceed null A on at least 6 of 8 plants AND the
median recovery must exceed 0.30.**

- **PASS.** Difference-of-means recovers a planted concept direction from re-read text.
  The pipeline works in the easy case. P2b (depth curve) proceeds, and the write-up states
  plainly that this is a necessary condition and not a validation for arbitrary traits.
- **FAIL, recovery above null but median below 0.30.** Partial recovery. Report the value
  with its null, run the depth curve, and describe the pipeline as weakly recovering.
- **FAIL, recovery at or below null, with the manipulation check PASSED.** This is the
  substantive negative: the plant demonstrably reached the text, and the extraction
  pipeline still did not recover it. **That is a real finding about a deployed method**
  and it is reported as the result for this study.
- **Manipulation check failed.** Void. Reported as void, not as a finding.

## Secondary

1. Recovery at 20% and 60%, reported per-strength. 60% is included because C48 found
   stronger steering there, and excluded from the primary because coherence had fallen to
   14/30 by 76% and 60% is on that slope.
2. Null B beside null A at every cell.
3. The steering rate per strength, so recovery can be read against how much the plant
   actually moved the text.

## Committed in advance

- **Every recovery number is reported beside null A.** A recovery figure without its null
  is not a result.
- **The manipulation check is reported first, before any recovery number**, in the paper
  and in the notebook.
- If this run is void for a third reason not anticipated here, that is reported as two
  void attempts and the study is reconsidered rather than re-run a fourth time on a new
  guess.
- No change to plants, prompts, strengths, nulls, or the criterion after seeing output.
- The necessary-condition framing appears wherever a PASS is reported. A pass is not
  evidence that persona extraction recovers arbitrary traits.
