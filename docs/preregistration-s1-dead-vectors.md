# Pre-registration: S-1, are dead vectors a read-position effect or a precision artefact?

**Filed 6 October 2026, before any S-1 run.** Paper 2, Part A. Script:
`experiments/kaggle_s3_positive_control.py` v2026-10-07a. Run sheet:
`kaggle/NEXT_SESSION_S1.md`.

## Question

C31 found that concept vectors read at the chat-template tail were inert on Qwen. They
were unit-norm, finite and moved first-token P(YES) at p = 3.7e-9, yet steered nothing.
Every run so far was in 4-bit NF4. A reviewer's first question is whether "dead vectors"
are a 4-bit artefact. S-1 crosses read position with precision and measures the one thing
a live vector must do: steer.

## Design

| factor | levels |
|---|---|
| read position / recipe | template tail (macar recipe, `--vector-pos template-tail`); concept token (macar, `--vector-pos concept`); sentence mean (`--vector-recipe aperture`) |
| precision, Gemma-3-12B-it | 4-bit NF4 with fp32 compute; 8-bit LLM.int8; unquantised fp16 |
| precision, Gemma-3-27B-it | 4-bit NF4 with fp32 compute |

- Layer: 0.6 of depth (`--layer -1`).
- Concepts: all 30.
- Unnormalised vectors, as in every run since C31.
- Stage `steer`: "Write a short story.", injection at every position, 60 new tokens.
- Grid: `--alpha-frac 0 0.25 0.5 1.0` of the measured residual norm.

**Amendment to the run plan, made now on feasibility grounds.** The plan put 27B at 4- and
8-bit and 12B at bf16. On the free tier neither works:
- 27B in 8-bit needs about 34 GB against 2×T4's 30;
- a T4 (compute capability 7.5) has no native bf16.

The precision axis therefore runs on 12B, with fp16 standing in for "unquantised". 27B keeps
one 4-bit cell, linking S-1 to C31. If fp16 overflows on 12B (`probe_finite` fails), that
cell is recorded as not runnable on free hardware. It is not replaced with fp32, which does
not fit.

## The gate and the outcome, fixed now

- **Steering pass** (per vector): `steered` is true at alpha-frac 0.5 and false at alpha 0.
  At alpha 0 the story is the same for every concept, so the baseline is "this story already
  mentions the concept or an associate". The 0.25 and 1.0 cells are reported beside it.
- **Outcome per cell:** the steering pass rate out of 30.

## Criterion and failure branch (from the run plan)

- **Criterion:** in every 12B precision cell, template-tail vectors fail the gate more often
  than concept-token vectors. Each cell is tested with an exact McNemar test on the 30
  concepts, one-sided. The criterion holds if each cell's difference is in the predicted
  direction and the three cells together give Fisher-combined p < 0.05.
- **Failure branch:** if tail vectors pass the gate unquantised (fp16) but not in 4-bit, the
  dead-vector result is a quantisation artefact, and Paper 2 reports it as that: a warning
  about quantised instruments, narrower than a warning about read position.

## Secondary, descriptive

- The sentence-mean (APERTURE) recipe's pass rate in each cell.
- Health statistics per arm, from the sidecars:
  - stability, held-out probe, logit steering check;
  - the AUC of each at predicting the gate (feeds S-2).
- Precision drift: |cos| between the same concept's vector across the three 12B precisions,
  per read position.
- Next-token KL and disclaimer rate by alpha.

## Not done

- No arm is re-run after its results are seen.
- No alpha is chosen from the data. The gate's 0.5 is fixed here.
