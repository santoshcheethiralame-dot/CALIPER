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

## Amendment 1 (7 October 2026, before any S-1 data)

**The grid used the wrong scale.** "Unnormalised vectors, as in every run since C31" and a
grid in fractions of the residual norm cannot both hold. The script multiplied the raw vector
by alpha = fraction x residual norm. S-2's first Qwen2.5-3B session showed the result:
perturbations about 56 times their stated size, with every non-zero alpha saturating next-token
KL at about 26 nats (S-2 Amendment 1).

**Resolution, script v2026-10-07b:**
- Vectors are still extracted unnormalised, as in C31.
- Under `--alpha-frac` they are unit-normalised before injection, so each perturbation is the
  stated fraction of the residual norm.
- Raw norms are kept in the per-vector health record. The C31-style absolute scale of any
  cell can therefore be reconstructed.
- The steering-pass rule, the arms and the precisions are unchanged.

## Amendment 2 (9 October 2026, before any S-1 data)

**The grid does not transfer to Gemma.** S-2's Gemma-3-4B session showed `--alpha-frac` 0.25
already saturating next-token KL (about 45 nats, random controls included) and the gate dose
leaving about 90% of generations incoherent (S-2 Amendment 3). S-1 runs only Gemma models, so
its grid, as filed, would label vectors in a broken model.

**Resolution, script v2026-10-09a:**
- Grid: `--calibrate-kl 0.05 0.5 5`, the same targets as S-2 Amendment 3, fixed there from
  content-free control rows of the Qwen sessions. No S-1 data exist.
- The calibration runs once per model and precision, so each precision cell of the 12B is
  compared at the same next-token disruption of a random direction, not at the same raw size.
  The calibrated alphas and the residual norm are both in every sidecar.
- **Steering pass:** steered at the 0.5-nat dose and not at alpha 0. The 0.05 and 5-nat cells
  are reported beside it.
- **Manipulation check:** at the 0.5-nat dose at least half of each cell's generations are
  coherent; a cell that fails is reported, not scored. The criterion, the failure branch, the
  arms and the precisions are unchanged.
- "No alpha is chosen from the data" still holds: the targets were fixed before any S-1 session,
  from another study's control rows.

## Amendment 3 (9 October 2026, before any S-1 data): the published operating point as a reference cell

**Why.** The released code of the introspection-mechanisms study (read 9 Oct) builds each
concept vector at the last token of the chat-templated "Tell me about {concept}" prompt minus
the mean over 100 baseline words, unnormalised, and injects strength x vector from the token
before "Trial" through every generated token. That read position is this study's template-tail
arm. The 27B session therefore gains one cell run exactly as released, so the paper can place
the published operating point on its own scale.

**Cell S-1M** (Gemma-3-27B-it, 4-bit NF4 with fp32 compute, layer 37):
- vectors by the released recipe (`--vector-recipe macar-release`: last-token read, 100
  baseline words, no normalisation);
- strengths 0, 4 and 8 (`--alphas 0 4 8`, raw multipliers, as released);
- injection from the token before "Trial" through generation (`--inject-from trial`);
- stage `steer`, the same 30 concepts.

**Outcome, descriptive (no prediction):** the steering-gate pass rate at strength 4 and 8, and
the median next-token KL of each strength, placed on the KL scale of the calibrated cells.
Precision differs from the release (4-bit here, bf16 there), and the paper says so.

**Engineering first (S-0b).** The two flags above are added to the script with tests before
the 27B session. The 12B session does not need them and can run first. Nothing else in this
document changes.

**Clarification, same day, before any data.** The steer stage injects at every position,
prompt and decode, so it already covers the released window; `--inject-from trial` matters
only where the prompt contains "Trial". S-1M therefore runs two stages with the released
vectors and strengths: `steer` (the gate, as above) and `forced` with `--inject-from trial`
(first-token P(YES) and next-token KL under the released injection window). Both are
descriptive. Script v2026-10-09b; tests in `tests/test_s3_script.py`.

## Amendment 4 (9 October 2026, before any S-1 data): the 5-nat dose as a co-primary

**Why.** S-2's Gemma-3-4B session on the KL-calibrated grid (notebook, 9 Oct) passed its
manipulation check but left the gate dose gentle for real vectors: concept vectors moved
next-token KL by 0.03 nats at the 0.5-nat-calibrated dose (random directions 0.11), and only 9
of 90 vectors steered there. At the 5-nat dose 36 of 90 did, with every generation of the real
arms still coherent. Random directions are far more disruptive than real ones at the same norm,
so the calibration's gate is a gentle dose for the vectors S-1 tests. S-1 runs the same family
on the same grid, and its McNemar test would have little power at the gate alone. Nothing from
S-1 has been seen.

**Change.**
- The criterion (tail vectors fail the steering pass more often than concept-token vectors in
  every 12B precision cell; each cell an exact one-sided McNemar on the 30 concepts; the cells
  Fisher-combined) is evaluated **at two doses**: the 0.5-nat gate, as filed, and the 5-nat
  dose. A vector passes at a dose if it is steered there and not at alpha 0.
- Two co-primaries split alpha: the Fisher-combined p must be below **0.025** at a dose for the
  criterion to hold at that dose. Both doses are reported whatever happens.
- The manipulation check applies at each dose separately: at least half of each cell's
  generations coherent.
- The failure branch is read at each dose: tail vectors passing in fp16 but not in 4-bit is a
  quantisation artefact.
- P2-G's secondary on S-1 keeps the 0.5-nat labels its own filing names; the 5-nat labels are
  reported beside them, descriptively.

**Analysis code.** `experiments/analyse_s1.py`, written and tested on synthetic files before any
S-1 zip is opened.
