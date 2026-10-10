# Pre-registration: P2-Q, the P2-R design on Qwen2.5-7B

**Filed 10 October 2026, before any P2-Q data.** Paper 2, claims C5 and C8 on a second model family.
The design, endpoints and scorer are P2-R's (`docs/preregistration-p2r-released-protocol.md`);
only the model, the vectors and the doses change. Script `experiments/kaggle_s3_positive_control.py`
v2026-10-10a. Scorer `experiments/analyse_p2r.py --preset qwen7b-concept` / `--preset qwen7b-tail`
(committed with this filing). Run sheet `kaggle/NEXT_SESSION_P2Q.md`.

## Why a second family

P2-R decides C8 (yes-bias against flattening) at the released operating point on Gemma-3-27B.
P2-F's probability-scale reading failed on both Gemma-3-27B and Qwen2.5-7B; the corrected
log-odds claim needs both. Qwen is also where dead vectors and the inverted P(YES) check live (C2,
C3), so the framing-specificity endpoint (P2) is most informative there.

## What runs

Qwen2.5-7B-Instruct, 4-bit NF4, fp32 compute, layer 17, as in S-2 and P2-F. Doses
`--alpha-frac 0 0.25 0.5 1.0`. Injection from the token before "Trial" (`--inject-from trial`), as
in P2-R, so every framing is injected over the same span. This differs from S-2, which injected
at every prompt position; the review flagged that difference in P2-F's Qwen cell. Framings: P2-R's
five (`--framing-set p2r`): the released introspective and matched neutral prompts and the
factual-NO, factual-YES and contested questions, all in the released two-turn frame.

| cell | vectors | stage | control |
|---|---|---|---|
| Q1 | concept token (`--vector-pos concept`) | forced | none |
| Q2 | template tail (`--vector-pos template-tail`) | forced | none |
| Q3 | concept-token norms | forced | `random` (norm-matched) |
| Q4 | concept-token KL | forced | `random-impact` (KL-matched on the introspective prompt) |
| Q5 | concept token | identify | none |
| Q6 | template tail | identify | none |

The content-free cells are built on the concept vectors, so they are matched to the concept arm.
For the tail arm they serve as an unmatched reference only, which is stated wherever they appear.

## Endpoints

P2-R's, with **0.5 of the residual norm** as the primary dose, the dose at which S-2 gated
liveness and P2-F scored its Qwen secondary.

- **P1** (yes-bias against flattening): median log-odds change on `factual_yes` at 0.5, Wilcoxon
  two-sided. Concept arm primary; tail arm secondary.
- **P2** (framing-specific excess): the introspective change minus the change predicted from the
  factual items' baselines, with a bootstrap interval. Concept arm primary; **tail arm reported
  beside it with equal prominence**, because the one framing-specific signal so far is the tail arm
  at 0.25 of the norm.
- **P3** (real against impact-matched random on the introspective prompt): concept arm only.

The manipulation checks are P2-R's: identical strength-0 rows across cells, KL 0 at strength 0,
YES+NO mass at least 0.5 per framing at strength 0, and opposite-signed factual baselines.

## Exclusions, stopping, cost

All 30 concepts, none excluded; completed cells are scored if the session stops early. Six cells
of 600 or 150 passes on a 7B 4-bit model: about 1.5-2.5 GPU-hours, the impact-matched cell being
the slowest.
