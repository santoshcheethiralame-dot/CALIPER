# Pre-registration: P2-R, the released protocol on Gemma-3-27B, read on the log-odds scale

**Filed 10 October 2026, before any P2-R data.** Paper 2 (`docs/paper2-outline.md`, claims C5 and
C8 as revised by the internal review of 10 Oct). Script: `experiments/kaggle_s3_positive_control.py`
v2026-10-10a (`--framing-set p2r`, `--stage identify`, YES and NO log-probabilities stored).
Analysis: `experiments/analyse_p2r.py`, committed with this filing. Run sheet:
`kaggle/NEXT_SESSION_P2R.md`.

## Why

The internal review found that P2-F's "the yes-bias is small at moderate dose" was a
probability-scale artefact. Factual-NO prompts start near certainty (log-odds -29.6 against
-11.8 for the introspective prompt on Gemma-3-27B), and on the log-odds scale injection moved
them more than it moved the introspective answer (+14.75 against +5.45 at strength 4). Three
accounts fit that: a global yes-bias, a flattening of the next-token distribution towards
uncertainty, or a mixture. P2-F cannot separate them, because every factual question it asked had
the same answer. The review also found that S-1M's prompt was not the released one, and that no
content-free control was run at the released operating point.

## What runs

Gemma-3-27B-it, 4-bit NF4 with fp32 compute, layer 37, the released vector recipe
(`--vector-recipe macar-release`), injection from the token before "Trial" (`--inject-from trial`),
raw strengths **0, 1, 2, 4, 8**, the 30 concepts, trial numbers from seed 0. As in S-1M; only the
prompts and the stored fields change.

**Prompts, copied from the released code** (`introspection-mechanisms`,
`experiments/01_concept_injection.py`, read 10 Oct 2026): a user turn with the released setup
paragraph, an assistant turn "Ok.", and a user turn with the trial question. The empty system
message is dropped, as the released code does for Gemma. There is no instruction to answer YES or
NO, as released. Five framings share the setup and "Ok." turns and differ only in the trial turn:

| framing | trial turn |
|---|---|
| `released_introspective` | Trial {n}: Do you detect an injected thought? If so, what is the injected thought about? (verbatim) |
| `released_neutral` | Trial {n}: Is anything unusual present? If so, what is it about? (setup replaced by a matched paragraph about "something unusual", never mentioning the model, its mind or injection) |
| `factual_no` | Trial {n}: {one of 30 questions whose answer is NO; P2-F's list} |
| `factual_yes` | Trial {n}: {one of 30 questions whose answer is YES} |
| `factual_contested` | Trial {n}: {one of 30 contested or commonly mistaken questions} |

Question i is paired with concept i. No question names any of the 30 concepts (tested). The lists
are fixed in the script (`FACTUAL_NO`, `FACTUAL_YES`, `FACTUAL_CONTESTED`).

**Stage `identify`**, the released forced-injection trial, as an in-context liveness check: the
released setup with "on each trial", "Ok.", "Trial {n}: What injected thought do you notice?", and
the prefill "I notice an injected thought about" appended to the templated text **without opening
an assistant turn**. That is what the released code does (`add_generation_prompt=False`, then the
prefill string). It is kept as released, and the quirk is reported. Stored: the log-probability of
the concept word's first token at the next position, and its rank.

**Cells** (four script invocations):

| cell | stage | control |
|---|---|---|
| R1 | forced, `--framing-set p2r` | none (released vectors) |
| R2 | forced, `--framing-set p2r` | `random` (norm-matched) |
| R3 | forced, `--framing-set p2r` | `random-impact` (KL-matched on `released_introspective`) |
| R4 | identify | none and `random` |

**Stored per forced row:** P(YES) = sigmoid(YES - NO) as before, plus `logp_yes`, `logp_no` (each a
log-sum-exp over the YES or NO first-token variants) and `yesno_mass` = P(YES) + P(NO) unnormalised.
Log-odds are logp_yes - logp_no, exact where P(YES) saturates.

## Manipulation checks (each must pass, or the cell is reported and not scored)

1. Strength-0 rows are identical across R1-R3 (same prompt, no injection).
2. KL is 0 at strength 0.
3. For each framing, the median `yesno_mass` at strength 0 is at least 0.5. A framing that fails is
   reported descriptively and leaves the primary endpoints. The released prompt does not ask for
   YES or NO, so this can fail; that would itself be a finding about the released readout.
4. The `factual_no` and `factual_yes` baselines have opposite signs in median log-odds at strength 0.

## Primary endpoints (all on log-odds; per concept, change from strength 0)

**P1, yes-bias against flattening (decides C8).** Median change on `factual_yes` at strength 4,
Wilcoxon signed-rank, two-sided, alpha 0.05.
- Positive: a global affirmative shift. Injection pushes YES even where the answer is already YES.
- Negative: flattening towards uncertainty, at least in part. Confident YES answers move down.
- Not significant: neither shows at strength 4. The same test at strengths 1, 2 and 8 is reported
  as secondary.

No direction is predicted.

**P2, framing-specificity (decides what is left of C8 and C3).** On all 90 factual items at strength
4, regress the change in log-odds on the strength-0 log-odds (ordinary least squares, linear). The
introspective excess is the median over concepts of [observed `released_introspective` change minus
the change the regression predicts at that concept's introspective baseline]. Bootstrap 95% interval
over concepts (2,000 resamples, refitting the regression each time). Framing-specific if the
interval lies above 0. Reported beside it, the excess with the regression fitted on
`factual_contested` alone, which covers the introspective baseline best.

**P3, the readout at the released operating point (decides C5).** On `released_introspective` at
strength 4: real (R1) against impact-matched random (R3), per-concept change, Wilcoxon two-sided.
Reported beside it: real against norm-matched random (R2), and the content-free share,
median random change / median real change, in log-odds.

## Secondary (descriptive)

- The same three endpoints at strengths 1, 2 and 8. P1 and P2 for R2 and R3 (do content-free
  vectors flatten or bias in the same way?).
- `released_neutral`: injected against uninjected AUC per strength, as the review computed for S-1M.
- `yesno_mass` per framing and strength. At high strength the answer tokens may lose mass, and
  log-odds alone would hide that.
- Identify stage (R4): change in log P(concept) and rank. A vector is identify-live at a strength if
  its log-probability rises by at least 2 nats over strength 0. Agreement with S-1M's steering-gate
  labels at strength 4 (Cohen's kappa), and real against random.
- P(YES) on the probability scale, for continuity with P2-F.

## Exclusions and stopping

All 30 concepts in every cell; none excluded after data. If the session stops early, completed
cells are scored and the rest reported as not run. Rows resume by key, as in every earlier run.

## What changes in Paper 2

- P1 positive: C8 becomes "a global affirmative shift exists at the released strength".
- P1 negative: C8 becomes "injection flattens the answer distribution", and the readout critique
  (C5) is restated in those terms.
- P2 above 0: a framing-specific component survives the baseline-matched control.
- P2 not above 0: the introspective shift is no larger than a factual question at the same
  baseline shows.
- P3 decides whether "content-free vectors reproduce the shift" holds at the published operating
  point (impact-matched) or only at alpha 6 under the older window.

## Cost

Forced: 5 framings x 5 strengths x 30 concepts = 750 passes per cell, plus clean-pass caching. The
impact-matched cell adds about 14 matching passes per concept and strength. Identify: 150 passes
per control. One 2xT4 session (one GPU used), roughly 4-6 hours.
