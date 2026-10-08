# Pre-registration: S-2, do the standard vector health checks detect dead vectors?

**Filed 6 October 2026, before any S-2 run.** Paper 2, Part A. Script:
`experiments/kaggle_s3_positive_control.py` v2026-10-07a. Run sheet:
`kaggle/NEXT_SESSION_S2.md`. The judge-scored readout waits for S-12 (kappa ≥ 0.6) and is
not part of this filing.

## Question

Papers that inject concept vectors vouch for them with checks: unit norm, finiteness,
distinctness from other vectors, a significant shift in first-token P(YES), and sometimes
stability or a probe. C31's dead vectors passed every one. S-2 asks, across many vectors and
three small models, how well each check separates vectors that do something (live) from
vectors that do not (dead). It is Paper 1's question asked of Paper 2's instrument: a
reliability check calibrated against a reference.

## Design

- **Models:**
  - Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct, unquantised fp16 (fp32 if `probe_finite`
    fails);
  - Gemma-3-4B-it, unquantised fp32 (Gemma overflows in fp16).
- **Layer:** 0.6 of depth. **Concepts:** all 30. **Trial seed:** 0.
- **Vector arms** (each a separate run, separate stem):

  | arm | flags | expected class |
  |---|---|---|
  | concept token | `--vector-pos concept` | mostly live |
  | template tail | `--vector-pos template-tail` | mostly dead (C31) |
  | sentence mean | `--vector-recipe aperture` | unknown |
  | random, norm-matched | `--control random` | content-free |
  | random, impact-matched | `--control random-impact` | content-free, KL-matched |
  | shuffled | `--control shuffle` | content-free |
  | span (on-manifold) | `--control span` | no single concept |

- **Stages per arm:**
  - `steer` (the gate);
  - `forced` (first-token P(YES), both framings);
  - `framing` (generated text, both framings).

  All use `--alpha-frac 0 0.25 0.5 1.0`.

## The reference label, fixed now

A real-vector arm's vector is **live** if it passes the S-1 steering gate (steered at
alpha-frac 0.5, not at 0). Otherwise it is **dead**. The content-free arms are not labelled
and enter only the secondary analyses.

## Primary endpoint

Pool the three real-vector arms per model (up to 90 vectors). For each health statistic,
take its AUC at telling live from dead:
- norm;
- max |cos| to another vector (low = distinct);
- stability;
- held-out probe;
- logit steering delta;
- the P(YES) shift: the paired t statistic of P(YES) at alpha-frac 0.5 against alpha 0,
  across the two framings, per vector. This is the "significant P(YES) shift" check.

**Prediction:**
- norm, distinctness and the P(YES) shift score near 0.5 (95% CI includes 0.5);
- the logit steering check scores at least 0.8.

Stability and the probe have no stated prediction. A model with fewer than 5 live or 5 dead
vectors is reported but not scored.

## Secondary, descriptive or as stated

1. Dose-response of detection, P(YES) and KL by arm.
2. Readout disagreement on identical trials: first-token P(YES) > 0.5 against rule-scored
   generated YES, as a 2×2 per arm.
3. **TOST:** at alpha-frac 0.5, is the random-impact arm's P(YES) equivalent to the live
   vectors' within ±0.10?
4. Disclaimer rate against alpha for live and random arms. Tests the A-R3 hypothesis that
   the perturbation, not the concept, switches the disclaimer off.
5. Off-list YES and leakage into NO, per framing.
6. Emotion concepts reported separately (the affect confound).

## Not done

- No arm, alpha or concept subset is chosen after seeing results.
- Judge scores do not enter any number here until S-12 passes.

## Amendment 1 (7 October 2026, before any S-2 data on the fixed script)

**What failed.** The first Qwen2.5-3B session ran script v2026-10-07a. It failed its
manipulation check. `--alpha-frac` set alpha = fraction x residual norm, but the injection
multiplied the raw concept vector (median norm 56) rather than a unit vector. Every
non-zero alpha was therefore about 56 times its stated size: 0.25 meant about 14 times the
residual norm. Every arm, including the random and shuffled controls, sat at a next-token KL
of about 26 nats.

**What was looked at.** Only the manipulation-check fields: KL by arm and alpha, the
impact-match records and file integrity. No endpoint of this prereg was computed. That
session's files are archived as a failed run and enter no analysis.

**The fix (v2026-10-07b).** Under `--alpha-frac` the injected direction is unit-normalised, so
the perturbation norm is the stated fraction of the residual norm. This is the grid's meaning
as written above, and APERTURE's convention (alpha x sigma x unit direction).
- The health checks still score the vectors as extracted. "Norm" in the primary endpoint is
  the raw norm.
- The controls are norm-matched to the unit vector.
- A test now checks that the injected vectors have unit norm.

Everything else in this document is unchanged. Qwen2.5-3B is re-run on v2026-10-07b.

## Amendment 2 (8 October 2026, before any re-run of Qwen2.5-7B)

**What failed.** The first Qwen2.5-7B session (script v2026-10-07b, fp16, unquantised) was
numerically broken at baseline:
- with no injection, greedy generations were mostly "!" (token 0), the signature of NaN logits;
- 238 of 420 alpha-0 forced-choice P(YES) values were NaN;
- next-token KL was NaN in up to 300 of 420 injected forced-choice cells.

The finiteness probe had checked one clean forward pass and passed. When `--compute-dtype` was
given explicitly, a failed probe was not acted on. Only file integrity and these NaN counts
were looked at; no endpoint was computed. The files are archived as a failed run.

**The fix.**
- Script v2026-10-08a also requires finite logits through a short uninjected generation.
- It refuses an explicit precision that fails the probe, rather than running.

**Qwen2.5-7B is re-run with 4-bit NF4 weights and fp32 compute.**
- fp32 unquantised needs about 30.5 GB for weights, more than two T4s hold.
- A T4 has no bf16.
- This is the same configuration as S-1's 4-bit cells.

Qwen2.5-3B (fp16, verified finite: no NaN values and no "!!!" in 2,520 generations) and
Gemma-3-4B (fp32) are unchanged. The per-model precision is reported with the results.

## Amendment 3 (9 October 2026, before any S-2 session on the new grid)

**What failed.** The Gemma-3-4B session (script v2026-10-08a, fp32) ran cleanly but failed its
manipulation check. At every non-zero alpha-frac, every arm, the content-free controls included,
sat at a median next-token KL of 43 to 75 nats, and at the gate dose (0.5) about 90% of the
steer-stage generations were incoherent. Qwen2.5-7B's concept arm sat at 0.18 / 0.36 / 5.9 nats
on the same grid.

**Why.** Alpha-frac scales by the residual norm at the read position. Gemma's is large and mostly
shared across tokens: median 31,450, against concept vectors of median norm 3,909 (12%). On
Qwen2.5-3B the ratio is 56 / 84 (67%). A fraction of the residual norm is therefore not a dose
that means the same thing on different models.

**What was looked at.** File integrity, the manipulation-check fields (KL and coherence by arm
and alpha) and, before the saturation was noticed, the primary output of `analyse_s2.py` for
that session. That session is reported as scored, next to its failed manipulation check
(notebook, 9 Oct). It is not pooled with the Qwen models.

**The fix (script v2026-10-09a, `--calibrate-kl`).**
- Each non-zero alpha is set so that a content-free probe, four seeded random unit directions,
  gives a fixed median next-token KL over three fixed prompts (the two forced-choice framings
  at trial 1 and the steer prompt). The calibration never sees a concept vector, an arm or an
  endpoint. It runs once per model, precision and layer, and every cell of that model reuses it.
- A target missed by more than 5% stops the run.
- **Targets: 0.05, 0.5 and 5 nats.** The gate target, 0.5 nats, is the geometric mean (0.44) of
  the random control's median KL at the old gate dose on the two Qwen models and two framings
  (4.6, 1.5, 0.03 and 0.20 nats), rounded. The other two sit one decade either side, close to
  the same controls' geometric means at the old 0.25 and 1.0 doses (0.015 and 7.8). These are
  content-free control rows only.
- **The gate becomes the second non-zero dose** (0.5 nats): a vector is live if steered there
  and not at alpha 0, and the P(YES) shift compares that dose with alpha 0. On the original grid
  the second non-zero dose is alpha-frac 0.5, so the rule as filed is unchanged for the Qwen
  runs. `analyse_s2.py` now reads the gate by grid position; on the three existing sessions its
  output is byte-identical to before.
- **A manipulation check, fixed now, for every calibrated session:** at the gate dose, at least
  half of the random-control steer generations are coherent. A model that fails it is reported
  and not scored.

**Runs.**
- **Gemma-3-4B is re-run on the calibrated grid** (fp32, all seven arms and three stages). That
  session is the Gemma-3-4B result for the primary endpoint and the predictions, unchanged.
- The Qwen sessions passed their manipulation check (graded KL) and stand as filed. A Qwen re-run
  on the calibrated grid is optional and, if run, is reported as a sensitivity analysis.
