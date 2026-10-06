# Pre-registration: S-2, do the standard vector health checks detect dead vectors?

**Filed 7 October 2026, before any S-2 run.** Paper 2, Part A. Script:
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
