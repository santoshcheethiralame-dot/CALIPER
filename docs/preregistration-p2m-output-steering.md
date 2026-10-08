# Pre-registration: P2-M, is the inverted P(YES) check output steering?

**Filed 9 October 2026, before this analysis has been run on any data.** The S-2 sessions it
uses already exist (Qwen2.5-3B, Qwen2.5-7B, Gemma-3-4B on the alpha-frac grid); their vectors
and P(YES) shifts have been analysed for S-2's own endpoints, not for this question. As with
C21, the criterion is filed before the analysis, and the paper says so. Paper 2.

## Question

On both Qwen models, vectors that steer nothing (dead) moved first-token P(YES) more than
vectors that steer (live): the "significant P(YES) shift" check is inverted (AUC 0.367 and
0.353).
APERTURE's thesis is that self-report answers are confounded by output steering. The simplest
version: a vector moves P(YES) because its direct path to the unembedding favours YES over NO,
whatever concept it carries. If dead vectors carry more of that component, the inversion is
output steering.

## Statistic

For each real-arm vector v (unit-normalised, as injected), its **logit-lens answer score**:

    a(v) = (mean_{t in YES} u_t - mean_{t in NO} u_t) . (gamma * v)

- u_t is the unembedding row of token t.
- YES and NO are the first-token id sets the script uses (`yes_no_ids`: casing and
  leading-space variants).
- gamma is the final norm's gain, in the model's own convention (Gemma: 1 + weight).

Computed offline from the released weights (unembedding and final norm only).

## Data

- **Primary:** S-2 Qwen2.5-3B and Qwen2.5-7B, 90 real-arm vectors each.
- **Secondary:** S-2 Gemma-3-4B on the KL-calibrated grid when it lands. The alpha-frac Gemma
  session failed its manipulation check and is reported only descriptively.

## Primary endpoint and prediction

Per model, Spearman correlation between a(v) and the vector's P(YES) shift statistic (the paired
t used by `analyse_s2.py`), with a bootstrap 95% CI over vectors (2,000 resamples, seed 0).
**Prediction: rho > 0 with the CI excluding 0, on both Qwen models.**

## Secondary

1. AUC of a(v) at telling live from dead (orientation: live > dead). Prediction: below 0.5
   (dead vectors carry more of the answer direction), CI excluding 0.5, on both Qwen models.
2. The P(YES)-shift AUC for live vs dead after regressing out a(v) (rank residuals),
   descriptive: does the inversion shrink towards 0.5?
3. By arm, descriptive.

## If it fails

If rho's CI includes 0 on either Qwen model, the direct-path account is not supported. Paper 2
then reports the inversion without a mechanism and lists indirect paths (later layers, the
template-tail position's role in the answer) as untested.

## Not done

- No change to the token sets, the norm convention or the statistic after the first output.
- A logit-lens score is a direct-path approximation; the paper says so.
