# Pre-registration: P2-L, does logit-lens accessibility tell live vectors from dead ones?

**Filed 9 October 2026, before this analysis has been run.** The vectors and their live labels
exist (S-2, S-1); they have not been scored by this statistic. As with P2-M, the criterion is
filed before the analysis, and the paper says so. Paper 2 (`docs/paper2-outline.md` C1).

## Question

Billa (2604.15557) reports that logit-lens accessibility predicts where steering vectors
succeed (rho 0.86-0.91). The standard health checks fail at liveness (S-2). If an accessibility
check works where they fail, Paper 2 recommends it; if it fails too, C1 is stronger.

## Statistic

For each real-arm vector v (unit, as injected), its **concept accessibility**:

    logits = W_U (gamma * v),   z(v) = (logits[t_c] - mean(logits)) / sd(logits)

where t_c is the first token of " {concept}" (leading space), W_U the unembedding and gamma the
final norm's gain in the model's own convention (Gemma: 1 + weight). Computed offline from the
released weights (unembedding and final norm only), as in P2-M. Orientation fixed: higher z
means more accessible, predicted live.

## Data (live labels exactly as filed for each study)

| session | labels | scored if |
|---|---|---|
| S-2 Qwen2.5-3B | S-2 gate | >= 5 live and >= 5 dead |
| S-2 Qwen2.5-7B | S-2 gate | same |
| S-2 Gemma-3-4B, KL grid | S-2 gate (0.5 nat) | same |
| S-1 Gemma-3-12B, 4-bit | S-1 gate (0.5 nat) | same |
| S-1 Gemma-3-27B, 4-bit | S-1 gate (0.5 nat): 3 live | not scored (reported) |
| S-2 Gemma-3-4B, alpha-frac (failed manipulation check) | S-2 gate | descriptive only |

## Primary endpoint and prediction

Per scored session, AUC of z(v) at telling live from dead (live > dead), stratified bootstrap
95% CI (2,000 resamples, seed 0). **Prediction: the CI's lower bound is above 0.5 in at least
three of the four scored sessions.**

## Secondary

1. Within-arm AUCs, descriptive.
2. Paired bootstrap difference against the existing logit-steering check (APERTURE's delta log
   P(concept) with a forward pass), per session.
3. The S-1 27B session at the 5-nat labels, and both Gemma-27B cells, descriptive.

## If it fails

Paper 2 reports that the best published steerability predictor, applied as a per-vector health
check, does not identify live vectors either, and C1 says "no available check" rather than "no
standard check".

## Not done

- No change to the token rule (first token of the space-prefixed concept), the z-score, or the
  orientation after the first output.
