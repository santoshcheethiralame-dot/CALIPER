# Pre-registration: P2-G, do dead steering vectors lie along the direction every concept shares?

**Filed 9 October 2026, before any S-2 Gemma-3-4B session on the KL-calibrated grid and before
any S-1 session.** Paper 2. Script: `experiments/explore_dead_geometry.py`, written and run on
the three existing S-2 sessions before this filing.

## Where the hypothesis comes from (exploratory, already seen)

On the three S-2 sessions run so far, a vector's **shared share**, |cos(v, m)|^2 with m the mean
of that model's 90 real-arm unit vectors, separated live from dead vectors, with dead vectors
higher:

| model | live / 90 | AUC (live > dead) | bootstrap 95% CI |
|---|---|---|---|
| Qwen2.5-3B | 34 | 0.349 | [0.242, 0.466] |
| Qwen2.5-7B | 32 | 0.229 | [0.136, 0.332] |
| Gemma-3-4B (alpha-frac, failed manipulation check) | 41 | 0.367 | [0.252, 0.484] |

This was found after those data were seen, and the statistic overlaps with the distinctness
health check (max |cos| to another vector). It is the Paper 2 analogue of Paper 1's
low-variance share: an answer that lives where the representation does not discriminate.

## Data for the test

- **Primary:** S-2 Gemma-3-4B on the KL-calibrated grid (`s2_gemma4b_kl`, S-2 Amendment 3),
  three real arms, 90 vectors, live labels from `analyse_s2.py` exactly as filed.
- **Secondary:** each S-1 cell set with at least 5 live and 5 dead vectors (Gemma-3-12B at
  each precision, Gemma-3-27B 4-bit), pooling its three arms, live labels from S-1's own gate
  (steered at the 0.5-nat dose, not at alpha 0). The S-1 labelling code is committed before
  any S-1 zip is opened.

## Primary endpoint and prediction

- AUC of the shared share at telling live from dead (orientation fixed: live > dead), with a
  stratified bootstrap 95% CI (2,000 resamples, seed 0), as in the script.
- **Prediction:** AUC <= 0.40 and the CI's upper bound below 0.5. That is, dead vectors carry
  more of the shared direction.
- Scored only if there are at least 5 live and 5 dead vectors.

## Secondary

1. Within-arm AUCs (concept token, template tail, sentence mean), descriptive.
2. **Does it add anything over distinctness?** Paired bootstrap of AUC(1 - shared share) minus
   AUC(distinctness), same resamples. Reported, no prediction.
3. Top-8 share and participation ratio, descriptive (no prediction: neither separated live
   from dead consistently on the three sessions).
4. The S-1 cells, as above, each with the primary's prediction.

## If it fails

If the Gemma-kl AUC's CI includes 0.5, the shared-direction account of dead vectors is
reported as found on three sessions and not confirmed, and Paper 2 does not use it as an
explanation.

## Not done

- No change to the statistic, the mean vector's definition (all 90 real-arm vectors of the
  model, unit-normalised as injected) or the orientation after data arrive.
- No arm is dropped.
