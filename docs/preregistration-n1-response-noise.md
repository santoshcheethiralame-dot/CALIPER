# Pre-registration: N-1, do the checks keep their ordering when the attainable fit is unknown?

**Filed 7 October 2026, before any N-1 run.** Paper 1, review round 1, item C1.

## Why

**The concern.** Two reviewers in the 7 Oct internal review raised the same objection.
Every Paper 1 unit's response is a noiseless function of w·s. So the true direction reaches
held-out R2 = 1, and any shortfall in R2 is estimator error.

**Why it matters.** In practice the attainable fit differs between units and is unknown, because
of noise or a misspecified link. A unit can then score a low R2 because its response is noisy,
not because its fit is wrong. That could take away held-out R2's advantage. N-1 tests this
directly.

## Design

**Units.** B-14's first 100 GPT-2 layer-6 units, drawn by `--neuron-pool 300 --neurons 100`
(the same draw B-15 used).

**Fixed settings.** Corrected estimator (`--independent-units`), 2 restarts, 8,000 tokens,
1,600 steps.

**Noise.** Additive Gaussian noise on each unit's response, at a signal-to-noise variance ratio
SNR. The attainable R2 is therefore SNR / (1 + SNR). Noise is seeded by unit id. The reference
direction w and the failure label (|cos| < 0.95) are unchanged.

**Arms:**

| arm | flag | attainable R2 |
|---|---|---|
| **N-1a (primary)** | `--snr-range 1 19`: each unit draws its own SNR, log-uniform | varies by unit, 0.50 to 0.95 |
| N-1b | `--snr 19` | 0.95 for every unit |
| N-1c | `--snr 4` | 0.80 for every unit |

## Primary endpoint

In N-1a: AUC(restart agreement) minus AUC(held-out R2) at predicting failure. The test is
DeLong, with a stratified bootstrap CI (2,000 resamples) and the paired permutation test of
`analyse_review_round1.py`.

**Prediction:** held-out R2 keeps its lead, with ΔAUC < 0 and a bootstrap CI excluding 0.

**If it fails, reported as such.** "Held-out fit is the better check" holds only where the
attainable fit is known or shared across units. Paper 1 then scopes its advice to that setting.

## Secondary

1. ΔAUC in N-1b and N-1c, against B-15's noiseless re-fits of the same units.
2. **SNR-normalised R2.** R2 divided by its unit's attainable ceiling, and its AUC. This is not a
   ground-truth-free check, because a practitioner does not know the ceiling. It is reported to
   show how much of R2's loss in N-1a comes from the unknown ceiling.
3. Failure rate per arm.
4. The converged-wrong / under-fitted split per arm, with the class boundary at 0.99 of each
   unit's ceiling (descriptive).

## Not done

- No SNR level, arm or unit subset is chosen after seeing results.
- The analysis script is written and tested on B-15's noiseless rows before any N-1 row exists.
