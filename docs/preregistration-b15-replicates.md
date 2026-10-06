# Pre-registration: B-15, how much a verdict moves when nothing about the unit changes

**Filed 7 October 2026, before the runs.** It supplies Paper 1's error budget on the fixed
estimator.

## Question

On the same 100 GPT-2 layer-6 units, how much do pass/fail verdicts and the reliability
signals move when one source of randomness changes and nothing about the unit does? On the
coupled estimator these numbers were confounded with batch composition (5 Oct audit). B-15
measures them cleanly.

## Design

The units are B-14's first 100 (`--neuron-pool 300 --neurons 100` reproduces B-14's draw). The
fixed estimator is used throughout (`--independent-units`), with directions saved. The
reference is B-14 itself (fit, corpus and split seeds 0; 2 restarts; token-level split). Each
arm changes **one** factor:

| arm | change | measures |
|---|---|---|
| **B-15a** | `--fit-seed 1 --restarts 5` | initialisation, and the restart curve (k = 2..5) with directions saved |
| **B-15b** | `--corpus-seed 1` | which tokens make the stimulus |
| **B-15c** | `--sequence-split` | held-out sequences shared with training vs not |

B-15a changes two things on purpose. Its first two restarts differ from B-14's (a new seed),
and its five restarts give the restart curve. The flip rate it reports is "a different
initialisation and more restarts", and the restart-count effect is read within B-15a by
subsampling k.

Defaults reproduce every earlier run. `tests/test_e01_seeds.py` checks that seed 0 matches
B-14's archived rows on two units.

## Endpoints, fixed now (analysis: `experiments/analyse_b15.py`, developed on B-14 ∩ B-7 L6)

1. **Per arm:** the verdict flip rate against B-14 on the shared units, with an exact
   (Clopper-Pearson) CI. Flips are counted at the 0.95 bar, with a sweep from 0.90 to 0.98.
2. **Per arm:** the AUC difference (restart minus held-out R2) and its change from B-14. Does
   the ordering survive every perturbation?
3. **Test-retest:** for alignment and each signal, the per-unit ICC(A,1) between B-14 and each
   arm.
4. **B-15a restart curve:** restart-agreement AUC from the first k of 5 restarts against the
   5-restart failure label. Also the two-class "unreachable units" fit on k = 2 vs 5 pass rates
   within the same run, which avoids pairing across runs.
5. **B-15c:** the change in held-out R2 under a sequence-level split, and whether R2's AUC
   survives it. It is the signal the paper recommends, so leakage would inflate it.

## How it is read

- No arm's flip rate is judged against a threshold. These are measurements for the error
  budget table.
- The one directional claim: if B-15c drops held-out R2's AUC below restart agreement's, the
  paper's recommendation is wrong under a clean split, and that is reported as the headline
  limitation.
- An ordering that survives all three arms is reported as robust to initialisation, token
  sample and split.

## Note added 7 October 2026, before any B-15 run

The cascade route's single-unit fits drew their head initialisation from torch's global RNG
until 7 Oct (`caliper/estimator.py`, `_Bottleneck`). The same unit, seed and thread count gave
cascade alignments of 0.976, 0.941 and 0.559 depending on run history. B-15 runs on the
corrected, seeded initialisation. B-14's cascade values therefore cannot be reproduced, and
B-14 vs B-15 comparisons carry this as a known extra source of variation. The initialisation
distribution is unchanged; only its seeding is. B-15 also records the thread count, since
float reduction order is a separate, smaller source.
