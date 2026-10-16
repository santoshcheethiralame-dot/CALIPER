# Pre-registration: E-2b, does MID also fail at the near-constant layers?

**Filed 16 October 2026, after E-2 was scored and before any E-2b fit.** Paper 1. Fits:
`experiments/e2_mid.py` unchanged. Scorer: `experiments/analyse_e2b.py`, developed on E-2's GPT-2
layer-6 arm only (`results/e2b_dev.json`, which reproduces E-2's 198 Euclidean and 199 identifiable
passes). Queue: `experiments/rerun_queue20.sh`, after queue 18b.

## Why

E-2 found that MID recovers 40 of the 41 GPT-2 units E-1's least-squares estimator failed, on the same
data, so E-1's failures (all under-fitted) belong to that estimator. The paper's other failure mode is
structural: per-coordinate standardisation magnifies a nearly constant layer-norm coordinate, and
every restart reproduces the wrong direction (B-17: GPT-Neo layer 6, 0 of 20, restart agreement 1.0;
B-17c: GPT-2 layer 3, 0 of 20). F-0's Proposition 5 says this depends only on the standardisation and
on the fit putting any fixed weight on that coordinate, not on the objective. MID standardises the
same way (`e2_mid.py`: `sd = S.std(0) + 1e-6`, directions mapped back by dividing by `sd`). E-2b tests
whether the failure mode is general across estimator families or specific to the least-squares one.

## Design

`e2_mid.py` with E-2's settings (8,000 corpus-seed-0 tokens, first 20% validation, 4,000 test tokens
from unseen documents, five random restarts, 1,600 steps), on:

| run | model, layer | units |
|---|---|---|
| E-2b-1 | GPT-Neo-125m L6 | B-17's 20 units |
| E-2b-2 | GPT-2 small L3 | B-17c's 20 units (`results/b17c_units.txt`) |

## Predictions

Share = squared norm of the selected MID direction on the layer's near-constant coordinates (SD < 1%
of the median SD over the run's own 8,000 tokens).

- **P1.** At GPT-Neo L6, at least 10 of 20 MID fits have share > 0.5.
- **P1b.** At GPT-2 L3, at least 5 of 20 MID fits have share > 0.5.
- **P2.** Among the GPT-Neo L6 fits with share > 0.5, median restart agreement is at least 0.99.

## Readings

- **P1 and P2 hold:** the failure mode is a property of per-coordinate standardisation, shared by two
  estimator families with different objectives; restart agreement certifies it in both. Paper 1 leads
  with it as estimator-general.
- **P1 fails:** MID's objective does not place weight on the near-constant coordinate, so the failure
  mode is specific to the least-squares estimator; Paper 1 states it as such, and MID becomes the
  recommended remedy alongside dropping the coordinate.

## Secondary (no predictions)

Pass counts and median alignment on both labels; restart agreement over all fits.

## Not done

No threshold (1%, 0.5, 10 and 5 of 20, 0.99), unit or setting is changed after seeing output. A unit
whose fit fails is reported as missing.
