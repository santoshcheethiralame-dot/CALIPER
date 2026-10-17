# Pre-registration: R-2, does flooring the scale repair the estimators?

**Filed 17 October 2026, before any R-2 fit.** Paper 1, the review's request to compare repairs.
Code: `CALIPER_SCALE_FLOOR=0.01` floors the per-coordinate scale at 1% of its median in
`caliper/estimator.py`, `caliper/batched.py` and `experiments/e2_mid.py`; unset, every path is as
before. Queue: `experiments/rerun_queue21.sh`. Scorer: `experiments/repair_table.py` extended
(`--r2`), committed with this filing.

## Runs

| run | estimator | units | compared with |
|---|---|---|---|
| R-2a | least squares, `e01_gate.py`, B-17c settings | B-17's 20 GPT-Neo L6 units | B-17b (coordinate dropped) |
| R-2b | least squares, B-17c settings | B-17c's 20 GPT-2 L3 units | B-17c (as usual), B-17d (dropped) |
| R-2c | MID, `e2_mid.py`, E-2 settings | B-17's units, GPT-Neo L6 | E-2b (as usual, and zeroed) |
| R-2d | MID | B-17c's units, GPT-2 L3 | E-2b |

## Predictions

- **P1.** In every run, the median share of the fitted directions on the near-constant coordinates is
  at most 0.05.
- **P2.** In every run, passes at the 0.95 Euclidean bar are within 3 of the comparison repair on the
  same units (the dropped re-fit for least squares, the zeroed E-2b fits for MID).

## Not done

No floor, unit, setting or threshold is changed after seeing output.
