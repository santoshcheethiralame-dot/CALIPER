# Pre-registration — B-11c, Pythia-1.4B on the fixed estimator

**Filed 6 October 2026, before the run.**

## Why

Pythia-1.4B is one of the six arms in Paper 1's pooled comparison (Table 2): B-11s, layer
12, 3,200 steps, 50 units, with 36/50 passing, restart AUC 0.746, held-out R2 AUC 0.996 and
a gap of -0.250. Like every arm so far, it was fitted with the coupled estimator: shared
early stopping and seeding by batch position. B-11c refits the same arm with
`--independent-units`, which removes both couplings. Its result replaces B-11s in the
pooled table.

## Design

Identical to B-11s except for the one flag:

```
experiments/e01_gate.py --model EleutherAI/pythia-1.4b --layer 12 --d-mlp 8192
    --restarts 2 --neurons 50 --steps 3200 --device cuda --independent-units
    --out b11c_pythia-14b_s3200_indep.jsonl
```

- Same 50 units. `--neuron-pool` is off, so the legacy draw `choice(8192, 50)` with seed 0
  matches B-11s.
- Same device class as B-11s (Kaggle T4, GPU). This arm is never pooled with a CPU arm at
  the unit level.
- New fields this run records: `route_agreement` (the ground-truth-free method check) and
  the fitted directions (`b11c_pythia-14b_s3200_indep_dirs/`).

## Endpoints (all fixed now)

1. **Primary for the pooled table:** DeLong, restart agreement minus held-out R2, failure =
   `align_selected < 0.95`. With roughly 10-15 failures expected, this arm is a component
   of the random-effects pooling, not a standalone test. It is reported with its CI and is
   not judged on its own p-value.
2. **Verdict agreement with B-11s** on the 50 shared units, exact McNemar. This is the size
   of the coupling at this scale.
3. **Failure classes:** wrong basin (held-out R2 > 0.99) vs under-fitted. B-11s found 0 of 14
   failures converged, all under-fitted. If that holds under per-unit stopping, the 1.4B
   failures are an optimisation-budget property, not silent failure, and the paper says so.
4. **Route agreement AUC,** the first ground-truth-free method-agreement number at this scale.

## Failure branches

- **Fewer than 5 failures:** the arm drops out of the pooling (rule fixed for every arm),
  and the paper reports it as "recovered at this budget".
- **Restart AUC above R2 AUC:** reported as is, and the pooled estimate is recomputed with it.
  The pooled claim stands or falls on the re-pooling, not on this arm.

## Analysis

`experiments/analyse_b14.py --rows <b11c rows> --reference data/b11/b11_pythia-14b_s3200.jsonl --no-model`.
The script is the one already tested on B-1b. `--no-model` is used because its nuisance
baselines are wired to GPT-2.
