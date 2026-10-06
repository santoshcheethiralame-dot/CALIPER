# Pre-registration: B-2c and B-8b, two coupled arms re-run on the fixed estimator

**Filed 6 October 2026, before either run starts.** These are the two runs the B-14 decision
selected (the reduced path, `LAB_NOTEBOOK.md` §6b, 6 Oct). Their coupled versions are in
Paper 1's pooled table and the claims lean on them. B-2c is the second model family; B-8b is
the family effect at fixed depth.

## Design

Each is identical to its coupled original except for `--independent-units`. Directions are
saved and route agreement is recorded, both now the default in `e01_gate.py`.

| run | replaces | command arguments | same units because |
|---|---|---|---|
| **B-2c** | B-2b | `--model EleutherAI/pythia-160m --layer 6 --d-mlp 3072 --neurons 300 --restarts 2 --independent-units` | legacy draw `choice(3072, 300)`, seed 0 |
| **B-8b** | B-8 | `--model EleutherAI/gpt-neo-125M --layer 10 --d-mlp 3072 --neurons 100 --restarts 2 --independent-units` | legacy draw `choice(3072, 100)`, seed 0 |

8,000 tokens, 1,600 steps and batch 32 (the `e01_gate.py` defaults, as in the originals).
Local CPU, the same machine as the originals; never pooled with a GPU arm at the unit level.
Run in sequence, B-2c first.

## Analysis (fixed now)

`experiments/analyse_b14.py --model <model> --layer <layer> --reference <original rows>`, the
script tested on B-1b and used once on B-14. Every pre-registered and Addendum 1 and 2
analysis applies unchanged:
- primary DeLong, with the decision table;
- McNemar against the coupled original;
- failure classes, and AUCs by class at cutoffs 0.98, 0.99 and 0.995;
- PR-AUC, threshold sweep, calibration;
- nuisance baselines with incremental CV AUC;
- controls;
- the LayerNorm-null ceiling;
- route agreement.

Each run then replaces its coupled row in `analyse_cross_arm.py`, and the random-effects
estimate is recomputed.

## How each is read

- **B-2c** has few failures (13 in B-2b). If it has fewer than 5, it drops out of the pooling
  by the rule fixed for every arm, and is reported as "recovered at this budget". Its own
  significance is not judged; it is a pooling component.
- **B-8b** is where B-14's class split matters most. B-8 had 24 converged-wrong failures, the
  largest such class in the programme. B-8b's by-class AUCs on that class are the best test of
  whether "neither check catches the converged-wrong fit" holds beyond GPT-2.
- **If either reverses the ordering** (restart AUC significantly above R2), it is reported as is,
  and the pooled claim stands or falls on the re-pooling.

## Cost

Roughly 12 h for B-2c and 5 h for B-8b at the B-14 rate. Both resume from their last finished
unit if interrupted.
