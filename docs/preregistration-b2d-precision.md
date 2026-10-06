# Pre-registration: B-2d, does Pythia's half-precision load move any verdict?

**Filed 6 October 2026, before the run.**

## What was found

transformers 5.13 (installed 4 July 2026) loads a checkpoint in the dtype its config stores.
Pythia-70m and Pythia-160m store float16, so every local Pythia run since July ran in half
precision. That includes the B-2 family (B-2c, in the pooled estimate). GPT-2 and GPT-Neo
store no dtype and ran in float32. Nothing recorded this. Measured on Pythia-160m layer 6
(20 units, 4,000 tokens):

| | fp16 load | fp32 load |
|---|---|---|
| ground-truth identity, max relative error | 5.1e-4 | 3.5e-7 |
| stimulus difference from fp32 (relative norm) | 0.70% | - |
| response difference from fp32 (relative norm) | 0.69% | - |
| weight columns | identical (the checkpoint is fp16) | - |

The reference direction is the same, and the identity still holds to 5e-4. Whether the
estimator's verdicts depend on the 0.7% perturbation is unmeasured.

## Design

```
python experiments/e01_gate.py --model EleutherAI/pythia-160m --layer 6 --d-mlp 3072 \
  --neuron-pool 300 --neurons 50 --restarts 2 --independent-units --dtype fp32 \
  --out results/b2d_pythia160m_fp32.jsonl
```

These are the first 50 of B-2c's units (the same draw, truncated). The comparison is with
B-2c's rows for the same units, on the **direct route only**: `align_direct` and `stability`.
The direct route was seeded identically in both runs. B-2c's cascade route ran before the
6 October head-seeding fix, so cascade and selected values differ for that reason too and
are not compared.

## Reading, fixed now

- Direct-route verdict flips (align_direct at the 0.95 bar), with an exact CI. The read is
  "no material effect" if flips ≤ 3 of 50 **and** the median |Δ align_direct| < 0.01.
- Otherwise the Pythia arm is re-run in fp32 before submission, and the pooled estimate
  uses the fp32 run.

Either way, Paper 1 states each arm's precision (Appendix A and Table 1).
