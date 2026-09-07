# One session, two runs (~35 min). Script version `2026-09-08a` — re-upload first.

Gemma under **Models**, `caliper-s3` at `2026-09-08a` under **Datasets**, GPU T4 x2,
Internet on, session restarted. Delete every existing cell.

## Cell 1 — only after a restart

```
pip install -q -U bitsandbytes accelerate transformers
```

## Cell 2 — RUN A: validate the refit vectors actually steer

**Why this is first.** C45/C46 inverted the paper's headline, and the vectors behind it
have never passed a steering control. The only two Gemma steer runs used alphas of 0.4
against a residual norm of 36,245 — a millionth of the norm — and their 0/180 means
nothing. Qwen's refit vectors were validated this way (25/150 against 5/150 before). If
Gemma's do not steer, C45/C46 is void the way C27-C30 was.

```python
import sys, glob
sys.argv = ["run", "--model", "gemma", "--compute-dtype", "fp32", "--stage", "steer",
            "--normalise", "--control", "none",
            "--alpha-frac", "0", "0.10", "0.20", "0.40", "0.76",
            "--out", "/kaggle/working/gval.jsonl"]
hits = glob.glob("/kaggle/input/**/*.py", recursive=True)
print(hits)
exec(open(hits[0]).read())
```

Read the `identify` column. Pre-refit Gemma gave 10/30 concept-in-text at a comparable
strength, so anything at or above that is a pass. **Flat at ~1/30 means the vectors carry
nothing and A3 must be withdrawn.**

## Cell 3 — RUN B: the on-manifold control (A-8)

**Why.** Every control tested so far is off-manifold. A Gaussian and a coordinate
permutation both point where the model's computation does not go, so C45/C46's
"random beats real" has two available explanations and no way to choose between them.

`--control span` builds a random unit-weighted combination of the 30 real concept
directions: **on-manifold by construction, carrying no single concept, matched in norm.**

```python
import sys, glob
sys.argv = ["run", "--model", "gemma", "--compute-dtype", "fp32", "--stage", "forced",
            "--normalise", "--control", "span",
            "--alpha-frac", "0", "0.01", "0.02", "0.05", "0.10", "0.20", "0.40",
            "--out", "/kaggle/working/g2.jsonl"]
hits = glob.glob("/kaggle/input/**/*.py", recursive=True)
exec(open(hits[0]).read())
```

Same alphas and same `--out` stem as C45/C46, so it drops straight into that comparison.

## What cell 3 decides

| span behaves like | reading |
|---|---|
| **real** (low P(YES), below baseline) | The signal tracks **manifold membership**, not concept content. "Detection" is an off-manifold alarm and a real concept is the least detectable perturbation there is |
| **random** (high P(YES)) | Manifold membership is not the axis. Something specific to real concept vectors suppresses the response, and the off-manifold account is wrong |
| between the two | Both contribute; report the split |

## Pre-flight, both cells

| line | must read |
|---|---|
| 1st | `kaggle_s3_positive_control 2026-09-08a` |
| model | `found N model dir(s) in /kaggle/input` |
| read position | `decoded ' table'` — **not** `'\n'` |
| alpha grid | `alpha grid from fractions [...] of residual norm 36245.0` |
| cell 3 only | `CONTROL span: vectors replaced` |

## Send back

`gval_steer_norm1.jsonl`, `g2_forced_span_norm1.jsonl`, and both `.config.json`.
