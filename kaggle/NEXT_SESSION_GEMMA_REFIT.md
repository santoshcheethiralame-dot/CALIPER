# A-12b — Gemma refit, the corrected sheet (~35 min) · **BLOCKS THE PREPRINT**

**Script version: `2026-09-07f`.** Re-upload as a New Version of `caliper-s3`.

> **What went wrong last time, so it cannot happen again.** The previous sheet told you
> to paste the residual norm into `R = 0.0` in cells 3 and 4. It was never pasted, so
> every alpha computed to 0 and the session ran seven identical alpha=0 conditions. That
> was the sheet's fault, not yours. Two fixes in `f`: **`--alpha-frac` takes fractions
> and computes the grid inside the run**, so there is nothing to paste; and a grid of
> all zeros now **refuses to run** instead of quietly producing 420 useless rows.

**What the lost session did establish, and it is kept.** Gemma's residual-stream norm at
the **concept** position is **36,244.97** (the template-tail figure was 58,932). The
alpha=0 baselines also reproduced C20 exactly: introspective 3.42e-05, neutral 0.188.

## Setup

Gemma 3 27B under **Models**, `caliper-s3` at **`2026-09-07f`** under **Datasets**,
GPU T4 x2, Internet on, **session restarted**.

## Cell 1 — after a restart

```
pip install -q -U bitsandbytes accelerate transformers
```

## Cell 2 — real vectors. No editing, no pasting.

```python
import sys, glob
sys.argv = ["run", "--model", "gemma", "--compute-dtype", "fp32", "--stage", "forced",
            "--normalise", "--control", "none",
            "--alpha-frac", "0", "0.01", "0.02", "0.05", "0.10", "0.20", "0.40",
            "--out", "/kaggle/working/g2.jsonl"]
hits = glob.glob("/kaggle/input/**/*.py", recursive=True)
print(hits)
exec(open(hits[0]).read())
```

## Cell 3 — the random control, identical but for one word

```python
import sys, glob
sys.argv = ["run", "--model", "gemma", "--compute-dtype", "fp32", "--stage", "forced",
            "--normalise", "--control", "random",
            "--alpha-frac", "0", "0.01", "0.02", "0.05", "0.10", "0.20", "0.40",
            "--out", "/kaggle/working/g2.jsonl"]
hits = glob.glob("/kaggle/input/**/*.py", recursive=True)
exec(open(hits[0]).read())
```

These are the same fractions used for Qwen in C33/C34, which is what makes the two
onsets comparable and the invariant test possible.

## Pre-flight

| Line | Must read |
|---|---|
| 1st line | `kaggle_s3_positive_control 2026-09-07f` |
| model source | `found N model dir(s) in /kaggle/input` |
| read position | `decoded ' table'` — **not** `'\n'` or a template marker |
| residual norm | `median 36244.9` (concept position) |
| **alpha grid** | **`alpha grid from fractions [...] of residual norm 36245.0`** followed by a list ending near **14498** |
| vectors | `30 vectors, median norm 1.00, non-finite 0` |
| cell 3 only | `CONTROL random: vectors replaced` |

If the printed alpha list is all zeros the run will now stop by itself.

## Time

~5 min load per cell, ~12 min per condition. About 35 minutes.

## Send back

`g2_forced_norm1.jsonl`, `g2_forced_random_norm1.jsonl`, and the `g2*.config.json`.

## What it decides

Whether Paper A's A1 headline survives a clean instrument — and, with C33/C34, whether
alpha* tracks the residual norm across models. Three outcomes for the headline:

- **Real still beats random near the onset** — it stands, on better measurement.
- **The separation grows** — the degraded vectors were masking it, and Paper A is stronger.
- **The separation vanishes** — A1 was an artifact of the read position, and the paper
  reverts to the content-free story. That would be the second reversal on this question
  and has to be reported as one.
