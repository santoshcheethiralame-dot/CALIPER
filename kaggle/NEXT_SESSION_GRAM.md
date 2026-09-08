# Concept-vector similarity — the 3-minute run that decides how to read P1b

**Script version `2026-09-08d`.** Re-upload as a New Version of `caliper-s3`.

## Why

P1b (C56) passed its manipulation check — the plant reached the text, median 6/16 at 40%
of the residual norm — and then failed its primary: median recovery 0.1557 against a
median null A of 0.1870, beating the null on only 3 of 8 plants.

The two nulls disagree sharply:

| null | median |
|---|---|
| null B — a random direction | **0.0066** |
| null A — a *different concept's* vector | **0.1870** |
| recovery — the planted concept | 0.1557 |

So the extracted difference sits ~24× above random. It lands in concept space. It simply
is not closer to the concept that was planted than to an unrelated one.

**That reading depends on a number nobody has measured: how similar the concept vectors
are to each other.** Null A was chosen as "conservative" without checking. If the vectors
share a large common component, null A ≈ 0.19 is just the typical inter-concept cosine —
the floor any concept-specific claim has to clear — and P1b means "recovers concept
space, not concept identity". If the vectors are near-orthogonal, null A ≈ 0.19 means
something quite different and the reading has to change.

This run measures it. No generation, no sweep, no alpha.

## Setup

Gemma 3 27B under **Models**, `caliper-s3` at **`2026-09-08d`** under **Datasets**,
GPU T4 x2, Internet on. **Delete every existing cell.**

## Cell 1 — only after a session restart

```bash
pip install -q -U bitsandbytes accelerate transformers
```

## Cell 2

```python
import sys, glob
sys.argv = ["run", "--model", "gemma", "--compute-dtype", "fp32", "--stage", "forced",
            "--normalise", "--alphas", "0",
            "--concepts", "30", "--out", "/kaggle/working/gram.jsonl"]
hits = glob.glob("/kaggle/input/**/*.py", recursive=True)
print(hits)
exec(open(hits[0]).read())
```

It builds the 30 vectors, prints the similarity line, writes the sidecar, then runs a
single alpha=0 forced-choice pass (30 trials, ~1 min) which we ignore. Building the
vectors is the whole point; the rest is the cheapest legal way to get the script there.

## The line to read

Right after `30 vectors, median norm 1.00`:

```
concept-vector similarity: median |cos| 0.XXXX, mean ..., max ..., min ...
```

## What it decides

| median \|cos\| | reading |
|---|---|
| **≈ 0.19** | Null A was the inter-concept floor. **P1b's finding stands**: difference-of-means recovers concept space but not concept identity — non-identifiability measured directly against planted ground truth |
| **≈ 0.01** (near-orthogonal) | Null A was a genuine null, and recovery ≈ null means the extraction recovered essentially nothing despite the plant reaching the text. A stronger negative, and a different claim |
| **≫ 0.19** | The vectors are so collinear that no method could separate them this way. **P1b is void again**, and the problem is the concept bank, not the extractor |

Sanity-checked offline before shipping: the statistic returns 0.011 on 30 random unit
vectors at d=3584 and 0.589 on vectors sharing a strong common component, so it
discriminates the cases above cleanly.

## Send back

`gram.config.json` — it carries `gram_offdiag_median_abs`, the mean, min, max, the 90th
percentile, and the ten most similar concept pairs. The `.jsonl` is throwaway.
