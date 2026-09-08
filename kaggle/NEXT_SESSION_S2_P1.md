# Study 2 P1/P2 — planted-direction recovery (~50 min)

**Script version `2026-09-08b`.** Re-upload as a New Version of `caliper-s3`.
Protocol frozen in `docs/preregistration-s2-p1.md`, filed before this run.

**What it decides.** Persona vectors are deployed and audited with, and validated only by
steering effect and finetuning correlation. Venkatesh & Kurapath (2602.06801) showed
steering cannot identify a direction. So: plant a known random direction, run the
extraction pipeline, and see whether it comes back.

**P1 is a hard gate.** If a planted direction cannot be recovered at the layer it was
planted in, nothing downstream in Study 2 is interpretable and cell 3 does not run.

## Setup

Gemma 3 27B under **Models**, `caliper-s3` at **`2026-09-08b`** under **Datasets**,
GPU T4 x2, Internet on, **session restarted**.

## Cell 1 — after a restart

```bash
pip install -q -U bitsandbytes accelerate transformers
```

## Cell 2 — P1, recovery at the plant layer

```python
import sys, glob
sys.argv = ["run", "--model", "gemma", "--compute-dtype", "fp32", "--stage", "plant",
            "--normalise", "--alpha-frac", "0.10", "0.20", "0.40",
            "--n-plants", "8", "--n-prompts", "16",
            "--out", "/kaggle/working/p1.jsonl"]
hits = glob.glob("/kaggle/input/**/*.py", recursive=True)
print(hits)
exec(open(hits[0]).read())
```

## Cell 3 — P2, the depth curve · **only if P1 passes**

```python
import sys, glob
sys.argv = ["run", "--model", "gemma", "--compute-dtype", "fp32", "--stage", "plant",
            "--normalise", "--alpha-frac", "0.40",
            "--n-plants", "8", "--n-prompts", "16",
            "--extract-layers", "39", "43", "47", "55",
            "--out", "/kaggle/working/p2.jsonl"]
hits = glob.glob("/kaggle/input/**/*.py", recursive=True)
exec(open(hits[0]).read())
```

## Pre-flight

| line | must read |
|---|---|
| 1st line | `kaggle_s3_positive_control 2026-09-08b` |
| model source | `found N model dir(s) in /kaggle/input` |
| read position | `decoded ' table'` |
| residual norm | `median 36245.0` |
| plants | `8 planted directions, unit norm, seed 0` |
| extraction | `extracting at layers [37] (plant layer 37)` |

## Reading the output yourself

The table prints one row per (strength, plant):

```
    alpha  %norm  layer  plant   |cos(diff, v)|   |cos(diff, null)|
```

**Compare the two right-hand columns.** `|cos(diff, v)|` is recovery of the direction
that was actually planted; `|cos(diff, null)|` is the same difference scored against an
unrelated random direction. Null is the floor — it is what "recovered nothing" looks like
at this dimensionality, and it will be small but not zero.

| what you see | meaning |
|---|---|
| recovery clearly above null, and above 0.30 | **P1 passes.** Run cell 3 |
| recovery ≈ null | **P1 fails. Do not run cell 3.** Send me the file; the finding is that a deployed extraction pipeline cannot recover a planted cause |
| recovery above null but under 0.30 | Partial. Run cell 3 anyway — the depth curve is still informative |

## Time

~6 min load, then P1 is 16 baseline generations plus 3 strengths x 8 plants x 16
generations = 400 generations at 40 tokens, each followed by one re-read forward pass.
Roughly 35 minutes. Cell 3 adds ~15.

## Send back

`p1_plant.jsonl` and `p1.config.json`, plus `p2_plant.jsonl` if cell 3 ran.
