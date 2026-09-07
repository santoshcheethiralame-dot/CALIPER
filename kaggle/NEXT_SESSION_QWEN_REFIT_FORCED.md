# A-13 — Qwen forced-choice at the concept position (~35 min) · **the invariant test**

**Script version: `2026-09-07e`.** Run this *after* A-12, not before — see "Why the
order matters".

**Why.** C27-C29 measured Qwen's detection through vectors that carried no content, so
that null measured nothing and the claim "the invariant hypothesis is dead" was
withdrawn. C32 fixed the vectors and proved they steer. This re-runs the forced-choice
arms properly.

**What it decides.** Whether **alpha\*** — the injection strength at which detection
switches on — tracks the residual-stream norm across models. Gemma's onset was at 13.9%
of its norm. If Qwen's lands near the same fraction, injection strength becomes
reportable as a fraction of the residual norm and Paper A gets a **calibration rule**
instead of three separate findings. If it lands elsewhere, the honest statement is that
the onset is model-specific and normalising by activation scale does not make models
comparable — still a real warning, but a weaker one.

## Why the order matters

Run A-12 first. Gemma's onset on **refitted** vectors is the reference point, and
comparing Qwen's new onset against Gemma's *old* template-tail onset would compare two
different instruments. Both numbers must come from `--vector-pos concept`.

## Setup

Qwen2.5-32B under **Models**, `caliper-s3` at `2026-09-07e` under **Datasets**,
GPU T4 x2, Internet on, **session restarted**.

Known and expected on this model: `unquantised modules do not fit at fp32 storage;
retrying at fp16 storage` (C26). Harmless.

## Cell 1 — after a restart

```
pip install -q -U bitsandbytes accelerate transformers
```

## Cell 2 — forced choice, real vectors

Qwen's residual norm at the concept token is **261.5** (from C32), so the fractions are
already worked out. Same fractions as the Gemma sheet.

| fraction | alpha |
|---|---|
| 0% | 0 |
| 1% | 3 |
| 2% | 5 |
| 5% | 13 |
| 10% | 26 |
| 20% | 52 |
| 40% | 105 |

```python
import sys, glob
sys.argv = ["run", "--model", "qwen", "--compute-dtype", "fp32", "--stage", "forced",
            "--normalise", "--control", "none",
            "--alphas", "0", "3", "5", "13", "26", "52", "105",
            "--out", "/kaggle/working/q_refit.jsonl"]
hits = glob.glob("/kaggle/input/**/*.py", recursive=True)
print(hits)
exec(open(hits[0]).read())
```

**Confirm the residual norm prints as 261.5.** If it differs, something changed since
C32 and the alpha grid above is wrong — send me the number before running cell 3.

## Cell 3 — the random control, same grid

```python
import sys, glob
sys.argv = ["run", "--model", "qwen", "--compute-dtype", "fp32", "--stage", "forced",
            "--normalise", "--control", "random",
            "--alphas", "0", "3", "5", "13", "26", "52", "105",
            "--out", "/kaggle/working/q_refit.jsonl"]
hits = glob.glob("/kaggle/input/**/*.py", recursive=True)
exec(open(hits[0]).read())
```

## Pre-flight

| Line | Must read |
|---|---|
| 1st line | `kaggle_s3_positive_control 2026-09-07e` |
| model source | `found N model dir(s) in /kaggle/input` |
| read position | `decoded ' table'` — not a template marker |
| residual norm | `median 261.5` |
| vectors | `30 vectors, median norm 1.00, non-finite 0` |
| cell 3 only | `CONTROL random: vectors replaced` |

## The confound to keep in mind when reading the result

C30 found Qwen refusing the premise in **30/30** clean trials — *"As an AI, I don't have
the capability to detect..."* — which is why its baseline P(YES) was 1.03e-09 against
Gemma's 3e-05. That refusal is a trained policy, not a measurement, and **it does not go
away because the vectors were fixed.**

So a low Qwen detection rate here has two available explanations, and this run cannot
separate them: no detection, or a model that will not answer the question. If the
detection numbers come back near zero while the steer control (C32) shows the vectors
plainly working, **that gap is the result** — the injection reaches the computation, the
model will not report it — and it needs saying that way rather than as "Qwen does not
detect".

## Send back

`q_refit_forced_norm1.jsonl`, `q_refit_forced_random_norm1.jsonl`, and the
`q_refit*.config.json` sidecars.
