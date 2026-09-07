# A-12 — Gemma refit at the concept position (~45 min) · **BLOCKS THE PREPRINT**

**Script version: `2026-09-07e`.** Re-upload as a New Version of `caliper-s3` if you
have not since C32.

**Why.** C32 proved the vector read position matters: refitting at the concept token
turned Qwen's flat 1/30 into a monotone 1 → 7 → 14 of 30. **Every Gemma run behind
Paper A — C15 to C24, including the A1 headline at alpha=8192/16384 — used
template-tail vectors.** They are not void, because Gemma's steered (10/30), but they
were measured with a degraded instrument. The onset, the real-vs-random comparison and
the content-free share can all move.

**Do not ship the preprint on the old numbers.**

## Setup

Gemma 3 27B under **Models**, `caliper-s3` at `2026-09-07e` under **Datasets**,
GPU T4 x2, Internet on, **session restarted**.

## Cell 1 — after a restart

```
pip install -q -U bitsandbytes accelerate transformers
```

## Cell 2 — steer control, and it also measures the new residual norm

```python
import sys, glob
sys.argv = ["run", "--model", "gemma", "--compute-dtype", "fp32", "--stage", "steer",
            "--normalise", "--control", "none",
            "--alphas", "0", "0.02", "0.05", "0.10", "0.20", "0.40",
            "--out", "/kaggle/working/g_refit.jsonl"]
hits = glob.glob("/kaggle/input/**/*.py", recursive=True)
print(hits)
exec(open(hits[0]).read())
```

> The alphas here are placeholders and will do nothing — **that is expected.** This cell
> is run for two things: the `read position check` line and the printed residual norm.
> Note them, then move to cell 3. (If you would rather not waste the trials, interrupt
> the cell once the norm has printed.)

**Two lines to capture:**

```
read position check: 'table' at token(s) [N] of M; decoded ' table' ...
residual norm at concept token(s): median R, min ..., max ...
```

If `decoded` is not `' table'`, stop — the fix did not take.

**Write down R.** It will not be 58,932; that was the template-tail figure.

## Cell 3 — the alpha grid, as fractions of R

Fill in R from cell 2. These are the same fractions used for Qwen in C32, which is what
makes the two models directly comparable and the invariant test possible.

| fraction | alpha |
|---|---|
| 1% | R × 0.01 |
| 2% | R × 0.02 |
| 5% | R × 0.05 |
| 10% | R × 0.10 |
| 20% | R × 0.20 |
| 40% | R × 0.40 |

The old onset was 13.9% of the tail norm. The grid brackets well below it because
refitted vectors are stronger and the onset may drop.

```python
import sys, glob
R = 0.0        # <-- paste the residual norm from cell 2
alphas = [f"{R*f:.0f}" for f in (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40)]
print("alphas:", alphas)
sys.argv = (["run", "--model", "gemma", "--compute-dtype", "fp32", "--stage", "forced",
             "--normalise", "--control", "none", "--alphas"] + alphas +
            ["--out", "/kaggle/working/g_refit.jsonl"])
hits = glob.glob("/kaggle/input/**/*.py", recursive=True)
exec(open(hits[0]).read())
```

## Cell 4 — the random control, same grid

Identical to cell 3 with `"--control", "random"`. **Do not change the alphas** — the
comparison is paired.

```python
import sys, glob
R = 0.0        # <-- the same R
alphas = [f"{R*f:.0f}" for f in (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40)]
sys.argv = (["run", "--model", "gemma", "--compute-dtype", "fp32", "--stage", "forced",
             "--normalise", "--control", "random", "--alphas"] + alphas +
            ["--out", "/kaggle/working/g_refit.jsonl"])
hits = glob.glob("/kaggle/input/**/*.py", recursive=True)
exec(open(hits[0]).read())
```

## Pre-flight, every cell

| Line | Must read |
|---|---|
| 1st line | `kaggle_s3_positive_control 2026-09-07e` |
| model source | `found N model dir(s) in /kaggle/input` |
| read position | `decoded ' table'` — **not** `'\n'` or a template marker |
| vectors | `30 vectors, median norm 1.00, non-finite 0` |
| cell 4 only | `CONTROL random: vectors replaced` |

## Time

~5 min load per cell, ~12 min per forced condition. About 45 minutes total.

## Send back

`g_refit_forced_norm1.jsonl`, `g_refit_forced_random_norm1.jsonl`, and every
`g_refit*.config.json`. The sidecars now carry `vector_read_position` and the new
residual norm, so the run is self-describing.

## What it decides

Whether Paper A's A1 headline survives a clean instrument. Three outcomes:

- **Real still beats random around the onset** — the headline stands, on better
  measurement, and the numbers get updated.
- **The separation grows** — the concept-specific component was being masked by the
  degraded vectors, and Paper A gets stronger.
- **The separation vanishes** — the A1 result was an artifact of the read position, and
  the paper returns to the content-free story it had before C23. That would be the
  second reversal on this question, and it must be reported as such.
