# A-14 — the validated-steering window (~35 min)

**Script version `2026-09-08a`.** Protocol frozen in `docs/preregistration-s3-forced-choice.md`,
Addendum 4, filed before this run.

**What it decides.** C45/C46 found real below random at 5/10/20% of the residual norm and
null at 40%. C48 then showed the vectors only steer measurably from 40% up — 2/30 and
5/30 semantic at 10% and 20% against a 3/30 baseline, 10/30 at 40%. So the significant
cells are where the vectors do almost nothing, and the one validated cell is null. This
run sweeps 30–60% with three conditions to find out which way that resolves.

Delete every existing cell. Nothing here needs editing.

## Cell 1 — only after a session restart

```bash
pip install -q -U bitsandbytes accelerate transformers
```

## Cell 2 — real vectors

```python
import sys, glob
sys.argv = ["run", "--model", "gemma", "--compute-dtype", "fp32", "--stage", "forced",
            "--normalise", "--control", "none",
            "--alpha-frac", "0", "0.30", "0.40", "0.50", "0.60",
            "--out", "/kaggle/working/gw.jsonl"]
hits = glob.glob("/kaggle/input/**/*.py", recursive=True)
print(hits)
exec(open(hits[0]).read())
```

## Cell 3 — random control

```python
import sys, glob
sys.argv = ["run", "--model", "gemma", "--compute-dtype", "fp32", "--stage", "forced",
            "--normalise", "--control", "random",
            "--alpha-frac", "0", "0.30", "0.40", "0.50", "0.60",
            "--out", "/kaggle/working/gw.jsonl"]
hits = glob.glob("/kaggle/input/**/*.py", recursive=True)
exec(open(hits[0]).read())
```

## Cell 4 — span control (on-manifold)

```python
import sys, glob
sys.argv = ["run", "--model", "gemma", "--compute-dtype", "fp32", "--stage", "forced",
            "--normalise", "--control", "span",
            "--alpha-frac", "0", "0.30", "0.40", "0.50", "0.60",
            "--out", "/kaggle/working/gw.jsonl"]
hits = glob.glob("/kaggle/input/**/*.py", recursive=True)
exec(open(hits[0]).read())
```

Run 2, then 3, then 4. Not Run All.

## Pre-flight, every cell

| line | must read |
|---|---|
| 1st line | `kaggle_s3_positive_control 2026-09-08a` |
| model source | `found N model dir(s) in /kaggle/input` |
| read position | `decoded ' table'` — **not** a template marker |
| residual norm | `median 36245.0` |
| alpha grid | `[0.0, 10873.5, 14498.0, 18122.5, 21747.0]` |
| vectors | `30 vectors, median norm 1.00, non-finite 0` |
| cells 3 and 4 | `CONTROL random:` / `CONTROL span: vectors replaced` |

If the alpha grid line is missing you are on an old script and the run is void.

## Send back

`gw_forced_norm1.jsonl`, `gw_forced_random_norm1.jsonl`, `gw_forced_span_norm1.jsonl`,
and the `gw*.config.json` sidecars.

## What each outcome means — written down before the numbers exist

- **B1, real below random.** The suppression survives where the instrument is validated.
  A3 becomes the headline and the 5–20% cells become supporting evidence.
- **B2, no difference.** The effect does not survive where the vectors demonstrably work.
  **The C45/C46 headline is then confined to strengths at which nothing measurable is
  happening** — a substantial weakening — and Paper A's centre of gravity moves to the
  readout-dependence and framing results.
- **B3, real above random.** Anomaly. Reported, not interpreted.

B2 gets the same billing as B1. That is in the pre-registration, not left to judgement on
the day.
