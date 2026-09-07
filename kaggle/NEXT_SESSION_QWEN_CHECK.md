# A-9 — the Qwen validity check (~12 minutes)

**Why this exists.** C27-29 found a flat null on Qwen: P(YES) never exceeds 1.36e-05
at any strength up to 56% of the residual-stream norm. That contradicts published work
on this same model family, and Qwen's baseline P(YES) is 1.03e-09 against Gemma's
3e-05 — a 30,000x gap that could mean the forced-choice readout is mis-specified for
this tokenizer rather than that the model has nothing to report.

**The null is embargoed until this runs.** It cannot go in the paper before then.

## What it decides

| What the output shows | What it means | Consequence |
|---|---|---|
| The concept appears in the generated text (`eagles`, `volcanoes`, ...) while forced-choice still reads ~1e-05 | The injection is functionally reaching the model and the model never reports it | **A real and strong Probe-Report Gap.** Reportable, and more interesting than the calibration rule we lost |
| The output is unchanged from alpha=0 | The injection is not doing what we think on this model | The null is an artifact. Do not report it. Debug the layer choice and the chat template |

## Setup

Same as before: Qwen under **Models**, `caliper-s3` under **Datasets** at version
`2026-09-07c`, GPU T4 x2, Internet on, session restarted.

## Cell 1 — only after a restart

```
pip install -q -U bitsandbytes accelerate transformers
```

## Cell 2 — generation at the top of the tested range

```python
import sys, glob
sys.argv = ["run", "--model", "qwen", "--compute-dtype", "fp32", "--stage", "control",
            "--normalise", "--control", "none",
            "--alphas", "0", "50", "100",
            "--out", "/kaggle/working/s3_qwen_gen.jsonl"]
hits = glob.glob("/kaggle/input/**/*.py", recursive=True)
print(hits)
exec(open(hits[0]).read())
```

`--stage control` generates text instead of scoring one token, so this is slower per
trial but only 90 trials. Budget ~5 min to load plus ~7 min to run.

## Pre-flight — same four lines as before

| Line | Must read |
|---|---|
| 1st line | `kaggle_s3_positive_control 2026-09-07c` |
| model source | `found N model dir(s) in /kaggle/input` |
| device budget | `{0: '14.5GiB', 1: '14.5GiB'} (no cpu offload)` |
| storage retry | `storage dtype float16 (compute stays float32)` — expected |

Plus: `residual norm at last token: median 177.8` should reproduce exactly. If it does
not, something changed between sessions and the comparison to C27-29 is unsafe.

## What to look at first, before sending anything

Read the `text` field of a few alpha=100 rows in the output. You do not need me for
this and it is the whole answer:

- **Concept words present** -> injection works, model does not report it. Send the file.
- **Text identical to alpha=0** -> injection is inert here. Send the file and say so;
  the next step is a layer sweep, not a write-up.

## Send back

- `s3_qwen_gen_control_norm1.jsonl` (or whatever suffix the script prints)
- its `.config.json` sidecar
