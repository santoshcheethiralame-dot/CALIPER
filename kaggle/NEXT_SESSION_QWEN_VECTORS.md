# A-10 — do the Qwen concept vectors work at all? (~10 minutes)

**Why.** C30 showed the injection never reaches Qwen's output: the concept appears in
0 of 30 responses even at 56% of the residual-stream norm, where Gemma produced bare
concept words in 10 of 30. Two causes are consistent with that, and they need
different fixes:

| Cause | Test result that indicates it | Fix |
|---|---|---|
| **The vectors are broken** on Qwen at layer 38 | concept still absent on a neutral prompt | vector construction, or sweep the layer |
| **Refusal training blocks it** — the model will not engage with introspective framing | concept appears once the framing is removed | the refusal becomes the finding (C30 Result 3) |

The introspective prompt is doing two jobs in every run so far: it frames the question
*and* it triggers Qwen's "As an AI, I don't have the capability..." policy in 30/30
clean trials. This run removes the framing so the injection is tested on its own.

## Setup

Unchanged: Qwen under **Models**, `caliper-s3` at `2026-09-07c` under **Datasets**,
GPU T4 x2, Internet on, **session restarted**.

## Cell 1 — only after a restart

```
pip install -q -U bitsandbytes accelerate transformers
```

## Cell 2 — steering on a neutral prompt

```python
import sys, glob
sys.argv = ["run", "--model", "qwen", "--compute-dtype", "fp32", "--stage", "steer",
            "--normalise", "--control", "none",
            "--alphas", "0", "25", "50", "100", "200",
            "--out", "/kaggle/working/s3_qwen_steer.jsonl"]
hits = glob.glob("/kaggle/input/**/*.py", recursive=True)
print(hits)
exec(open(hits[0]).read())
```

> **If `--stage steer` is rejected as an invalid choice**, the script does not have it
> yet — tell me and I will add it before you run. It should print the accepted values
> in the error.

Note alpha=200 (112% of the residual norm) is included deliberately. If a perturbation
larger than the residual stream itself still leaves the output untouched, the vectors
are certainly not carrying content and no prompt change will rescue them.

## Pre-flight

| Line | Must read |
|---|---|
| 1st line | `kaggle_s3_positive_control 2026-09-07c` |
| model source | `found N model dir(s) in /kaggle/input` |
| residual norm | `median 177.8` — must match C27-C30 exactly |
| vectors | `30 vectors, median norm 1.00, non-finite 0` |

## Read the answer yourself

Look at the `text` of a few alpha=100 or alpha=200 rows. You do not need me:

- **Concept words present** (`elephant`, `volcano`, ...) -> the vectors work.
  The refusal is the blocker, C30 Result 3 becomes a real finding, and the Qwen
  measurement needs a prompt that the model will actually engage with.
- **Still nothing** -> the vectors do not encode concepts on Qwen at layer 38.
  Next step is a layer sweep, not a prompt change. Qwen is then out of Paper A
  entirely and is future work.

## Send back

`s3_qwen_steer*.jsonl` and its `.config.json`.
