# A-11 — refit the vectors at the concept position, then re-run the steer control

**Script version required: `2026-09-07e`.** Re-upload as a New Version of `caliper-s3`.

**What changed.** C31 showed the concept vectors were read at the chat-template tail
(`<|im_start|>assistant` on Qwen) rather than at the concept word, and on Qwen they
carried nothing: 1/30 concept-in-text at 112% of the residual norm, identical to
injecting nothing. `--vector-pos concept` is now the default and averages the word's own
token positions. `--vector-pos template-tail` reproduces the old behaviour.

**What this run decides.**

| Steer result | Meaning | Next |
|---|---|---|
| Concept appears well above the 1/30 baseline | The extraction bug was the whole problem. Qwen comes back into scope and C27-C30 can be re-run properly | Re-run forced-choice on Qwen; the invariant test is live again |
| Still ~1/30 | The template tail was not the only issue. Layer 38 or the difference-of-means method itself is wrong for Qwen | Layer sweep, or Qwen stays out of Paper A |

## Setup

Qwen under **Models**, `caliper-s3` at **`2026-09-07e`** under **Datasets**,
GPU T4 x2, Internet on, **session restarted**.

## Cell 1 — after a restart

```
pip install -q -U bitsandbytes accelerate transformers
```

## Cell 2 — the steer control, refitted vectors

```python
import sys, glob
sys.argv = ["run", "--model", "qwen", "--compute-dtype", "fp32", "--stage", "steer",
            "--normalise", "--control", "none",
            "--alphas", "0", "25", "50", "100", "200",
            "--out", "/kaggle/working/s3_qwen_refit.jsonl"]
hits = glob.glob("/kaggle/input/**/*.py", recursive=True)
print(hits)
exec(open(hits[0]).read())
```

Same alphas as C31 so the two are directly comparable — the only change is where the
vector is read.

## The line that proves the fix took

After `building concept vectors`, you must see:

```
read position check: 'table' at token(s) [N] of M; decoded ' table' (template tail is '\n')
```

**If `decoded` is not the word itself, stop the run.** That is the C31 bug still live and
nothing after it is worth reading.

Other pre-flight lines, unchanged:

| Line | Must read |
|---|---|
| 1st line | `kaggle_s3_positive_control 2026-09-07e` |
| model source | `found N model dir(s) in /kaggle/input` |
| stage banner | `STAGE steer: neutral prompt, injection at EVERY position ...` |
| vectors | `30 vectors, median norm 1.00, non-finite 0` |

Note the residual norm will now be reported **at the concept token**, so it will differ
from the 177.8 measured at the template tail. That is expected, not a bug — and it is
itself worth recording, since 177.8 was the number in every Qwen table so far.

## Read the answer yourself

Look at the `text` of a few alpha=100 rows. Concept words present in far more than 1 of
30 means the fix worked.

## Send back

`s3_qwen_refit_steer_norm1.jsonl` and `s3_qwen_refit.config.json`. The sidecar now
carries `vector_read_position`, so every future run is self-describing on this.
