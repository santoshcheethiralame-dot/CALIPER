# Kaggle session: S-1, dead vectors by read position × precision

Prereg: `docs/preregistration-s1-dead-vectors.md`. Script: `kaggle_s3_positive_control.py`
**v2026-10-08a**. About 2-3 GPU-hours per model.

## Before you start (once)

1. **Upload the current script.** Make a new version of the `caliper-s3` dataset containing
   `experiments/kaggle_s3_positive_control.py` from the repo at or after the S-1 prereg
   commit. The cell below refuses an older copy.
2. **Inputs:**
   - Models: **Gemma 3, `gemma-3-12b-it`** (Transformers). Session 2 swaps it for
     `gemma-3-27b-it`. Accept the Gemma licence on Kaggle if asked.
   - Datasets: `caliper-s3`.
3. Session options: **GPU T4 x2**, Internet **On**.

## Cell 1 (after any restart)

Every block below is one notebook **code cell**, pasted as is. A `%%bash` first line
makes the cell run as shell. Without it, Kaggle runs the lines as Python and fails
with `SyntaxError`.

```
%%bash
pip install -q -U bitsandbytes accelerate transformers
```

## Cell 2: load the script by name, not by position

```python
import glob, os, sys, gc, torch, logging, transformers
# Quiet the per-call library warnings. On 7 Oct a bitsandbytes cast warning, repeated
# 468,832 times, overflowed papermill's output queue and failed S-11's first attempt.
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
WANT, VER = "kaggle_s3_positive_control.py", "2026-10-08a"
hits = [h for h in glob.glob("/kaggle/input/**/*.py", recursive=True)
        if os.path.basename(h) == WANT]
assert len(hits) == 1, f"expected one {WANT}, found {hits}"
SRC = open(hits[0]).read()
assert f'VERSION = "{VER}"' in SRC, "stale caliper-s3 dataset: upload the current script"
NS = {"__name__": "caliper_s3"}     # not "__main__", so the file defines main() only
exec(compile(SRC, WANT, "exec"), NS)
main = NS["main"]
print("loaded", VER)
```

## Cell 3, session 1: Gemma-3-12B, nine cells

```python
GRID = ["--alpha-frac", "0", "0.25", "0.5", "1.0"]
ARMS = {"tail": ["--vector-pos", "template-tail"],
        "concept": ["--vector-pos", "concept"],
        "sentence": ["--vector-recipe", "aperture"]}
PRECS = {"4bit": ["--quant", "4bit", "--compute-dtype", "fp32"],
         "8bit": ["--quant", "8bit"],
         "fp16": ["--quant", "none", "--compute-dtype", "fp16"]}
for prec, pflags in PRECS.items():
    for arm, aflags in ARMS.items():
        sys.argv = ["run", "--model", "gemma12", "--stage", "steer", *GRID, *pflags, *aflags,
                    "--out", f"/kaggle/working/s1_gemma12_{prec}_{arm}.jsonl"]
        print("\n######", prec, arm, flush=True)
        try:
            main()
        except SystemExit as e:          # e.g. fp16 overflow: recorded, not retried
            print("CELL STOPPED:", e, flush=True)
        gc.collect(); torch.cuda.empty_cache()
```

If the fp16 cells stop with "contain inf or nan", that is the prereg's "not runnable on free
hardware" outcome. Do not switch them to fp32.

## Cell 3, session 2: Gemma-3-27B, 4-bit only

Same cell, with `PRECS = {"4bit": ...}` only, `--model gemma` and `s1_gemma27_...` names.

## Cell 4: package the outputs

```python
import zipfile
TAG = "s1_gemma12"                     # session 2: "s1_gemma27"
with zipfile.ZipFile(f"/kaggle/working/{TAG}.zip", "w") as z:
    for f in sorted(glob.glob(f"/kaggle/working/{TAG}_*")):
        z.write(f, os.path.basename(f))
print(f"DOWNLOAD /kaggle/working/{TAG}.zip")
```

Download before the session expires. Each cell writes a `.jsonl`, a `.config.json` (alphas
used, precision actually loaded, library versions, per-vector health) and a `.vectors.npz`.

## What healthy output looks like

- `read position check:` shows the concept word for the concept arm and a template token
  for the tail arm.
- `alpha grid from fractions ...` gives four distinct numbers, the first 0.
- `STAGE steer: ...` and then `10 rows, ...` progress lines; 120 rows per cell.
- The summary table has an `med KL` column that is 0.000 at alpha 0.
