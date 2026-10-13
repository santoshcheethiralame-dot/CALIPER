# Kaggle session: P2-E, eval-awareness steering against matched controls (Llama-3.1-8B)

Prereg: `docs/preregistration-p2e-eval-awareness.md` (filed 13 Oct, before data). Scripts:
`kaggle_p2e_eval_awareness.py` **v2026-10-13a**, which imports `kaggle_s3_positive_control.py`
**v2026-10-11b**. Forced choice only (first-token YES/NO log-odds, no generation). About 3-5
GPU-hours for all three doses; dose 0.5 runs first.

## Before you start (once)

1. **Upload both scripts** as one new version of the `caliper-s3` dataset:
   `experiments/kaggle_p2e_eval_awareness.py` and `experiments/kaggle_s3_positive_control.py`.
   They must sit in the same folder; the P2-E script imports the other by name.
2. **Inputs:**
   - Model: **Llama 3.1, `8b-instruct`** (Transformers), from Kaggle Models. Accept Meta's licence
     on Kaggle if asked. The script finds it by the tokens `llama` and `8b` in the mount path.
   - Dataset: `caliper-s3`.
3. Session options: **GPU T4 x2**, Internet **On**.

## Cell 1 (after any restart)

```
%%bash
pip install -q -U bitsandbytes accelerate transformers
```

## Cell 2: load both scripts by name and check versions

```python
import glob, os, sys, gc, torch, logging, transformers
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
NEED = {"kaggle_p2e_eval_awareness.py": "2026-10-13a",
        "kaggle_s3_positive_control.py": "2026-10-11b"}
dirs = set()
for name, ver in NEED.items():
    hits = [h for h in glob.glob("/kaggle/input/**/*.py", recursive=True)
            if os.path.basename(h) == name]
    assert len(hits) == 1, f"expected one {name}, found {hits}"
    assert f'VERSION = "{ver}"' in open(hits[0]).read(), f"stale {name}: upload the current one"
    dirs.add(os.path.dirname(hits[0]))
assert len(dirs) == 1, f"the two scripts must be in one folder, found {dirs}"
sys.path.insert(0, dirs.pop())
import kaggle_p2e_eval_awareness as p2e
print("loaded", p2e.VERSION, p2e.s3.VERSION)
```

## Cell 3: run

```python
OUT = "/kaggle/working/p2e_llama8b.jsonl"
FLAGS = ["--quant", "none", "--compute-dtype", "fp16", "--out", OUT]
sys.argv = ["run", *FLAGS]
try:
    p2e.main()
except SystemExit as e:
    print("STOPPED:", e, flush=True)
```

**If it stops with `M1 FAILED`** (non-finite logits in fp16), delete `OUT` and its sidecar, restart
the session, run Cells 1-2 again, and use the filed fallback:

```python
FLAGS = ["--quant", "4bit", "--compute-dtype", "fp32", "--out", OUT]
```

Say which one ran when you send the zip. Rows resume by key, so a restarted session continues
where it stopped (same flags only).

## Cell 4: package

```python
import zipfile
with zipfile.ZipFile("/kaggle/working/p2e_llama8b.zip", "w") as z:
    for f in sorted(glob.glob("/kaggle/working/p2e_llama8b*")):
        if not f.endswith(".zip"):
            z.write(f, os.path.basename(f))
print("DOWNLOAD /kaggle/working/p2e_llama8b.zip")
```

## What healthy output looks like

- `kaggle_p2e_eval_awareness 2026-10-13a` and `kaggle_s3_positive_control 2026-10-11b`, then
  `YES ids [...], NO ids [...]`.
- `layer L; AUROC x`. **If the AUROC is below 0.70, say so**: the run continues as filed, but the
  low value goes beside every endpoint.
- Three `dose {...}` lines. **Each should say `'matched': True`.** A dose with `False` is unscored
  under M5. Report it; do not rerun with other targets.
- Then `dose 0.5 done`, `dose 0.1 done`, `dose 2.0 done`. 120 clean rows first, then 4,140 rows
  per dose (120 prompts x 21 arms, plus 90 prompts x 18 unrelated impact-matched arms): 12,540 in
  all.

## After the run

Send `p2e_llama8b.zip`. Scoring runs once:
`python experiments/analyse_p2e.py --file <unzipped>/p2e_llama8b.jsonl`.
