# Next Kaggle session — A-1b and A-1c

Script version required: **2026-09-07a**. Anything else and the run is void.

## Before you start

1. kaggle.com -> Datasets -> **caliper-s3** -> New Version -> upload
   `experiments/kaggle_s3_positive_control.py`. Save.
2. Open the notebook. Right panel -> Input. You need **two** entries:
   - **Gemma 3 / gemma-3-27b-it** under *Models* (this is what makes it load with
     no token and no download)
   - **caliper-s3** under *Datasets*, refreshed to the version you just uploaded
3. Session options -> Accelerator **GPU T4 x2**, Internet **On**.
4. **Run -> Restart session** before the first load. Stale GPU memory from a previous
   session is the single most common cause of a spurious OOM here.

## Cell 1 — only after a restart

```
pip install -q -U bitsandbytes accelerate transformers
```

## Cell 2 — A-1b, the grid extension

```python
import sys, glob
sys.argv = ["run", "--model", "gemma", "--compute-dtype", "fp32", "--stage", "forced",
            "--normalise", "--control", "none",
            "--alphas", "2048", "4096", "8192", "16384", "32768",
            "--out", "/kaggle/working/s3_unit_ext.jsonl"]
hits = glob.glob("/kaggle/input/**/*.py", recursive=True)
print(hits)
exec(open(hits[0]).read())
```

## Cell 3 — A-1c, the random control, same grid

```python
import sys, glob
sys.argv = ["run", "--model", "gemma", "--compute-dtype", "fp32", "--stage", "forced",
            "--normalise", "--control", "random",
            "--alphas", "2048", "4096", "8192", "16384", "32768",
            "--out", "/kaggle/working/s3_unit_ext.jsonl"]
hits = glob.glob("/kaggle/input/**/*.py", recursive=True)
print(hits)
exec(open(hits[0]).read())
```

Run cell 2, wait for it to finish, then cell 3. Not Run All.

## Four lines to check before letting it run

| Line | Must read | If it doesn't |
|---|---|---|
| 1st line of output | `kaggle_s3_positive_control 2026-09-07a` | The dataset didn't refresh. Remove the input and re-add it. **Nothing after a wrong stamp is worth reading** |
| GPU memory | `cuda:0 ~15.5 of 15.6 GB free before load` on **both** cards | Stop and restart the session. Do not let it load |
| after loading | `probe: max\|h\| = 51436.5, all finite = True` | fp32 didn't take; check the `--compute-dtype fp32` flag is present |
| after vectors | `30 vectors, median norm 1.00, non-finite 0` | `--normalise` didn't take. Norm must be 1.00, not 5002 |

In cell 3 you must **also** see `CONTROL random: vectors replaced` after the vector
step. If that line is missing it ran the real vectors and the result is meaningless.

## What to send back

Both files, downloaded from the right panel (**Output -> /kaggle/working**, download
arrow — not `FileLink`, it 404s here):

- `s3_unit_ext_forced_norm1.jsonl`
- `s3_unit_ext_forced_random_norm1.jsonl`

Plus **the `.scalars.json` sidecar for each** — that is the new bit in this version, and
it carries the residual-stream norm at the read position. That number has never been
reported for this model and layer, and it is what converts Macar et al.'s alpha=4 into a
relative perturbation size.

## Time budget

~5 min to load, ~12 min per condition (300 forward passes each, no generation).
About 35 minutes total. Well inside a 12-hour session.

## If it fails

| Symptom | Cause | Fix |
|---|---|---|
| OOM at ~6% loaded | previous session's weights still resident | Run -> Restart session, then retry |
| `median norm nan` | fp16 overflow (Gemma peaks at 51436 against fp16's 65504 ceiling) | `--compute-dtype fp32` is missing |
| stamp is older than `2026-09-07a` | dataset didn't refresh | remove input, re-add, re-run |
| `ImportError: bitsandbytes` | transformers cached the absence at import | restart the kernel, run cell 1 first |
| Session dies partway | — | The out file appends. Just re-run the same cell; it continues |
