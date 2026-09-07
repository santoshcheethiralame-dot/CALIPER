# S1 multi-seed confirmatory run — on Kaggle, not your laptop

**Protocol frozen in `docs/preregistration-s1-multiseed.md` (7 Sep, plus a same-day
clarification on the k=2 arm). Do not edit the script to change any of it.**

This is CPU-or-GPU work with no model download beyond GPT-2 small (~500 MB), so it fits
free Kaggle comfortably and leaves your machine alone. On a T4 it should be far quicker
than the ~4 h it would take locally.

## One-time setup (~3 minutes)

The CALIPER repo is **private**, so Kaggle cannot pip-install it from GitHub the way the
APERTURE notebook did. Upload the code as a dataset instead — it is 56 KB.

1. Zip the folder `kaggle/bundle/` from the repo. It contains:
   - `caliper/` (`estimator.py`, `activations.py`, `runtime.py`, `__init__.py`)
   - `e01_gate.jsonl` — the 100 unit ids and C13's numbers
   - `run_s1_multiseed.py` — the frozen runner
2. kaggle.com → **Datasets** → **New Dataset** → drag the zip in.
   Title it exactly **`caliper-s1`**. Create.
3. New notebook → right panel → **Add Input** → Datasets → `caliper-s1` → Add.
4. Session options → Accelerator **GPU T4 x2** (one is plenty), Internet **On**.
   Internet is needed once, for the GPT-2 download.

## Cell 1

```python
import glob, shutil, os
src = glob.glob("/kaggle/input/caliper-s1/**/run_s1_multiseed.py", recursive=True)[0]
root = os.path.dirname(src)
shutil.copytree(root, "/kaggle/working/bundle", dirs_exist_ok=True)
print(os.listdir("/kaggle/working/bundle"))
```

The copy exists because Kaggle inputs are read-only and the script writes its resume file
beside itself. Confirm the listing shows `caliper`, `e01_gate.jsonl`,
`run_s1_multiseed.py`.

## Cell 2

```python
!python /kaggle/working/bundle/run_s1_multiseed.py
```

## What you should see

```
device: cuda
  Tesla T4
100 units from the C13 gate
  stimulus (20000, 768)
  1/100  n49 0.9991 (gate 1.0000)  ...s  eta ...m
```

Then a progress line per unit with a live ETA, and at the end a block reporting the
primary endpoint against its pre-registered criterion.

**If `device:` says `cpu`**, the accelerator is not attached. Stop and set it — the run
still works but takes hours instead of minutes.

## If the session dies

Re-run cell 2. The script appends one line per unit and skips units already present, so
it resumes where it stopped. **Download `/kaggle/working/s1_multiseed.jsonl` before the
session expires** — a Kaggle output that expires is gone (section 8).

## What comes back

The script prints the verdict itself, so you will know the answer before sending me
anything:

```
PRIMARY  multi-seed pass rate  XX/100
         Wilson 95% [...]
         criterion: lower bound > 0.90  ->  PASS / FAIL
REFERENCE  C13 single-fit pass rate on the same units  77/100
SECONDARY  paired change ... improved N/100, worsened N/100
           units C13 passed that this loses: N
```

That last line is the one to read carefully. A rule that repairs failures by breaking
healthy units is not a repair, and the pre-registration commits to reporting it either
way.

## Send back

`s1_multiseed.jsonl`. Nothing else is needed — every per-unit number, including C13's for
comparison, is in it.
