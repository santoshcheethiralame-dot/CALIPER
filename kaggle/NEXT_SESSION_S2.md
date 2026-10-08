# Kaggle sessions: S-2, the instrument audit on small models

Prereg: `docs/preregistration-s2-instrument-audit.md`. Script: `kaggle_s3_positive_control.py`
**v2026-10-09b**. Qwen2.5-3B and 7B are done (alpha-frac grid, as filed). **Gemma-3-4B is
re-run on the KL-calibrated grid** (prereg Amendment 3); its alpha-frac session failed the
manipulation check.

## Before you start

- **`caliper-s3`:** current script uploaded, as for S-1.
- **Models:** exactly **one** per session:
  - `qwen2.5` → `3b-instruct` or `7b-instruct`;
  - `gemma-3` → `gemma-3-4b-it`.
- **Session options:** GPU T4 x2, Internet On.
- **Cells 1 and 2:** the same as in `NEXT_SESSION_S1.md` (pip install; load the script by name
  with the version check).

## Cell 3: every arm × stage for one model

```python
MODEL, QUANT, DTYPE = "gemma4b", "none", "fp32"
RUN = "gemma4b_kl"                    # file prefix: keeps the failed alpha-frac files apart
# qwen7b: "4bit", "fp32" (fp16 overflows; fp32 does not fit a T4 pair; Amendment 2)
# gemma4b: "none", "fp32" (Gemma overflows in fp16)
GRID = ["--calibrate-kl", "0.05", "0.5", "5"]     # Amendment 3
# The Qwen sessions used ["--alpha-frac", "0", "0.25", "0.5", "1.0"]; RUN = MODEL there.
ARMS = {"concept":  ["--vector-pos", "concept"],
        "tail":     ["--vector-pos", "template-tail"],
        "sentence": ["--vector-recipe", "aperture"],
        "random":   ["--control", "random"],
        "impact":   ["--control", "random-impact"],
        "shuffle":  ["--control", "shuffle"],
        "span":     ["--control", "span"]}
for stage in ("steer", "forced", "framing"):
    for arm, aflags in ARMS.items():
        sys.argv = ["run", "--model", MODEL, "--quant", QUANT, "--compute-dtype", DTYPE,
                    "--stage", stage, *GRID, *aflags,
                    "--out", f"/kaggle/working/s2_{RUN}_{arm}.jsonl"]
        print("\n######", stage, arm, flush=True)
        try:
            main()
        except SystemExit as e:
            print("CELL STOPPED:", e, flush=True)
        gc.collect(); torch.cuda.empty_cache()
```

- **Output files.** Each arm keeps one `--out` across the three stages. The script derives
  `_steer`, `_forced` and the control tag itself, so nothing collides.
- **Resuming.** Every stage resumes by key, so re-running the cell after a disconnect continues
  where it stopped.

## Cell 4: package the outputs

```python
import zipfile
with zipfile.ZipFile(f"/kaggle/working/s2_{RUN}.zip", "w") as z:
    for f in sorted(glob.glob(f"/kaggle/working/s2_{RUN}_*") +
                    glob.glob("/kaggle/working/kl_calibration_*.json")):
        z.write(f, os.path.basename(f))
print(f"DOWNLOAD /kaggle/working/s2_{RUN}.zip")
```

## Expected cost

On a 3B model a cell runs in a few minutes. `random-impact` adds about 14 forward passes per
concept and alpha. `framing` generates 2 framings × 4 alphas × 30 concepts. Expect about
2-3 h for 3B, 4-5 h for 7B and about 3 h for Gemma-4B in fp32. The KL calibration adds a few
minutes to the first cell only.

## What healthy output looks like (calibrated grid)

- First cell: `KL calibration (...s)`; every later cell: `KL calibration reused`.
- `alpha grid from KL targets [0.05, 0.5, 5.0] nats:` with four increasing numbers, the same
  in every cell.
- In the steer cells at the 0.5-nat dose, most generations still read as English.
