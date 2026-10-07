# Kaggle sessions: S-2, the instrument audit on small models

Prereg: `docs/preregistration-s2-instrument-audit.md`. Script: `kaggle_s3_positive_control.py`
**v2026-10-07b**. Plan one session per model: Qwen2.5-3B, Qwen2.5-7B, Gemma-3-4B.

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
MODEL, DTYPE = "qwen3b", "fp16"        # qwen7b fp16; gemma4b fp32 (Gemma overflows in fp16)
GRID = ["--alpha-frac", "0", "0.25", "0.5", "1.0"]
ARMS = {"concept":  ["--vector-pos", "concept"],
        "tail":     ["--vector-pos", "template-tail"],
        "sentence": ["--vector-recipe", "aperture"],
        "random":   ["--control", "random"],
        "impact":   ["--control", "random-impact"],
        "shuffle":  ["--control", "shuffle"],
        "span":     ["--control", "span"]}
for stage in ("steer", "forced", "framing"):
    for arm, aflags in ARMS.items():
        sys.argv = ["run", "--model", MODEL, "--quant", "none", "--compute-dtype", DTYPE,
                    "--stage", stage, *GRID, *aflags,
                    "--out", f"/kaggle/working/s2_{MODEL}_{arm}.jsonl"]
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
with zipfile.ZipFile(f"/kaggle/working/s2_{MODEL}.zip", "w") as z:
    for f in sorted(glob.glob(f"/kaggle/working/s2_{MODEL}_*")):
        z.write(f, os.path.basename(f))
print(f"DOWNLOAD /kaggle/working/s2_{MODEL}.zip")
```

## Expected cost

On a 3B model a cell runs in a few minutes. `random-impact` adds about 14 forward passes per
concept and alpha. `framing` generates 2 framings × 4 alphas × 30 concepts. Expect about
2-3 h for 3B, 4-5 h for 7B and about 3 h for Gemma-4B in fp32.
