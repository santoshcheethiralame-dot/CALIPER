# Kaggle sessions: P2-F, the factual-NO control

Prereg: `docs/preregistration-p2f-factual-no.md`. Script: `kaggle_s3_positive_control.py`
**v2026-10-09c**. Forced choice only (first-token P(YES), no generation), so each session is
short: about 1-1.5 GPU-hours for 27B, under an hour for Qwen-7B.

## Before you start

- **Upload v2026-10-09c** to the `caliper-s3` dataset (new version) and refresh the input.
- One model per session: session 1 **`gemma-3-27b-it`**, session 2 **`qwen2.5` -> `7b-instruct`**.
- GPU T4 x2, Internet On.

## Cells 1 and 2

As in `NEXT_SESSION_S1.md`, with `VER = "2026-10-09c"` in Cell 2.

## Cell 3, session 1: Gemma-3-27B at the released operating point (primary)

```python
for control in ("none", "random"):
    sys.argv = ["run", "--model", "gemma", "--stage", "forced", "--quant", "4bit",
                "--compute-dtype", "fp32", "--layer", "37",
                "--vector-recipe", "macar-release", "--alphas", "0", "4", "8",
                "--inject-from", "trial", "--framing-set", "factual-no",
                "--control", control, "--no-health",
                "--out", "/kaggle/working/p2f_gemma27.jsonl"]
    print("\n###### P2-F gemma27", control, flush=True)
    try:
        main()
    except SystemExit as e:
        print("CELL STOPPED:", e, flush=True)
    gc.collect(); torch.cuda.empty_cache()
```

## Cell 3, session 2: Qwen2.5-7B, as in S-2 (secondary)

```python
GRID = ["--alpha-frac", "0", "0.25", "0.5", "1.0"]
ARMS = {"concept": ["--vector-pos", "concept"],
        "tail": ["--vector-pos", "template-tail"],
        "random": ["--control", "random"]}
for arm, aflags in ARMS.items():
    sys.argv = ["run", "--model", "qwen7b", "--stage", "forced", "--quant", "4bit",
                "--compute-dtype", "fp32", *GRID, *aflags,
                "--framing-set", "factual-no", "--no-health",
                "--out", f"/kaggle/working/p2f_qwen7b_{arm}.jsonl"]
    print("\n###### P2-F qwen7b", arm, flush=True)
    try:
        main()
    except SystemExit as e:
        print("CELL STOPPED:", e, flush=True)
    gc.collect(); torch.cuda.empty_cache()
```

## Cell 4: package

```python
import zipfile
TAG = "p2f_gemma27"                    # session 2: "p2f_qwen7b"
with zipfile.ZipFile(f"/kaggle/working/{TAG}.zip", "w") as z:
    for f in sorted(glob.glob(f"/kaggle/working/{TAG}*")):
        z.write(f, os.path.basename(f))
print(f"DOWNLOAD /kaggle/working/{TAG}.zip")
```

## What healthy output looks like

- `loaded 2026-10-09c`.
- 27B: `YES ids [...], NO ids [...]`, then 90 rows per control (30 questions x 3 strengths);
  files end `_forced_random_macar-release_factualno_fromtrial` and
  `_forced_macar-release_factualno_fromtrial`.
- Qwen-7B: 120 rows per arm (30 x 4 doses), files ending `_forced_factualno`.
- The summary table's P(YES) at alpha 0 should be near 0 (the manipulation check: the questions
  are unambiguous NOs). If it is not, say so when you send the zip.
