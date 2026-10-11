# Kaggle session: B1, what Gemma-3-27B says when YES and NO lose their mass (P2-R Amendment 1)

Prereg: `docs/preregistration-p2r-released-protocol.md`, Amendment 1 (filed 11 Oct, before data).
Script `kaggle_s3_positive_control.py` **v2026-10-11b**. About 2-3 GPU-hours.

## Before you start

- Upload the current `experiments/kaggle_s3_positive_control.py` to `caliper-s3` as a new dataset
  version, then in the notebook's Input panel update `caliper-s3` to that latest version. Cell 2
  refuses anything older than **2026-10-11b**.
- Model input: **`gemma-3-27b-it`**, as in the P2-R session (`NEXT_SESSION_P2R.md`).
- GPU T4 x2, Internet On.

## Cells 1 and 2

As in `NEXT_SESSION_S1.md`, with `VER = "2026-10-11b"` in Cell 2.

## Cell 3: two cells (each resumes by key after a restart)

```python
COMMON = ["--model", "gemma", "--quant", "4bit", "--compute-dtype", "fp32", "--layer", "37",
          "--vector-recipe", "macar-release", "--alphas", "0", "1", "2", "4", "8",
          "--inject-from", "trial", "--no-health", "--store-top-k", "20", "--continuation", "20",
          "--out", "/kaggle/working/b1_gemma27.jsonl"]
RUNS = [("R1 forced, released vectors", ["--stage", "forced", "--framing-set", "p2r", "--control", "none"]),
        ("R2 forced, norm-matched random", ["--stage", "forced", "--framing-set", "p2r", "--control", "random"])]
for name, flags in RUNS:
    sys.argv = ["run", *COMMON, *flags]
    print(f"\n###### B1 {name}", flush=True)
    try:
        main()
    except SystemExit as e:
        print("CELL STOPPED:", e, flush=True)
    gc.collect(); torch.cuda.empty_cache()
```

## Cell 4: package

```python
import zipfile
TAG = "b1_gemma27"
with zipfile.ZipFile(f"/kaggle/working/{TAG}.zip", "w") as z:
    for f in sorted(glob.glob(f"/kaggle/working/{TAG}*")):
        if not f.endswith(".zip"):
            z.write(f, os.path.basename(f))
print(f"DOWNLOAD /kaggle/working/{TAG}.zip")
```

## What healthy output looks like

- First line `kaggle_s3_positive_control 2026-10-11b`.
- 750 rows per cell; files end `_forced_macar-release_p2r_tok_fromtrial` and
  `_forced_random_macar-release_p2r_tok_fromtrial`.
- The per-framing summary's `med mass` should match P2-R's (introspective about 1.0 at strength 0,
  about 1e-4 at 4).

## After the run

Send `b1_gemma27.zip`. Scoring runs once: `python experiments/analyse_b1_tokens.py --dir <unzipped>`.
