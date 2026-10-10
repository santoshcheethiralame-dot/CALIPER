# Kaggle session: P2-R, the released protocol on Gemma-3-27B (log-odds)

Prereg: `docs/preregistration-p2r-released-protocol.md` (filed 10 Oct, before data). Script:
`kaggle_s3_positive_control.py` **v2026-10-10a**. Forced choice and the identify stage only (no
free generation). About 4-6 GPU-hours on one T4; the impact-matched cell is the slow one.

## Before you start

- **Upload v2026-10-10a** to the `caliper-s3` dataset as a new version and refresh the input. The
  old 10-09c copy has no `--framing-set p2r` and stores no YES/NO logits; Cell 2 refuses it.
- Model input: **`gemma-3-27b-it`** (as in the S-1M and P2-F sessions).
- GPU T4 x2, Internet On.

## Cells 1 and 2

As in `NEXT_SESSION_S1.md`, with `VER = "2026-10-10a"` in Cell 2.

## Cell 3: the four cells (run in this order; each resumes by key if the session restarts)

```python
COMMON = ["--model", "gemma", "--quant", "4bit", "--compute-dtype", "fp32", "--layer", "37",
          "--vector-recipe", "macar-release", "--alphas", "0", "1", "2", "4", "8",
          "--inject-from", "trial", "--no-health", "--out", "/kaggle/working/p2r_gemma27.jsonl"]
RUNS = [("R1 forced, released vectors", ["--stage", "forced", "--framing-set", "p2r", "--control", "none"]),
        ("R4 identify, released vectors", ["--stage", "identify", "--control", "none"]),
        ("R2 forced, norm-matched random", ["--stage", "forced", "--framing-set", "p2r", "--control", "random"]),
        ("R4 identify, norm-matched random", ["--stage", "identify", "--control", "random"]),
        ("R3 forced, impact-matched random", ["--stage", "forced", "--framing-set", "p2r", "--control", "random-impact"])]
for name, flags in RUNS:
    sys.argv = ["run", *COMMON, *flags]
    print(f"\n###### P2-R {name}", flush=True)
    try:
        main()
    except SystemExit as e:
        print("CELL STOPPED:", e, flush=True)
    gc.collect(); torch.cuda.empty_cache()
```

R3 runs last because it is the slowest. If the session ends early, Cell 4 still packages whatever
finished, and the rest is reported as not run.

## Cell 4: package

```python
import zipfile
TAG = "p2r_gemma27"
with zipfile.ZipFile(f"/kaggle/working/{TAG}.zip", "w") as z:
    for f in sorted(glob.glob(f"/kaggle/working/{TAG}*")):
        z.write(f, os.path.basename(f))
print(f"DOWNLOAD /kaggle/working/{TAG}.zip")
```

## What healthy output looks like

- `loaded 2026-10-10a`, then `YES ids [...], NO ids [...]`.
- Forced cells: 750 rows each (5 framings x 5 strengths x 30). The summary table now has
  `med logodds` and `med mass` columns. At strength 0:
  - `factual_no` log-odds clearly negative and `factual_yes` clearly positive;
  - `med mass` per framing. **If a framing's mass at strength 0 is below 0.5, say so when you send
    the zip.** The released prompt does not ask for YES or NO, so the model may start its answer
    with another word; that is a filed check, not a failure of the run.
- Identify cells: 150 rows each, with `med logp(concept)` rising with strength for the released
  vectors.
- Files end `_forced_macar-release_p2r_fromtrial`, `_forced_random_macar-release_p2r_fromtrial`,
  `_forced_random-impact_macar-release_p2r_fromtrial`, `_identify_macar-release_fromtrial` and
  `_identify_random_macar-release_fromtrial`.

## After the run

Send `p2r_gemma27.zip`. Scoring runs once: `python experiments/analyse_p2r.py --dir <unzipped>`.
