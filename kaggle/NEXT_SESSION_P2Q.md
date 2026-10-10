# Kaggle session: P2-Q, the P2-R design on Qwen2.5-7B

Prereg: `docs/preregistration-p2q-qwen-logodds.md` (filed 10 Oct, before data). Script
`kaggle_s3_positive_control.py` **v2026-10-10a**, the same upload as P2-R. About 1.5-2.5 GPU-hours.
It can run in a separate session from P2-R.

## Before you start

- `caliper-s3` must hold **v2026-10-10a** (the P2-R upload). Cell 2 refuses anything older.
- Model input: **`qwen2.5` -> `7b-instruct`** (as in the S-2 and P2-F Qwen sessions).
- GPU T4 x2, Internet On.

## Cells 1 and 2

As in `NEXT_SESSION_S1.md`, with `VER = "2026-10-10a"` in Cell 2.

## Cell 3: six cells in order (each resumes by key after a restart)

```python
COMMON = ["--model", "qwen7b", "--quant", "4bit", "--compute-dtype", "fp32",
          "--alpha-frac", "0", "0.25", "0.5", "1.0", "--inject-from", "trial", "--no-health"]
RUNS = [
    ("Q1 concept, forced", "concept", ["--stage", "forced", "--framing-set", "p2r", "--vector-pos", "concept"]),
    ("Q2 tail, forced", "tail", ["--stage", "forced", "--framing-set", "p2r", "--vector-pos", "template-tail"]),
    ("Q3 random, forced", "random", ["--stage", "forced", "--framing-set", "p2r", "--control", "random"]),
    ("Q5 concept, identify", "concept", ["--stage", "identify", "--vector-pos", "concept"]),
    ("Q6 tail, identify", "tail", ["--stage", "identify", "--vector-pos", "template-tail"]),
    ("Q4 impact-matched random, forced", "impact", ["--stage", "forced", "--framing-set", "p2r",
                                                  "--control", "random-impact"]),
]
for name, tag, flags in RUNS:
    sys.argv = ["run", *COMMON, *flags, "--out", f"/kaggle/working/p2q_qwen7b_{tag}.jsonl"]
    print(f"\n###### P2-Q {name}", flush=True)
    try:
        main()
    except SystemExit as e:
        print("CELL STOPPED:", e, flush=True)
    gc.collect(); torch.cuda.empty_cache()
```

## Cell 4: package

```python
import zipfile
TAG = "p2q_qwen7b"
with zipfile.ZipFile(f"/kaggle/working/{TAG}.zip", "w") as z:
    for f in sorted(glob.glob(f"/kaggle/working/{TAG}*")):
        z.write(f, os.path.basename(f))
print(f"DOWNLOAD /kaggle/working/{TAG}.zip")
```

## What healthy output looks like

- Forced cells: 600 rows each (5 framings x 4 doses x 30). At dose 0, `factual_no` log-odds
  negative, `factual_yes` positive, and `med mass` per framing. **Report any framing with mass below
  0.5 at dose 0.**
- Identify cells: 120 rows each.
- Files end `_forced_p2r_fromtrial`, `_forced_random_p2r_fromtrial`,
  `_forced_random-impact_p2r_fromtrial` and `_identify_fromtrial`.

## After the run

Send `p2q_qwen7b.zip`. Scoring runs once per arm:
`python experiments/analyse_p2r.py --preset qwen7b-concept --dir <unzipped> --out results/p2q_concept.json`
and the same with `--preset qwen7b-tail --out results/p2q_tail.json`.
