# Kaggle session: P2-Q2, P2-Q with an answer instruction (P2-Q Amendment 1)

Prereg: `docs/preregistration-p2q-qwen-logodds.md`, Amendment 1 (filed 11 Oct, before data).
Script `kaggle_s3_positive_control.py` **v2026-10-11a**. About 1.5-2 GPU-hours.

## Before you start

- Upload the current `experiments/kaggle_s3_positive_control.py` to the `caliper-s3` dataset
  (new version). Cell 2 refuses anything older than **2026-10-11a**.
- Model input: **`qwen2.5` -> `7b-instruct`**, as in P2-Q.
- GPU T4 x2, Internet On.

## Cells 1 and 2

As in `NEXT_SESSION_S1.md`, with `VER = "2026-10-11a"` in Cell 2.

## Cell 3: five cells in order (each resumes by key after a restart)

```python
ANS = ["--answer-instruction", "Start your answer with YES or NO."]
COMMON = ["--model", "qwen7b", "--quant", "4bit", "--compute-dtype", "fp32",
          "--alpha-frac", "0", "0.25", "0.5", "1.0", "--inject-from", "trial", "--no-health"]
RUNS = [
    ("Q1 concept, forced", "concept", ["--stage", "forced", "--framing-set", "p2r", "--vector-pos", "concept", *ANS]),
    ("Q2 tail, forced", "tail", ["--stage", "forced", "--framing-set", "p2r", "--vector-pos", "template-tail", *ANS]),
    ("Q3 random, forced", "random", ["--stage", "forced", "--framing-set", "p2r", "--control", "random", *ANS]),
    ("Q7 random, identify", "random", ["--stage", "identify", "--control", "random"]),
    ("Q4 impact-matched random, forced", "impact", ["--stage", "forced", "--framing-set", "p2r",
                                                  "--control", "random-impact", *ANS]),
]
for name, tag, flags in RUNS:
    sys.argv = ["run", *COMMON, *flags, "--out", f"/kaggle/working/p2q2_qwen7b_{tag}.jsonl"]
    print(f"\n###### P2-Q2 {name}", flush=True)
    try:
        main()
    except SystemExit as e:
        print("CELL STOPPED:", e, flush=True)
    gc.collect(); torch.cuda.empty_cache()
```

## Cell 4: package

```python
import zipfile
TAG = "p2q2_qwen7b"
with zipfile.ZipFile(f"/kaggle/working/{TAG}.zip", "w") as z:
    for f in sorted(glob.glob(f"/kaggle/working/{TAG}*")):
        if not f.endswith(".zip"):
            z.write(f, os.path.basename(f))
print(f"DOWNLOAD /kaggle/working/{TAG}.zip")
```

## What healthy output looks like

- The first log line says `kaggle_s3_positive_control 2026-10-11a`.
- Forced cells: 600 rows each. **Look at `med mass` per framing at dose 0 in the summary: it should
  now be well above 0.5.** If it is still near 0, stop after Q1 and send the log; the remaining
  cells would be unscorable too.
- Identify cell: 120 rows.
- Files end `_forced_p2r_ans_fromtrial`, `_forced_random_p2r_ans_fromtrial`,
  `_forced_random-impact_p2r_ans_fromtrial` and `_identify_random_fromtrial`.

## After the run

Send `p2q2_qwen7b.zip` (download it before the session ends). Scoring runs once per arm:
`python experiments/analyse_p2r.py --preset qwen7b-concept-ans --dir <unzipped> --out results/p2q2_concept.json`
and `--preset qwen7b-tail-ans --out results/p2q2_tail.json`.
