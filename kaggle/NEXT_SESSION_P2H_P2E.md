# Kaggle session: P2-H then P2-E, both on Llama-3.1-8B-Instruct

Preregs (both filed 13 Oct, before data): `docs/preregistration-p2h-hahami-llama.md` and
`docs/preregistration-p2e-eval-awareness.md`. Scripts: `kaggle_p2h_hahami.py` **v2026-10-13a**,
`kaggle_p2e_eval_awareness.py` **v2026-10-13a**, both importing `kaggle_s3_positive_control.py`
**v2026-10-11b**. Forced choice plus a short steering gate (P2-H only). About 1-1.5 GPU-hours for
P2-H, then 3-5 for P2-E. P2-H runs first because Paper 2's ICML case needs it (review ruling 9).

## Before you start (once)

1. **Upload three scripts** as one new version of the `caliper-s3` dataset, in one folder:
   `experiments/kaggle_p2h_hahami.py`, `experiments/kaggle_p2e_eval_awareness.py`,
   `experiments/kaggle_s3_positive_control.py`.
2. **Inputs:** Model **Llama 3.1, `8b-instruct`** (Transformers) from Kaggle Models (accept Meta's
   licence on Kaggle if asked); dataset `caliper-s3`.
3. Session options: **GPU T4 x2**, Internet **On** (P2-H downloads Hahami et al.'s released
   vectors at a pinned commit for the fidelity check; a failed download is recorded, not fatal).

## Cell 1 (after any restart)

```
%%bash
pip install -q -U bitsandbytes accelerate transformers
```

## Cell 2: load the scripts by name and check versions

```python
import glob, os, sys, gc, torch, logging, transformers
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
NEED = {"kaggle_p2h_hahami.py": "2026-10-13a",
        "kaggle_p2e_eval_awareness.py": "2026-10-13a",
        "kaggle_s3_positive_control.py": "2026-10-11b"}
dirs = set()
for name, ver in NEED.items():
    hits = [h for h in glob.glob("/kaggle/input/**/*.py", recursive=True)
            if os.path.basename(h) == name]
    assert len(hits) == 1, f"expected one {name}, found {hits}"
    assert f'VERSION = "{ver}"' in open(hits[0]).read(), f"stale {name}: upload the current one"
    dirs.add(os.path.dirname(hits[0]))
assert len(dirs) == 1, f"the scripts must be in one folder, found {dirs}"
sys.path.insert(0, dirs.pop())
import kaggle_p2h_hahami as p2h
import kaggle_p2e_eval_awareness as p2e
print("loaded", p2h.VERSION, p2e.VERSION, p2h.s3.VERSION)
PREC = ["--quant", "none", "--compute-dtype", "fp16"]
```

## Cell 3: P2-H

```python
sys.argv = ["run", *PREC, "--out", "/kaggle/working/p2h_llama8b.jsonl"]
try:
    p2h.main()
except SystemExit as e:
    print("STOPPED:", e, flush=True)
p2h._CLEAN.clear(); gc.collect(); torch.cuda.empty_cache()
```

## Cell 4: P2-E

```python
sys.argv = ["run", *PREC, "--out", "/kaggle/working/p2e_llama8b.jsonl"]
try:
    p2e.main()
except SystemExit as e:
    print("STOPPED:", e, flush=True)
p2e.s3._CLEAN.clear(); gc.collect(); torch.cuda.empty_cache()
```

**If either stops with `M1 FAILED`** (non-finite logits in fp16): delete that run's `.jsonl` and
`_sidecar.json`, restart the session, run Cells 1-2, set
`PREC = ["--quant", "4bit", "--compute-dtype", "fp32"]` (the filed fallback), and rerun that cell.
Say which precision each run used when you send the zip. Rows resume by key with the same flags.

## Cell 5: package (run whenever you stop, even partway)

```python
import zipfile
with zipfile.ZipFile("/kaggle/working/p2h_p2e_llama8b.zip", "w") as z:
    for f in sorted(glob.glob("/kaggle/working/p2[he]_llama8b*")):
        if not f.endswith(".zip"):
            z.write(f, os.path.basename(f))
print("DOWNLOAD /kaggle/working/p2h_p2e_llama8b.zip")
```

## What healthy output looks like

**P2-H**
- `kaggle_p2h_hahami 2026-10-13a`, then `fidelity {...}`: ten cosines (Dust_0 ... Illusions_19).
  Values near 1 mean our rebuild matches their release; anything below 0.95 or `unavailable`,
  say so.
- `layer 0 alpha 5.0 done` first (the primary cell), then alphas 3, 1, 2, 4, then layer 19, then
  `steering gate done`. 120 clean rows, 360 per layer and strength, 150 steering rows: 3,870 rows.

**P2-E**
- `layer L; AUROC x` (say if below 0.70), three `dose {...}` lines that should each end
  `'matched': True` (say if not), then `dose 0.5 done`, `dose 0.1 done`, `dose 2.0 done`: 12,540 rows.

## After the run

Send `p2h_p2e_llama8b.zip`. Each run is scored once:
- `python experiments/analyse_p2h.py --file <unzipped>/p2h_llama8b.jsonl`
- `python experiments/analyse_p2e.py --file <unzipped>/p2e_llama8b.jsonl`
