# Kaggle session: flagship substrate pilot, round 3 (OPT only; discarded)

Round 2 (`results/flagship_pilot2/`, read 10 Oct) asked OPT-1.3b for 64k and 128k tokens, but the
corpus sample held only about 34k, so both cells ran on the same ~34k tokens and gave identical
rows. `e01_gate.py` now draws the whole corpus pool for budgets above 32k (about 1,080 documents,
~135k tokens for OPT) and stops if it gets fewer tokens than asked. Budgets of 32k or less keep
their old sample, so no earlier run changes. About 2-3 GPU-hours.

## Before you start

1. **Upload the rebuilt bundle**: Datasets -> **caliper-bundle** -> **New Version** ->
   `kaggle/caliper-bundle.zip` (rebuilt 10 Oct, night; the same upload serves the F-4/F-6 session). The cell refuses older versions.
2. Notebook -> **Input:** attach **caliper-bundle**, refreshed to the new version.
3. Session options: **GPU T4 x2** (one is used), **Internet On**.

## The cell (paste as the only cell)

```python
import sys, os, glob, subprocess, shutil, json

hits = glob.glob("/kaggle/input/**/caliper/estimator.py", recursive=True)
assert hits, "bundle not found under /kaggle/input - is caliper-bundle attached?"
root = os.path.dirname(os.path.dirname(hits[0]))
sys.path.insert(0, root); os.chdir(root); os.environ["PYTHONPATH"] = root
src = open("experiments/e01_gate.py").read()
assert "(300 if a.tokens <= 32000 else 2000)" in src, \
    "STALE BUNDLE: upload the 10 Oct (evening) caliper-bundle.zip as a new version and re-attach"
print("root:", root, "| bundle OK (corpus fix)")

import torch
assert torch.cuda.is_available(), "no GPU - turn on the T4 accelerator"

OPT = ["--model", "facebook/opt-1.3b", "--layer", "12", "--d-mlp", "8192"]
for tokens in (64000, 120000):
    out = f"/kaggle/working/pilot3_f3_opt13b_l12_{tokens // 1000}k.jsonl"
    print(f"\n###### OPT {tokens} tokens", flush=True)
    r = subprocess.run([sys.executable, "experiments/e01_gate.py", *OPT,
                        "--neurons", "16", "--tokens", str(tokens), "--steps", "3200",
                        "--restarts", "2", "--independent-units", "--device", "cuda",
                        "--out", out], check=False)
    print(f"exit {r.returncode}", flush=True)

for f in sorted(glob.glob("/kaggle/working/pilot3_*.jsonl")):
    rows = [json.loads(l) for l in open(f)]
    al = [r["align_selected"] for r in rows]
    print(os.path.basename(f), len(rows), "units, pass", sum(a >= 0.95 for a in al),
          "| median align", round(sorted(al)[len(al) // 2], 3) if al else None)

z = shutil.make_archive("/tmp/flagship_pilot3", "zip", "/kaggle/working")
shutil.copy(z, "/kaggle/working/flagship_pilot3.zip")
print("DOWNLOAD /kaggle/working/flagship_pilot3.zip")
```

120k rather than 128k: the pool holds about 135k OPT tokens and full 128-token windows in
batches of eight lose a little at the end. If a cell prints `corpus gave ... fewer than
--tokens`, send the log; that is the new guard working.

## After the run

Send `flagship_pilot3.zip`. Scoring: `python experiments/analyse_flagship_pilot.py results/flagship_pilot3`.
