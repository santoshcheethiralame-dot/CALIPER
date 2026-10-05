# Next Kaggle session: B-11c, Pythia-1.4B on the fixed estimator

Prereg: `docs/preregistration-b11c-pythia14b-indep.md`. One run, about 3-4.5 hours on one
T4 (B-11s took ~4.3 h; per-unit stopping should shorten it).

## Before you start

1. **Upload the rebuilt bundle.** On kaggle.com, open Datasets, then **caliper-bundle**, then
   **New Version**, and drag in `kaggle/caliper-bundle.zip` (503 KB, built 6 Oct). The old
   version predates the stopping fix, route agreement and direction saving. A run on it
   would repeat the coupled estimator. Kaggle unzips it automatically.
2. Open a notebook (new, or the one used for B-11s). Right panel, **Input:** attach
   **caliper-bundle**, refreshed to the version you just uploaded. No model input is needed:
   Pythia downloads from the Hugging Face Hub.
3. Session options: **GPU T4 x2** (one is used), **Internet On** (the Pythia download).
4. **Restart session** before running, so no GPU memory is held from earlier.

## The cell (self-contained; paste as the only cell)

```python
import sys, os, glob, subprocess, shutil

hits = glob.glob("/kaggle/input/**/caliper/estimator.py", recursive=True)
assert hits, "bundle not found under /kaggle/input - is caliper-bundle attached?"
root = os.path.dirname(os.path.dirname(hits[0]))
sys.path.insert(0, root); os.chdir(root); os.environ["PYTHONPATH"] = root
src = open("experiments/e01_gate.py").read()
assert "--independent-units" in src and "route_agreement" in src, \
    "STALE BUNDLE: upload the 6 Oct caliper-bundle.zip as a new version and re-attach"
print("root:", root, "| bundle OK (fixed estimator)")

import torch
assert torch.cuda.is_available(), "no GPU - turn on the T4 accelerator"
print("cuda:", torch.cuda.get_device_name(0))

out = "/kaggle/working/b11c_pythia-14b_s3200_indep.jsonl"
subprocess.run([sys.executable, "experiments/e01_gate.py",
                "--model", "EleutherAI/pythia-1.4b", "--layer", "12", "--d-mlp", "8192",
                "--restarts", "2", "--neurons", "50", "--steps", "3200",
                "--device", "cuda", "--independent-units", "--out", out], check=False)

# Zip to /tmp first: archiving /kaggle/working into itself can swallow the partial zip.
z = shutil.make_archive("/tmp/b11c_results", "zip", "/kaggle/working")
shutil.copy(z, "/kaggle/working/b11c_results.zip")
print("DOWNLOAD /kaggle/working/b11c_results.zip")
```

Use **Save Version, then Save & Run All (Commit)**, so it runs headless and survives
closing the tab.

## What healthy output looks like

- `bundle OK (fixed estimator)`. If you see `STALE BUNDLE`, stop and re-upload.
- `stimulus (8000, 2048)`, then `alive: 50/50`.
- Progress lines `32/50 neurons ...` and then `50/50`.
- A summary block that ends in `VERDICT ...`. The verdict is the gate's own 0.90 bar, not
  B-11c's endpoint, so ignore it.

## After the run

1. From the version's **Output** tab, download `b11c_results.zip`. It holds the `.jsonl`,
   the `_summary.json` and the `_dirs/` folder of saved directions.
2. Put it in `C:\Users\carbo\projects\caliper\data\b11\` and tell me. I'll unzip it, run the
   pre-registered analysis, and update the pooled table in Paper 1.
