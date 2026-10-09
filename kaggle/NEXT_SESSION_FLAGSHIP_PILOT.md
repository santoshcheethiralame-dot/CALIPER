# Kaggle session: flagship substrate pilot (dial setting, discarded)

Plan: `docs/flagship-plan.md` §5 (F-3, F-4, F-6). This is a **pilot**: it sets each new
substrate's token budget so that a pre-registration can name a primary setting with a failure
rate between 20% and 80%. Its fits are not evidence and are not reused. About 4-6 GPU-hours.
It can run while S-1 sessions run on the same account, if Kaggle allows two sessions.

## Before you start

1. **Upload the rebuilt bundle.** Datasets -> **caliper-bundle** -> **New Version** -> drag in
   `kaggle/caliper-bundle.zip` (508 KB, built 9 Oct). The 6 Oct version has none of the new
   targets; the cell refuses it.
2. Notebook -> **Input:** attach **caliper-bundle**, refreshed to the new version. No model
   input: OPT, Qwen, GPT-2 and Pythia download from the Hugging Face Hub (all ungated).
3. Session options: **GPU T4 x2** (one is used), **Internet On**.

## The cell (self-contained; paste as the only cell)

```python
import sys, os, glob, subprocess, shutil, json

hits = glob.glob("/kaggle/input/**/caliper/estimator.py", recursive=True)
assert hits, "bundle not found under /kaggle/input - is caliper-bundle attached?"
root = os.path.dirname(os.path.dirname(hits[0]))
sys.path.insert(0, root); os.chdir(root); os.environ["PYTHONPATH"] = root
src = open("experiments/e01_gate.py").read()
assert '"glu"' in src and '"unembed"' in src, \
    "STALE BUNDLE: upload the 9 Oct caliper-bundle.zip as a new version and re-attach"
print("root:", root, "| bundle OK (substrates F-3, F-4, F-6)")

import torch
assert torch.cuda.is_available(), "no GPU - turn on the T4 accelerator"
print("cuda:", torch.cuda.get_device_name(0))

CELLS = [  # tag, args
    ("f3_opt13b_l12", ["--model", "facebook/opt-1.3b", "--layer", "12", "--d-mlp", "8192"]),
    ("f4_qwen05b_l12", ["--model", "Qwen/Qwen2.5-0.5B", "--layer", "12", "--d-mlp", "4864",
                        "--target", "glu"]),
    ("f6_gpt2", ["--model", "gpt2", "--target", "unembed"]),
    ("f6_pythia410m", ["--model", "EleutherAI/pythia-410m", "--target", "unembed"]),
]
for tokens in (8000, 32000):
    for tag, args in CELLS:
        out = f"/kaggle/working/pilot_{tag}_{tokens // 1000}k.jsonl"
        print(f"\n###### {tag} {tokens} tokens", flush=True)
        r = subprocess.run([sys.executable, "experiments/e01_gate.py", *args,
                            "--neurons", "16", "--tokens", str(tokens), "--steps", "3200",
                            "--restarts", "2", "--independent-units", "--device", "cuda",
                            "--out", out], check=False)
        print(f"exit {r.returncode}", flush=True)

for f in sorted(glob.glob("/kaggle/working/pilot_*.jsonl")):
    rows = [json.loads(l) for l in open(f)]
    al = [r["align_selected"] for r in rows]
    print(os.path.basename(f), len(rows), "units, pass", sum(a >= 0.95 for a in al),
          "| median align", round(sorted(al)[len(al) // 2], 3) if al else None)

z = shutil.make_archive("/tmp/flagship_pilot", "zip", "/kaggle/working")
shutil.copy(z, "/kaggle/working/flagship_pilot.zip")
print("DOWNLOAD /kaggle/working/flagship_pilot.zip")
```

Use **Save Version, then Save & Run All (Commit)** so it runs headless.

## What healthy output looks like

- `bundle OK (substrates F-3, F-4, F-6)`.
- Per cell: `stimulus (8000, d)` or `(32000, d)`, `alive: 16/16` (an OPT unit that never fires
  is screened and reported, which is expected), then `16/16 neurons`.
- The closing table: units and passes per cell. What the pilot is looking for is a cell whose
  pass rate sits between 20% and 80% for each substrate.

## After the run

Download `flagship_pilot.zip` and send it. I read only the pass rates and fit quality to set
each substrate's budget, write that into its pre-registration, and archive the pilot as
discarded.
