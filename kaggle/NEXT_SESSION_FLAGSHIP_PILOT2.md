# Kaggle session: flagship substrate pilot, round 2 (dial setting, discarded)

Round 1 (`results/flagship_pilot1/`, read 10 Oct) set nothing for F-3 or F-4: OPT was under-fitted
at both budgets and the gated-unit (Qwen) cells crashed after 2 units on a direction-saving bug,
fixed in the bundle rebuilt 10 Oct. This round brackets the budgets again. As before, the fits are
not evidence and are not reused. About 4-6 GPU-hours.

## Before you start

1. **Upload the rebuilt bundle.** Datasets -> **caliper-bundle** -> **New Version** -> drag in
   `kaggle/caliper-bundle.zip` (508 KB, rebuilt 10 Oct). The 9 Oct version still has the bug; the
   cell refuses it.
2. Notebook -> **Input:** attach **caliper-bundle**, refreshed to the new version.
3. Session options: **GPU T4 x2** (one is used), **Internet On**.

## The cell (self-contained; paste as the only cell)

```python
import sys, os, glob, subprocess, shutil, json

hits = glob.glob("/kaggle/input/**/caliper/estimator.py", recursive=True)
assert hits, "bundle not found under /kaggle/input - is caliper-bundle attached?"
root = os.path.dirname(os.path.dirname(hits[0]))
sys.path.insert(0, root); os.chdir(root); os.environ["PYTHONPATH"] = root
src = open("experiments/e01_gate.py").read()
assert "w=p.weights[..., i]" in src, \
    "STALE BUNDLE: upload the 10 Oct caliper-bundle.zip as a new version and re-attach"
print("root:", root, "| bundle OK (10 Oct, gated-unit fix)")

import torch
assert torch.cuda.is_available(), "no GPU - turn on the T4 accelerator"
print("cuda:", torch.cuda.get_device_name(0))

QWEN = ["--model", "Qwen/Qwen2.5-0.5B", "--layer", "12", "--d-mlp", "4864", "--target", "glu"]
OPT = ["--model", "facebook/opt-1.3b", "--layer", "12", "--d-mlp", "8192"]
CELLS = [  # tag, args, tokens
    ("f4_qwen05b_l12", QWEN, 8000),
    ("f4_qwen05b_l12", QWEN, 16000),
    ("f4_qwen05b_l12", QWEN, 32000),
    ("f3_opt13b_l12", OPT, 64000),
    ("f3_opt13b_l12", OPT, 128000),
    ("f6_gpt2", ["--model", "gpt2", "--target", "unembed"], 16000),
    ("f6_pythia410m", ["--model", "EleutherAI/pythia-410m", "--target", "unembed"], 2000),
]
for tag, args, tokens in CELLS:
    out = f"/kaggle/working/pilot2_{tag}_{tokens // 1000}k.jsonl"
    print(f"\n###### {tag} {tokens} tokens", flush=True)
    r = subprocess.run([sys.executable, "experiments/e01_gate.py", *args,
                        "--neurons", "16", "--tokens", str(tokens), "--steps", "3200",
                        "--restarts", "2", "--independent-units", "--device", "cuda",
                        "--out", out], check=False)
    print(f"exit {r.returncode}", flush=True)

for f in sorted(glob.glob("/kaggle/working/pilot2_*.jsonl")):
    rows = [json.loads(l) for l in open(f)]
    al = [r["align_selected"] for r in rows]
    print(os.path.basename(f), len(rows), "units, pass", sum(a >= 0.95 for a in al),
          "| median align", round(sorted(al)[len(al) // 2], 3) if al else None)

z = shutil.make_archive("/tmp/flagship_pilot2", "zip", "/kaggle/working")
shutil.copy(z, "/kaggle/working/flagship_pilot2.zip")
print("DOWNLOAD /kaggle/working/flagship_pilot2.zip")
```

Use **Save Version, then Save & Run All (Commit)** so it runs headless. If a cell prints a
non-zero `exit`, send the session log with the zip: the subprocess output is in the log, not
the zip.

## What healthy output looks like

- `bundle OK (10 Oct, gated-unit fix)`.
- Every cell ends `exit 0` with `16/16 neurons` (an OPT unit that never fires is screened and
  reported).
- Raw pass counts for GPT-2's unembedding stay near 0: its final layer norm has a near-zero gain,
  and those fits are scored offline with the null direction removed (round 1: 0/16 at 8k and
  5/16 at 32k on that label).

## After the run

Download `flagship_pilot2.zip` and send it, with the log if any cell failed. Scoring:
`python experiments/analyse_flagship_pilot.py results/flagship_pilot2`.
