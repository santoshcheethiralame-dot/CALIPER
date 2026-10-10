# Kaggle session: F-4 (gated units) and F-6 (unembedding), main runs

Preregs: `docs/preregistration-f4-gated-units.md` and `docs/preregistration-f6-unembedding.md`
(filed 10 Oct, before data). Five fits on one T4: F-4 fits A and B (100 units), F-6 fits A and B
(100 rows), and the Pythia-410m control (30 rows). Roughly 4-6 GPU-hours; each fit resumes by
unit if the session restarts.

## Before you start

1. **Upload the rebuilt bundle**: Datasets -> **caliper-bundle** -> **New Version** ->
   `kaggle/caliper-bundle.zip` (rebuilt 10 Oct, night). The cell refuses older versions.
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
assert '"--corpus-docs"' in src, "STALE BUNDLE: upload the 10 Oct (night) caliper-bundle.zip and re-attach"
import torch
assert torch.cuda.is_available(), "no GPU - turn on the T4 accelerator"
print("bundle OK | cuda:", torch.cuda.get_device_name(0))

BASE = ["--steps", "3200", "--restarts", "2", "--independent-units", "--device", "cuda"]
QWEN = ["--model", "Qwen/Qwen2.5-0.5B", "--layer", "12", "--d-mlp", "4864", "--target", "glu",
        "--neurons", "100", "--tokens", "8000"]
GPT2 = ["--model", "gpt2", "--target", "unembed", "--neurons", "100", "--tokens", "32000"]
RUNS = [
    ("F-6 GPT-2 fit A", GPT2, []),
    ("F-6 GPT-2 fit B", GPT2, ["--corpus-seed", "1", "--exclude-corpus-seed", "0", "--corpus-docs", "2000"]),
    ("F-4 Qwen fit A", QWEN, []),
    ("F-4 Qwen fit B", QWEN, ["--corpus-seed", "1", "--exclude-corpus-seed", "0"]),
    ("F-6 Pythia-410m control", ["--model", "EleutherAI/pythia-410m", "--target", "unembed",
                                 "--neurons", "30", "--tokens", "8000"], []),
]
OUT = {"F-6 GPT-2 fit A": "f6_gpt2_a", "F-6 GPT-2 fit B": "f6_gpt2_b", "F-4 Qwen fit A": "f4_qwen05b_a",
       "F-4 Qwen fit B": "f4_qwen05b_b", "F-6 Pythia-410m control": "f6_pythia410m_control"}
for name, model, extra in RUNS:
    out = f"/kaggle/working/{OUT[name]}.jsonl"
    print(f"\n###### {name}", flush=True)
    r = subprocess.run([sys.executable, "experiments/e01_gate.py", *model, *BASE, *extra, "--out", out],
                       check=False)
    print(f"exit {r.returncode}", flush=True)

for f in sorted(glob.glob("/kaggle/working/f[46]_*.jsonl")):
    rows = [json.loads(l) for l in open(f)]
    al = [r["align_selected"] for r in rows]
    print(os.path.basename(f), len(rows), "units, Euclidean pass", sum(a >= 0.95 for a in al))

z = shutil.make_archive("/tmp/flagship_f4f6", "zip", "/kaggle/working")
shutil.copy(z, "/kaggle/working/flagship_f4f6.zip")
print("DOWNLOAD /kaggle/working/flagship_f4f6.zip")
```

Use **Save Version, then Save & Run All (Commit)** so it runs headless.

## What healthy output looks like

- Each fit prints `stimulus (8000, 896)` (Qwen) or `stimulus (32000, 768)` (GPT-2), then
  `100/100 neurons`; the control prints `30/30`.
- GPT-2's Euclidean pass count will be near 0. That is the layer-norm null direction, and the
  identifiable label is scored offline. The Pythia control should pass at least 27 of 30.
- A cell that prints `corpus gave ... fewer than --tokens` is the token guard; send the log.

## After the run

Send `flagship_f4f6.zip`. Scoring, once:
`python experiments/analyse_substrate.py --a results/flagship_f4f6/f6_gpt2_a --b results/flagship_f4f6/f6_gpt2_b --model gpt2 --norm transformer.ln_f --out results/f6_analysis.json`,
`... --a .../f4_qwen05b_a --b .../f4_qwen05b_b --model Qwen/Qwen2.5-0.5B --out results/f4_analysis.json`,
and the control with `--model EleutherAI/pythia-410m --norm gpt_neox.final_layer_norm`.
