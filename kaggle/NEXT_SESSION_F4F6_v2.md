# Kaggle session: F-4 and F-6, re-planned (F-6 Amendment 1)

Replaces `NEXT_SESSION_F4F6.md`, whose 4-6 hour estimate ignored the pilot's measured 425 s per
GPT-2 row. Measured costs: F-4 Qwen 27 s/unit (8k), Pythia-410m control 84 s/row (8k), F-6 GPT-2
425 s/row (32k). This session: about 7.5 hours, with a deadline that packages whatever is done.

## Before you start

- **Stop the old session** (F-6 fit A, started with the 4-6 hour sheet). Its output is not needed.
- Input: **caliper-bundle**, the **17 Oct** version: upload `kaggle/caliper-bundle.zip` as a new
  version first (the 10 Oct one predates the OPT/Gemma norm lookup and the unscoreable-unit screen;
  the cell refuses it).
- **GPU T4 x2** (both are used now), **Internet On**.

## The cell (paste as the only cell; Save Version -> Save & Run All)

```python
import sys, os, glob, subprocess, shutil, json, time

T0 = time.time()
DEADLINE = T0 + 11.0 * 3600            # stop fits here, leaving time to package
hits = glob.glob("/kaggle/input/**/caliper/estimator.py", recursive=True)
assert hits, "bundle not found under /kaggle/input - is caliper-bundle attached?"
root = os.path.dirname(os.path.dirname(hits[0]))
sys.path.insert(0, root); os.chdir(root); os.environ["PYTHONPATH"] = root
assert "unscoreable (silent on the validation rows)" in open("experiments/e01_gate.py").read(), "STALE BUNDLE: upload the 17 Oct caliper-bundle.zip"
import torch
assert torch.cuda.device_count() >= 2, "turn on GPU T4 x2"
print("bundle OK |", torch.cuda.device_count(), "GPUs", flush=True)

BASE = ["--steps", "3200", "--restarts", "2", "--independent-units", "--device", "cuda"]
QWEN = ["--model", "Qwen/Qwen2.5-0.5B", "--layer", "12", "--d-mlp", "4864", "--target", "glu",
        "--neurons", "100", "--tokens", "8000"]
GPT2 = ["--model", "gpt2", "--target", "unembed", "--neurons", "50", "--neuron-pool", "100",
        "--tokens", "32000", "--batch", "10"]
DISJOINT = ["--corpus-seed", "1", "--exclude-corpus-seed", "0"]
PHASES = [
    [("f4_qwen05b_a", QWEN, []), ("f4_qwen05b_b", QWEN, DISJOINT)],
    [("f6_pythia410m_control", ["--model", "EleutherAI/pythia-410m", "--target", "unembed",
                                "--neurons", "30", "--tokens", "8000"], [])],
    [("f6_gpt2_a", GPT2, []), ("f6_gpt2_b", GPT2, DISJOINT + ["--corpus-docs", "2000"])],
]

def start(name, model, extra, gpu):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    log = open(f"/kaggle/working/{name}.log", "w")
    cmd = [sys.executable, "experiments/e01_gate.py", *model, *BASE, *extra,
           "--out", f"/kaggle/working/{name}.jsonl"]
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)

for phase in PHASES:
    procs = {name: start(name, model, extra, gpu) for gpu, (name, model, extra) in enumerate(phase)}
    print(f"\n###### started {list(procs)} at {(time.time()-T0)/3600:.2f} h", flush=True)
    while any(p.poll() is None for p in procs.values()):
        time.sleep(300)
        for name in procs:
            f = f"/kaggle/working/{name}.jsonl"
            n = sum(1 for _ in open(f)) if os.path.exists(f) else 0
            print(f"  {(time.time()-T0)/3600:.2f} h  {name}: {n} rows", flush=True)
        if time.time() > DEADLINE:
            print("DEADLINE: stopping fits; finished rows are kept", flush=True)
            for p in procs.values():
                if p.poll() is None:
                    p.terminate()
            time.sleep(30)
    for name, p in procs.items():
        print(f"  {name} exit {p.returncode}", flush=True)
    if time.time() > DEADLINE:
        break

for f in sorted(glob.glob("/kaggle/working/f[46]_*.jsonl")):
    rows = [json.loads(l) for l in open(f) if l.strip().endswith("}")]
    print(os.path.basename(f), len(rows), "rows, Euclidean pass", sum(r["align_selected"] >= 0.95 for r in rows))
z = shutil.make_archive("/tmp/flagship_f4f6", "zip", "/kaggle/working")
shutil.copy(z, "/kaggle/working/flagship_f4f6.zip")
print("DOWNLOAD /kaggle/working/flagship_f4f6.zip", flush=True)
```

## What healthy output looks like

- Every 5 minutes a progress line per running fit. F-4 fits finish in about 45 minutes, the
  control in about 45 minutes, then both F-6 fits run together for about 6 hours.
- If the two parallel F-6 fits slow each other down, the deadline at 11 hours stops them and the
  session still packages; the row count printed at the end is what gets scored.
- Pythia control: at least 27 of 30 should pass.

## After the run

Send `flagship_f4f6.zip`. Scoring is unchanged (see `NEXT_SESSION_F4F6.md`, "After the run").
