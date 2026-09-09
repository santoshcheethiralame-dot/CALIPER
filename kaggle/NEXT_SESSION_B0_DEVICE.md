# B-0 — is the device a confound? The run that unlocks every other Kaggle run

**Run this before moving any B-series work to a GPU.** ~40 minutes, one session.

---

## Why this exists

The obvious move is to put the B-series on Kaggle's GPU and stop waiting four hours per
run. The reason not to do it blind:

**The study is about which units fail.** A failing unit is one where the optimiser landed
in the wrong basin, so failing units sit near basin boundaries *by construction* — and
that is exactly the population where a difference in floating-point reduction order can
flip the answer. CPU and GPU do not agree bitwise.

So a study reporting "GPT-2 fails on 23%, Pythia on 7%" where one ran on CPU and the other
on GPU has a hardware term inside its headline number, and no way to separate it.

C31 is the precedent and it cost this project a set of runs: a vector carrying no content
passed every health check anyone reports, and only a positive control caught it. Same
treatment here. **Measure the device, do not assume it.**

There is also an upside worth knowing precisely: this run reports the **actual speedup**,
so the rest of the programme can be scheduled against a measured number rather than a hope.

---

## Pre-registered criterion — filed before the run

| outcome | meaning | consequence |
|---|---|---|
| **PASS** — no unit changes pass/fail side **and** max \|Δalignment\| < 0.01 | the device is not a confound | B-series runs may mix devices freely. Move everything to GPU |
| **FAIL** — any unit flips side, **or** max \|Δalignment\| ≥ 0.01 | the device is a confound | Every set of runs that gets compared shares one device. The device is reported beside every number in the paper |

A FAIL is not a wasted run. It is a reportable methodological finding — *estimator failure
classification is hardware-dependent* — and it is the kind of thing this bench exists to
catch. It also has a direct practical consequence: anyone reproducing an interpretability
result on different hardware may not reproduce which units failed.

---

## Setup

**`caliper-bundle` dataset** — already uploaded; attach it, no new version needed.
**GPU T4 x2. Internet ON.**

*Corrected 10 Sep: an earlier version of this sheet said internet could stay off. Wrong.*
The **corpus** needs no network — the Gutenberg texts ship in the bundle. But
`load_model` calls `from_pretrained("gpt2")`, which fetches the weights from the Hub. Only
the stimulus is cached, never the model.

## Cell 1

```python
# Locate the bundle by a marker file, not by guessing the mount depth. Kaggle mounts a
# dataset at a path that depends on how it was created and attached - a previous session
# died on every command because the sheet assumed /kaggle/input/<one-level>/ and the
# bundle was actually a level deeper.
import sys, os, glob

print("what is actually mounted:")
for p in sorted(glob.glob("/kaggle/input/**", recursive=True))[:40]:
    print("   ", p)

hits = glob.glob("/kaggle/input/**/caliper/estimator.py", recursive=True)
assert hits, "caliper/estimator.py not found under /kaggle/input - is the dataset attached?"
root = os.path.dirname(os.path.dirname(hits[0]))
sys.path.insert(0, root)
os.chdir(root)

# PYTHONPATH, not just sys.path. Every run below is launched with `!python ...`, which is a
# SUBPROCESS - it inherits the working directory but not the kernel's sys.path. Python puts
# the *script's* directory (experiments/) on the path, never the bundle root, so `import
# caliper` fails with ModuleNotFoundError even though Cell 1 looks like it worked. This
# line is what the local run_queue.sh does as `export PYTHONPATH=.`.
os.environ["PYTHONPATH"] = root

print("root:", root)
print("contents:", sorted(os.listdir(root)))
for need in ("caliper/estimator.py", "caliper/batched.py", "experiments/e01_gate.py"):
    assert os.path.exists(need), f"missing {need} - rebuild with build_bundle.py and re-upload"

# Prove a subprocess can import the package, here, rather than discovering it four
# tracebacks later. This is the exact condition every run below executes under.
import subprocess
p = subprocess.run([sys.executable, "-c", "import caliper, caliper.batched; print('subprocess import OK')"],
                   capture_output=True, text=True)
assert p.returncode == 0, "a subprocess cannot import caliper: " + p.stderr
print(p.stdout.strip())
print("bundle OK")
```

## Cell 2

```python
import torch
print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("threads:", torch.get_num_threads())
```

If `cuda` is False, stop — the run is meaningless without both devices.

## Cell 3

```python
!python experiments/kaggle_device_equivalence.py --neurons 16 --restarts 2 --out /kaggle/working/device_equivalence.json
```

24 units drawn as the **first 24 of the same 100 the gate drew**, same rng and seed, so
the sample contains known passes and known failures rather than an easy subset.

---

## What to read

```
  max |d alignment|.............. 0.000123   (tolerance 0.01)
  units flipping pass/fail....... 0
  GPU speedup.................... 14.3x
  VERDICT........................ PASS
```

Three numbers matter and each has a use:

1. **units flipping** — the criterion. Any flip is a FAIL regardless of the tolerance.
2. **max |Δalignment|** — how much the answer moves at all.
3. **speedup** — schedules the rest of the programme. If it is under ~3x, GPU is not worth
   the confound risk even on a PASS, and Kaggle CPU sessions are the better route.

## Send back

`device_equivalence.json`. It carries the verdict, both per-unit tables, the flips, and
the timings.

---

## What happens next, either way

**On PASS** — the whole of Tier 3 moves to GPU and fits in one or two sessions:

| run | what it answers |
|---|---|
| B-7 layer sweep | "you looked at one layer" |
| B-8 third family, GPT-Neo-125m | "two models is not a family effect" |
| B-9 **Pythia-1.4b** | **"only 124M and 160M models"** — the strongest objection |
| B-10 required-N token sweep | "your operating point is arbitrary" |

**On FAIL** — those runs go to **Kaggle CPU sessions**, which draw no GPU quota and are
device-consistent with B-1/B-2 running locally. Slower per run — Kaggle gives 4 cores
against this machine's 10, so roughly 11 h where local takes 4.4 h, which only just fits a
12 h session — but several sessions run at once, and `Checkpoint` resumes a run that gets
cut off mid-session rather than restarting it.

B-9 is the exception. Pythia-1.4b at d_model 2048 is roughly 2.7x wider than GPT-2 and may
not fit a 12 h CPU session at all. **If B-0 fails, B-9 runs on GPU as its own
device-consistent set** — Pythia-1.4b on GPU compared against nothing else — and the paper
reports it as a separate scale probe rather than folding it into the CPU table.
