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

**`caliper-bundle` dataset** — upload `kaggle/caliper-bundle.zip` as a new dataset if you
have not already (see `UPLOAD.md`). **GPU T4 x2. Internet can stay off** for this run: it
uses GPT-2, and the corpus is the cached prose shipped in the bundle.

## Cell 1

```python
import sys, glob, os
root = glob.glob("/kaggle/input/*/")[0]
sys.path.insert(0, root)
os.chdir(root)
print(root, os.listdir(root))
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
!python experiments/kaggle_device_equivalence.py --neurons 24 --out /kaggle/working/device_equivalence.json
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
