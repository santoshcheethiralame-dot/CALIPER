# Kaggle session — B-0 then the Pythia scale ladder

**One GPU session. B-0 first (~40 min), then four ladder rungs.** Runs in parallel with the
local queue and touches nothing it owns.

Pre-registrations: `preregistration-b11-pythia-ladder.md` (the ladder) and the criterion
block in `NEXT_SESSION_B0_DEVICE.md` (B-0).

---

## Why these two, and not the other queued runs

The local queue's remaining runs — B-7 layer sweep, B-8 GPT-Neo — are **comparative**. B-7
is scored against layer 6, B-8 against GPT-2 and Pythia. Moving either to a GPU puts a
hardware term inside the comparison, which is the whole reason B-0 exists.

These two are different:

- **B-0** is *about* the device. It cannot be confounded by it.
- **The ladder is internally device-consistent** — every rung on the same GPU — so the
  scaling curve stands on its own without reference to the local numbers.

And the ladder carries the one run that **cannot** be done locally at all: **pythia-1.4b**
needs ~5.6 GB in fp32 against 16 GB total with 2.6 GB free under load.

---

## Setup

`caliper-bundle` at its **newest version** (rebuilt 8 Sep; earlier versions are missing
`batched.py` and will not run). **GPU T4 x2. Internet ON** — Pythia is downloaded from the
Hub and is not gated, so no token is needed.

## Cell 1 — paths

```python
import sys, glob, os
root = glob.glob("/kaggle/input/*/")[0]
sys.path.insert(0, root); os.chdir(root)
print(root, os.listdir(root))
```

## Cell 2 — confirm the GPU is real

```python
import torch
print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

If this prints False, stop. Both runs need it.

## Cell 3 — B-0, the device check

```python
!python experiments/kaggle_device_equivalence.py --neurons 24 \
    --out /kaggle/working/device_equivalence.json
```

**Read the verdict line.** It does not block the ladder either way — the ladder is
internally consistent regardless — but it decides whether the *layer sweep and GPT-Neo*
runs can ever move to Kaggle, and it reports the measured speedup that schedules
everything else.

## Cell 4 — the ladder

Each rung is depth-matched to 0.5 relative depth and runs at **2 restarts**, matching the
primary arm and the incumbent AUCs per Addendum 1.

```python
RUNGS = [
    ("EleutherAI/pythia-70m",   3,  2048),
    ("EleutherAI/pythia-160m",  6,  3072),
    ("EleutherAI/pythia-410m",  12, 4096),
    ("EleutherAI/pythia-1.4b",  12, 8192),
]
for model, layer, d_mlp in RUNGS:
    tag = model.split("/")[-1].replace(".", "")
    print(f"\n{'=' * 70}\n  {model}  layer {layer}  d_mlp {d_mlp}\n{'=' * 70}", flush=True)
    !python experiments/e01_gate.py --model {model} --layer {layer} --d-mlp {d_mlp} \
        --restarts 2 --neurons 100 --device cuda \
        --out /kaggle/working/b11_{tag}.jsonl
```

Rungs are independent, so if the session dies partway the completed ones are still valid —
and each is resumable, because `Checkpoint` reads its own output file.

## Cell 5 — package before the session expires

```python
import shutil, glob
for f in glob.glob("/kaggle/working/*.jsonl") + glob.glob("/kaggle/working/*.json"):
    print(f, sum(1 for _ in open(f)) if f.endswith("jsonl") else "")
shutil.make_archive("/kaggle/working/ladder_results", "zip", "/kaggle/working")
```

**Download the zip before the session ends.** Kaggle outputs are ephemeral and this project
has already lost a set of run artifacts that way.

---

## What to read, per rung

```
  n_passing................. 93
  wilson_95_lower........... 0.863
  wilson_95_upper........... 0.966
  median_alignment.......... 0.9994
  min_alignment............. 0.0761
  seconds_per_neuron........ ...
```

Three things matter:

1. **The Wilson interval per rung.** The trend across rungs is the primary endpoint.
2. **min_alignment beside median_alignment.** C53 found a median of 0.9994 hiding a unit
   recovered at 0.0761. If aggregate quality keeps improving with scale while the worst
   unit stays bad, that is the bench's central claim holding across a 20x parameter range.
3. **The 160m rung against C53's local 93/100.** A free aggregate device cross-check at
   n=100. Weaker than B-0's paired test, informative either way.

## The one branch to watch

Per the filing: **if the failure rate rises with scale, do not report it yet.** Wider models
may simply be harder to fit at a fixed 1600 steps. The pre-committed check is to re-run the
top rung at `--steps 3200` and report both. A rate that falls with more steps is an
optimiser artefact, not a scale effect.

## Send back

`ladder_results.zip` — the four `.jsonl` files, their config sidecars, and
`device_equivalence.json`.
