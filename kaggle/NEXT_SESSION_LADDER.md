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

**Upload `kaggle/caliper-bundle.zip` as a NEW dataset named `caliper-bundle`** — it does
not exist yet; the existing `caliper-s3` and `caliper-s1` are different packages. Steps are
in `UPLOAD.md`.

Then: **GPU T4 x2. Internet ON** — Pythia is pulled from the Hub and is not gated, so no
token is needed. The *corpus* needs no internet: the bundle ships the cached Gutenberg
texts and `sample_corpus` only reads them.

## Cell 1 — paths

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

## Cell 2 — confirm the GPU is real

```python
import torch
print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

If this prints False, stop. Both runs need it.

## Cell 3 — B-0, the device check

```python
!python experiments/kaggle_device_equivalence.py --neurons 16 --restarts 2 \
    --out /kaggle/working/device_equivalence.json
```

> **Run the ladder FIRST if the session is short.** B-0's CPU arm is its expensive half -
> Kaggle's 4-core CPU is roughly 2.5x slower than the local machine - and the GPU idles
> throughout it. The ladder is the run with the result in it; B-0 only decides whether
> *future* comparative runs may move to Kaggle. Cell order here is convenience, not
> dependency: **the ladder does not depend on B-0 in any way.**

**Read the verdict line.** It does not block the ladder either way — the ladder is
internally consistent regardless — but it decides whether the *layer sweep and GPT-Neo*
runs can ever move to Kaggle, and it reports the measured speedup that schedules
everything else.

## Cell 4 — the ladder

Each rung is depth-matched to 0.5 relative depth and runs at **2 restarts**, matching the
primary arm and the incumbent AUCs per Addendum 1.

```python
# 50 UNITS PER RUNG, not 100 - see preregistration-b11-addendum-1-n50.md. fit_cascade is
# 24x the work of fit_batch (151,040 unbatched steps per 32 units against 6,400 batched),
# it runs one unit at a time, and e01_gate.py:72 calls it without device= so it stays on
# CPU. The GPU accelerates about 4% of the job. At 100 units the four rungs need ~8.4 h
# against a 12 h cap; at 50 they need ~4.6 h and all four land.
#
# ORDER MATTERS. Measured on a T4: 38.6 s/unit at d_model 512, and cost scales with
# d_model, so the four rungs are about 8.4 h against a 12 h session cap. The LAST rung is
# the one at risk of being cut off, so 1.4b goes third, not fourth:
#   - 1.4b cannot run locally at all (5.6 GB fp32 against 2.6 GB free) and is the rung that
#     answers "only small models". Losing it means losing it.
#   - 410m fits locally at ~1.6 GB, so if the session dies before it, it is recoverable.
RUNGS = [
    ("EleutherAI/pythia-70m",   3,  2048),
    ("EleutherAI/pythia-160m",  6,  3072),
    ("EleutherAI/pythia-1.4b",  12, 8192),
    ("EleutherAI/pythia-410m",  12, 4096),
]
for model, layer, d_mlp in RUNGS:
    tag = model.split("/")[-1].replace(".", "")
    print(f"\n{'=' * 70}\n  {model}  layer {layer}  d_mlp {d_mlp}\n{'=' * 70}", flush=True)
    !python experiments/e01_gate.py --model {model} --layer {layer} --d-mlp {d_mlp} \
        --restarts 2 --neurons 50 --device cuda \
        --out /kaggle/working/b11_{tag}.jsonl
```

Rungs are independent, so if the session dies partway the completed ones are still valid —
and each is resumable, because `Checkpoint` reads its own output file.

## Cell 5 — package before the session expires

```python
import shutil, glob, os

rows = {f: sum(1 for _ in open(f)) for f in sorted(glob.glob("/kaggle/working/*.jsonl"))}
for f, n in rows.items():
    print(f"{os.path.basename(f):<28} {n:>4} rows" + ("   <- SHORT" if n < 50 else ""))

# Refuse to package nothing. A previous session zipped an empty directory, and the
# resulting archive looked like a successful download until it was opened.
assert rows, "no .jsonl files in /kaggle/working - the ladder has not produced anything yet"
assert all(n >= 50 for n in rows.values()), "a rung is short - check which, before packaging"

# Write the archive OUTSIDE the directory being archived. Writing it into /kaggle/working
# and then zipping /kaggle/working makes the archive contain itself.
shutil.make_archive("/kaggle/ladder_results", "zip", "/kaggle/working")
print("wrote /kaggle/ladder_results.zip",
      os.path.getsize("/kaggle/ladder_results.zip"), "bytes")
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
