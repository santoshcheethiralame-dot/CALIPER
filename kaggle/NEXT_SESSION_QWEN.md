# Qwen session — A-3, the invariant test

**What this decides.** On Gemma the detection signal switches on at **13.9% of the
residual-stream norm** (norm 58,932, onset alpha*=8,192). If Qwen's onset also lands
near 14% of *its own* norm, injection strength becomes reportable as a fraction of
the residual norm and the paper gets a calibration rule instead of a list of
findings. If it lands somewhere else, we report that it varies by model.

Script version required: **2026-09-07b**. (`a` fails to load Qwen — see the note below.)

> **Re-upload `kaggle_s3_positive_control.py` as a New Version of `caliper-s3` before
> starting.** Version `2026-09-07a` loads Gemma fine but dies on Qwen with
> *"Some modules are dispatched on the CPU or the disk"*. Two bugs, both fixed in `b`:
> the per-GPU budget was hardcoded to 13 GiB (tuned for Gemma at ~19 GB; Qwen needs
> ~21 GB) and is now measured from free memory, and the loader offered accelerate a
> 12 GiB CPU budget, which let it place modules on CPU — something bitsandbytes 4-bit
> refuses outright. The CPU budget is gone.

---

## Step 0 — get the weights the right way (read this first)

**Mount Qwen from Kaggle *Models*, do not let it download from HuggingFace.**
Qwen2.5-32B in bf16 is roughly 65 GB of safetensors. 4-bit quantisation happens
*after* the download, so the full file still has to land on disk, and that will
either exhaust the container or eat most of the session.

1. In the notebook, right panel -> **Add Input** -> **Models** -> search
   `Qwen2.5-32B-Instruct` -> add the **transformers** variant.
   *(Confirmed working: it mounts at
   `/kaggle/input/models/qwen-lm/qwen2.5/transformers/32b-instruct/1`.)*
2. Keep **caliper-s3** attached under *Datasets* (refresh to `2026-09-07a`).
3. You may leave Gemma attached or remove it; the loader picks by `--model-path`.

**If Qwen2.5-32B is not on Kaggle Models**, do not fall back to downloading it.
Use `Qwen2.5-14B-Instruct` instead and tell me — 14B is a weaker test of the
invariant but a real one, and it is honest as long as the size is reported. A
failed 65 GB download costs the whole session.

Session options: **GPU T4 x2**, Internet **On**. Then **Run -> Restart session**.

---

## Cell 1 — after the restart

```
pip install -q -U bitsandbytes accelerate transformers
```

## Cell 2 — the probe (~8 min)

Loads the model, measures the residual norm, writes it to disk, and runs 60 trials
at alpha=0. Those 60 trials are not filler: they are the sanity gate.

```python
import sys, glob
sys.argv = ["run", "--model", "qwen", "--compute-dtype", "fp32", "--stage", "forced",
            "--normalise", "--control", "none", "--alphas", "0",
            "--out", "/kaggle/working/s3_qwen_probe.jsonl"]
hits = glob.glob("/kaggle/input/**/*.py", recursive=True)
print(hits)
exec(open(hits[0]).read())
```

### Four lines to check before going further

| Line | Must read | If not |
|---|---|---|
| 1st line | `kaggle_s3_positive_control 2026-09-07b` | dataset didn't refresh; remove and re-add the input |
| device budget | `device budget: {0: '14.5GiB', 1: '14.5GiB'} (no cpu offload)` |
| layer stack | `model.layers (64 layers)` then `layer 38 of 64` | tell me the printed path; `find_layers` needs Qwen's tree added |
| vectors | `30 vectors, median norm 1.00, non-finite 0` | `--normalise` didn't take, or fp16 overflowed |
| **residual norm** | `residual norm at last token: median ...` | **write this number down; it drives cell 3** |

### The sanity gate — do not skip this

Look at the alpha=0 rows the probe just wrote. On Gemma, with nothing injected:

- introspective P(YES) = **0.0000**
- neutral_matched P(YES) = 0.1882

If Qwen's alpha=0 **introspective** mean is above ~0.05, **stop**. The chat template
is not being applied correctly and every number after it is meaningless. That is the
single most likely way this session goes wrong, and it is invisible unless you look.

Check it with:

```python
import json
rows = [json.loads(l) for l in open("/kaggle/working/s3_qwen_probe_forced_norm1.jsonl")]
for fr in ("introspective", "neutral_matched"):
    g = [r["p_yes"] for r in rows if r["framing"] == fr]
    print(fr, round(sum(g)/len(g), 4))
```

---

## Cell 3 — real vectors, grid computed from the measured norm

No arithmetic by hand. This reads the probe's sidecar and builds the grid at the
same *fractions of the residual norm* that bracketed Gemma's onset.

```python
import sys, glob, json
N = json.load(open("/kaggle/working/s3_qwen_probe.config.json"))["residual_norm_at_read_median"]
fracs = [0.01, 0.035, 0.07, 0.14, 0.28, 0.56]
alphas = [str(round(N * f)) for f in fracs]
print(f"residual norm {N:,.0f}  ->  alphas {alphas}")
sys.argv = ["run", "--model", "qwen", "--compute-dtype", "fp32", "--stage", "forced",
            "--normalise", "--control", "none", "--alphas", *alphas,
            "--out", "/kaggle/working/s3_qwen.jsonl"]
hits = glob.glob("/kaggle/input/**/*.py", recursive=True)
exec(open(hits[0]).read())
```

## Cell 4 — random control, identical grid

```python
import sys, glob, json
N = json.load(open("/kaggle/working/s3_qwen_probe.config.json"))["residual_norm_at_read_median"]
alphas = [str(round(N * f)) for f in [0.01, 0.035, 0.07, 0.14, 0.28, 0.56]]
sys.argv = ["run", "--model", "qwen", "--compute-dtype", "fp32", "--stage", "forced",
            "--normalise", "--control", "random", "--alphas", *alphas,
            "--out", "/kaggle/working/s3_qwen_rand.jsonl"]
hits = glob.glob("/kaggle/input/**/*.py", recursive=True)
exec(open(hits[0]).read())
```

Cell 4 must print `CONTROL random: vectors replaced` after the vector step. Without
that line it ran real vectors and the file is worthless.

**Note the separate `--out` names.** C23/C24 shared one and the config sidecar of the
first run was silently overwritten by the second. Two names, two sidecars.

Run 1, then 2, then 3, then 4 — one at a time, never Run All.

---

## What to send back

Four files from **Output -> /kaggle/working** (download arrow; `FileLink` 404s here):

- `s3_qwen_probe.config.json`  <- has the residual norm
- `s3_qwen_forced_norm1.jsonl`
- `s3_qwen_rand_forced_random_norm1.jsonl`
- `s3_qwen.config.json`

Zip them and drop the zip in chat, same as last time.

---

## Time budget

| | |
|---|---|
| load | ~5 min (mounted) |
| cell 2 probe | ~3 min |
| cell 3 real | ~18 min (360 forward passes) |
| cell 4 random | ~18 min + reload |
| **total** | **~50 min** |

Well inside a 12-hour session. If the session dies mid-run the out files append, so
re-running the same cell continues rather than restarting.

---

## What each outcome means

| Qwen's onset | Reading |
|---|---|
| **near 14% of its own norm** | **The invariant holds.** The paper gets a calibration rule: report alpha as a fraction of the residual norm, and the onset is model-independent. This is the strong outcome and it explains why published alpha values across this literature are incomparable |
| a different but clean fraction | Onset is model-specific but *norm-relative* framing is still the right way to report it. Weaker, still useful |
| no onset anywhere in the grid | Either Qwen doesn't show the effect, or the window sits outside 1-56%. Check the top row is actually perturbing at all before concluding anything — if P(YES) at 56% is still ~0, suspect the injection path, not the model |
| real never beats random | Gemma's A1 does not generalise. That is a real result and it goes in the paper: the concept-specific component is model-specific |

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| alpha=0 introspective P(YES) high | chat template wrong for Qwen | **stop**, send me the probe file |
| `could not locate decoder layers` | Qwen's module path missing | send me the error; one-line fix |
| OOM at load | stale GPU memory, or the model is not 4-bit | Restart session; confirm both cards read ~15.5/15.6 GB free |
| download starts instead of mounting | Qwen attached from HF not Kaggle Models | stop it; go back to step 0 |
| `Some modules are dispatched on the CPU or the disk` | running version `a` | re-upload the script; `b` fixes it |
| `median norm nan` | fp16 overflow | `--compute-dtype fp32` missing |
