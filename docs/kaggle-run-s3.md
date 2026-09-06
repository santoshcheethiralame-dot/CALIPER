# Kaggle Run Guide — Study 3

**Positive control, then the neutral-framing experiment. Written to be followed
start to finish without decisions.**

---

## What you are running and why

Anthropic's group (Macar, Yang, Wang, Wallich, Ameisen & Lindsey, arXiv:2603.21396) report
that a language model can **detect** when a concept vector is injected into its residual
stream — 10.8% detection with 0% false positives on Gemma3-27B — and they trace it to a
circuit that only exists after post-training. That detection claim is the *defended* claim
in the introspection debate. Two critique papers attack *identification* (what the concept
was); nobody attacks detection.

We are going to attack it, with a control nobody has run: ask the same question **without
ever mentioning the model, its mind, or injection**. If a neutral "is anything unusual
here?" detects injections as well as "do you detect an injected thought?", then the
defended claim is generic anomaly sensitivity, not self-access.

**But first we must show the effect exists in our setup.** We run in 4-bit on free GPUs;
they did not report their precision. Stage 1 is a positive control: reproduce their
detection before trying to explain it away. If Stage 1 fails, Stage 2 is meaningless.

---

## Step 1 — Create the notebook

1. kaggle.com → **Create** → **New Notebook**
2. Right sidebar → **Session options**:
   - **Accelerator: GPU T4 x2** ← not P100. A 32B model in 4-bit needs ~17–20 GB;
     one T4 has 16 GB. Two T4s give 32 GB and the loader splits the model across them.
   - **Internet: On** (needed to download the model)
3. Leave Persistence off. The script writes results to `/kaggle/working/` which you
   download at the end.

---

## Step 2 — Pick the model

| flag | model | gated? | why it qualifies |
|---|---|---|---|
| `--model qwen` **(start here)** | Qwen2.5-32B-Instruct | no | Vogel (2025) replicated concept-injection detection on this model class |
| `--model olmo` | OLMo-2-32B-Instruct | no | Macar et al. used it for their post-training analysis |
| `--model gemma` | Gemma3-27B-it | **yes** | their primary model — 10.8% / 0% FPR at L=37 |

**`qwen` is the default — just run the script.** It is ungated, so there is nothing to
configure, and it has a published detection result. Gemma is the cleaner comparison to
their headline number, but a run that happens today beats the perfect model. If the effect
is real it should show in all three.

---

## Step 2b — The easiest route for Gemma: Kaggle Inputs

Instead of a HuggingFace token, add the weights through Kaggle itself:

**Right sidebar → Input → Add Input → Models → search "gemma 3" → 27b-it → Add.**
Accept the licence *on Kaggle*. That replaces the HF token completely — no secret, no
`HF_TOKEN`, no gated-repo error.

The script auto-detects it: it scans `/kaggle/input` for a directory containing
`config.json`, prefers one whose path matches `--model`, and loads from disk. Weights are
already local, so **the 8–15 minute download disappears too.**

Run with `sys.argv = ["run", "--model", "gemma"]` and it will find the input.

---

## Step 3 — Cell 1 (setup)

```python
!pip install -q -U bitsandbytes accelerate transformers
```

Kaggle ships `transformers` but **not** `bitsandbytes`, and 4-bit loading needs it
(version 0.46.1 or newer). If you skip this cell the run dies with
`ImportError: Using bitsandbytes 4-bit quantization requires bitsandbytes`.

The script now installs it itself as a safety net, so a forgotten cell no longer kills the
run. Running the cell anyway is still faster, because the install then happens while you
are setting up rather than inside the timed run.

**Only if using `--model gemma` and you did *not* add it through Inputs,** add this too:

```python
from kaggle_secrets import UserSecretsClient
import os
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

---

## Step 4 — Cell 2 (the experiment)

Paste the entire contents of `experiments/kaggle_s3_positive_control.py` and run it.
**Nothing to add, nothing to configure.** The default is Qwen2.5-32B-Instruct, which is
ungated — no licence, no token.

Only if you want to change something, add ONE line at the very bottom *before* `main()`
is reached — note `"run"` must be the first element:

```python
import sys; sys.argv = ["run", "--model", "gemma"]      # or --stage framing, etc.
main()
```

---

## Step 4b — Watch it run

**Use the interactive editor, not "Save & Run All".** In a committed run, Kaggle only
flushes the Logs tab when a *cell finishes* — so during a 10-minute model load you see
`Running for 230s` and `Output 0 B`, which looks like a hang but is not.

Open the notebook → **Edit** → **Run All**. Output streams into the cell as it happens,
and you can see the load, the vector building, and per-trial progress live.

---

## Step 5 — What you should see

```
  token present: False
  file access to Qwen/Qwen2.5-32B-Instruct: OK
loading Qwen/Qwen2.5-32B-Instruct in 4-bit ...
  cuda:0 8.4 GB used
  cuda:1 8.1 GB used
  layer stack: model.layers (64 layers)
  layer 38 of 64 (0.60 depth, matching L=37/62)
building concept vectors at layer 38 ...
  30 vectors, median norm 12.47
  10 trials, 134s
  ...
```

**Timings.** Model download ~8–15 min the first time. Vector building ~2 min. Then
~10–14 s per trial; 120 trials ≈ 25 min. Total **under an hour**, well inside the
12-hour session cap and the 30 h weekly quota.

---

## Step 6 — Read the result

The script prints a table and a verdict:

```
framing          alpha    n   detect  identify  coherent
introspective        0   30     0.0%      0.0%    100.0%  <- FPR
introspective        2   30     6.7%      3.3%     96.7%
introspective        4   30    13.3%      6.7%     93.3%
introspective        8   30    20.0%      3.3%     46.7%

  TARGET (Macar et al., L=37, alpha=4): detection 10.8%, FPR 0%
  OURS (4-bit):                        detection 13.3%, FPR 0.0%
  VERDICT: effect SURVIVES quantisation - proceed to --stage framing
```

### Decision table

| What you see | What it means | Do this |
|---|---|---|
| detection ≥ 5% at α=4, FPR ≤ 5% | **Effect survives 4-bit.** The gate is open | Go to Step 8 |
| detection ~0% at every α | Injection ineffective or α mis-scaled | `--normalise`, then sweep `--layer` |
| detection high at **α=0** | False positives — the prompt is leading the model | Stop. Inspect the `text` field in the results file |
| coherent < 60% at α=4 | fp16 overflow or α too large in raw units | Use `--alphas 0 1 2`, or `--normalise` |
| detection rises smoothly with α | Good — this is the dose–response their paper reports | Continue |

### If it fails before any table appears

| Error | Cause | Fix |
|---|---|---|
| `ImportError: ... requires bitsandbytes` | Not installed; `transformers` caches the answer at *its* import, so installing afterwards needs a kernel restart | Run Step 3's cell, then **Run All** again. The script also self-installs now |
| `Need ~20 GB ... this session has 16 GB` | Accelerator is P100 or single T4 | Session options → **GPU T4 x2** |
| `GatedRepoError` / 401 | Gemma licence or token | Step 2b (Inputs) is easier than Step 7 (token) |
| `Running for 200s`, `Output 0 B` | Committed runs only flush logs per completed cell | Step 4b — use Edit → Run All |

**The bar is not their exact number.** We use 30 concepts against their 500 and
rule-based scoring against a GPT-4.1-mini judge. The bar is *the effect exists and false
positives stay near zero*.

### Two ambiguities you may need to sweep

Their paper does not state either, and both change what α means:

- **Vector normalisation** — whether `v_c` is L2-normalised before scaling. Try
  `--normalise` if raw vectors give incoherence or nothing at all.
- **Layer index** — L=37 of 62 assumed 0-indexed. The script uses 0.60 of depth. If
  detection is flat everywhere, try `--layer 30`, `--layer 45`.

Sweep these *before* concluding the effect is absent. A null result from an
unswept parameter is not a result.

---

## Step 7 — Only if you want Gemma (the gated model)

The 401 you hit is authentication, not compute. In likelihood order:

1. **Accept the licence on the exact repo**: huggingface.co/google/gemma-3-27b-it.
   Accepting for **gemma-2 does not cover gemma-3** — this is the usual cause.
2. **Attach the secret to this notebook.** Adding `HF_TOKEN` under Add-ons → Secrets is
   not enough; there is a per-notebook toggle, and "Save & Run All" fails silently if the
   secret is not attached.
3. **Token scope**: needs *Read access to contents of all public gated repos*.

The script preflights access and fails in ~2 seconds with this list printed, rather than
60 seconds into a download. It also prints `token present: True/False`, which tells you
whether the problem is a missing token or a token without access.

---

## Step 8 — Stage 2, the actual experiment

Only after Stage 1 passes. Change the last line to:

```python
import sys; sys.argv = ["run", "--model", "qwen", "--stage", "framing",
                        "--alphas", "0", "2", "4"]
main()
```

This adds the neutral framing and runs both. Same injection, same scoring, two questions:

- **introspective** — *"Do you detect an injected thought? If so, what is it about?"*
- **neutral** — *"Is there anything unusual or out of place here? If so, what is it
  about?"*

The neutral prompt never mentions the model, a mind, thoughts, injection, or internal
states. It cannot invite a report *about the self*.

### What the outcomes mean

| Result | Interpretation | Value |
|---|---|---|
| neutral ≈ or > introspective | **The defended detection claim is anomaly sensitivity, not self-access** | The headline. A control nobody ran, on the model where the effect is claimed |
| neutral << introspective | Detection is specific to self-directed questioning | Calibration evidence **for** their account — publishable, not a headline |
| both ≈ 0 | Detection did not survive our setup | Stage 1 should have caught this; recheck |

**Both branches are publishable, which is why this is worth running.** Write down which
you expect before you look — see the pre-registration note below.

---

## Step 9 — Save the results

```python
import pandas as pd
df = pd.read_json("/kaggle/working/s3_results.jsonl", lines=True)
df.to_csv("/kaggle/working/s3_results.csv", index=False)
df.groupby(["framing", "alpha"])[["detected", "identified", "coherent"]].mean()
```

Download `s3_results.jsonl` from the **Output** tab (right sidebar) and paste the summary
table back into the conversation. The raw file keeps every generated response in the
`text` field, which is what you need if anything looks wrong.

---

## Before Stage 2: pre-register

This project has produced five wrong conclusions from proxy measurements in two weeks,
every one caught only because ground truth was available. Stage 2 has no such safety net —
it is a comparison between two conditions with no correct answer to check against.

So before running it, write into `docs/` and commit:

- the exact criterion (e.g. *"neutral detection at α=4 within 3 points of introspective, or
  higher"*),
- the number of trials,
- and **what you will conclude if it goes the other way.**

A criterion chosen after seeing the numbers is not a criterion.

---

## Notes

- **Resumable.** Results are appended per trial with `fsync`, and re-running skips
  completed trials. A disconnect or the 12-hour cap costs one trial, not the run.
- **If detection is too weak to compare framings**, Macar et al. report that ablating the
  refusal direction lifts detection from 10.8% to 63.8%. An abliterated model gives ~6×
  the signal, and testing their *elicited* setting engages their own under-elicitation
  argument on its own terms.
- **Quota.** ~1 hour per full run against 30 h/week. You can afford to sweep.
