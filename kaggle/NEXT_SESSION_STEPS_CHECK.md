# B-11 steps check — is the scale effect an optimiser artefact?

**Executing a check pre-committed in `preregistration-b11-pythia-ladder.md`. No new
filing needed; the branch and its interpretation were fixed before the ladder ran.**

---

## Why this run exists, and why it must happen before anything is claimed

The ladder found the failure rate rising sharply with scale:

| rung | d_model | pass | rate |
|---|---|---|---|
| 70m | 512 | 48/50 | 96% |
| 160m | 768 | 48/50 | 96% |
| 410m | 1024 | 48/50 | 96% |
| **1.4b** | 2048 | **31/50** | **62%** |

That is the most interesting outcome available and therefore the one to distrust first.
The filing says so in advance:

> **if the failure rate rises with scale, do not report it yet.** Wider models may simply
> be harder to fit at a fixed 1600 steps. Re-run the top rung at 3200 steps and report
> both. A rate that falls with more steps is an optimiser artefact, not a scale effect.

**The artefact reading is currently the more likely one.** Three rungs sit at *exactly*
48/50 across a doubling of width, and then only the widest breaks — at 4x the width of the
smallest, fitted with the same 1600 steps into the same width-64 bottleneck. That is the
shape of an under-optimised fit, not obviously the shape of a scale effect.

## What is being compared, and why it is paired

The unit draw depends only on `--neurons` and the seed — **not** on `--steps`. So the
3200-step run scores **the same 50 units** as the 1600-step run. This is a paired design:
compare with **McNemar's exact test on the discordant units**, not a two-proportion test.

Note the contrast with `--neurons`, which *does* change the whole draw (notebook section 8).
`--steps` is safe to vary; `--neurons` is not.

## Cell A — required

```python
!python experiments/e01_gate.py --model EleutherAI/pythia-1.4b --layer 12 --d-mlp 8192 \
    --restarts 2 --neurons 50 --steps 3200 --device cuda \
    --out /kaggle/working/b11_pythia-14b_s3200.jsonl
```

~4.3 h. **Note the distinct output file.** `--steps` is part of a run's identity, so it
gets its own file — never resume a 3200-step run into a 1600-step checkpoint.

## Cell B — the control, and it is worth the extra time

```python
!python experiments/e01_gate.py --model EleutherAI/pythia-410m --layer 12 --d-mlp 4096 \
    --restarts 2 --neurons 50 --steps 3200 --device cuda \
    --out /kaggle/working/b11_pythia-410m_s3200.jsonl
```

~2.1 h. **Run Cell A first**; this one only if the session has room.

**Why it earns its time.** Cell A alone cannot separate two explanations:

- *more steps helps everything* — in which case a 1.4b improvement says nothing about width
- *more steps helps the wide model specifically* — which is the under-optimisation claim

410m is the right control: adjacent in scale, and the cleanest rung in the ladder (min
alignment 0.9145, no severe failures at all). If 1.4b jumps toward 96% while 410m barely
moves, the width-dependent optimisation deficit is established directly rather than
inferred.

## Cell C — package

```python
import shutil, glob, os
rows = {f: sum(1 for _ in open(f)) for f in sorted(glob.glob("/kaggle/working/*.jsonl"))}
for f, n in rows.items():
    print(f"{os.path.basename(f):<34} {n:>4} rows" + ("   <- SHORT" if n < 50 else ""))
assert rows, "nothing to package"
shutil.make_archive("/kaggle/steps_check", "zip", "/kaggle/working")
print("wrote /kaggle/steps_check.zip", os.path.getsize("/kaggle/steps_check.zip"), "bytes")
```

---

## How the result will be read — fixed now, before the numbers exist

| outcome at 3200 steps | reading |
|---|---|
| **1.4b rises toward ~96%, 410m roughly unchanged** | **Optimiser artefact, width-dependent.** The scale "finding" is withdrawn. The ladder is reported as flat across 20x parameters *once each rung is adequately optimised*, and the paper gains a real methodological point: **a fixed step budget silently under-fits wider models, and would have been read as a scale effect** |
| **1.4b rises and 410m rises comparably** | More steps help everywhere. 1600 was simply too few for this bottleneck. Report the ladder at 3200 and say the 1600-step numbers were under-optimised throughout |
| **1.4b stays near 62%** | **The scale effect survives its own strongest challenge.** Reportable, and considerably stronger for having been filed in advance and tested |
| **1.4b falls further** | Unexpected. Report as measured, fit no story to it, and do not run a third step budget hunting for a preferred answer |

**Committed:** both step budgets are reported side by side whichever way this comes out,
and the 1600-step ladder table stays in the paper with its numbers intact. This check
cannot be used to quietly replace an inconvenient result with a convenient one.
