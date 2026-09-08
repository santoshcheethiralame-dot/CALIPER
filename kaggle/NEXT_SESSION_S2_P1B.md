# Study 2 P1b — planted **concept** direction recovery (~2 h, headless is fine)

**Script version `2026-09-08c`.** Re-upload as a New Version of `caliper-s3`.
Protocol frozen in `docs/preregistration-s2-p1b.md`, filed before this run.

**Why this supersedes P1.** The first attempt planted random unit directions and is
recorded VOID (C54/C55) — recovery sat *below* the null at every strength and every
depth. A random direction has no natural representation, so the text it produces carries
no consistent signal for difference-of-means to recover. The pipeline was asked for
something that cannot happen. This run plants **real concept vectors**, which C48 showed
demonstrably steer Gemma at 40% of the residual norm.

**What it can and cannot show.** A concept direction is the *easy* case — the model
already represents it. So a PASS is a **necessary condition**, not a validation for
arbitrary traits. A FAIL, with the manipulation check passed, is a real finding about a
deployed method.

## Setup

Gemma 3 27B under **Models**, `caliper-s3` at **`2026-09-08c`** under **Datasets**,
GPU T4 x2, Internet on, **session restarted**.

## Cell 1 — after a restart

```bash
pip install -q -U bitsandbytes accelerate transformers
```

## Cell 2 — P1b

```python
import sys, glob
sys.argv = ["run", "--model", "gemma", "--compute-dtype", "fp32", "--stage", "plant",
            "--normalise", "--plant-source", "concept",
            "--alpha-frac", "0.20", "0.40", "0.60",
            "--n-plants", "8", "--n-prompts", "16",
            "--out", "/kaggle/working/p1b.jsonl"]
hits = glob.glob("/kaggle/input/**/*.py", recursive=True)
print(hits)
exec(open(hits[0]).read())
```

**Only this cell.** There is no cell 3 — the depth curve is a separate run and only
happens if P1b passes. Headless runs every cell present, so leaving a stale cell 3 in the
notebook will burn an hour on a run that may be void.

## Pre-flight

| line | must read |
|---|---|
| 1st line | `kaggle_s3_positive_control 2026-09-08c` |
| read position | `decoded ' table'` |
| residual norm | `median 36245.0` |
| plants | `8 planted CONCEPT vectors: ['elephant', 'spider', ...]` |
| nulls | `null A (different concept): ['harbor', 'violin', ...]` |

If it says `planted RANDOM directions`, `--plant-source` did not take and the run repeats
the void one.

## Reading the output

```
    alpha %norm  lyr     plant  steered  recovery   nullA   nullB
```

**Read `steered` first — it gates everything else.** It is the count of 16 generations
containing the concept or a concept-specific associate. Baseline is roughly 3/30 in C48's
measurement, so about 1–2 of 16 by chance.

| what you see | meaning |
|---|---|
| `steered` at 40% is clearly above ~2/16, and `recovery` > `nullA` on 6+ of 8 with median > 0.30 | **PASS.** The pipeline recovers a planted concept. Send the file; the depth curve is next |
| `steered` clearly above chance, but `recovery` ≈ `nullA` | **The substantive negative.** The plant reached the text and extraction still missed it. This is a real finding about a deployed method |
| `steered` at or near chance | **VOID again.** The plant never reached the text; no recovery number means anything. Send it and say so — per the pre-registration we reconsider the study rather than re-run on a third guess |

`nullA` is a *different concept's* vector — conservative, since concept vectors share
structure. `nullB` is a random direction and will be smaller. Recovery must beat `nullA`,
not just `nullB`.

## Time

~8 min load, then 16 baseline + 3 strengths × 8 plants × 16 = 400 generations, each
followed by a re-read pass. Roughly **1 h 50 m**. Well inside the 12 h headless limit.

## Send back

`p1b_plant.jsonl` and `p1b.config.json`. The jsonl now carries the generated text for
every plant at the plant layer, so the manipulation check is auditable rather than
inferred — that is the gap that made the first run uninterpretable.
