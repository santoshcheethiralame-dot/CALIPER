# Next Kaggle session: S-3, finish APERTURE's F1 (confound hardening)

Governed by the frozen pre-registration in APERTURE:
`projects/mirror/docs/prereg/2026-07-30-f1-confound-hardening.md` (frozen at `9267351`).
Nothing in that design changes. These sessions run the 13 missing files, plus c05 neutral
again (see "The c05 change" below), so 14 files in all.

## The c05 change (decided 7 Oct 2026, before any new F1 data)

The 11 finished files came from an older library stack, and their versions were not
recorded. Kaggle's image has moved on since: S-11 ran on transformers 5.18 and then 5.19.
c00 to c04 have both framings on the old stack, so their within-config difference D(c) is
clean. c05 does not: its neutral file is old-stack, and its introspective file would be
new-stack. That would put version drift (5 of 192 answers in A-F1 vs A-R11) inside D(c05).
So c05 neutral is run again on the new stack and becomes the scored file. The old c05
neutral is kept, and its agreement with the new one is reported as a drift measurement. P2's
pooled difference still spans two stacks (c00-c04 old, c05-c11 new); this is reported as a
limitation. Library versions are printed this time (recovery cell).

**State before the session:** 11 of 24 files are complete and hash-verified locally, 96 rows
each. These are c00 to c04 for both framings, plus c05 neutral. Missing: c05 introspective
and c06 to c11 for both framings. **Expect two sessions:** the notebook stops itself at 8
hours, and the last batch managed 11 files in 8.8 h.

**Do not score the partial data.** The frozen design scores all 12 configs once, with
`aperture/f1_score.py`.

## Before you start (once)

1. **Upload the finished files as a Kaggle Dataset.** Use
   `C:\Users\carbo\Downloads\aperture-f1-partial.zip` (33 files, 4.5 MB) and name the dataset
   exactly **`aperture-f1-partial`**. The recovery cell finds the old c05 neutral
   by that name. Do **not** attach the outputs of the August F1 notebook versions: they hold
   the old c05 neutral under another path, and the recovery cell would stop the run.
2. **Import the notebook:** `C:\Users\carbo\projects\caliper\kaggle\f1_s3_session.ipynb`
   (File > Import Notebook). This is APERTURE's `f1_confound_hardening.ipynb` with one
   change: the recovery cell (the 6th cell, right after the one that defines `NAMES` and
   `CONFIGS`) is replaced by the code below. Every other cell is byte-identical. Do not edit
   any cell for session 1.

   *Why a file:* on 7 Oct the hand edit went into the 5th cell, which is the `NAMES` cell,
   because Kaggle counts cells from 1. Cell 6 then failed with `NameError: NAMES` before any
   data was made.
3. For reference, the replaced recovery cell:

   ```python
   import glob, os, shutil, importlib.metadata as md

   SESSION = 1                  # set to 2 for the second session
   OLD = "aperture-f1-partial"  # the dataset with the 11 files from the old library stack

   recovered, left_out = 0, 0
   for f in sorted(glob.glob("/kaggle/input/**/f1_*", recursive=True)):
       if OLD in f and os.path.basename(f).startswith("f1_c05_neutral"):
           left_out += 1
           continue
       shutil.copy(f, os.path.basename(f))
       recovered += 1
   print(recovered, "files recovered;", left_out, "old-stack c05-neutral files left out")
   assert left_out == 3, "attach the aperture-f1-partial dataset"
   assert os.path.exists("f1_c05_neutral.jsonl") == (SESSION == 2), \
       "session 1 must not start with a c05 neutral file; session 2 must"
   print({p: md.version(p) for p in ("torch", "transformers", "bitsandbytes", "accelerate")})
   ```
4. **Inputs** (right panel, Add Input):
   - Models: Gemma 2, `google/gemma-2-2b-it`, Transformers. Cell 3 asserts it.
   - Datasets: `aperture-f1-partial`.
5. **Secrets:** `HF_TOKEN` attached (Cell 1 reads it).
6. **Session options:** GPU T4 (x1 is enough for 2B in 8-bit), Internet **On**. Cell 0 installs
   `aperture` from GitHub `main`, which is `b5bb2fb` (checked 7 Oct).

## Session 1

**Save Version > Save & Run All (Commit)**, so it runs headless and survives closing the
tab. Cell 6 stops itself after 8 hours, at a file boundary.

Healthy output:
- Recovery cell: `30 files recovered; 3 old-stack c05-neutral files left out`, then the
  versions. If you also see `33 files recovered from earlier versions`, the old recovery
  cell is still in the notebook: stop and re-import.
- Cell 6: `skip (complete)` for the 10 files c00-c04, then
  `=== c05 [paraphrase] neutral -> f1_c05_neutral.jsonl (t+0.00h)`.
- One `extracting 16 vectors at layer ..., seed ...` line per new (layer, seed).
- About 48 minutes per file, so about 10 files: c05 to c09, both framings. Then
  `budget reached, stopping cleanly`.
- Cell 7: `broken: []` and `DOWNLOAD /kaggle/working/artifacts/F1.zip`.

Do not open or tally any new file.

## Session 2

1. Open the same notebook and set `SESSION = 2` in the recovery cell (the one starting
   `import glob, os, shutil, importlib.metadata`). That is the only edit.
2. Add Input: session 1's **output** (Your Work > the notebook > Output, or Add Input >
   Notebook Output). Keep `aperture-f1-partial` attached.
3. Save & Run All (Commit) again. The recovery cell now picks up the new c05 neutral from session 1's
   output. Cell 6 skips 20 files and runs c10 and c11 (4 files, about 3.2 h).
4. Cell 7 should print `24 of 24 configs complete` and `broken: []`.

Total: about 11.5 GPU-hours across both sessions.

## After each session

Download the version's output (or `artifacts/F1.zip`) and send me the zip, as with S-11.
Do not unzip it over `projects/mirror/runs/F1/`: that would overwrite the old c05 neutral,
which is kept for the drift check.

## When all 24 are in

Tell me and I will run the frozen scorer once, `aperture/f1_score.py`, against P1 to P4
and the decision rule. I will register the run as A-F1 / S-3 in CALIPER's notebook, next
to C20.

## Outcome (7 Oct 2026)

One session finished all 14 files in about 45 minutes, so session 2 was not needed. Scored
once with the frozen scorer: P1-P4 all hold. Details in the notebook (§3 S-3 / A-F1, §4).
