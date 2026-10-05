# Next Kaggle session: S-3, finish APERTURE's F1 (confound hardening)

Governed by the frozen pre-registration in APERTURE:
`projects/mirror/docs/prereg/2026-07-30-f1-confound-hardening.md` (frozen at `9267351`).
Nothing in that design changes. This session only runs the 13 files that are missing.

**State before the session:** 11 of 24 files are complete and hash-verified locally, 96 rows
each. These are c00 to c04 for both framings, plus c05 neutral. Missing: c05 introspective
and c06 to c11 for both framings. **Expect two sessions:** the notebook stops itself at 8
hours, and the last batch managed 11 files in 8.8 h.

**Do not score the partial data.** The frozen design scores all 12 configs once, with
`aperture/f1_score.py`.

## Before you start (once)

1. **Make the finished files available as an input.** Either attach the output of the last
   F1 notebook version you ran on Kaggle, or upload
   `C:\Users\carbo\Downloads\aperture-f1-partial.zip` (33 files, 4.5 MB) as a new private
   Kaggle Dataset, e.g. `aperture-f1-partial`.
2. Open the F1 notebook (`projects/mirror/notebooks/f1_confound_hardening.ipynb`). Upload it
   if it is not already on Kaggle.
3. Right panel, Input: you need **two** entries.
   - **Models:** Gemma 2, `google/gemma-2-2b-it` (Transformers). The model cell asserts it.
   - **Datasets:** the F1 files from step 1.
4. Add-ons, Secrets: `HF_TOKEN` must exist (cell 1 reads it).
5. Session options: Accelerator **GPU T4 x2** (one T4 is enough for 2B in 8-bit; either is
   fine), Internet **On** (cell 0 installs `aperture` from GitHub main, which matches local
   commit `b5bb2fb`).

## Run

**Run All**, or better, **Save Version, Save & Run All (Commit)**, so it runs headless and
survives closing the tab.

What healthy output looks like:
- cell 5 prints `33 files recovered from earlier versions` (or 33 plus any others);
- cell 6 prints `skip (complete): f1_c00_neutral.jsonl` for the 11 finished files;
- cell 6 then starts at `f1_c05_introspective.jsonl`;
- vector extraction lines appear once per (layer, extract seed).

## After each session

1. Download `/kaggle/working/artifacts/F1.zip` (cell 7 prints `DOWNLOAD ...`) **before the
   session expires**.
2. Unzip into `projects/mirror/runs/F1/` and check that cell 7 printed `broken: []`.
3. Session 2: attach session 1's output version as an input too, so its new files are
   recovered, and run again. It resumes from where it stopped.

## When all 24 are in

Tell me and I will run the frozen scorer once, `aperture/f1_score.py`, against P1 to P4
and the decision rule. I will register the run as A-F1 / S-3 in CALIPER's notebook, next
to C20.
