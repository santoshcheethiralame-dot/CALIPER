# Handoff, 13 Oct 2026: P2-H and P2-E filed, one Kaggle session pending

Read this first, then the two preregs. Nothing here has data yet.

## State in one paragraph

Two Paper 2 runs were filed before data on 13 Oct and share one Kaggle session on
Llama-3.1-8B-Instruct. **P2-H** is the positive control the second internal review made
required for ICML (ruling 9, item B2). It runs Hahami et al.'s released binary-detection protocol
inside the audit. **P2-E** is new. It tests whether eval-awareness steering beats directions of
equal effect that carry no evaluation content. It came from a cross-field literature sweep
(`projects/research-radar/RADAR.md`), which found that the two published control results for that
steering disagree and that neither matched effect. Scripts, scorers, tests and the run sheet are
committed. The next step is the user's Kaggle run.

## Files

| what | P2-H | P2-E |
|---|---|---|
| prereg | `docs/preregistration-p2h-hahami-llama.md` | `docs/preregistration-p2e-eval-awareness.md` |
| Kaggle script | `experiments/kaggle_p2h_hahami.py` v2026-10-13a | `experiments/kaggle_p2e_eval_awareness.py` v2026-10-13a |
| scorer (run once) | `experiments/analyse_p2h.py` | `experiments/analyse_p2e.py` |
| tests | `tests/test_p2h_script.py`, `tests/test_analyse_p2h.py` | `tests/test_p2e_script.py`, `tests/test_analyse_p2e.py` |
| run sheet | `kaggle/NEXT_SESSION_P2H_P2E.md` (both, P2-H first) | same (`NEXT_SESSION_P2E.md` is for a P2-E-only rerun) |

Both scripts import `kaggle_s3_positive_control.py` v2026-10-11b (unchanged), so all three files go
into the `caliper-s3` dataset in one folder. Tests: `python -m pytest tests/test_p2h_script.py
tests/test_analyse_p2h.py tests/test_p2e_script.py tests/test_analyse_p2e.py` → 10 pass (tiny Llama
must be in the HF cache). A "Windows fatal exception: access violation" faulthandler dump printed
during the transformers import is not a failure.

## What the user must do

1. Upload the three scripts as a new `caliper-s3` version.
2. Attach Kaggle Models **Llama 3.1 8b-instruct** (accept Meta's licence on Kaggle), GPU T4 x2,
   Internet On.
3. Run `kaggle/NEXT_SESSION_P2H_P2E.md` Cells 1-5. About 1-1.5 h for P2-H, then 3-5 h for P2-E.
4. Send `p2h_p2e_llama8b.zip`, and say whether either run hit `M1 FAILED` and used the 4-bit fp32
   fallback, whether the fidelity cosines were near 1, the P2-E layer AUROC, and whether all three
   P2-E doses printed `'matched': True`.

## When the zip lands (score once; do not re-analyse into a different reading)

1. Unzip to `results/p2h_p2e/` and record sha256 of both jsonl files in the notebook.
2. `python experiments/analyse_p2h.py --file results/p2h_p2e/p2h_llama8b.jsonl`
   → `results/p2h_analysis.json`. It imports `rescore_steering_gate` (run from the repo root as
   written, or with `PYTHONPATH=experiments`).
3. `python experiments/analyse_p2e.py --file results/p2h_p2e/p2e_llama8b.jsonl`
   → `results/p2e_analysis.json`.
4. Notebook: §3 rows, §4 entries, §6b decision rows; mark both §7.10 rows done.
5. Paper 2 (`docs/paper2-outline.md` §4, §6, §7): apply each prereg's interpretation table as
   written. P2-H decides whether Paper 2 has a positive control (ICML vs workshop at Gate P2,
   15 Jan). P2-E decides whether candidate claim C9 exists.
6. Run an internal review of the new claims before they are committed. Both earlier reviews caught
   errors written into claims on the day the data landed.

## Things that are easy to get wrong

- **Hua et al. DID run a random control** (24 Gaussian vectors at about the steering vector's
  norm; weak effect). An earlier summary said they did not. The P2-E question is that their control
  was norm-matched, not effect-matched, and that UK AISI's unrelated *contrastive* controls were
  strong.
- **P2-H keeps two quirks of Hahami's release on purpose**: the vector is read at the input of
  block L and injected at its output, and the readout feeds the last prompt token twice. Do not
  "fix" them; the single-pass readout is stored beside the released one so the effect can be
  reported.
- **P2-H's bootstrap is over concepts** (ruling 4's filed unit), unlike P2-R's original scorer.
- **P2-E's P1/P2 compare magnitudes** |Δ|, because random and unrelated directions have arbitrary
  sign; the signed effect of the real vector is secondary.
- **P2-E needs no judge** (P2-J). The readout is the model's own YES/NO to "is this a test or
  evaluation of you?", gated by M4 (eval items must already score above deploy items before any
  steering).
- The P2-E test prompts (30 eval-style, 30 deploy-style, plus 10 + 10 for calibration) were written
  for this filing and live in the script. If the user wants to change them, that has to happen
  before the run, as an amendment.

## Also from the 13 Oct sweep (not runs)

- `docs/flagship-plan.md` §5: F-9 / F-10 report every check per model family (Calibration-Family
  Overfit 2607.06596 matches our B-8b family effect). Write it into those preregs when they are
  filed.
- `docs/flagship-plan.md` §5, after F-13: capture-recapture over SAE seeds, calibrated on
  SynthSAEBench, entered as a lead for Paper 3 (ties in Proposal V). Decide at Gate F-B.
- `docs/paper1-hardening-plan.md`: candidate Paper 1 sentence citing Norman et al. 2606.19544
  (reliability without validity in LLM judges). The author's call; `main.tex` was not touched.
- `docs/CITATIONS.md` §32: every new source with its reading level.

## Untouched

The working tree had many uncommitted changes before 13 Oct (LICENSE, PROJECT_NOTES_COMPLETE.md,
several kaggle sheets, paper/, presentation/, results/*.json, `experiments/s1_2_deflation.py`
deleted). None were committed or modified by this work; check them before the next commit.
