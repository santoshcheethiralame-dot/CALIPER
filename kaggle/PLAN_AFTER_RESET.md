# Kaggle plan for the next weekly quota (written 15 Oct 2026)

About 30 GPU-hours a week; one session stops at 12 hours. Three sessions, about 16 to 17 hours in
all, in this order, leaving about 13 hours for a failed session or a re-run.

| # | Sheet | What it answers | GPU-hours | Why this order |
|---|---|---|---|---|
| 1 | `NEXT_SESSION_P2H_P2E.md` | P2-H, the positive control (Hahami et al.'s protocol on Llama-3.1-8B); then P2-E, eval-awareness steering | 1-1.5, then 3-5 (deadline guard packages before the cap) | Paper 2's readiness gate for TACL needs both; P2-H decides the readout claims |
| 2 | `NEXT_SESSION_B1.md` | B1: what Gemma-3-27B emits when YES and NO lose their probability (P2-R Amendment 1) | 2-3 | Paper 2 C5, the second review's ruling 10 |
| 3 | `NEXT_SESSION_F4F6_v2.md` | F-4 gated units (Qwen2.5-0.5B) and F-6 unembedding (GPT-2, 32k tokens) | about 7.5 (deadline guard) | Flagship Gate F-B (31 Jan) needs three new substrates |

## Checked on 15 Oct

- **Script versions match the sheets.** `kaggle_s3_positive_control.py` v2026-10-11b,
  `kaggle_p2h_hahami.py` and `kaggle_p2e_eval_awareness.py` v2026-10-13a. Each sheet's Cell 2
  refuses a stale upload.
- **Sessions 1 and 2 share one upload.** Put the three scripts in one folder as a new version of the
  `caliper-s3` dataset (the P2-H/P2-E sheet's "Before you start"); B1 uses the same
  `kaggle_s3_positive_control.py`.
- **Session 3 uses `kaggle/caliper-bundle.zip` (10 Oct).** Neither `experiments/e01_gate.py` nor
  `caliper/` has changed since, so the bundle is current; its stale-bundle check still runs.
- **P2-H needs Meta's licence accepted on Kaggle** for the Llama 3.1 8B-Instruct model input, and
  Internet on (the fidelity check downloads Hahami et al.'s vectors; a failed download is recorded,
  not fatal).

## After each session

Download the output zip and say which session it was. Each run is scored once with its filed
scorer: `analyse_p2h.py`, the P2-E scorer, the B1 analysis, and the F-4/F-6 analyses. Paper 2's
tables fill themselves from the scored JSONs.

## If quota is left

Candidates, in order: re-run any session that hit its deadline guard; an F-4 arm at 16k tokens
(the pilot passed 10 of 16 there against 7 of 16 at 8k); nothing else is ready to run on a GPU.
