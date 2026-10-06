# Level 3 Part B brief: calibrating self-report trust checks (S-5 to S-10)

**Owners:**
- Member 4: S-5, S-6, S-9, plus S-12;
- the S-7, S-8 and S-10 owner is to be settled by the team.

**Plan of record:** `docs/RUN_PLAN_L2_L3.md` §3, Part B. **Gate B:** mid-March 2027.

## The question

When a model reports on itself, practitioners trust the report if it is stable: the same
answer across samples, paraphrases or temperatures. That is restart agreement again, and
Paper 1 showed it is the weakest check at Level 1. Part B builds self-report tasks whose
correct answer is computed exactly from the model's own internals or outputs. It then scores
the trust checks against that answer.

**Checks to calibrate:**
- agreement: mode frequency over 10 samples, paraphrase consistency, seed or temperature
  stability;
- other: verbalised confidence, and forced-choice or logit scoring (the functional check).

**Report with every task:** a privileged-access index, the model's accuracy minus the best
outside observer's accuracy on the same text.

**Models:** Qwen2.5-0.5B/1.5B/3B/7B-Instruct and Gemma-3-1B/4B-it. Ground truth is always
computed on the **same quantised model** that reports.

| ID | task | exact truth | cost |
|---|---|---|---|
| S-5 | predict its own answer | its logits on the object-level prompt | ~4 GPU-h |
| S-6 | grammaticality (BLiMP minimal pairs) | its own string probabilities | ~2 GPU-h |
| S-7 | which of N inputs was perturbed; which injection was stronger | the injection log, S-2's **live** vectors only | ~4 GPU-h |
| S-8 | source: injected vector vs misleading prompt vs nothing | the condition | ~2 GPU-h |
| S-9 | its own sampling temperature (negative control) | the setting | ~1 GPU-h |
| S-10 | PLANTED: can a verbaliser name a planted concept for the right reason, and say nothing when nothing is planted? | the plant | ~8 GPU-h; needs `readouts.py` |

## Dependencies, in order

1. **S-12 first.** Judge-scored numbers wait for kappa ≥ 0.6
   (`docs/team/S12_grading_protocol.md`). The 200-item sample is drawn and ready.
2. **S-7 and S-8 wait for S-2.** Only vectors that pass the steering gate may be injected.
   A dead vector makes "the model could not tell" uninterpretable.
3. **S-10 needs `readouts.py`:** logit lens, Patchscopes-style and SelfIE-style verbalisers.
   APERTURE's E13 design is the spec.

## Injection apparatus

Use `experiments/kaggle_s3_positive_control.py` v2026-10-07a (S-0):
- `--quant` for precision;
- `--vector-recipe` for the extraction arm;
- per-trial next-token KL in every row, so reports can be read inside coherence bands;
- `--control random-impact` for content-free vectors matched on effect.

Write new tasks as stages beside the existing ones. Do not fork the script; the S-0 tests
(`tests/test_s3_script.py`) must keep passing.

## Rules that bite here

- Score forced-choice identification with APERTURE's gamma prior-null estimator, so that a
  model naming high-frequency concepts is not credited with access.
- Report emotion concepts separately (the affect confound, A-R4 to A-R6).
- The framing must not cue the answer. APERTURE R12: the more the prompt talks about
  injection, the lower the identification.
- Pre-register each task before its first data. Write same-day, append-only notebook entries.
  Commits carry no generated-by or co-author trailers.
