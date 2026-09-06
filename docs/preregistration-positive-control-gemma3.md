# Pre-registration — Positive Control: Reproduce Introspective Detection in Gemma3-27B under 4-bit

**Filed 2 September 2026, before the run. Nothing below changes after seeing results.**

## Why a positive control before anything else

The only path to a jaw-dropping Study 3 runs through a neutral-framing *detection* control on a
model where detection is actually reported. That model is Gemma3-27B (Macar, Yang, Wang,
Wallich, Ameisen & Lindsey, 2603.21396): **10.8% detection, 0% false positives, at layer 37,
α = 4.** Nobody reports the numerical precision they ran at. We can only afford 4-bit on Kaggle.
If quantisation suppresses the effect, every downstream experiment is uninterpretable — so the
first run must show the effect exists in our setup, and nothing else.

**This run tests the pipeline, not a hypothesis.** It has no neutral-framing arm. Mixing the two
would make a failure to reproduce indistinguishable from a real finding.

## Protocol (frozen, following 2603.21396 §2, App. B.2, Table 4)

| Element | Setting | Source |
|---|---|---|
| Model | `google/gemma-3-27b-it` | their primary model |
| Precision | 4-bit NF4, double-quant, fp16 compute | **our constraint — the thing under test** |
| Hardware | Kaggle 2×T4, `device_map="auto"` | |
| Layer | residual stream after decoder block 37 (0-indexed) | their L = 37 |
| Vector | `v_c = h_c − mean(h_baseline)` at the **last token** of `"Tell me about {c}"` via chat template; baseline = 100 unrelated common nouns | their App. B.2 |
| Injection | `h ← h + α·v_c`, raw (no renormalisation), **prompt positions only** — not during generation | their §2 |
| α | {0, 4} | their headline setting |
| Prompt | Lindsey-style preamble + `"Trial {n}: Do you detect an injected thought? If so, what is the injected thought about?"` | their Table 4 |
| Concepts | 40 (20 concrete, 20 abstract), fixed list in the notebook | subset of their 500 |
| Trials | 40 concepts × 2 α × 3 seeds = 240 generations, sampled at T = 1.0, 60 new tokens | |
| Scoring | Rule-based YES on the first sentence + coherence filter; **raw text saved for a later LLM-judge pass**; first-token P(yes) recorded as a judge-free secondary | their scorer is GPT-4.1-mini — a deviation, flagged |

## Pre-registered criteria

**PASS** — all four hold:
1. Detection rate at α = 4 in **[0.05, 0.25]** (theirs: 0.108; band allows quantisation and prompt drift)
2. False-positive rate at α = 0 **≤ 0.02** (theirs: 0.00)
3. Coherence rate at α = 4 **≥ 0.90** (rules out "brain-damaged" affirmatives)
4. **No NaN/Inf** in any forward pass (fp16 on T4 is a known Gemma risk)

**FAIL-LOW** — detection < 0.05: run the pre-specified follow-up cell (L ∈ {36, 38}, α ∈ {6, 8}).
If still < 0.05, **4-bit quantisation suppresses the effect on this hardware** and the Study 3
path is closed at this compute. Report that.

**FAIL-FPR** — false positives > 0.05: the scorer or prompt is broken. Fix and re-run *this
control*; do not proceed.

**FAIL-COHERENCE** — coherence < 0.90 at α = 4: quantisation noise makes α = 4 too strong. Try
α ∈ {2, 3} in the follow-up cell.

## Stated in advance

- **We expect PASS**, with detection somewhat below 10.8% because quantisation adds noise and
  our rule-based scorer is stricter than a judge.
- A PASS licenses exactly one next step: the neutral-framing detection control, under its own
  pre-registration.
- A FAIL of any kind does not license changing the criteria. It licenses the follow-up cell
  named for that failure, once, and then a report.
- No concept is dropped after the fact. The 40 are fixed in the notebook before download.

## Deviations from their protocol, declared

1. **4-bit instead of their (unstated) precision** — this is the variable under test.
2. **Rule-based scoring instead of GPT-4.1-mini** — raw text is saved so the judge can be
   applied afterwards; the first-token P(yes) provides a scorer-independent check.
3. **40 concepts instead of 500** — adequate for a ±5-point band at n = 120 per arm.
4. **Preamble reconstructed from Lindsey (2025)** — if their Table 4 preamble differs
   materially, substitute it verbatim before running and note the substitution.
