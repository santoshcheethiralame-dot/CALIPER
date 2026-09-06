# Pre-registration — PC1: Positive Control for Introspective Detection on Gemma3-27B in 4-bit

**Filed 2 September 2026, before the run. Nothing below may change after seeing results.**

## Why this run exists

Study 3's jaw-dropping path requires a neutral-framing *detection* control on a model where
untrained detection is documented. The smallest such model is Gemma3-27B (Macar, Yang, Wang,
Wallich, Ameisen & Lindsey, 2603.21396): **10.8% detection, 0% false positives at L=37, α=4.**

Kaggle's 2×T4 (32 GB) holds Gemma3-27B only in 4-bit. **Nobody reports the precision they ran
at.** Detection is a ~10% effect with 0% FPR; quantisation noise could suppress it or inflate
false positives. If the published number does not reproduce under our compute, no downstream
framing experiment is admissible — and we want to know that before spending the quota.

## Protocol (frozen — matches 2603.21396 §2, App. B.2, Table 4 as extracted)

- **Model:** `google/gemma-3-27b-it`, 4-bit NF4 (bitsandbytes), fp16 compute, `device_map="auto"`
  across 2×T4.
- **Layer:** L = 37, residual stream output of decoder block 37 (0-indexed), 62 layers total.
- **Concept vector:** `v_c = h_c − h̄_baseline`, where `h_c` is the last-token residual at L for
  the chat-templated prompt *"Tell me about {c}"*, and `h̄_baseline` is the mean last-token
  residual over 100 unrelated common nouns. No normalisation.
- **Injection:** `h ← h + α·v_c` at layer L, **prompt positions only** (prefill), not during
  generation. α ∈ {0, 4}. α = 0 is the matched control (same prompt, no injection).
- **Detection prompt:** researcher preamble stating that neural patterns ("thoughts") may be
  injected on 50% of trials, then *"Trial {n}: Do you detect an injected thought? If so, what
  is the injected thought about?"* — via the model's chat template.
- **Generation:** sampling, temperature 1.0, max 60 new tokens.
- **Design:** 40 concepts × 6 trials × 2 conditions = 480 generations, seeds fixed.
- **Scoring (first pass):** rule-based — YES iff the first two sentences contain an affirmative
  detection and no negation, AND the response is coherent (repetition ratio < 0.5, ≥ 8 words).
  All raw generations saved so an LLM judge can be run afterwards if the rule-based pass is
  ambiguous.

## Pre-registered criterion

> **PASS** if the α=4 detection rate's Wilson 95% interval **includes 10.8% or has lower bound
> above 5%**, **and** the α=0 false-positive rate's Wilson 95% upper bound is **below 3%**.

At n = 240 per condition a true rate of 10.8% gives an interval of roughly [7%, 15%], so the
criterion can distinguish the published effect from zero while tolerating the concept-subset
and judge deviations below.

## Declared deviations from the source protocol

1. **Judge.** Rule-based first pass instead of GPT-4.1-mini. Raw text is saved for a later LLM
   judge; the pre-registered verdict is on the rule-based score.
2. **Concept set.** 40 concepts, not 500. Chosen before the run, listed in the notebook, spanning
   concrete nouns, abstract nouns and emotions, including Lindsey-style items.
3. **Baseline nouns.** Our own list of 100 common nouns; the source list is not reproduced
   verbatim. The vector construction method is identical.
4. **Precision.** 4-bit NF4 with fp16 compute. Source precision unstated. **This is the variable
   under test.**

## Stated in advance

- **Expected:** detection 7–15% at α=4, FPR ≤ 1% at α=0.
- **If detection ≈ 0 at α=4:** the effect did not survive our compute or a protocol detail is
  off. Permitted follow-ups, *stated now*: (a) an α sweep {2, 4, 8}; (b) a check for fp16
  overflow (NaNs / degenerate text); (c) verification of the prompt preamble against the source
  repository. **Not permitted:** changing the criterion, the layer, or the scoring rule to reach
  a pass.
- **If FPR is high at α=0:** the judge or generation is degenerate; report it. Do not proceed to
  the framing experiment.
- **This run makes no claim about introspection.** It establishes only whether the published
  detection effect reproduces under Kaggle 4-bit compute.
