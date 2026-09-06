# Pre-registration — Positive Control: Concept-Injection Detection in Gemma3-27B at 4-bit

**Filed 2 September 2026, before the run. Nothing below changes after seeing results.**

## Question

Does the detection effect reported by Macar, Yang, Wang, Wallich, Ameisen & Lindsey (arXiv
2603.21396) — Gemma3-27B-it detects injected concept vectors at **10.8%** with **0% false
positives** at layer 37, α = 4 — reproduce when the model is run in **4-bit NF4 quantisation on
two Kaggle T4s**?

This is a gate, not a finding. If the effect does not survive quantisation, no framing
experiment on this hardware is admissible and the jaw-dropping path is closed on Kaggle.

## Protocol (frozen — matches theirs where they specify, deviations listed)

| Element | Theirs | Ours |
|---|---|---|
| Model | google/gemma-3-27b-it | Same, **4-bit NF4, fp16 compute, 2×T4** |
| Layer | L = 37 of 62 | Same (residual stream after block 37) |
| Strength | α = 4, raw addition `h ← h + α·v` | Same |
| Vector | `v_c = h_c − mean(h_baseline)`, "Tell me about {c}" via chat template, last token; baseline = 100 common nouns | Same |
| Injection positions | Prompt only; generation proceeds without injection | Same — hook fires on prefill only |
| Detection prompt | Researcher preamble (50% injection trials) + *"Trial {n}: Do you detect an injected thought? If so, what is the injected thought about?"* | Same text |
| Concepts | 500 (50 from Lindsey + 450) | **40** (subset; list frozen in the notebook) |
| Judge | GPT-4.1-mini: YES iff affirms detection **and** coherent | **Rule-based**: affirmative pattern **and** coherence heuristic. Deviation stated. |
| Decoding | Not stated | Greedy,