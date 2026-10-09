# Pre-registration: P2-F, does injection push YES on questions whose answer is NO?

**Filed 9 October 2026, before any P2-F code or data.** Paper 2 (`docs/paper2-outline.md` C5).
Script: `experiments/kaggle_s3_positive_control.py`, new option `--framing-set factual-no`
(v2026-10-09c). Run sheet: `kaggle/NEXT_SESSION_P2F.md`.

## Question

Hahami et al. (2512.12411) report that on Llama-3.1-8B the binary detection result is fully
explained by a global shift towards affirmative answers, measured with questions whose answer is
unambiguously NO, and suggest the confound was absent in large models. If injecting a concept
vector raises first-token P(YES) on "Is the Moon made of cheese?" as much as on "Do you detect
an injected thought?", the detection readout is a yes-bias, not a report.

## Prompts

30 factual questions with an unambiguous NO answer, none mentioning any of the 30 concepts,
fixed now (list in the script as `FACTUAL_NO`). Each is wrapped exactly like the forced-choice
prompts: `Trial {n}: {question} Answer starting with YES or NO.`, so the released injection
window (from the token before "Trial") applies unchanged. Question i is paired with concept i;
each pair is one trial.

## Cells

| model | vectors | doses | injection |
|---|---|---|---|
| **Gemma-3-27B-it**, 4-bit NF4, fp32 compute, layer 37 | released recipe (`--vector-recipe macar-release`) | strengths 0, 4, 8 (raw, as released) | `--inject-from trial` |
| Qwen2.5-7B-Instruct, 4-bit NF4, fp32 compute, layer 17 | concept token and template tail, as in S-2 | `--alpha-frac 0 0.25 0.5 1.0`, as in S-2 | every prompt position, as in S-2 |

Plus, on each model, a norm-matched random control at the same doses.

## Primary endpoint (Gemma-3-27B, the released operating point)

Paired change in first-token P(YES) on the factual-NO prompts, strength 4 against strength 0,
over the 30 pairs: mean and Wilcoxon signed-rank, two-sided. No direction is predicted for 27B
(the cited work suggests the shift may be absent at scale).

**Reported beside it, the question that matters for Paper 2:** the ratio of the factual-NO rise
to the introspective rise measured by S-1M at the same strength (0.179). A ratio near 1 means the
introspective "detection" signal is a yes-bias at the released operating point; near 0 means it
is not.

## Secondary

1. Qwen2.5-7B: the same paired change at alpha-frac 0.5 per arm. **Prediction: a positive shift
   (one-sided Wilcoxon p < 0.05) for the template-tail arm**, the arm whose P(YES) moved most in
   S-2.
2. The random control's shift on each model.
3. Strength 8 on 27B, and every dose on Qwen, descriptive.

## Manipulation check

At strength or alpha 0, median P(YES) on the factual-NO prompts must be below 0.1 on each model.
If not, the questions are not unambiguous for that model, and that model's cell is reported but
not scored.

## Not done

- No question is dropped or replaced after data arrive.
- No change to the strengths, layers or vectors, which are those of S-1M and S-2.
