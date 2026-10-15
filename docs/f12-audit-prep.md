# F-12 prep: auditing directions in use with the validated checks

**9 October 2026.** Preparation only; F-12 runs after F-10 says which checks work, and gets its
own pre-registration then. Plan: `docs/flagship-plan.md` §5.

## What F-12 asks

There is no ground truth for a published steering or monitoring direction. F-12 takes the
checks the flagship validates against exact truth, applies them to directions the field uses,
and reports where those directions fall. A direction that fails a validated check is not
proven wrong; it is flagged as not trustworthy by the evidence that works.

## Targets

| target | source | recipe | models we can run | notes |
|---|---|---|---|---|
| Refusal direction | Arditi et al., NeurIPS 2024; `github.com/andyrdt/refusal_direction` | difference of means between harmful and harmless instructions at post-instruction positions, candidate (layer, position) picked by an ablation and steering score | Qwen chat models in the repo; Qwen2.5-0.5B/1.5B-Instruct on CPU | Only activations on the prompt sets are needed; no harmful completions are generated. The repo warns its data contain offensive text |
| Persona vectors | Chen et al. 2025; `github.com/safety-research/persona_vectors` (`generate_vec.py`) | difference of means of response activations between trait-eliciting and trait-suppressing system prompts; Qwen2.5-7B-Instruct layer 20, Llama-3.1-8B layer 16 | Qwen2.5-7B on Kaggle; Qwen2.5-0.5B on CPU for development | The pipeline generates prompts and scores with an API judge. Use its released trait files if present; otherwise our T-0 harness builds the contrast sets |
| Concept vectors (introspection) | Macar et al.; `github.com/safety-research/introspection-mechanisms` | last-token read of the templated "Tell me about {c}" minus the mean of 100 baseline words (read 9 Oct) | Gemma-3-27B (S-1M produces them); Gemma-3-4B / 12B | Already in our pipeline as `--vector-recipe macar-release` |
| Our concept vectors | S-1, S-2 | concept-token, template-tail, sentence-mean | Qwen2.5-3B/7B, Gemma-3-4B/12B/27B | Live/dead labels exist, so they also test the checks |

## Checks to apply (subject to F-10's verdicts)

- **Geometric:** the direction's share in the lowest-variance directions of that layer's
  activations. Use the scale-free definition fixed after the F-6 pilot: massive dimensions
  break the raw "bottom 1% of variance" rule (notebook, 9 Oct).
- **Resampling:** split-half and document-disjoint agreement of the recipe. X-1 says this is
  weak; report it, do not lean on it.
- **Dose:** next-token KL per unit norm against random and on-manifold controls (P2-D's
  finding that real vectors are 3-20x gentler than random ones).
- **Function:** steering effect at a matched KL, and for refusal, the ablation effect.

## Engineering needed (none started)

1. A residual-stream activation collector over arbitrary prompt sets at a chosen position,
   reusing `caliper/activations.py` hooks.
2. Loaders for the two repos' released prompt or trait files, kept outside the repo
   (offensive content); only hashes and counts are committed.
3. The check functions from F-10, imported, not rewritten.

## Risks

- Licence and content: read both repos' licences before copying any data; commit nothing
  offensive.
- The checks are validated on MLP-unit and readout substrates; directions in the residual
  stream are a transfer. F-12 states this, as Paper 1 states its own scope.

## Shortlist (16 October 2026)

Ranked by what can run without new downloads or Kaggle time, and by what the result would mean.
What changed since 9 October: F-0 (`docs/f0-theory.md`) says which check can see which error, and
E-1 (scoring when queue 16 finishes) ranks the checks on exact truth. F-12's check list is taken from
E-1's ranking, not chosen now.

| rank | target | model | cost | why this rank |
|---|---|---|---|---|
| 1 | Refusal direction (Arditi et al. recipe) | Llama-3.2-1B-Instruct (cached, 3.5 GB; the gated licence is already accepted) | CPU, a few hours | The most used direction in the field, and it has a functional test (ablation should remove refusals, judged by Arditi's refusal-phrase match), so the checks can be read against function. Only prompt activations; no harmful completions are generated |
| 2 | Our concept vectors (S-1, S-2, P2-N) | Qwen2.5-3B/7B, Gemma-3, Qwen3.5 (vectors and live/dead labels on disk) | none: the health checks are scored | The control: checks must rank live vectors above dead ones before they are trusted on targets without labels. Paper 2's health table is that calibration (distinctness about 0.6-0.8 across models; logit-lens 0.56-0.62 on Qwen3.5) |
| 3 | Released concept vectors (Macar et al.) | Gemma-3-27B (S-1M vectors on disk; the released tail-read vectors are live at the released strength) | one Kaggle session for the layer's activations (the geometric check); the dose check is done | Same family as rank 2 but someone else's recipe and model size; joins a Kaggle session already planned rather than getting its own |
| 4 | Persona vectors (Chen et al.) | Qwen2.5-7B-Instruct (config only in the cache) | Kaggle, and an API judge in the released pipeline | Dropped unless rank 1 and 3 finish early: the most expensive and the only one that needs generation and a judge |

**Checks, as F-0 constrains them.**
- Difference-of-means recipes are deterministic, so restart agreement does not exist for them;
  resampling (split-half of the prompt sets) is the only agreement check, and Proposition 2 says it
  sees variance, not bias. Report it, do not rank by it.
- The geometric check is the scale-free low-variance share. Add the analogue of Proposition 5: the
  direction's share on massive-activation or near-constant coordinates of that layer, which a
  standardised probe would magnify (the refusal recipe does not standardise, so this is a check on
  the layer, not the recipe).
- Dose (KL per unit norm against random and on-manifold controls) and function (ablation and
  steering at matched KL) as listed above.

**Before any F-12 code.** (1) Read the refusal repo's licence and its data sources' licences; prompt
files stay outside the repo, only hashes and counts are committed. Fetching the repo needs Santosh's
OK. (2) E-1 scored, so the check list is fixed from data. (3) A pre-registration with the
target-by-check table and what each outcome means.

**Earliest start.** A CPU pilot of rank 1 fits in the first idle CPU window after queue 19 (late
October), ahead of the April slot in the timeline; it would move F-12 from "planned" to "piloted" in
the flagship draft.
