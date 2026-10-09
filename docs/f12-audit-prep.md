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
