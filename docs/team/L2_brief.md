# Level 2 brief: the trait level (T-0, T-1, T-2)

**Owner:** Member 2 (proposal, run plan §6). **Plan of record:** `docs/RUN_PLAN_L2_L3.md`
§2. **Gate A:** 31 December 2026.

## The question

Level 1 (Paper 1) found that restart agreement predicts a unit-direction estimator's
failures worse than held-out fit. Level 2 asks the same of **trait vectors**: the
difference-of-means directions behind CAA, persona vectors and steering. When one is wrong,
which ground-truth-free check notices?

## Why a final-layer trait has an exact answer

For a trait expressed as "the model predicts token set A rather than B", the residual
direction the model reads it along at the last layer is exact:

    w = P(g * (mean U_A - mean U_B))

U is the unembedding, g the final norm's gain, and P removes the all-ones direction a
LayerNorm cannot see. This is the trait-level analogue of Level 1's weight column.

`caliper/traits.py` implements it (`readout_direction`). `tests/test_traits.py` checks it
against the model's own logits on GPT-2: max relative error 1.3e-5, correlation
0.9999999999. Load models in float32. `caliper.activations.load_model` now does this by
default, because transformers 5 loads Pythia in its stored float16, and at half precision
the same check only reaches a correlation of 0.993.

## What is built and what is yours

**Done:**
- `readout_direction`, `readout_logit_gap`, the LayerNorm-null projection;
- `diffmean`;
- `planted_fixture` (a known direction in synthetic activations, with an anisotropy dial);
- the `Trait` dataclass.

**Yours (T-0).** Each stub raises `NotImplementedError` and has a strict-xfail test. Remove the
`xfail` mark when it passes.

| function | test |
|---|---|
| `contrastive_prompts` | balanced A/B prompt sets from templates |
| `logistic_probe_direction`, `lda_direction` | recover the planted direction (LDA under anisotropy) |
| `whiten`, `causal_inner_product` | whitening is idempotent |
| `split_half_cosine`, `bootstrap_cosine` | agreement checks run on a 2-trait toy |
| `heldout_probe_accuracy`, `separation_snr` | functional checks run on a 2-trait toy |
| `save_direction` | every run saves its directions |

Two rules come from Level 2's failed pilots (P1, P1b):
1. Every reference is a direction the model provably reads or writes. Random plants are not.
2. Every cosine is reported raw **and** whitened, against a null built in the same space. A
   concept bank with median |cos| 0.42 made the old null sit above the recovery.

## Then

1. **T-1** (about 2 h CPU). Build a fully enumerable template grid, so the DiffMean over every
   combination is the exact estimand. Then subsample 8-256 prompts. Criterion: split-half
   cosine predicts the distance to the exact estimand with Spearman > 0.8. This is the
   **precision** control: agreement checks should work here.
2. **T-2 pilot** (20 traits). Families: sentiment, yes/no, topic vocabulary,
   language/register. Use trait-specific answer tokens, never A/B multiple choice, which
   gives every trait the same direction. Choose the failure bar τ so that 20-80% of fits fail
   at the primary setting. Then file `docs/preregistration-t2-readout-bench.md` **before**
   the full sweep.
3. **T-2 full sweep** (about 200 traits; GPT-2 small and Pythia-160m first, then SmolLM2-360M
   and Qwen2.5-0.5B). Primary endpoint: DeLong, best agreement check minus best functional
   check, clustered by trait.

**Gate A:** the failure rate lies in 20-80% and at least one check's AUC CI excludes 0.5. If
not, Paper 3 becomes an L3-only paper. That is a fine outcome, but decide it on the date, not
later.

## House rules (they apply to everyone)

- Pre-register before every run.
- Develop the analysis on old or synthetic data.
- Write a dated, append-only entry in `docs/LAB_NOTEBOOK.md` in the same session as the run.
- Commits in this repo carry no generated-by or co-author trailers.
