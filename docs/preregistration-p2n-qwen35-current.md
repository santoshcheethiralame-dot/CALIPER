# Pre-registration: P2-N, liveness, health checks and dose on a current model family (Qwen3.5)

**Filed 14 October 2026, before any P2-N data.** Paper 2, claims C1 (no health check certifies
liveness) and C4 (dose has no transferable unit). Script: `experiments/kaggle_s3_positive_control.py`
v2026-10-11b, unchanged, run locally on CPU. Scorer: `experiments/analyse_p2n.py`, committed with this
filing; it calls the unchanged S-2 scorer `experiments/analyse_s2.py`.

## Why

Paper 2's general claims about the instrument rest on Qwen2.5-3B/7B and Gemma-3-4B/12B, chosen in
September 2026 and kept for comparability across sessions. Two newer families now exist (Qwen3.5,
Gemma 4). A reviewer may fairly ask whether C1 and C4 hold on a current model. Qwen3.5 is ungated,
small sizes run on the laptop's CPU, and its architecture differs from Qwen2.5 in a way that matters
for an audit of injection: most of its layers use linear attention (gated delta rule), with full
attention every fourth layer.

## Design

**Models.** Qwen3.5-2B (primary) and Qwen3.5-0.8B (secondary), the text decoder only (loaded as
`Qwen3_5ForCausalLM`), unquantised float32 on CPU. The linear-attention layers use the library's
torch implementation (the optional fused kernels are not installed); this changes speed, not the
computation. Smoke test (14 Oct, before filing): logits finite, 24 decoder layers found, first-token
YES+NO mass 0.994 on a probe prompt.

**Settings, S-2's design.** Layer at 0.6 of depth (layer 14 of 24 for both sizes), the 30 concepts,
doses as fractions of the residual norm 0, 0.25, 0.5 and 1.0, the gate at 0.5. Three recipes: concept
token, template tail, sentence mean (the APERTURE recipe). Stages: `steer` (the steering gate, with
the health statistics computed for every vector) and `forced` (first-token P(YES) under the
introspective and matched-neutral framings, for the P(YES)-shift check). S-2's `framing` stage
(generated-text detection) is not run: it is not needed for C1 or C4 and is the costliest stage on a
CPU. One content-free arm: norm-matched random directions, `forced` stage only, for the real-against-
random KL comparison of C4.

**Command form** (2B; the 0.8B runs are identical with its own snapshot and output names):
`python experiments/kaggle_s3_positive_control.py --model qwen3b --model-path <snapshot> --quant none
--compute-dtype fp32 --alpha-frac 0 0.25 0.5 1.0 --stage <steer|forced> [--vector-pos concept |
--vector-pos template-tail | --vector-recipe aperture | --control random] --out
results/p2n/s2_qwen35_2b_<arm>.jsonl`. The `--model` key only selects the script's memory guard; the
loaded model is the `--model-path` snapshot, recorded in each config as `model_id`.

## Liveness label

A vector is live if its gate-dose story is steered and its dose-0 story is not. **Primary label: the
strict gate** (`rescore_steering_gate.py`'s rule: a whole-word match of the concept or an associate that
does not already occur in that concept's dose-0 story), adopted after Paper 2's second internal review
found that the original substring rule made some concepts unscorable. The original substring label is
reported beside it.

## Predictions

**P1, S-2's filed primary, on Qwen3.5-2B, pooled over the three recipes (strict label).** AUC of each
health check at telling live from dead vectors, with stratified bootstrap 95% intervals:
- norm, distinctness and the P(YES) shift near 0.5 (the interval includes 0.5);
- logit steering at least 0.8.
Stability and the probe have no stated prediction. With fewer than 5 live or 5 dead vectors the model
is reported and not scored (S-2's rule).

**P2, C4 on a current family.** The median next-token KL at 0.25 of the residual norm differs between
Qwen3.5-0.8B and Qwen3.5-2B by more than a factor of 2 (as Qwen2.5-3B and -7B differed about 25-fold);
and on Qwen3.5-2B, norm-matched random directions move the next token more than real concept vectors
at 0.5 of the norm (median random KL over median real KL above 1, as on every earlier model).

**Readings.**
- P1 as predicted: C1's pooled pattern replicates on a current family.
- P1 differs: reported per check; Paper 2 states which checks behave differently on Qwen3.5.
- P2 holds: C4's "no transferable unit" extends to a current family.

## Secondary (descriptive)

- The same AUCs on Qwen3.5-0.8B; both labels; live counts per recipe and dose.
- Within one recipe and concrete concepts (20 vectors per cell), the AUC of each check, as in Paper 2's
  within-cell table.
- Logit-lens accessibility (P2-L's check) computed offline from the saved vectors, if the unembedding
  can be read.
- The KL at each fraction for every recipe, beside the Qwen2.5 and Gemma-3 values in Paper 2's dose
  table.

## Exclusions and stopping

All 30 concepts, every recipe, both models; nothing excluded after data. Rows resume by key. Local
queue 15; the 2B model runs first.

## Cost

Measured in the smoke test: about 1 s per forward pass and 15 s per 40 generated tokens for the 2B
model on 12 CPU threads. The steering gate generates 60 tokens for 30 concepts at 4 doses per recipe;
roughly 3 hours for the 2B model and about half that for the 0.8B model, reported from the logs.
