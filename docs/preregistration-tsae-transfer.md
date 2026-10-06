# Pre-registration: T-SAE, does the calibration transfer to SAE latents?

**Filed 6 October 2026, before the run.** Paper 1's transfer experiment, answering the
objection that the reference standard covers only MLP neurons reading their own layer.

## The reference standard

A ReLU sparse-autoencoder latent's activation is f_j = ReLU((s - b_dec) . W_enc[:, j] +
b_enc[j]), so its pre-activation reads exactly one direction of the residual stream, the
encoder column. This is the same kind of object as an MLP neuron's input weight column,
but for a feature of the residual stream rather than of an MLP's input.

- **SAE:** jbloom's GPT-2 small residual SAEs (`jbloom/GPT2-Small-SAEs-Reformatted`), block 6
  `hook_resid_pre`, 24,576 latents, ReLU, cached locally (`caliper/sae.py`).
- **Stimulus:** the residual stream entering block 6, **mean-centred per token**. Checked on our
  corpus before filing: the SAE's reconstruction explains 0.83 of the variance on centred
  inputs and 0.54 on raw ones, so the SAE was trained on the centred stream (TransformerLens
  centres writing weights).
- **Reference:** the encoder column with its mean removed. The centred stimulus has no component
  along the all-ones direction, so that part of the column is not identifiable.
- **Verification:** `tests/test_sae_target.py` checks that the projection reproduces the
  pre-activation at correlation > 0.999999.

## Design

`e01_gate.py --target sae --layer 6 --neurons N --restarts 2 --independent-units` (8,000
tokens, 1,600 steps, batch 32, fit, corpus and split seeds 0). Directions are saved and
route agreement is recorded.

- **Latent draw:** `caliper.sae.draw_latents`. Latents firing on at least 10^-3 of tokens are
  eligible (about 8+ firings in 8,000 tokens), split into four equal-count density quartiles,
  N/4 from each, then shuffled once with seed 0.
- **Density** is recorded on every row and is the difficulty dial.
- **N = 200**, fixed now. At the GPT-2 MLP rate (~200 s/unit) that is about 11 CPU-hours. If the
  2-latent smoke timing shows more than 400 s/latent, N drops to 120, recorded as a deviation
  before the run starts.
- **Link function:** this changes from GELU to ReLU. That is part of what transfers or doesn't,
  and the paper says so.

## Endpoints, the same as B-14 (analysis: `analyse_b14.py --no-model`)

1. **Primary:** DeLong, restart agreement minus held-out R2 at predicting failure
   (alignment < 0.95), with the B-14 decision table read for this substrate:
   - diff <= -0.10, p < 0.05: the R2-over-restart ordering transfers;
   - diff < 0, p < 0.05, diff > -0.10: transfers with a smaller gap;
   - p >= 0.05: does not transfer at this power;
   - diff > 0, p < 0.05: reverses.
   If there are fewer than 20 failures, the result is reported as underpowered and not extended.
2. **Failure classes and AUCs by class** (Addendum 2): wrong basin (held-out R2 > 0.99) vs
   under-fitted, at cutoffs 0.98, 0.99 and 0.995.
3. **Route agreement AUC** (ground-truth-free).
4. **Pass rate by density quartile.**
5. **Threshold sweep, PR-AUC, calibration, permutation and ground-truth-signal controls**, as for
   B-14.

## Prediction, recorded and not a criterion

Single-index theory: ReLU is monotone, unlike GELU's negative lobe where the failing MLP units
sit. So we expect fewer wrong-basin failures than GPT-2 MLP units (16/300). Most failures should
be under-fitting in the sparsest quartile. If held-out R2's advantage is again confined to the
under-fitted class, that is evidence the B-14 class split is general rather than GELU-specific.

## What will not change after seeing output

The SAE, layer, centring, eligibility rule, N (subject only to the timing rule above, decided
before the run), the bar, the endpoints and the decision table.
