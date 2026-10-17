# Pre-registration: E-3b, the standard probe on two current RMSNorm models

**Filed 17 October 2026, before any E-3b fit.** Paper 1. Script: `experiments/e3_probe.py --set e3b`,
E-3's code unchanged apart from the layer set, the predictions and the output file
(`results/e3b_probe.json`).

## Why

E-3 showed that `StandardScaler` + `LogisticRegression` puts 95-99.8% of its direction on the
near-constant coordinate in three small LayerNorm models. The weights-only map
(`results/layer_map_weights.json`, validated on 70 of 70 layers) says the exposure is widest in current
RMSNorm families. E-3b runs the same probe in two of them, and tests one further prediction: a
coordinate that is exactly constant has zero variance, which sklearn's scaler maps to scale 1, so no
magnification and no collapse should occur there (our own estimators, which add a small epsilon to the
scale, would not be immune).

Stimulus maps used to choose the layers (from the activations alone, no probe fitted):
- Llama-3.2-1B-Instruct: layer 3 has near-constant MLP-input coordinates (min SD / median 4e-4);
  layer 10 has none (0.86).
- Gemma-3-270m: layer 13 has one near-constant, non-zero coordinate (163, SD 7.2e-3 of the median) and
  no exactly constant one; layer 5 has two exactly constant coordinates (SD exactly 0, gain exactly 0)
  and no other near-constant one.

## Design

As E-3: capital-letter property, the output of the norm feeding the MLP, 8,000 training tokens of
corpus seed 0, 4,000 test tokens of unseen corpus-seed-1 documents, sklearn defaults with
`max_iter=5000`, share on the coordinates with SD below 1% of the median.

## Predictions

- **P1.** Llama-3.2-1B L3: share > 0.5 and |cos| with the unscaled probe < 0.3.
- **P2.** Gemma-3-270m L13: share > 0.5.
- **P3.** Gemma-3-270m L5 (exactly constant coordinates only): share <= 0.05.
- **P4.** Llama-3.2-1B L10 (control): share <= 0.05 and |cos| with the unscaled probe > 0.7.
- **P5.** At Llama L3 and Gemma L13, the scaled probe's test accuracy is within 0.01 of the unscaled one.

## Readings

- **P1, P2 hold:** the failure reaches the standard probe in current models; Paper 1 says so.
- **P3 holds:** the failure needs a coordinate that is nearly, not exactly, constant for sklearn's
  scaler; implementations that add epsilon to the scale are exposed to both. Paper 1's remedy section
  states the difference.
- **Any fails:** reported as such.

## Not done

No threshold, layer, property or setting is changed after seeing output.
