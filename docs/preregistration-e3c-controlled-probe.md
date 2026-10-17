# Pre-registration: E-3c, the standard probe on a property with a known direction, and its repairs

**Filed 17 October 2026, before any E-3c fit.** Paper 1, answering the review's second and third
points. Script: `experiments/e3c_probe_controlled.py`, which fits and scores once.

## Why

E-3 and E-3b show that the scaled probe's direction concentrates on a near-constant coordinate, but the
capitalisation property has no true direction, so they show disagreement, not error. E-3c uses a
property whose direction is known: "MLP readout j is in its top 10%", label 1[w_j . s > tau_j], whose
true direction is w_j (modulo the layer norm's null direction 1/gamma). It also compares the repairs.

## Design

Data as E-3 (MLP input; 8,000 training tokens, corpus seed 0; 4,000 test tokens of unseen seed-1
documents). Layers: flagged GPT-Neo L6, GPT-2 L3, OPT-125m L5, Llama-3.2-1B-Instruct L3, Gemma-3-270m
L13; controls GPT-Neo L10, GPT-2 L6, OPT L6, Llama L10. Ten readouts per layer, `default_rng(20261019)`.
Probe variants (logistic regression, `max_iter=5000`): unscaled; standardised (scikit-learn's rule);
scale floored at 1% of the median; near-constant coordinates dropped, then standardised;
standardised, then near-constant coordinates zeroed; standardised with C = 0.01 and C = 100;
standardised with L1; PCA whitening. Score: |cos| with w_j on the identifiable label (none to remove in
RMSNorm models); accuracy of the direction thresholded at its training 90th percentile.

## Predictions

- **P1.** At every flagged layer, the standardised probe's median |cos| with w_j is at most 0.3.
- **P2.** At every flagged layer, the unscaled probe's is at least 0.7.
- **P3.** At every control, the standardised probe's is at least 0.7.
- **P4.** At every flagged layer, the floor, drop and zero repairs each reach at least 0.7.

Regularisation, L1 and whitening: reported with no prediction.

## Not done

No threshold, layer, unit draw, variant or setting is changed after seeing output.
