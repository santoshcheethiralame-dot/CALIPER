# Pre-registration: E-3, does a standard linear probe fail the same way?

**Filed 17 October 2026, before any E-3 fit.** Paper 1. Script: `experiments/e3_probe.py`, which fits
and scores once; it was checked only for syntax and imports before filing.

## Why

The standardisation failure is now shown for two direction estimators on MLP units, in three model
families (B-17/B-17c, F-3a, E-2b). Most practitioners never fit MLP units; they train linear probes, and
a common pipeline standardises each coordinate first (sklearn `StandardScaler` before
`LogisticRegression`). Proposition 5 applies to any fit made in standardised coordinates and mapped
back. E-3 asks whether that everyday pipeline produces a probe direction that collapses onto the
near-constant coordinate while its accuracy and cross-validation agreement look perfect.

No ground-truth direction exists for a probed property. The references are the same probe fitted
without the scaler and the difference of class means, neither of which magnifies any coordinate.

## Design

- **Property:** the current token starts with a capital letter (decoded token, stripped).
- **Stimulus:** the output of the norm feeding the layer's MLP (the same stimulus as Paper 1).
- **Data:** 8,000 tokens of corpus seed 0 for training; 4,000 tokens of unseen corpus-seed-1
  documents for testing; position 0 of each 128-token window dropped.
- **Probes:** `make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))`, sklearn defaults
  otherwise; the same without the scaler; the difference of class means. Directions in the stimulus'
  own coordinates (coefficients divided by the scaler's scale).
- **Layers:** flagged by the layer map: GPT-Neo-125m L6, GPT-2 L3, OPT-125m L5; controls: GPT-Neo L10,
  GPT-2 L6, OPT-125m L6.
- **Share:** the standardised direction's squared norm on the layer's near-constant coordinates (SD
  below 1% of the median over the training tokens), or on its lowest-SD coordinate where there is none.
- **Agreement:** median pairwise |cos| of the standardised directions over 5 folds (KFold, shuffled,
  seed 0), the practitioner's stability check.

## Predictions

- **P1.** At every flagged layer, the standardised probe's test accuracy is within 0.01 of the
  unstandardised probe's.
- **P2.** The standardised direction has share > 0.5 at two or more of the three flagged layers, and
  share <= 0.05 at all three controls.
- **P3.** At every flagged layer where it collapses, its 5-fold agreement is at least 0.95.
- **P4.** At those layers, |cos| between the standardised and unstandardised directions is below 0.3;
  at all controls it is above 0.7.

## Readings

- **P2 to P4 hold:** a standard probing pipeline reports a direction that is almost entirely one
  meaningless coordinate, with accuracy and cross-validation that give no warning. Paper 1 states the
  failure mode as a property of the preprocessing, with this as its practitioner-facing case.
- **P2 fails:** the logistic loss with L2 regularisation does not put weight on the near-constant
  coordinate, and the failure is confined to the direction estimators studied; reported as such.
- **P1 fails:** the collapse also costs accuracy, so accuracy does warn; reported.

## Not done

No threshold, property, layer or setting is changed after seeing output.
