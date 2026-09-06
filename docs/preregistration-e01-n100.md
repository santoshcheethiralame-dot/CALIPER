# Pre-registration — E0.1 at n = 100

**Filed 19 August 2026, before the run. Nothing below may be changed after seeing results.**

Filed because this project has produced four wrong conclusions in two days — a response
transform that halved the pass rate, two phantom findings from a mis-specified binning
metric, and an operating point chosen backwards. Every one was caught by the analytic
control. A criterion chosen after seeing the data would not be.

## Question

Does the estimator, under the protocol fixed below, recover the analytically known direction
for language-model MLP neurons?

## Protocol (frozen)

- Model GPT-2, layer 6. Stimulus: post-`ln_2` residual stream. Response: post-GELU activation.
- 100 neurons drawn uniformly at random from the 3,072 in the layer, seed 0.
- Liveness screen: response std > 1e-4. Screened-out units are reported, not silently dropped.
- **8,000 tokens, 1,600 steps, 2 restarts** — the E0.3b operating point, chosen on worst-case
  recovery rather than median.
- **Dual protocol:** fit direct and cascade, select by held-out R².
- Raw stimulus space. No PCA truncation, no whitening, no response transform.

## Primary criterion

Per neuron, a **pass** is `|v₁ · ŵ| > 0.95` **and** `k=2 gain < 0.01`.

> **The run passes if the lower bound of the 95% Wilson interval on the pass rate exceeds
> 0.90.**

At n = 100 that needs ~95 passes. A bare fraction over 20 neurons cannot resolve a 0.95
threshold — one failure is exactly 0.95, two is 0.90 — which is why the earlier gate was
uninformative regardless of outcome.

## Secondary quantities, reported regardless

Median and minimum alignment · median k=2 gain · fraction where direct and cascade disagree by
more than 0.05 alignment (the reliability flag) · regret against the oracle choice ·
per-neuron effective N · wall-clock per neuron.

## Stated in advance

1. **Expected outcome.** ~95% pass. E0.1i gave 7/8 under selection with an oracle of 8/8.
2. **If it fails**, the residual failure rate is a property of the method, not a bug, and
   becomes a headline result. It does **not** license further post-hoc estimator changes
   followed by re-running this gate; a changed estimator requires a fresh pre-registration.
3. **The selection rule is known to be imperfect.** n1989 showed held-out R² preferring a less
   accurate direction by a decisive margin. Units where the two methods disagree are reported
   as lower-confidence rather than excluded.
4. **No neuron is dropped after the fact** for any reason other than the liveness screen
   declared above.
