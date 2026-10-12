# Pre-registration: G-0, does a weight-derived gain predict where the layer-norm null direction breaks direction recovery?

**Filed 12 October 2026, before any G-0 run.** Paper 1, section on identifiability.

## Why

Paper 1 explains most converged-wrong fits by the layer norm's null direction u = 1/gamma, and says
the smallest gain of a layer gives the map: a near-zero gain makes a coordinate nearly constant,
and per-coordinate standardisation magnifies the fit's arbitrary component along u. An exploratory
analysis on 12 Oct (`experiments/gain_law.py`) proposes a sharper, weights-only law: a unit's damage
scales with G_j = ||1/gamma|| ||gamma (.) w_hat_j|| / sqrt(d), with an estimator constant calibrated on
the GPT-2 layer-6 arms (G about 1). Calibrated there only, it matched GPT-Neo layer 10 out of sample
(forecast share 0.22 against 0.22, 0.22 and 0.19 observed) but has no skill near G = 1 and its form
was chosen after the arms were seen. GPT-2 layer 0 separates the two rules: its smallest gain (0.045)
is ordinary, like layer 6's (0.042), so the smallest-gain rule calls it safe, while its median G is
4.57 (layer 6: 1.08).

## Design

**Units.** 50 GPT-2 layer-0 MLP units, `numpy.random.default_rng(20261013)` over the layer's units minus
every id in any earlier result file and D-1's and D-3's draws (`experiments/g0_units.py` →
`results/g0_units.txt`, committed with this filing).

**Fit.** B-14's command with `--layer 0` and the unit list: `--restarts 2 --steps 1600 --tokens 8000
--batch 32 --independent-units`, fit seed 0, corpus seed 0, float32, directions saved, as D-1 except
the layer. Output `results/g0_gpt2_l0.jsonl`. Local CPU queue 13 (`experiments/rerun_queue13.sh`),
after queue 12 (D-3, then F-1b); about 2 hours.

## Primary quantity and rival predictions

Among fits that pass the identifiable label (u = 1/gamma of layer 0's `ln_2` removed from fit and
reference, |cos| ≥ 0.95), the share S that fail the filed Euclidean label (|cos| < 0.95). With n the
number of identifiable passes:

- **Gain law:** S = 0.41 for these 50 units (point forecast); band = central 95% of S simulated at n
  by resampling the calibration constant (800 calibration fits) and the units (for example
  [0.23, 0.57] at n = 30, [0.25, 0.58] at n = 40).
- **Smallest-gain rule:** S at the pooled GPT-2 layer-6 rate, 54/707 = 0.076; band = central 95% of
  the binomial at n (for example [0, 0.17] at n = 30 and 40).

## Readings

- Fewer than 20 identifiable passes: underpowered; reported descriptively.
- S inside the gain-law band and outside the smallest-gain band: **gain law confirmed**; Paper 1
  states the gain as a filed, weights-only rule for where the null direction breaks a standardising
  estimator.
- S inside the smallest-gain band and outside the gain-law band: **gain law not replicated**; the gain
  stays exploratory and Paper 1 says it did not transfer.
- Otherwise: **partial**, reported as such.

## Secondary

1. The median calibration constant c√d of the layer-0 fits against the calibration's (1.47): does it
   move, as it did at GPT-2 layer 3?
2. Median restart agreement on the filed failures.
3. Counts: identifiable passes, filed failures, converged-wrong fits on both labels.

## Analysis

`experiments/analyse_g0.py`, committed with this filing; run once when all 50 rows exist. The
forecast code reproduces the exploratory analysis it ports (layer 0 median G 4.55 and forecast 0.40
over all 3,072 units; layer 6, 1.08 and 0.06).

## Not done

No unit, threshold, band rule, calibration set or reading is changed after seeing output.
