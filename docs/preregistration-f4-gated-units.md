# Pre-registration: F-4, gated (SwiGLU) units in Qwen2.5-0.5B

**Filed 10 October 2026, before any F-4 main-run data.** Flagship substrate F-4
(`docs/flagship-plan.md` §5, §13). Pilot (discarded, budgets only): `results/flagship_pilot1/`,
`results/flagship_pilot2/`. Scorer: `experiments/analyse_substrate.py`, committed with this filing.
Run sheet: `kaggle/NEXT_SESSION_F4F6.md`.

## Substrate

Qwen2.5-0.5B, layer 12 (d_model 896, d_mlp 4864). A gated unit's response is
SiLU(g . s) x (u . s), with s the RMSNorm output; its reference is the plane span(g, u), read from
the weights (`e01_gate.py --target glu`). The estimator fits a k = 2 subspace; alignment is the
mean cosine of the principal angles between the fitted plane and the reference plane. Qwen uses
RMSNorm, which subtracts no mean, so there is no layer-norm null direction: the Euclidean label is
the primary label.

## Units, budget and settings

100 units drawn with `default_rng(0)` (the script's own draw). 8,000 tokens (the pilot put 8k at
9/16 failures, the middle of the 20-80% band; 16k at 6/16), 3,200 steps, 2 restarts, independent
units, as in the pilot.
- **Fit A:** corpus seed 0.
- **Fit B:** corpus seed 1 with `--exclude-corpus-seed 0`: the same units on disjoint documents.

## Primary endpoint and prediction

On fit A: AUC(restart agreement) minus AUC(held-out R2) for failure (alignment < 0.95), DeLong and
stratified bootstrap. **Prediction: negative**, held-out fit predicting failure better, as on every
GELU substrate in Paper 1.

## Secondary

1. **Gate F-B:** the failure rate with a Wilson interval, and whether it lies in the 20-80% band.
2. **The calibration table's contrast (F-10):** AUC(cross-sample agreement between fits A and B)
   minus AUC(restart agreement), and minus AUC(held-out R2).
3. The class split (converged-wrong: held-out R2 > 0.99).
4. Restart agreement on k = 2 planes was low in the pilot (0.42-0.52), so its distribution is
   reported.

## Not done

The budget, units and label are fixed now. If fit A's failure rate falls outside 20-80%, it is
reported as such; there is no second budget.
