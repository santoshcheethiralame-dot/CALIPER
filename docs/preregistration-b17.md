# Pre-registration: B-17, why GPT-Neo layer 6 fails, read unit by unit

**Filed 6 October 2026, before the run.**

## Question

A 4-unit smoke run on GPT-Neo-125m layer 6 (4 Oct, coupled estimator) gave median alignment
0.0026 with 0/4 passing. That is below the random-direction median of about 0.024, and the
same smoke run on GPT-2 layer 6 passed 3/4. Correction 8 (5 Oct) says this needs 20 units with
per-unit diagnostics before it is called a property of the model. B-17 decides between three
explanations:

1. **Pipeline fault.** The captured stimulus and response do not share the ground truth at this
   layer.
2. **Geometry.** The fitted direction differs from w only in directions the stimulus barely
   occupies, so it predicts the response as well as w does. That is a property of the model's
   residual stream, and it limits what any stimulus-response estimator can recover.
3. **Optimisation.** The fit falls short of the response the true direction explains.

## Already ruled out (weights only, no fits, 6 Oct)

The LayerNorm-null ceiling is not the cause. The cosine ceiling from w's unidentifiable 1/γ
component is at least 0.9999 for all 3,072 layer-6 units (median 1.0). Per-layer ground-truth
exactness was checked on 5 Oct (relative error 3e-7).

## Design

```
python experiments/e01_gate.py --model EleutherAI/gpt-neo-125M --layer 6 --d-mlp 3072 \
  --neurons 20 --restarts 2 --independent-units --out results/b17_gptneo125m_l6.jsonl
python experiments/diagnose_units.py --rows results/b17_gptneo125m_l6.jsonl \
  --model EleutherAI/gpt-neo-125M --layer 6 --out results/b17_diagnostics.json
```

The run uses the fixed estimator, the default operating point (8k tokens, 1,600 steps) and
saved directions. The diagnostic was developed on the first 10 rows of B-8b (layer 10). This
is recorded in the notebook, and those rows feed only the diagnostic's development, not B-8b's
pre-registered analysis.

## Reading rules, fixed now

For each unit, using the selected route's direction:
- **Pipeline fault:** `r2_exact` < 0.999. If this holds for **any** unit, B-17 is a pipeline
  fault, and every GPT-Neo layer-6 statement is withdrawn until it is found.
- Otherwise, each failing unit (alignment < 0.95) is classed as:
  - **geometry** if `sigma_align` ≥ 0.99 and `r2_fitted` ≥ `r2_true` − 0.01;
  - **optimisation** if `r2_fitted` < `r2_true` − 0.01;
  - **other** otherwise (reported individually).
- **Verdict on the layer:** the majority class among failures. With fewer than 5 failures,
  the 4-unit smoke run is not replicated, and that is the finding.
- **The below-chance median:** if it recurs (median alignment < 0.05), report `cos_pc1` and
  `sigma_align` for those units. A fitted direction that is functionally equivalent yet
  orthogonal in weight space would be the sharpest case of explanation 2.

## Exploratory, flagged as such

`diagnose_units.py` is also run on every arm with saved directions (B-2c, B-8b, B-11c, B-15).
It gives the class shares (geometry vs optimisation) per arm. These shares are descriptive
only. The pre-registered failure-class split (B-14 Addendum 2) remains the one defined by
held-out R2.
