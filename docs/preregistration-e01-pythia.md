# Pre-registration — E0.1 gate on a second model family (Pythia-160m)

**Filed 8 September 2026, before the run. Nothing below changes after seeing results.**

## Why

C13 scored a standard rank-1 subspace estimator against free ground truth on 100 GPT-2
small units and found 77/100 recovered, Wilson 95% [0.679, 0.842], failing its own
pre-registered ≥0.90 bar. Outline v3 makes that the paper's lead result, which makes
"one model family" the obvious reviewer objection.

This run repeats the gate on a different family.

## Why Pythia-160m specifically

- **Different family and training data.** GPT-NeoX architecture, trained on the Pile.
  GPT-2 is a different architecture trained on WebText. Nothing shared but the idea.
- **Identical dimensions.** d_model 768, d_mlp 3072, 12 layers — the same shape as GPT-2
  small. So a difference in recovery rate cannot be attributed to width, depth, or the
  number of parameters per unit. That is a stronger comparison than a larger model would
  give.
- **The identity holds** (verified 8 Sep, before this filing): the MLP reads
  `post_attention_layernorm` directly under NeoX's parallel residual, and
  `s·W + b` reproduces the pre-activation with correlation **1.0000000000**, relative
  error 1.7e-04 mean / 5.3e-04 max. The absolute error is larger than GPT-2's 3.3e-06
  purely because of float32 accumulation order — float64 halves it and the correlation is
  unchanged.

## Protocol (frozen)

| element | value |
|---|---|
| model | `EleutherAI/pythia-160m` |
| layer | **6 of 12** — the same relative depth as GPT-2 small layer 6 of 12 |
| units | 100, drawn by `np.random.default_rng(0).choice(3072, 100, replace=False)` — the same draw procedure as C13 |
| corpus | `sample_corpus(n_docs=300, seed=0)`, **8,000 tokens**, `skip_first=1` |
| estimator | dual protocol: direct k=1 and `fit_cascade`, selected by held-out R², **2 restarts, 1600 steps, batch 32** — identical to C13 |
| pass | alignment > 0.95 **and** k2_gain < 0.01, identical to C13 |

Everything is held to C13 except the model. That is the point.

**Config corrected before filing.** An earlier draft of this table said 20,000 tokens,
3 restarts and 2500 steps. Checking `results/e01_gate.log` shows C13 actually ran at the
script defaults - `stimulus (8000, 768)`, 1600 steps, 2 restarts - at 63.7 s per neuron.
The 20k/2500/3 figure is from the separate throughput measurement in `plan.md` section 1
and was conflated with the gate config in the lab notebook section 2.1, which is
corrected in the same commit as this filing. **Matching C13 exactly is the whole value of
this run, so the numbers here are the log's, not the notebook's.**

## Primary endpoint

Pass rate with a 95% Wilson interval.

**Criterion, fixed now: the lower bound exceeds 0.90** — the same bar C13 was scored
against and failed.

I already know GPT-2's answer, so this is not a blind prediction. What the pre-registration
fixes is the criterion, the protocol, and what each outcome means, none of which can be
adjusted afterwards.

- **FAIL, at a rate close to GPT-2's 77%.** The silent-failure result generalises across
  model families. This is the strongest outcome for the paper and the one the abstract
  would lead with.
- **FAIL, at a materially different rate.** The failure is real but its magnitude is
  model-dependent. Report both rates and drop any claim about "roughly a quarter" in
  favour of a range.
- **PASS.** The failure is specific to GPT-2 small. **This substantially weakens the
  paper's lead result** and the honest framing becomes "an estimator that is well behaved
  on one model and silently wrong on another, with no way to tell in advance which you
  have" — still publishable, materially less strong, and reported as such.

## Secondary endpoints

1. **Does the disagreement flag transfer?** AUC of the two-route disagreement statistic
   against failure, compared to GPT-2's 0.915. The flag is the paper's deliverable, so a
   flag that works on one family only is a serious limitation.
2. **Does the pre-fit z_mean predictor transfer?** GPT-2 gave AUC 0.877. C7's account was
   that failures sit deep in GELU's non-monotone region; Pythia uses the same activation,
   so the account predicts it should transfer.
3. **Median alignment across all units**, testing whether failures are equally invisible
   in aggregate (GPT-2: 0.993).
4. **Restart agreement on the failures** (GPT-2: 0.85–0.99), testing whether they are
   equally confidently wrong.

## Committed in advance

- Both rates are reported side by side whichever way this falls, and the GPT-2 77% is not
  retired or re-described.
- No change to layer, unit count, estimator config, or pass criterion after seeing any
  part of the output. A run that dies partway resumes on the same frozen settings.
- If the flag's AUC drops materially, that is reported in the abstract, not buried in
  limitations — it is the deliverable.
- Layer 6 was chosen before running as the depth-matched analogue of C13. No layer sweep
  will be run and then reported selectively; if a sweep happens later it is labelled
  exploratory.
