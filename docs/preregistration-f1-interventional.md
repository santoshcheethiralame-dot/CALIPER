# Pre-registration: F-1, does interventional data repair converged-wrong fits?

**Filed 9 October 2026, before any F-1 code or data.** Flagship (`docs/flagship-plan.md` §5).
Runs on the laptop CPU after queue 6 (B-17b/c/d).

## Question

Converged-wrong fits differ from the reference w almost only in the stimulus's lowest-variance
directions (Paper 1, identifiability section). Observational data cannot pin a direction down
where the data do not vary; linear causal representation learning proves interventions are
needed for identifiability. F-1 asks whether adding interventional samples, stimuli pushed into
the poorly sampled directions with the unit's exact response, repairs those fits, and whether
it does so because the samples are interventional rather than because there are more of them.

## Units

From two existing runs whose fits, labels and stimuli are known:
- B-8b (GPT-Neo-125M layer 10, `results/b8b_gptneo125m_indep`): all 20 converged-wrong and all
  18 under-fitted units, plus 20 passing units drawn with `numpy.random.default_rng(0)`.
- B-15a (GPT-2 small layer 6, `results/b15a_fitseed1`): all 6 converged-wrong and all 12
  under-fitted units, plus 20 passing units drawn the same way.

96 units in all. Labels are the base runs' own: pass = Euclidean alignment >= 0.95;
converged-wrong = fail with held-out R2 > 0.99; under-fitted = fail otherwise.

## Arms

Every arm re-fits each unit with the base run's settings (`--restarts 2 --independent-units`,
same fit seed, same 8,000-token natural stimulus and held-out split) and adds m = 2,000 samples
to the **fitting** set only. The held-out set stays natural, so held-out R2 is comparable.

1. **Targeted interventions.** s' = s + d for 2,000 stimulus rows s drawn at random from the
   fitting set (seed 0). d = U_low z, where U_low holds the eigenvectors of the stimulus
   covariance C that carry the bottom 1% of its variance, z ~ N(0, I), and d is rescaled to the
   median norm of the centred stimulus. Response: the unit's exact function,
   GELU_erf(w . s' + b), as the pipeline records it.
2. **Random interventions.** As arm 1, with d drawn isotropically over all 768 directions,
   rescaled to the same norm.
3. **More natural tokens.** 2,000 extra tokens from documents disjoint from the base corpus
   (`--exclude-corpus-seed 0`), with responses from the model.

Secondary arm, GPT-2 units only: **residual interventions**, d added to the residual stream
before the layer norm, which then recomputes s'. This is what a practitioner can do.

## Primary endpoint and prediction

On the 26 base converged-wrong units:
- the share that pass (alignment >= 0.95) under arm 1. **Prediction: at least 21 of 26 (>= 80%).**
- arm 1 against arm 3, exact one-sided McNemar on the 26 paired verdicts. **Prediction:
  arm 1 > arm 3, p < 0.05.**

Both must hold for the primary to hold.

## Secondary

1. Arm 2 against arm 1 and arm 3 (McNemar), descriptive: does targeting matter, or does any
   off-manifold variation help?
2. Harm: passing units that fail under arm 1. Prediction: at most 2 of 40.
3. Under-fitted units recovered per arm, descriptive. Under-fitting is a sparsity limit, so no
   arm is predicted to fix it.
4. After augmentation: restart agreement, the low-variance share of the fitted direction and
   held-out R2, and how each check's AUC at predicting the new failures changes.
5. The residual-intervention arm against arm 1 on the GPT-2 units.

## If it fails

- If arm 3 does as well as arm 1, the limit is sample size, not observation. The flagship
  reports "more data" as the remedy, and the interventional claim is dropped.
- If no arm repairs the fits, converged-wrong is not a data problem at this budget. The
  flagship leads with detection (low-variance share, cross-sample agreement), and F-1 is
  reported as a tested, negative remedy. This is Gate F-A (15 Dec).

## Not done

- No change to m, the 1% subspace, the perturbation scale or the unit draw after data arrive.
- Code is written and tested on two units of B-15a before the run; those two fits are
  discarded and re-run in the queue.
- No unit is dropped. A unit whose re-fit fails to run is reported as missing.

## Clarification (9 October 2026, before any F-1 data; the pilot fits are discarded)

- "The base run's settings" governs. B-8b: `--restarts 2`, fit seed 0. **B-15a: `--restarts 5
  --fit-seed 1`** (the parenthetical above wrongly gave restarts 2 for both).
- Unit lists, drawn as specified and fixed now: `results/f1_units_b8b.json` (20 converged-wrong,
  18 under-fitted, 20 passing) and `results/f1_units_b15a.json` (6, 12, 20). The comma lists the
  queue reads are `results/f1_units_*.txt`.
- Implementation: `experiments/e01_gate.py --augment {targeted,random,natural} --augment-n 2000`.
  The held-out rows stay the first 1,600 of the natural stimulus (test_frac is rescaled so the
  split does not move). The exact response is the model's own MLP input layer followed by the
  erf GELU; on 500 natural tokens it reproduces the recorded responses to 2.4e-7.
- Order: queue 7 (`experiments/rerun_queue7.sh`), after queue 6. GPT-2 arms first.
- The residual-intervention arm (secondary 5) needs the pre-norm residual, which the pipeline
  does not capture; it is built after the primary arms run, before its own data exist.

## Addendum 1 (9 October 2026, before any F-1 row exists): the layer-norm null direction

The internal review (round 2) found that most converged-wrong fits differ from w along one exactly
unidentifiable direction: 1/gamma of the layer norm feeding the MLP, along which s . (1/gamma) is
constant on every real token. The targeted arm perturbs inside the bottom-1% subspace, which
contains 1/gamma, so its synthetic samples can carry information the natural distribution never
can. To keep F-1 from crediting that as a repair:

- **Added secondary (no prediction):** every verdict is also scored with 1/gamma projected out of
  the fit and of w (the identifiable target), for every arm, beside the filed primary.
- **Added descriptive:** the share of each fit on 1/gamma before and after augmentation, per arm.
- The primary endpoint, prediction and arms are unchanged.
