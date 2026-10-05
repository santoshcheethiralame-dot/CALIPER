# Pre-registration — B-14, the primary endpoint re-run on independent units

**Filed 5 October 2026, before the run.** Governs whether B-1b's headline survives the
estimator fix.

## Why this run exists

B-1b (9 Sep, n=300, GPT-2 L6) is the bench's headline: held-out R2 predicts silent
failure at AUC 0.942, restart agreement at 0.778, DeLong p = 1.8e-07 (recomputed 5 Oct).

On 5 Oct two couplings were found in `fit_batch`, and B-1b ran with both:

1. **Shared early stopping.** The patience counter reset whenever any unit in the batch
   improved, so a unit kept training until its slowest batch-mate stopped. Measured on a
   synthetic set: 3,200 steps in a batch of 16 against as few as 952 alone.
2. **Initialisation keyed on batch position.** The default stacked draw ties a unit's start
   to the batch size, and `--per-neuron-seed` ties it to the position within the batch.
   Neither ties it to the neuron.

Together they move 9-10 of 50 verdicts between batch sizes (B-12). The fix is
`--independent-units`: seeding by neuron id and a per-unit stopping counter. It is pinned
by `tests/test_batched.py::test_independent_units_fit_alone_equals_fit_in_batch`: a unit
fitted alone matches the same unit inside a shuffled batch at agreement 1.000000, against
0.9728 under the old rule. The direct route is the only thing that changes. The cascade
route (`fit_cascade`) was always fitted one unit at a time.

## Disclosure, made before the run

I know B-1b's numbers and the B-12 trend: units passed more often in larger batches (40 /
36 / 37 of 50 at batch 32 / 8 / 1), which suggests the shared rule trained units longer.
**Expectation, recorded and not a criterion:** the failure count rises somewhat, and
held-out R2 stays ahead of restart agreement, by a smaller margin.

## Design

Identical to B-1b except for the one flag.

```
python experiments/e01_gate.py --neurons 300 --restarts 2 --steps 1600 --tokens 8000 \
    --batch 32 --layer 6 --independent-units --out results/b14_primary_gpt2_indep.jsonl
```

- Model GPT-2 small, layer 6, same 300 units (legacy draw `choice(3072, 300)`, seed 0;
  `--neuron-pool` is left off so the draw is byte-identical to B-1b's).
- Same corpus, same 8,000 tokens (`collect`, seed 0), same 2 restarts, 1,600 steps.
- Local CPU, the same machine and thread count as B-1b. Nothing here is pooled with a GPU arm.
- Failure = `align_selected < 0.95`, the label B-1b's AUCs used.

## Primary endpoint

DeLong test on correlated ROC curves, restart agreement minus held-out R2, same units.
The same endpoint as B-1b, so the two are directly comparable.

## Decision rule, fixed now

| outcome | reading |
|---|---|
| diff <= -0.10 and p < 0.05 | **Headline stands on the fixed estimator.** The paper reports B-14, and B-1b goes to the appendix as the run that found the coupling |
| diff < 0 and p < 0.05, but diff > -0.10 | Ranking stands, gap is smaller than B-1b claimed. Report both runs and quote B-14's effect size |
| p >= 0.05 | **B-1b's significance was partly a stopping artefact.** Report it that way. The recommendation becomes "R2 is no worse and is free", not "R2 is significantly better" |
| diff > 0, p < 0.05 | Reversal. Stop drafting and work out why |

If failures fall below 36, the power figure filed in addendum 1 no longer holds. The run is
reported as underpowered, not extended after the fact.

## Secondary, all pre-specified and descriptive

1. **Verdict agreement with B-1b** on the 300 shared units, with an exact McNemar test on
   the discordant pairs. This is the size of the coupling on the headline arm.
2. **Failure classification.** Each failure is labelled *wrong basin* if its held-out R2
   exceeds 0.99, the convergence line B-11's steps check used, and *under-fitted*
   otherwise. Both counts are reported, and B-1b's failures are classified the same way.
   This is what answers "is the silent-failure rate an optimiser artefact?" A failure
   class that is mostly under-fitted is not silent failure in the paper's sense.
3. **Paired bootstrap** on the AUC difference (2,000 resamples of units, stratified by
   failure), reported beside DeLong. It is not a second primary test.
4. **AUCs of disagreement and r2_spread**, as in B-1b, with Benjamini-Hochberg across the
   secondary family.
5. **Seconds per unit**, since per-unit stopping should make the run cheaper.

## What will not change after seeing output

The unit draw, the token sample, the bar (0.95), the failure label, the primary endpoint
and the decision table. Any extra analysis is labelled exploratory in the notebook.
