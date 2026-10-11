# Pre-registration: D-1, the primary comparison on fresh GPT-2 units, identifiable label

**Filed 11 October 2026, before any D-1 run.** Paper 1, internal review round 3, item D1.

## Why

The filed primary (B-14) re-fits B-1's 300 units after a bug fix, so its confirmatory evidence
is on units already seen. On units never fitted before, the gap between the checks was smaller:
X-1a −0.035 [−0.116, +0.049] and X-1b −0.097 on GPT-Neo layer 10, and −0.068 on B-1's own draw.
Separately, the identifiable label (B-14 Addendum 1, 5 Oct: 1/γ of `ln_2` removed from fit and
reference) has been scored only on re-fits of seen GPT-2 units (B-15a/b/c) and on the B-14r
replicate. D-1 asks whether held-out R²'s lead over restart agreement holds on fresh GPT-2 units,
scored on the identifiable label.

## Design

**Units.** 200 GPT-2 layer-6 MLP units never fitted before: drawn with
`numpy.random.default_rng(20261011)` from the layer's 3,072 units minus every unit id appearing
in any earlier result file (858 ids from 64 files; ids from other models are excluded too, which
only shrinks the pool). `experiments/d1_units.py` → `results/d1_units.txt` and
`results/d1_units.json`, committed with this filing.

**Fit.** The filed B-14 command, unchanged except the unit list and `--out`:
`--restarts 2 --steps 1600 --tokens 8000 --batch 32 --layer 6 --independent-units --units <list>`;
fit seed 0, corpus seed 0, float32; directions saved. Output `results/d1_gpt2_l6_fresh.jsonl`.

**Queue.** Local CPU queue 10 (`experiments/rerun_queue10.sh`), which starts when queue 9 (F-1b)
writes "queue 9 done". About 12 hours of CPU.

## Definitions

- **Identifiable label.** A fit fails when |cos| between the fit and w, both with u = 1/γ removed,
  is below 0.95. Converged-wrong: failing with held-out R² > 0.99; under-fitted otherwise.
- **Euclidean label.** The filed label: |cos(v̂, w)| < 0.95, as in B-14.
- **Checks.** Restart agreement (`stability`) and held-out R² (`r2_k1`) of the selected fit, as
  stored, scored exactly as in B-14.

## Prediction

**P1 (primary).** On the identifiable label, over all failures vs passes, AUC(restart agreement)
minus AUC(held-out R²) is below 0, and the stratified bootstrap 95% interval (2,000 resamples)
lies below 0. DeLong p and the paired permutation p are reported beside it.

**Readings.**
- Fewer than 10 identifiable failures: P1 is reported as underpowered and not extended.
- **Holds** (interval below 0): the lead replicates on fresh units; Paper 1 states this beside
  the primary in the abstract.
- **Not replicated** (interval spans 0): Paper 1 states that the lead is established on re-fits
  of seen units only, and the abstract says so.
- **Reversed** (interval above 0): the abstract's general claim is withdrawn.

## Secondary, reported as stated

1. The same comparison on the Euclidean label (the filed primary's label).
2. Converged-wrong vs passing units, both labels, when at least 2 converged-wrong fits exist.
3. The direct route scored against its own held-out R² (best over restarts), as in round-3 B1.
4. Class counts (pass, converged-wrong, under-fitted) on both labels.

## Analysis

`experiments/analyse_d1.py`, committed with this filing, developed on B-15a before any D-1 row
existed (`results/d1_dev.json`): it reproduces B-15a's row of the null-direction table
(identifiable −0.175 [−0.325, −0.065]; Euclidean −0.169). It is run once, on all 200 rows.

## Not done

- No unit, threshold, label, class boundary or reading is changed after seeing output.
- The primary label is not switched: if P1 and secondary 1 disagree, both are reported, with
  P1 as primary.
- D-1 does not replace B-14 as Paper 1's filed primary; it is a fresh-unit replication.
