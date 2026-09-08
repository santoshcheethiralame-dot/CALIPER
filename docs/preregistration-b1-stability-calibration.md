# Pre-registration — B-1 / B-2, calibrating restart agreement against free ground truth

**Filed 8 September 2026, before the run.** Governs the bench's headline result.

## The question

Restart agreement — refit from several random initialisations, check the answers
agree — is the default reliability check in this literature. **Does it predict
whether the recovered direction is correct?**

Nobody knows, because answering it needs a known-correct direction. We have one for
free: an MLP neuron's input weight column is exactly the direction its pre-activation
reads. So this is answerable here and, as far as the 8 September scout found, nowhere
else.

## Disclosure, made before the run

A pilot exists on disk and I have looked at it. `results/s1_seed_lottery.json`,
8 units at 10 seeds each, run earlier for a different purpose (the seed lottery,
C43). Reading it as a detector:

| group | unit | median alignment | spread over 10 seeds | verdict at the 0.95 bar |
|---|---|---|---|---|
| worst-fail | 527 | 0.9163 | 0.0474 | FAIL, and stable |
| worst-fail | 1503 | 0.3251 | 0.9169 | FAIL, and unstable |
| worst-fail | 2723 | 0.9668 | 0.0377 | pass |
| worst-fail | 2023 | 0.7343 | 0.3309 | FAIL, and unstable |
| worst-fail | 1180 | 0.9883 | 0.0215 | pass |
| worst-fail | 1625 | 0.9131 | 0.0658 | FAIL, and stable |
| pass | 3066 | 0.9999 | 0.0003 | pass |
| pass | 2053 | 1.0000 | 0.0001 | pass |

Two of four failures are stable to within 0.05 while being wrong; passing units are
stable to within 0.0003. **My expectation, recorded now: restart agreement will
separate the severe failures and miss the subtle ones — moderate AUC, poor operating
point at high specificity.** That expectation is not a criterion, and the criteria
below were chosen so that any of the three outcomes is reportable.

n = 8 across two hand-picked groups is not a sample. It does not license a claim and
it is why the confirmatory run exists.

## Protocol (frozen)

| element | value |
|---|---|
| script | `experiments/e01_gate.py`, unchanged except that per-fit stability is now recorded |
| substrate B-1 | GPT-2 small, layer 6, the **same 100 units** as C13/C47 (`rng.default_rng(0).choice(3072, 100)`) |
| substrate B-2 | Pythia-160m, layer 6, the same 100 units as C53 |
| method | fitted rank-1 bottleneck, dual protocol (direct + cascade), selection by held-out R2 |
| operating point | 8000 tokens, 1600 steps — the E0.3b point, unchanged from C13 |
| restarts | **5** (C13 ran 2; 2 restarts gives one pairwise comparison, which is too thin a stability estimate to calibrate) |
| ground truth | `w / ||w||`, the neuron's input weight column |
| failure | `abs(subspace_alignment(selected, w)) <= 0.95` — the C13 bar, unchanged |

## Signals scored, all computed without the ground truth

1. **`stability`** — median pairwise subspace alignment across the 5 restarts.
   `Fit.stability`, already implemented at `caliper/estimator.py:29`.
2. **`r2_spread`** — max minus min held-out R2 across restarts. The cheaper
   stability variant a practitioner would actually reach for.
3. **held-out R2** — incumbent, AUC 0.906 / 0.995.
4. **two-route disagreement** — incumbent, AUC 0.802 / 0.975.
5. **k2 gain** — incumbent, AUC 0.367, carried as a negative control. A signal
   registry that cannot show a signal failing is not measuring anything.

## Primary endpoint

**AUC of `stability` for predicting failure, on B-1's 100 units, against the AUC of
held-out R2 on the same 100 units.** Reported with a DeLong or bootstrap interval on
the difference, not as two bare numbers.

**Criteria, fixed now:**

- **`stability` AUC < 0.70 and significantly below held-out R2.** The field's default
  check does not discriminate failures on this substrate. This is the headline: a
  widely used check, calibrated for the first time, found wanting, with a
  better free alternative already in hand.
- **`stability` AUC within the interval of held-out R2.** The two are equivalent
  detectors. The result becomes cost-benefit — restart agreement costs 5 fits,
  held-out R2 costs zero — and is reported that way, without the corrective framing.
- **`stability` AUC significantly above held-out R2.** The incumbent recommendation
  is wrong and the deliverable changes: the bench recommends restart agreement and
  says so. Finding 2 in the notebook is amended.

All three are publishable. None of them is a failed run.

## Secondary

1. The same five signals on B-2 (Pythia), for transfer. The caveat from C53 travels
   with it: Pythia's failures are more severe (median 0.794 vs 0.889, min 0.076 vs
   0.145), and extreme failures are easier to separate, so a gain there is partly
   about what the detector was asked to detect.
2. **Operating characteristics, not just AUC.** For each signal, at the threshold
   catching 73% of failures, how many good units are discarded. This is the number a
   practitioner needs and the reason C35 preferred held-out R2 despite both AUCs
   being high.
3. Does `stability` add anything **on top of** held-out R2? Logistic on both,
   5-fold CV, compared against R2 alone by cross-validated AUC. A signal that is
   merely redundant is a different finding from a signal that is uninformative, and
   the bench should distinguish them.
4. Whether the 5-restart `stability` and the cheap 2-restart version agree. If they
   do, the bench recommends the cheap one.

## Committed in advance

- The 100 units are the C13/C47 set and are not reselected. The alignments will
  differ from C13 because 5 restarts is a better optimiser than 2; **the pass rate
  is therefore expected to move, and a changed pass rate is not a finding here.**
  The gate result stands at its own filed configuration.
- Every AUC is reported beside its interval and beside the incumbents on the same
  units. An AUC without a comparison is not a result.
- k2 gain is reported even though it is expected to fail.
- If `stability` and `r2_spread` disagree sharply, both are reported and neither is
  dropped for being inconvenient.
- No change to the bar, the units, the operating point, or the criteria after seeing
  output.
- The pilot table above is reproduced in the paper wherever the confirmatory result
  is quoted, so a reader can see what was known before the criteria were set.
