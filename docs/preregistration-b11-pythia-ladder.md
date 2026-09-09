# Pre-registration — B-11, the Pythia scale ladder

**Filed 9 September 2026, before the run.**

---

## Why this run exists

Two reviewers' objections, one finding, one run.

**The objection.** "Your ground truth only works on 124M and 160M models. Nobody deploys
those." It is fair, it is the strongest attack on the bench, and no amount of argument
answers it. A scale curve does.

**The finding, if it is there.** Pythia publishes 70m / 160m / 410m / 1b / 1.4b **trained
on identical data in identical order**. That is the only model suite where scale can be
varied with the training corpus and curriculum held fixed. Running one protocol across
rungs therefore measures *silent-failure rate as a function of scale*, controlled — not
"we also tried a bigger model".

If the failure rate falls with scale, the bench's warning is a small-model artefact and we
say so plainly. If it holds or rises, the bench matters more at deployment scale than at
the scale we measured it, which is the stronger result and the one that is not yet known.

## Why it goes on Kaggle, and why that is not a confound

Runs that are **compared to each other** must share a device — floating-point reduction
order differs between CPU and GPU, and failing units sit near basin boundaries by
construction, so the comparison is where the risk lives (see `NEXT_SESSION_B0_DEVICE.md`).

**Every rung of this ladder runs on the same GPU.** The scaling curve is therefore
internally device-consistent, and its claim — how failure rate moves with scale — is
answerable without reference to the local runs at all.

Two consequences, both stated in advance:

- The ladder's absolute rates are **not** pooled with the local `n=300` arms. They are
  reported as their own table, with the device named.
- The **160m rung is directly comparable to C53**, which ran the same protocol locally on
  CPU (93/100 at 2 restarts). That is a free aggregate device check at n=100. It is
  weaker than B-0's paired per-unit test and does not replace it, but a large discrepancy
  there is informative and is reported either way.

## Protocol (frozen)

| element | value |
|---|---|
| suite | `EleutherAI/pythia-{70m,160m,410m,1.4b}`, revision `main` |
| depth | **0.5 relative depth at every rung** — the same choice as C53's layer 6 of 12 |
| units | 100 per rung, drawn by the same rng and seed as every other gate run |
| restarts | **2** — matches the primary arm and the incumbent AUCs (Addendum 1) |
| tokens | 8000, the E0.3b operating point |
| steps | 1600 |
| bar | alignment > 0.95, unchanged |
| device | one GPU session, all rungs |

Per-rung configuration, depth-matched:

| model | d_model | d_mlp | layers | `--layer` |
|---|---|---|---|---|
| pythia-70m | 512 | 2048 | 6 | 3 |
| pythia-160m | 768 | 3072 | 12 | 6 |
| pythia-410m | 1024 | 4096 | 24 | 12 |
| pythia-1.4b | 2048 | 8192 | 24 | 12 |

`pythia-1b` is optional and runs only if the session has time; its 16 layers make it the
least clean rung and it is not needed for the trend.

**1.4b is the reason this run is on a GPU at all.** It needs ~5.6 GB in fp32 against a
local machine with 16 GB total and 2.6 GB free under load. It is not attempted locally.

## Primary endpoint

**Silent-failure rate per rung, as a Wilson 95% interval**, and the sign of the trend
across the four rungs tested by Cochran–Armitage on the ordered rungs.

Filed before the run:

- **Rate falls monotonically and 1.4b's interval excludes the 160m point.** The phenomenon
  attenuates with scale. Reported as a limitation on the bench's reach, in the abstract,
  and the paper says the calibration was measured where the failures are.
- **Rate is flat — intervals overlap across rungs.** Silent failure is scale-invariant over
  a 20x parameter range with training data held fixed. This is the result we do not have
  and the one that makes the bench matter at deployment scale.
- **Rate rises with scale.** The strongest outcome, and the one to be most sceptical of:
  before reporting it, check that the wider models are not simply harder to fit at a fixed
  1600 steps. **That check is pre-committed** — re-run the top rung at 3200 steps and
  report both. A rate that falls with more steps is an optimiser artefact, not a scale
  effect.
- **Non-monotone.** Reported as measured, with no story fitted to it. Four points do not
  support a shape claim.

## Secondary

1. Every signal AUC per rung — `stability`, held-out R2, disagreement, `r2_spread` — so
   the calibration's transfer across scale is visible, not assumed. Compared by DeLong
   within a rung, never across devices.
2. Median and minimum alignment per rung. C53's aggregate-invisibility finding — median
   0.9994 hiding a unit at 0.0761 — is checked at every rung. If aggregate quality keeps
   improving while the worst unit stays bad, that is the bench's central claim holding
   across scale.
3. Seconds per unit per rung, so the cost model for future work is measured.

## Committed in advance

- **No pooling with the local arms.** Device is named beside every number.
- The 100 units per rung are drawn by the same rng and seed and are not reselected. Units
  differ across rungs because `d_mlp` differs; that is inherent and disclosed.
- **If a rung fails to load or OOMs, it is reported as not run** — not quietly dropped, and
  not replaced with a different model to fill the slot.
- The steps-artefact check is run if and only if the rate rises, and its result is reported
  whichever way it comes out.
- If the trend is significant but driven entirely by 70m — the rung least like a deployed
  model — that is stated, and the trend is re-reported over the three larger rungs.
