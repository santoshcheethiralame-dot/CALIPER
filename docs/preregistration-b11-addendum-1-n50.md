# Addendum 1 to the B-11 ladder filing — 50 units per rung, not 100

**Filed 9 September 2026, mid-session, before any rung completed.**

---

## Disclosure

**No rung had produced a completed result when this was written.** Two attempts at the
70m rung were made; the first was abandoned when the session was restarted, the second was
interrupted deliberately. **No alignment, no pass rate, and no signal AUC has been seen for
any rung.** This amendment is made on *timing* information alone.

## Why

The ladder was costed at 100 units per rung on the assumption that a GPU would carry the
fitting. It does not carry most of it.

Measured on a T4: **38.6 s/unit at d_model 512**. The reason is now precisely known and it
is structural, not a configuration mistake:

| per batch of 32 units | steps | batched? |
|---|---|---|
| `fit_batch` (k=1 and k=2, 2 restarts) | 6,400 | yes, across 32 units |
| `fit_cascade` (720-point grid + 5 x 800 polish, **per unit**) | 151,040 | **no, one unit at a time** |

**The cascade is 24x the work of the batched fit**, it is unbatched, and `e01_gate.py:72`
calls it without `device=`, so it inherits `fit()`'s `device="cpu"` default. The GPU is
accelerating about 4% of the job, on a machine whose CPU is weaker than the local one.

At 100 units per rung the four rungs need ~8.4 h against a 12 h session cap, with the
largest rung last. That does not fit reliably. At 50 it is ~4.6 h and all four land.

## The change

**50 units per rung, all four rungs, one device, one session.** Nothing else moves: same
depth-matching, same 2 restarts, same 8000 tokens, same 1600 steps, same 0.95 bar, same
rng and seed, same primary endpoint.

## What this costs, stated honestly

Wilson intervals roughly sqrt(2) wider. At an expected pass rate near 0.9, a 50-unit rung
gives about [0.78, 0.96] where 100 units would give about [0.83, 0.95].

**This weakens the per-rung estimate and it weakens the trend test.** Cochran-Armitage on
four rungs of 50 detects only a large monotone effect. The consequence is committed in
advance:

- **A null or ambiguous trend at n=50 will NOT be reported as "the failure rate is
  scale-invariant."** It will be reported as **underpowered and inconclusive**, with the
  intervals shown.
- Only a trend whose per-rung intervals separate is reported as a trend.
- If the result is inconclusive and the question still matters after the primary arms land,
  the rungs are extended to 100 by re-running — every rung is resumable, so extending is
  additive rather than a fresh run.

**A wider interval reported as a wider interval is honest. A narrow claim from a wide
interval is not.** This amendment buys four rungs instead of two, and the price is stated
here rather than discovered at review.

## Committed in advance

- The 50 units are the **first 50 of the same draw**, same rng and seed. No reselection,
  and a later extension to 100 is nested rather than parallel.
- The 70m rung restarts from zero. Its two abandoned attempts produced no recorded result
  and are not analysed.
- **The device stays uniform across all four rungs.** If a rung cannot complete on the GPU
  it is reported as not run, and is not substituted with a local CPU run to fill the slot.
- The `fit_cascade` device default is **not** changed mid-ladder. Fixing it would speed
  future runs and is worth doing, but changing the numerics between rungs would put a code
  version inside the scaling curve, which is the confound this ladder was designed to avoid.
