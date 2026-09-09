# Addendum 1 to the B-1 filing — statistical power, and a second arm at 2 restarts

**Filed 9 September 2026, 08:00 IST. Amends `preregistration-b1-stability-calibration.md`.**

---

## Disclosure first, because it governs how this should be read

**This addendum was written after seeing partial B-1 output.** At 64 of 100 units the
interim values were:

| signal | interim AUC at n=64 |
|---|---|
| held-out R2 | 0.891 |
| `stability` (restart agreement) | 0.822 |
| disagreement | 0.718 |
| `r2_spread` | 0.618 |

6 failures among 64 units. **Nothing in this addendum changes the bar, the units, the
signals, the primary endpoint or its decision rule.** What changes is sample size and the
addition of a second arm. The direction of the primary criterion is untouched, and the
interim numbers are reproduced above so any reader can see exactly what was known when
this was written.

The original filing committed to "no change to the bar, the units, the operating point, or
the criteria after seeing output". That commitment is kept. It did not, and could not,
commit to never discovering that the design was underpowered.

## What the original filing got right, and should be credited with

It anticipated the pass-rate change:

> The alignments will differ from C13 because 5 restarts is a better optimiser than 2;
> **the pass rate is therefore expected to move, and a changed pass rate is not a
> finding here.**

That is exactly what happened — C13's 22 failures per 100 became roughly 6 per 64 — and it
is filed as **not a finding**. It is not reported as one. An earlier session note called
it "a real finding"; that was wrong and is corrected here.

## What it did not anticipate: the power consequence

A lower failure rate means fewer positive cases, and the primary endpoint is a
**comparison of two AUCs on the same units**. Its power is driven by the size of the
smaller class.

Published sample-size work on ROC AUC differences (Obuchowski–McClish for discrete,
Hanley–McNeil for continuous, two-sided alpha = 0.05, 80% power) puts the requirement at
**36–142 cases per group to detect Delta AUC = 0.10**, and 909–3,709 for Delta AUC = 0.02.
Strong inter-test correlation helps substantially — rho = 0.8 cuts the requirement by
roughly half to two thirds — and these signals are correlated, being computed from the
same fits. But the interim gap is Delta AUC ~ 0.07 against **about 10 expected failures at
n = 100**.

That is not a comparison. Reporting it as one, in either direction, would be exactly the
kind of underpowered claim this bench exists to expose in other people's work.

## The amendment

### Primary arm moves to `restarts = 2`, `n = 300`

Three independent reasons, and the first is the one that matters:

1. **The incumbents were measured at 2 restarts.** Held-out R2's 0.906 and disagreement's
   0.802 come from the C13/C47 configuration, which ran 2 restarts. Scoring a 5-restart
   `stability` against a 2-restart R2 AUC compares signals measured under different
   optimisers. **The fair comparison holds the optimiser fixed.**
2. **It is what practitioners do.** A 2-restart stability estimate is a single pairwise
   alignment — noisy, and precisely the number a practitioner running two seeds actually
   holds. The bench's claim is about the check people use, so the primary arm should
   measure the check people use.
3. **It is adequately powered.** At C13's 22% failure rate, n = 300 yields roughly 66
   failures, inside the range the power work says is sufficient for Delta AUC = 0.10.

Cost: about 5.3 hours locally, from C13's measured 63.7 s per unit at 2 restarts.

### The 5-restart arm is retained and re-labelled

B-1 as run (5 restarts, n = 100) becomes the **sensitivity arm**, not the primary. It
answers two questions the primary cannot:

- Does a better optimiser reduce silent failure? (It appears to. Reported descriptively,
  with the original filing's non-finding clause attached.)
- Does an expensive 5-restart stability estimate beat a cheap 2-restart one — the original
  filing's secondary 4?

**Secondary 4 was not computable from what B-1 recorded**, which saved only the median
pairwise alignment. `Fit.stability_pairs` and a `stability_pairs` column were added on
9 Sep so every restart subsample is recoverable offline. B-1's own rows lack it; B-2
onward carry it, and B-1 is re-runnable at no conceptual cost if the question turns out to
matter.

## Criterion — unchanged in form, restated at the new n

Primary: **AUC of the 2-restart `stability` for predicting failure at n = 300, against the
AUC of held-out R2 on the same units**, compared by **DeLong's test for two correlated
ROC curves** (correlated because both are computed from the same fits on the same units),
with Benjamini–Hochberg correction across the signals tested.

- **`stability` AUC significantly below held-out R2** — the field's default check is
  dominated by a free alternative the fit already computes. The practice recommendation
  stands and is now calibrated.
- **The two statistically indistinguishable at adequate power** — a genuine equivalence
  result, and the recommendation becomes cost-based rather than accuracy-based.
- **`stability` AUC significantly above held-out R2** — Finding 2 is wrong and is amended
  in the notebook and the paper. Filed as a live outcome, not a formality.

## Committed in advance

- **DeLong's test, not overlapping confidence intervals.** Two AUCs on the same units are
  correlated; independent intervals would overstate the evidence for a difference.
- Benjamini–Hochberg across the signal set, since four to five signals are being compared
  on one dataset.
- **The 300 units extend the C13 draw rather than replacing it** — same rng, same seed,
  same layer. The first 100 are the existing set, so the original arm is nested inside
  the new one and no unit is reselected.
- Both arms are reported. The 5-restart arm is not dropped for being underpowered; it is
  reported as the sensitivity arm it now is, with its n and its restart count stated.
- **This disclosure travels with the result.** Wherever the primary AUC comparison is
  quoted, the paper states that the sample size was increased after an underpowered
  interim look, and reproduces the interim table above.
- If n = 300 still yields fewer than 30 failures, the comparison is reported as
  **underpowered and inconclusive** rather than extended a second time on a third guess.
