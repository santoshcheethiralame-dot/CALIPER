# B-14 Addendum 2: failure-class-stratified AUCs, fixed before unblinding

**Filed 6 October 2026 (morning), while B-14 is at 230/300 rows and before any of its
outputs have been viewed.** The primary endpoint and Addendum 1 are unchanged.

## Why

An exploratory analysis of runs that were already unblinded (B-1b, B-2b, B-8, B-11c) splits
failures into the pre-registered classes, wrong basin (held-out R2 > 0.99) and under-fitted:

| arm | wrong basin | under-fitted | R2 / restart AUC, wrong basin vs pass | R2 / restart AUC, under-fitted vs pass |
|---|---|---|---|---|
| B-1b GPT-2 L6 (coupled) | 12 | 40 | 0.776 / 0.729 | 0.992 / 0.793 |
| B-8 GPT-Neo L10 (coupled) | 24 | 17 | 0.555 / 0.477 | 0.940 / 0.731 |
| B-11c Pythia-1.4B (fixed) | 0 | 19 | n/a | 0.973 / 0.847 |

Held-out R2's advantage comes mostly from the under-fitted class. On the wrong-basin class,
both checks are weak. The R2 AUC on that class is attenuated by construction, because the
class is defined by R2 > 0.99. The restart-agreement AUC is not attenuated that way.

## Added secondary analyses for B-14 (exploratory, specified now)

1. The four signals' AUCs computed separately for wrong-basin vs pass and under-fitted vs
   pass, with stratified bootstrap CIs. They are reported only if the class has at least 5
   units.
2. The same split for B-1b on the same units, so the effect of the stopping fix on each class
   can be seen.
3. A sensitivity of the class boundary: R2 cutoffs at 0.98, 0.99 and 0.995.

## How it will be read

None of this changes the primary decision table. If restart agreement and held-out R2 are
both near chance on wrong-basin failures, the paper says so plainly. The checks catch the
under-fitted failures and none catches the converged-but-wrong ones. That moves the
contribution from "R2 beats restart agreement" to "neither check sees the hard case, and here
is the hard case's size".
