# Scoring plan: B-14r as a replicate of the primary

**Filed 11 October 2026, 02:45, while B-14r is running (142 of 300 rows written).** Before
this filing, the first 63 rows were compared with B-14 for row identity on 10 Oct (notebook §6b,
10 Oct): every direct-route fit matched, no cascade fit did, and 13 of 63 verdicts flipped. No
comparison of the checks (AUC, DeLong, the identifiable label) has been computed on B-14r rows.
Scorer: `experiments/land_b14r.py` (committed 10 Oct), which runs once when all 300 rows exist.

## What B-14r is

The filed B-14 command, unchanged except `--out`, on the current code. B-14 ran before the cascade
route's head was seeded per unit (babd0f5, 6 Oct), so B-14's cascade draws cannot be reproduced.
B-14r is therefore **a replicate of the primary that differs only in the cascade's initial draw**,
not B-14's rows with directions added. B-14 stays the filed primary.

## Endpoints (all reported, none replaces the filed primary)

1. **Row identity by route**: direct-route fits identical, cascade fits not (expected from the
   code; reported as counts).
2. **Cascade-initialisation component of the error budget**: verdict flips between B-14 and
   B-14r, Cohen's kappa, and flips by selected route. This is the "no-change re-fit" baseline
   against which B-15's seed, sample and split flips are read.
3. **The replicate's own pre-registered comparison**: AUC(restart agreement) minus AUC(held-out
   R2) on the Euclidean label, DeLong and stratified bootstrap, beside B-14's -0.127.
4. **The identifiable label** of B-14 Addendum 1 (filed 5 Oct): 1/gamma of `ln_2` removed from fit
   and reference, scored on B-14r: failures, class split, and the same comparison.

## Reading rules

- Endpoint 3 is a replication of the primary's sign: it holds if the difference is negative with
  the bootstrap interval excluding 0.
- Endpoint 2 is descriptive; it enters the paper's error budget as its own component.
- Endpoint 4 is reported beside the Euclidean result for the primary replicate; it does not
  replace the filed label.
