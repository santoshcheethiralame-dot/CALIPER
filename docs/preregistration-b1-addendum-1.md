# Addendum 1 to the B-1 pre-registration — the comparison test, power, and the extension rule

**Filed 8 September 2026, while B-1 is running and before any row has been written.**
`results/b1_stability_gpt2.jsonl` does not yet exist at the time of filing; the run is in
its first batch. No outcome has been observed.

The parent filing is `preregistration-b1-stability-calibration.md`. This addendum fixes
three things it left underspecified. All three were found by a literature scout done
after the run started, which is itself worth recording: the scout should have preceded
the run.

---

## 1. The comparison test was never named

The parent filing says the primary endpoint is `stability`'s AUC *"against the AUC of
held-out R2"*, and its branches turn on whether stability is **"significantly below"** the
incumbent. It does not say by what test. Left unfixed, that is an open door: several tests
would give several answers and the choice could be made after seeing the numbers.

**Fixed now. The test is DeLong's test for two correlated ROC curves** (DeLong, DeLong &
Clarke-Pearson, 1988).

This is not a stylistic choice. Both signals are scored **on the same units**, so the two
ROC curves are correlated, and comparing them with independent confidence intervals
would be wrong — it ignores the covariance and is conservative in a way that could hide
a real difference. DeLong's test is the standard non-parametric method for exactly this
design.

- Two-sided, alpha = 0.05.
- Reported as: AUC difference, its standard error, z, and p.
- Implemented in-repo from the covariance form of the statistic, with a unit test
  against a case whose answer is known analytically. No new dependency.

## 2. Multiplicity

Five signals are scored, on two models. Every pairwise comparison against held-out R2 is
a test, and reporting the most favourable one without correction would be a
multiple-comparisons error of the ordinary kind.

**Fixed now.** The primary comparison is exactly one: **`stability` versus held-out R2 on
B-1 (GPT-2)**. It is not corrected, because it is a single pre-specified test.

Every other comparison — the other signals, and everything on B-2 (Pythia) — is
**secondary and Benjamini–Hochberg corrected** across the full family of secondary
comparisons, reported with both raw and adjusted p.

## 3. The run may be underpowered, and that has to be stated before the numbers arrive

A power scout after launch returned figures that change how the result must be read.
For comparing two correlated AUCs at 80% power, two-sided alpha = 0.05, required sample
size is driven almost entirely by the size of the difference:

| difference to detect | cases required |
|---|---|
| ΔAUC = 0.10 | 36–142 |
| ΔAUC = 0.02 | 909–3,709 |

Correlation between the two tests helps substantially — strong correlation (rho = 0.8)
cuts the requirement by roughly half to two thirds — and our signals are computed from
the same fits, so they will be correlated. Even so:

**B-1 at n = 100, with roughly 20 failures expected, can detect a large gap and cannot
detect a small one.**

### The rule, fixed now

- **If |ΔAUC| ≥ 0.10** — the comparison is adequately powered. Report the DeLong result
  as the primary outcome and take the parent filing's branch.
- **If |ΔAUC| < 0.10** — the comparison is **underpowered, and is reported as
  underpowered**, not as a null. The phrase "no significant difference" is forbidden in
  this case; the reportable statement is "the difference is smaller than this run can
  resolve." We then **extend to n = 300 units** on the same model, same protocol, same
  seed policy, and re-test once. That extension is authorised here, in advance, so it
  cannot be a reaction to a disappointing p-value.
- **The extension happens at most once.** If n = 300 still cannot resolve it, the honest
  conclusion is that the two signals are close enough that the choice between them should
  be made on cost, not on discrimination — and cost already favours held-out R2, which is
  free, over restart agreement, which is 5x the compute.

---

## 4. One free analysis the parent filing did not anticipate

B-1 stores **five** restarts per unit. Practitioners commonly run two or three. The
stability statistic can therefore be recomputed at 2, 3, 4 and 5 restarts **from the data
already collected**, at no additional compute, giving the operating characteristic of the
check *as it is actually used* rather than only at our chosen budget.

**Pre-specified:** this is secondary, exploratory, and reported as a curve (AUC versus
restart count) rather than tested. Its purpose is descriptive — if the check only works
at five restarts, that is a materially different practical recommendation from it working
at two, and the paper should say which.

Subsampling is by the first *k* restarts in recorded order, not a favourable subset.

---

## 5. What this addendum does not change

- The primary endpoint, its direction, and the three branches in the parent filing.
- The failure label: `alignment < 0.95`, direction recovery, alignment alone. (Not the
  E0.1 gate criterion, which also requires `k2_gain < 0.01` and would make `k2_gain`
  predict itself. This distinction is recorded in the notebook and in
  `b1_signal_calibration.py`.)
- The disclosed n = 8 pilot and the expectation drawn from it, which remains an
  expectation and not a criterion.

## 6. Committed

- This addendum is filed before any B-1 row exists, and the file's absence at filing time
  is stated above and verifiable from the commit's timestamp against the run log.
- The DeLong implementation is unit-tested before it is pointed at real data.
- If the extension to n = 300 is triggered, it is labelled an extension in the paper, with
  this filing cited as its authorisation.
- No change to the test, the correction, or the extension rule after seeing output.
