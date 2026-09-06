# CALIPER — Plan of Record

**Written 19 August 2026, day 2 of execution. Supersedes the timeline in the project
specification; the experiment definitions there still stand except where noted.**

---

# 1. Where we actually are

Two days in. Phase 0 is the calibration phase, and it is doing exactly what it was designed
to do: **the first instrument we scored turned out to be broken, in a specific and fixable
way, and nothing downstream would have revealed it.**

| Experiment | Status | Result |
|---|---|---|
| **E0.2** Random-direction null | **PASSED** | Detection threshold R² = 0.0443 |
| **E0.3a** Recovery ceiling | **DONE** | 0.976–0.998 at every sparsity — no information limit |
| **E0.5** Classical baselines | **PASSED** | 1/30 recovered by closed form vs 26/30 fitted |
| **E0.1** Analytic control | **FAILING → fix identified** | 0.80 passing; cause found, remedy built |
| E0.3 Sample-complexity table | Not started | Machinery exists in E0.3a |

**Measured throughput (the number the rest of this plan turns on):** 160 s per neuron for
`k=1` plus `k=2` at 20k tokens, d=768, 3 restarts, 2500 steps, single-threaded CPU. The dual
direct+cascade protocol roughly doubles that to ~5–6 min per neuron.

---

# 2. What the last two days changed

Five things in the specification are now wrong, and one thing is new. All were forced by
measurement, none by argument.

**2.1 PCA truncation is out.** MLP read-directions retain 0.352 of their norm in the top-64
PCA subspace against 0.267 for random directions — barely above chance. Retaining 80% needs
D ≈ 512, which is not a reduction. The spec's "reduce to D ≈ 50–100" would have destroyed the
signal it was meant to preserve.

**2.2 Whitening is out.** Round-trips at 1.000 on synthetic data, returns 0.02 on the real
residual stream. The covariance is near-singular at d = 768 and the transform amplifies noise
in near-zero-variance directions. Raw space is the protocol.

**2.3 Sharpee's correlated-stimulus warning does not apply here.** The spec built two mandatory
gates around the claim that random projections explain ~60% of a response at D ~ 10³. Measured:
they explain at most 4.4%. The residual stream is far better behaved than natural images.

**2.4 The sample-complexity table must be indexed by events, not positions.** These units sit
90–99% of the time below GELU's zero, so a position count overstates the information available
by an order of magnitude.

**2.5 The E0.1 pass criterion is unusable at n = 20.** One failure is exactly 0.95, two is 0.90.
The threshold has finer granularity than the measurement. It must be restated as a pass rate
with a confidence interval.

**2.6 NEW — a methodological finding worth publishing on its own.** A rank-1 bottleneck search
on real language-model units converges to a **confidently wrong direction on ~20% of neurons**.
Restart agreement is 0.85–0.99 and gives no warning. A perfect solution exists — R² at the true
direction is 1.000 — and the k=2 subspace already contains the true direction at 0.997. Fitting
one extra dimension and descending into it recovers the answer: n2977 from 0.452 to 0.999,
n230 from 0.866 to 0.999.

---

# 3. The binding constraint, and the fix

**Phase A as specified is not feasible at measured throughput.**

E1.1 sweeps the stimulus layer for each response unit. For GPT-2 at layer 6 that is ~7 stimulus
depths. At ~10³ units and ~5 min per unit per configuration:

```
1000 units x 7 depths x 5 min  =  583 hours  =  24 days of continuous CPU
```

And that is one model, one layer, one variable. The spec asks for two model families and a
Pythia checkpoint ladder. The honest figure is **months of wall-clock**, on a machine that also
has to run everything else.

### 3.1 The fix: batch the fits over the shared stimulus

Every neuron in a layer reads **the same stimulus matrix**. The current code fits them one at a
time, so it recomputes the dominant cost — the `X @ V` projection over 16,000 × 768 — once per
neuron. Batching gives each neuron its own `V` (768 × k) and its own small nonlinearity, then
projects all of them in a single einsum:

```
X:  (n_samples, 768)                shared, projected once
V:  (n_neurons, 768, k)             per-neuron subspaces
Z = einsum('sd,ndk->snk', X, V)     one matmul instead of n_neurons matmuls
f:  grouped MLP, n_neurons in parallel
```

The nonlinearities are tiny (64-wide, k inputs) and run as a grouped batch. Expected speedup at
64 neurons per batch is **20–40×**, which turns 583 hours into 15–30 hours. That is the
difference between infeasible and routine.

**This is the highest-value engineering task in the project and it is now the top priority.**
It is roughly one person-week and it unblocks every subsequent phase. Nothing in Phase A should
start before it lands.

### 3.2 Secondary levers, in order of payoff

1. **Float32 throughout and torch threading** — the fits are already fp32 but thread settings
   are untuned. Cheap check, possibly 2×.
2. **Fewer restarts once the dual protocol is in.** The cascade makes restart-0 informative;
   3 restarts may be more than needed. Measure before cutting.
3. **Free cloud GPU** (Colab/Kaggle) for the Pythia ladder specifically. Not needed for GPT-2
   scale once batching lands, but the checkpoint sweep is inherently 8× the work.
4. **Raise this with the mentor as a resourcing question**, not as a blocker. Department GPU
   access, if it exists, changes Phase B and D materially.

---

# 4. Phase 0 completion plan

Four steps, in strict order. Nothing in Phase A begins until step 4.

### Step 1 — close the estimator protocol (in flight, ~30 min)

E0.1i tests whether **held-out R² selects the better of the direct and cascade fits without
using ground truth**, which matters because Phase A will not have any. Two neurons in so far and
the rule is working with wide margins (n2977: 0.550 vs 0.869 → picks cascade → 0.995;
n230: 0.686 vs 0.999 → picks cascade → 1.000).

Watch for: **degenerate margins.** If any neuron has both fits at R² = 1.000000 while their
alignments differ materially, the objective cannot arbitrate there and selection is a coin flip.
n1561 is the suspect (0.991 direct vs 0.917 cascade, both R² ≈ 1.000).

- **If the rule holds** → estimator default becomes fit-both-keep-better; proceed to step 2.
- **If margins are degenerate** → that is a sharper finding than a clean pass: *the objective
  cannot always identify the correct direction even when the correct direction is reachable.*
  It bears on every method in this family. Report it, add a tie-break rule (prefer the fit with
  higher restart stability, or the direct fit on ties), and proceed.

### Step 2 — batching (1 person-week)

Section 3.1. Build it, verify it reproduces single-neuron results exactly on a fixed seed, and
add a regression test asserting batched and unbatched fits agree to numerical precision.

### Step 3 — E0.1 at n = 100, properly (≈2 h batched, ≈8 h unbatched)

The gate run. Report **pass rate with a Wilson confidence interval**, not a bare fraction.
Pre-register the criterion before running: *the lower bound of the 95% CI on the pass rate
exceeds 0.90.* At n = 100 that requires ~95 passes, which is a real test rather than an
artefact of sample granularity.

### Step 4 — E0.3 sample-complexity table (2–3 person-weeks)

Extend E0.3a's ceiling sweep into the deliverable the spec promises: **required N for a given
recovery precision, indexed by events rather than positions**, across K ∈ {1,2,3} and D. Settles
which of the two published scaling accounts governs correlated text — the linear
`N ∝ K·D` account or the exponential histogram account. Reusable regardless of what else lands,
and nobody has it for language models.

**Phase 0 then closes.** E0.4 (retrodiction on the known circular weekday feature) stays
EXTENDED and is skipped unless there is slack.

---

# 5. Phase A, revised

Unchanged in intent. Three changes in execution:

- **Stimulus space is raw**, not whitened or truncated (§2.1, §2.2).
- **Estimator is the dual protocol** (direct + cascade, selected by held-out R²).
- **Batching is a prerequisite**, not an optimisation.

Order within Phase A, revised so that the cheapest decisive experiment runs first:

| | Experiment | Why this order |
|---|---|---|
| A1 | **E1.1 stimulus depth sweep** | The boundary condition is known exactly (K = 1 at distance 1), so it doubles as an ongoing correctness check on real data. Produces the dimensionality-vs-computational-distance curve, which is a novel object in its own right. |
| A2 | **E1.4 causal validation** | Ablate the recovered subspace against matched random and top-K-PC controls. Until this passes, every K is a curve fit rather than a measurement. Reuses APERTURE's intervention machinery. |
| A3 | **E1.2/E1.3 dimensionality distribution** | The headline descriptive artefact. Only meaningful after A2. |
| A4 | **E1.8 absorption-notch test** | Cheap, quotable, and the one component a frontier lab would adopt. Pulled early per the audit. |
| A5 | E1.6 kurtosis heuristic, E1.7 response characterisation | Rounding out the paper. |

**Battery stays at two variables.** The audit measured the earlier plan at ~2× team capacity
and that finding has not been superseded.

---

# 6. Publication strategy

We already have a coherent, defensible paper — and it is not the one the spec was aiming at.

### 6.1 Paper 1 — "Calibrating subspace estimation for language-model units"

Everything in it is measured and in the repository:

1. **Rank-1 search fails on ~20% of real units**, converging confidently to a wrong direction
   with no warning from restart agreement; a perfect solution exists and is reachable; a
   cascade from k+1 recovers it. *(E0.1, E0.1b–i)*
2. **Classical spike-triggered methods fail completely** — 1/30 recovered against 26/30 for the
   fitted estimator, because the residual stream is non-Gaussian and GELU is non-monotone over
   the occupied range, driving the Bussgang constant toward zero. *(E0.5)*
3. **The random-direction null is tight** in language-model residual streams — 4.4%, against
   the ~60% the vision literature warns of. Detection threshold established. *(E0.2)*
4. **MLP read-directions are near-isotropic** in the residual stream's PCA basis, so
   dimensionality reduction destroys what it is meant to preserve. *(E0.1 ceiling curve)*
5. **The recovery ceiling is flat across sparsity**, so failures are not information limits.
   *(E0.3a)*
6. **A required-N table** for language models. *(E0.3, pending)*

That is a methods paper about a technique the field is beginning to adopt, and every result is a
negative or a caution that is genuinely hard to obtain — because each one **required knowing the
answer in advance**, which is the project's whole thesis. Venue: BlackboxNLP, ICML MechInterp,
or NeurIPS ATTRIB. Target the first deadline reachable after step 4.

### 6.2 Paper 2 — the measurement itself

Phase A's dimensionality distribution, causal validation, and the SAE adjudication. This is the
main-track attempt. Unchanged from the specification.

### 6.3 Why this ordering is better than the original

The spec assumed Phase 0 would be throat-clearing before the real work. It has instead produced
the more defensible paper, because calibration results are exactly what a field with no
measurement standard is missing. Paper 1 also **de-risks Paper 2** — it establishes in print
that our instrument is characterised, which is the first thing a reviewer will ask about any
number in Paper 2.

---

# 7. What to bring the mentor

Not a status update — a decision request on three things.

1. **The calibration-first design is vindicated, concretely.** The first instrument we scored
   was broken on a fifth of its inputs, silently. Show the n2977 trace: 0.452 → 0.999.
2. **Ask for compute.** Whether department GPU access exists changes Phase B and the Pythia
   ladder materially. Frame it with the measured numbers in §3, not as a general request.
3. **Propose Paper 1.** A methods paper from Phase 0 was not in the original plan. It is
   reachable this year, it de-risks the main paper, and it gives the capstone a publication
   milestone well before the thesis.

---

# 8. Risks, revised

| Risk | Change since the spec | Response |
|---|---|---|
| **Compute** | **Promoted to the binding constraint.** Measured, not estimated. | Batching (§3.1) before Phase A. Escalate GPU access to the mentor. |
| **Estimator correctness** | Was unknown; now characterised and largely fixed | Dual protocol; re-verify at n = 100 with a CI |
| Cheap proxy metrics misleading us | **New, and it has bitten three times** — equal-count binned R² produced two phantom findings and one bad candidate selection | Any screen is a screen: shortlist, then decide with the real objective. Now encoded in `fit_cascade`. |
| Underpowered diagnostics | **New, bit repeatedly** — four separate n = 20 tests all returned p ≈ 0.13 | No diagnostic below n = 100 gets a conclusion attached to it |
| Orphaned background jobs skewing throughput | New, operational | `nohup` not foreground `timeout`; check for orphans before quoting any timing |
| Scooping | Unchanged | Instrument framing is scoop-resistant; Paper 1 is date-stampable soon |
| Scope 2× capacity | Unchanged | Battery stays at two variables; Phase B still gated on E2.5 |

---

# 9. Decision points

| When | Question | Branch |
|---|---|---|
| **~30 min** | Does held-out R² select the better fit? | Yes → dual protocol is the default. Degenerate → publish that, add a tie-break, continue. |
| After batching | Does batched == unbatched to precision? | Must be exact, or batching is not usable and Phase A needs re-scoping. |
| After E0.1 n=100 | Is the CI lower bound above 0.90? | Yes → Phase 0 closes. No → the residual failure rate is a property of the method and becomes a headline result rather than a bug. |
| After E1.4 | Does ablating the recovered subspace collapse the response? | No → the measurement is a curve fit and Phase A stops until the estimator is causal rather than observational. |

---

## Immediate queue

1. E0.1i completes → protocol decision *(in flight)*
2. Update spec and README with §2's six corrections *(1 hour, no compute)*
3. Build batching *(1 person-week — the unblocker)*
4. E0.1 at n = 100 with a CI *(the gate)*
5. E0.3 required-N table *(closes Phase 0)*
