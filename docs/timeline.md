# CALIPER — Full Programme Timeline

**August 2026 → May 2028 · 21 months · written 19 Aug 2026, day 2 of execution**

---

## Assumptions this timeline rests on

Stated first because if any of them is wrong, the dates move.

| Assumption | Basis | Risk if wrong |
|---|---|---|
| ~66 person-weeks of core work | External audit costing, unchanged | The single largest scheduling risk |
| ~50–70 productive person-weeks available across 4 people | Audit estimate net of coursework, exams, placement | Timeline is ~1.0× capacity — no slack |
| **All four members contribute** | Project structure | **Currently false.** One person is executing. This is the biggest live risk and it is not technical |
| Compute is CPU + Kaggle (30 GPU-h/week) | Measured today | A local GPU compresses Phase A and B materially |
| Measured throughput 63 s/neuron | Gate run, 19 Aug | Already 6.5× better than day-1 code |

**Academic calendar:** sem 5 now (Jul–Nov 2026) · sem 6 (Jan–May 2027) · **sem 7 (Jul–Nov 2027,
placement season — assume ~40% productivity)** · sem 8 (Jan–May 2028) · graduation mid-2028.

---

## Phase 0 — Calibration · Aug–Oct 2026

**Goal: every instrument carries a measured error rate before it is pointed at anything unknown.**

| | Work | Status |
|---|---|---|
| Aug (done) | E0.2 null · E0.3a ceiling · E0.5 classical baselines · estimator built, characterised, fixed | **Complete** |
| Aug–Sep | E0.1 gate — re-run with the operating-point transfer question pre-registered | Failing at 8k config; re-run pending |
| Sep | E0.3 required-N table, indexed by events | Not started |
| Sep–Oct | Batched + GPU port verified; cascade screens moved off numpy | Batching done (4.5×) |

**Gate:** Wilson 95% lower bound on the E0.1 pass rate > 0.90. Phase A does not begin until it
clears, or until the residual failure rate is characterised and declared a known property.

**Slip risk: moderate.** Phase 0 was budgeted 3 months and is on schedule, but it has already
produced four wrong turns. Each cost 1–2 days and each was caught. Budget for two more.

---

## Paper 1 — the calibration methods paper · Oct 2026 – Mar 2027

Six results, all already measured, all negative or cautionary, all requiring the answer to be
known in advance:

1. Rank-1 subspace search fails silently on ~20–25% of real units; a perfect solution exists
   and is reachable; a cascade from k+1 recovers it
2. Held-out R² is not a reliable proxy for direction recovery
3. Classical spike-triggered estimators recover 1 unit in 30
4. The random-direction null is tight in LM residual streams, contra the vision literature
5. MLP read-directions are near-isotropic in the residual PCA basis
6. Required-N for language models

| | |
|---|---|
| **Oct–Nov 2026** | Complete result set; figures; internal draft |
| **Dec 2026** | arXiv preprint — date-stamp the cascade finding |
| **Jan–Feb 2027** | Submit: ICML 2027 (~Jan) or ACL 2027 (~Feb); fallback BlackboxNLP 2027 |
| **Mar 2027** | Reviews / camera-ready |

**Why this ordering matters:** Paper 1 de-risks Paper 2. A reviewer's first question about any
number in the main paper is whether the instrument is characterised. Having that in print
answers it before it is asked.

---

## Phase A — Unit-level measurement · Nov 2026 – Jun 2027

The first phase that measures something unknown. Scoped to **300 units × 4 stimulus depths**
(~22 h compute) rather than the specification's 1,000 × 7 (~130 h).

| | Experiment | Gate |
|---|---|---|
| Nov–Dec 2026 | **E1.1** stimulus depth sweep — dimensionality vs computational distance | Boundary condition K=1 at distance 1 must hold on real data |
| Jan–Feb 2027 | **E1.4** causal validation — ablate recovered subspace vs matched random and top-K PC | **Hard gate.** If ablation doesn't collapse the response, every K is a curve fit and Phase A stops |
| Feb–Apr 2027 | **E1.2/E1.3** dimensionality distribution across layers, scales, checkpoints | Survives the E0.2 null; FDR controlled |
| Mar 2027 | **E1.8** absorption-notch test | Pulled early — cheap and the most quotable single result |
| Apr–Jun 2027 | **E1.6** kurtosis heuristic · **E1.7** response characterisation | — |

Concurrent: **Phase C calibration** (InterpBench FDR, planted-latent networks, training-trajectory
null). Every headline number in Phase A must carry an error rate measured here.

---

## Sem 7 — the constrained window · Jul–Nov 2027

**Placement season. Assume ~40% productivity and plan accordingly.**

| | |
|---|---|
| Jul–Aug 2027 | Paper 2 draft from Phase A results. Writing survives interruption better than experiments |
| Sep 2027 | **ICLR 2028 deadline (~Sept)** — the primary target for Paper 2 |
| Oct–Nov 2027 | **Phase B** (communication subspaces), *conditional on* E2.5 showing the measurement isn't recoverable from weights alone |

**Decision point, Jul 2027:** if Phase A is behind schedule, Phase B is cut entirely and moves to
Paper 3 or is dropped. It is the only major component the programme can lose without damaging
the thesis.

---

## Sem 8 — consolidation · Jan–May 2028

| | |
|---|---|
| Jan–Feb 2028 | Paper 2 revisions; **ICML 2028 (~Jan)** as fallback venue |
| Feb–Mar 2028 | **Phase D** — planted-prior manipulation, *if* the delta paragraph against Cacioli's work still holds |
| Mar–Apr 2028 | Thesis assembly; artefact release (datasets, JAX estimator library) |
| Apr–May 2028 | Defence |

---

## Publication targets, ranked by realism

| Venue | Deadline | Paper | Odds |
|---|---|---|---|
| arXiv preprint | Dec 2026 | Paper 1 | Certain — date-stamps the cascade finding |
| BlackboxNLP / ICML MechInterp / NeurIPS ATTRIB | rolling 2027 | Paper 1 | High |
| ACL / ICML 2027 main | Jan–Feb 2027 | Paper 1 | Moderate |
| **ICLR 2028** | **Sept 2027** | **Paper 2** | **The main shot** |
| ICML 2028 | Jan 2028 | Paper 2 | Fallback |
| NeurIPS 2028 | May 2028 | Paper 2 | Too close to graduation |

Audit consensus: 8–15% main-track as originally framed, 25–45% restructured. Phase 0's results
push the low end up, because a characterised instrument is the thing reviewers ask for.

---

## Hard gates — the programme stops or changes at each

| When | Gate | If it fails |
|---|---|---|
| **Sep 2026** | E0.1 pass rate CI lower bound > 0.90 | Characterise the residual failure rate and declare it; do not keep re-running |
| Oct 2026 | E0.3 required-N table exists | Phase A cannot be scoped without it |
| **Feb 2027** | E1.4 — ablating the recovered subspace collapses the response | **Phase A stops.** The measurement is observational, not causal, and needs rethinking |
| Jul 2027 | Phase A on schedule | Cut Phase B |
| Oct 2027 | E2.5 — data-manifold rank below weight-derived rank | Cut Phase B (overlaps *Talking Heads*) |
| Feb 2028 | Cacioli delta paragraph still holds | Cut Phase D |

---

## The three risks that actually decide this

**1. Team.** Four-person capstone, one person executing. The plan has four separable tracks —
extraction pipeline, estimator and nulls, causal patching, calibration harnesses — and each
member needs a defensible contribution at review. This is the largest risk in the programme and
it is not technical. **Resolve in weeks, not months.**

**2. Capacity.** 66 person-weeks of core work against 50–70 available is ~1.0× — no slack. Two
mitigations already applied: battery cut to two variables, Phase A scoped to 300 units. A third
is held in reserve: cut Phase B.

**3. Proxy metrics.** Four wrong turns in two days, every one from trusting a cheap proxy — a
transform validated on 3 neurons, a mis-specified binning metric twice, an operating point
validated on synthetic units. Standing rule now: **a screen is a screen; decide with the real
objective, and check that synthetic conclusions transfer before betting on them.**

---

## What happens in the next four weeks

1. E0.1 gate — new pre-registration with the operating-point transfer question stated in advance
2. E0.3 required-N table → **Phase 0 closes**
3. Assign the other three team members to tracks
4. Mentor meeting: calibration-first vindicated · GPU ask · propose Paper 1
5. Begin Paper 1 figures while Phase A infrastructure is built
