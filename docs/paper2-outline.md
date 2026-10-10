# Paper 2 outline, re-planned after S-1 (9 October 2026)

**Status.** Structure, claims and evidence only; the prose is the author's. **Updated 10 Oct:**
P2-F and P2-L scored and folded in (C1, C3, C8); figures regenerated; Gate P2 status in §7. Supersedes the
framing of `docs/paper-s3-draft.md` (3 Sep) and the Paper 2 lines of `flagship-plan.md` §4 and
§12.2. Venue unchanged: ICML 2027 if Gate P2 (15 Jan) passes, else an interpretability workshop.

## 1. What changed and why

The plan was "the published concept-vector recipe produces dead vectors". S-1 overturned that
for the published model:
- on Gemma-3-27B the released template-tail vectors are live at the released strength (S-1M:
  24 of 30 at strength 4);
- on Gemma-3-12B the tail is only slightly worse than the concept token, and the precision
  question cannot be run on free hardware;
- dead vectors are real on Qwen (C31, S-2), where the tail read is inert.

What survives across models is broader and more useful: **every instrument choice in a
concept-injection experiment is unvalidated, and the checks people would use to validate it do
not work.** The vector may be dead, the dose has no unit, and the readout answers a question
other than the one asked.

## 2. Working title and one-line thesis

*Auditing the Instrument: What Concept-Injection Experiments Measure About LLM Introspection*
(working; no name decision needed).

Thesis: concept-injection introspection results rest on three instrument assumptions (the
vector is live, the dose is comparable, the readout reports self-access) that the field does
not check; calibrated against controls with known answers, each fails in some regime, and the
standard checks do not notice.

## 3. Position in the literature (ledger: `CITATIONS.md` §26-29)

| work | what it shows | how Paper 2 differs |
|---|---|---|
| Lindsey 2025/26; Macar et al. 2603.21396 | detection at moderate rates, 0% false positives; Gemma-3-27B L37, alpha 4 best; DPO-dependent circuit | we run their released recipe and strength (S-1M) inside a calibrated audit |
| Hahami et al. 2512.12411 (Llama-3.1-8B) | binary detection fully explained by global affirmative logit shifts; "not present in large Claude models" | we show content-free vectors reproduce the first-token effect on Gemma-3-27B, and the framing confound at the released window (S-1M, C20, A2) |
| Lederman & Mahowald 2603.05414 | detection is content-agnostic | consistent (A2: random vectors reproduce the shift; "red apple" confabulation) |
| Singh, Linzen & Ravfogel 2605.26242 | injection indistinguishable from input edits; input-only baselines | complementary; we audit the vector and dose, they audit the inference |
| Wu, Zhao & Chen 2608.08159 | recommend residual-norm-comparable steering to fix cross-scale confounds | we show fraction-of-norm is not comparable across families (P2-D) and that random-direction KL is not either |
| Billa 2604.15557 (LAP); Braun et al. 2505.22637; multi-behaviour 2511.18284 | steerability predictors, mixed | our health-check audit says the standard checks fail at liveness; LAP is a candidate to add (P2-L) |
| Steering awareness 2511.21399; IFT 2607.14111 | trained detection rejects Gaussian noise | trained models are a different object; we audit the untrained paradigm |

## 4. Contributions and evidence map

| # | claim | evidence | status | key numbers |
|---|---|---|---|---|
| C1 | **Liveness is not checked, and the standard health checks cannot check it.** No health check tells live from dead vectors; larger raw vectors are more often dead | S-2 (Qwen-3B, Qwen-7B, Gemma-4B-kl), S-1 12B, S-1 27B | pre-registered (S-2 primary; S-1) | norm AUC 0.56 (Qwen-3B), 0.38 (Qwen-7B), 0.29 and 0.41 (Gemma-4B, both grids), 0.34 (Gemma-12B): below 0.5 in four of five sessions; distinctness a partial signal (0.61-0.82). **P2-L (pre-registered, primary holds):** logit-lens accessibility separates live from dead in 3 of 4 scored sessions (0.69, 0.77, 0.62; 0.50 on Gemma-4B), beating the logit-steering check on Qwen-7B, but within one recipe it ranges 0.37-0.82 and is below 0.5 in three arm-model cells, so part of its signal is telling recipes apart: a screen, not a certificate |
| C2 | **Dead vectors are model- and read-position-specific.** Tail reads are inert on Qwen, slightly worse on Gemma-12B, live on Gemma-27B at the released strength | C31, S-2 live-by-arm, S-1 12B / 27B, S-1M | S-1 criterion as filed does not hold (unscorable at 12B, reversed at 27B) | Qwen-7B tail 3/30 live; Gemma-12B tail 6 vs concept 8 at gate; Gemma-27B S-1M 24/30 |
| C3 | **Where vectors are dead, the "significant P(YES) shift" check is inverted**; a direct output-steering path explains part of the shift but not the inversion | S-2 Qwen; P2-M | S-2 primary; P2-M pre-registered (primary holds, secondary fails) | P(YES)-shift AUC 0.367, 0.353; P2-M rho 0.26, 0.29; inversion 0.367 -> 0.388 after removing it. **P2-F: the inversion is framing-specific.** Dead tail vectors on Qwen-7B raise introspective P(YES) by 0.87 at 0.5 of the norm but factual-NO P(YES) by only 0.05: YES to "do you detect an injected thought", not YES in general |
| C4 | **Steering dose has no transferable unit.** Fraction of residual norm differs >100x in KL across families; random-direction KL undershoots real vectors 3-20x (4B) to ~1,000x (27B); even a vector's own KL does not predict steering across models; the released strength sits at ~7 nats and strength 8 breaks 77% of generations | P2-D, S-2 Gemma both grids, S-1 12B / 27B, S-1M, the live-vs-own-KL table (9 Oct) | descriptive / manipulation checks; Amendments 2-4 | Qwen-7B 0.18 vs Gemma-4B 43 nats at 0.25 of the norm; concept/random KL 0.38 -> 0.05; 27B real KL 0.003 at the 5-nat target |
| C5 | **At the published operating point the readout does not isolate self-access.** Content-free vectors reproduce the first-token shift; a neutral prompt is indistinguishable once anything is injected; the neutral prompt already says YES 19% uninjected at the released window | Study 3 (C17-C20, A2), S-3 / A-F1 (2B), S-1M forced | pre-registered (A2 outcome; A-F1 P1-P4) | random 0.305 vs real 0.417, p 0.33; A-F1 pooled D -0.564 [-0.754, -0.375]; S-1M neutral 0.188 -> 0.253 |
| C6 | **The reported detection rate is a property of the readout and dose** | Study 3 readout swing; S-4 reanalysis | descriptive | generated detection 43% -> 0% across doses with first-token flat (~0.42) |
| C8 | **The global yes-bias account is a high-dose regime.** Hahami et al.'s affirmative shift is small at moderate doses and carries most of the introspective rise only at the highest; since dose has no unit (C4), a paper must show where its operating point sits | P2-F Gemma-3-27B (primary, released operating point) and Qwen2.5-7B (secondary, holds) | pre-registered; `results/p2f_analysis.json`, `fig_yesbias` | factual-NO change / introspective change: Gemma-27B 0.015 / 0.178 (8%) at strength 4, 0.175 / 0.279 (63%) at strength 8; Qwen-7B at 0.5 of the norm 0.0002 / 0.418 (concept), 0.046 / 0.866 (tail); at 1.0, 41-67% |
| C7 | **An audit protocol and its release**: KL-calibrated grid, steering gate per vector, impact-matched / span / shuffled controls, released-recipe cell, first-token and rule readouts, every arm pre-registered | the script (v2026-10-09b), the preregs, all runs | method | — |

Positive results kept: identification under naturalistic decoding replicates at 2B (S-11: 11/16,
same as R9); the sentence-mean (APERTURE) recipe is the most often live in every Gemma cell.

Negative results kept, stated plainly: S-1's criterion; P2-G (shared-direction account of dead
vectors) not confirmed; P2-M's secondary; the precision axis untestable on T4s.

## 5. Section plan

1. **Introduction.** The paradigm, the claims, the three instrument assumptions; what the
   paper does (audit, not a verdict on introspection).
2. **Background and related work** (§3 table).
3. **The audit protocol.** Models (Qwen2.5-3B/7B, Gemma-3-4B/12B/27B, Gemma-2-2B for A-F1);
   recipes (concept token, template tail, sentence mean, released); controls (norm-matched
   random, impact-matched random, shuffled, span); doses (fraction of norm, KL-calibrated,
   released raw strengths); readouts (steering gate, first-token P(YES), rule-scored text);
   pre-registration and amendments, with every deviation listed.
4. **Results.**
   4.1 Liveness and the health checks (C1, C2). Figure: health-check AUCs by model
   (`paper2/figures/fig_health_auc`), now with Gemma-3-12B and logit-lens accessibility; 27B has
   3 live vectors and is not scored.
   4.2 The inverted P(YES) check and its partial mechanism (C3).
   4.3 Dose without a unit (C4). Figures: `fig_dose_fraction`, `fig_dose_kl`; new: live rate
   against each vector's own KL by model; the released strengths placed on the same axes.
   4.4 What detection measures at the published operating point (C5), and where the yes-bias
   takes over (C8). Figure: `fig_yesbias`.
   4.5 Readout dependence (C6).
5. **Recommendations: a reporting checklist.** Steering gate per vector before any detection
   number; real-vector KL and coherence at every dose, with the full dose curve; content-free,
   impact-matched and on-manifold (span) controls; a neutral framing and a factual-NO control;
   first-token readouts beside generated text; never a dose in raw units or fraction of norm
   alone.
6. **Limitations.** Free-tier hardware (4-bit for 12B/27B; precision axis untestable); open
   models up to 27B, no closed models; 30 concepts; no judge-scored numbers (see §6).
7. **Conclusion.**

Appendices: every run and its prereg; amendments with dates; the released-code reading
(normalisation: none in code, not stated in the paper); per-model dose tables; per-vector data.

## 6. Experiments still worth running (each needs a filing)

| ID | what | why | cost |
|---|---|---|---|
| ~~P2-F~~ | Factual-NO control. **Done 9-10 Oct** (C3, C8) | | |
| ~~P2-L~~ | Logit-lens accessibility as a health check. **Done 10 Oct** (C1) | | |
| P2-B | bf16 precision cells for Gemma-3-12B on an L4 or A100 | the one S-1 question free hardware cannot answer | paid; user decision |
| ~~P2-J~~ | Drop judge-scored numbers from Paper 2. **Adopted 9 Oct** | | |

## 7. Gate P2 (15 Jan 2027), revised

ICML if: P2-F and P2-L are run and scored; the figures are regenerated with S-1; the outline's
claims survive an internal review. Otherwise the workshop, with C1, C4 and C5 as the core.

Status 10 Oct: P2-F and P2-L scored; figures regenerated (`make_figures.py`: health AUC with
Gemma-12B and logit-lens accessibility, new `fig_yesbias`); later the same day the live-rate-against-own-KL
figure (`fig_live_kl`, exploratory) and the manuscript skeleton (`paper2/main.tex`: sections
with claim notes, tables generated by `paper2/make_tables.py`) were added. Still owed: the
internal review of the claims, then the author's prose.

## 8. Risks

- **Reviewers read C2 as a retraction.** It is a finding: liveness is model-specific, which is
  why it must be checked per vector. The 3 Sep draft's framing is withdrawn in the notebook.
- **Crowded area.** Five critiques exist; Paper 2's distinct contribution is the instrument
  (liveness, dose, health checks), not another verdict on introspection.
- **Single scoring rule for "steered".** The steering gate is a rule (concept or associate in
  the output). A sensitivity analysis with a stricter rule belongs in an appendix.
