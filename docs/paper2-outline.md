# Paper 2 outline, re-planned after S-1 (9 October 2026)

**Status.** Structure, claims and evidence only; the prose is the author's.

> **Internal review, 10 Oct 2026: Major Revision (5 of 5).** Rulings, verified against the rows:
> **C8 withdrawn as written** (on the log-odds scale injection moves factual-NO answers more than
> the introspective answer: +14.75 vs +5.45 at strength 4 on Gemma-27B, 27/30 concepts);
> **C5 corrected** (the p 0.33 result is at alpha 6; at the published alpha 4 real beats random,
> p 0.004; at S-1M the neutral prompt does not respond to injection); **C1 overstated** (the norm
> sentence is a concrete-vs-abstract concept confound; distinctness works, 0.73-0.82);
> **C3 narrower** (concept arm on both Qwen models, tail arm on Qwen-7B only; drop
> "framing-specific"); C4's direction was written backwards (fixed below). Roadmap: offline A1-A9,
> literature B, new compute C1-C4 (one Gemma-27B session with log-odds, factual-YES and
> baseline-matched controls, random controls on every framing). The claim table below is not yet
> revised; the roadmap governs. **Updated 10 Oct:**
P2-F and P2-L scored and folded in (C1, C3, C8); figures regenerated; Gate P2 status in §7.
**Updated 11 Oct: P2-R scored** (`results/p2r_analysis.json`, `results/p2r_secondaries.json`); C1, C4,
C5 and C8 revised below; `tab_p2r` and `fig_p2r` added. Supersedes the
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
vector is live, the dose is comparable, the readout reports self-access). Calibrated against
controls with known answers, each holds in some regimes and fails in others, and no health check
in use tells which regime a given experiment is in. The paper states this as a
regime-by-assumption table (model, recipe, dose, window, readout against live / comparable /
self-access), so that every cell is a checkable claim rather than "fails in some regime" (review
DA-7).

## 3. Position in the literature (ledger: `CITATIONS.md` §26-29)

| work | what it shows | how Paper 2 differs |
|---|---|---|
| Lindsey 2025/26; Macar et al. 2603.21396 | detection at moderate rates, 0% false positives; Gemma-3-27B L37, alpha 4 best; DPO-dependent circuit | we run their released recipe, strength and (P2-R) prompt inside a calibrated audit; S-1M is a 4-bit reconstruction of a bf16 original |
| Hahami et al. 2512.12411 (Llama-3.1-8B; FULL, 10 Oct) | binary detection is explained by a global shift in logit(YES) - logit(NO), measured against factual-NO controls (r = 0.999); partial introspection only on differential tasks and early layers. Claude is not mentioned (the earlier attribution is withdrawn) | our corrected C8 agrees with them on the log-odds scale at the released point on Gemma-3-27B; P2-R adds the factual-YES arm that separates a yes-bias from flattening, which their NO-only control cannot |
| Lederman & Mahowald 2603.05414 | detection is content-agnostic | consistent at alpha 6 (A2), not at the published alpha 4, where real beats random (C5 as revised) |
| Singh, Linzen & Ravfogel 2605.26242 | injection indistinguishable from input edits; input-only baselines | complementary; we audit the vector and dose, they audit the inference |
| Ferrara 2608.20569 (FULL-MAIN, 10 Oct) | eight open models, intervention-vs-sham pairs scored by d' and AUROC: report discrimination at chance (0.5007); impact-matched random directions on two models; a LoRA model trained to report reaches AUROC ~1.0 | the closest critique, and it already has SDT scoring, impact matching and a trained positive control. Paper 2's distinct part: per-vector liveness gating, dose transfer across families, the health-check audit, and the published operating point and prompt |
| Pearson-Vogel et al. 2602.20031 (FULL, 10 Oct) | detection of prior injections from the KV cache, read from P(yes)/P(no); factual controls shift under 0.2% on the probability scale | readout sensitivity is not new; their factual controls are read on the scale C8's correction shows to hide large log-odds shifts |
| Zou et al. 2609.35108 (FULL-MAIN, 10 Oct) | gate and router heads for localisation (44-66%, chance 10%) on Qwen3-4B, Llama-3.1-8B, Gemma-3-12B; dose alpha x ||h|| chosen per model; norm-matched random controls; first-token readout | doses set as a fraction of the activation norm, which C4 shows does not transfer; no per-vector liveness check; a published small-model setting the audit could re-run (review EIC F5) |
| Godet 2025 (confusion; localization) | in the ledger, not yet positioned | position against C5 (random close to real is not new) |
| Wu, Zhao & Chen 2608.08159 | recommend residual-norm-comparable steering to fix cross-scale confounds | we show fraction-of-norm is not comparable across families (P2-D) and that real vectors are 3-1,000x gentler than random ones at equal norm |
| Tan et al. 2407.12404; Braun et al. 2505.22637; Billa 2604.15557 (LAP); Bas & Novak 2511.18284; Davarmanesh et al. 2602.00333 | steering reliability and steerability predictors | per-vector liveness screens tested as such (C1, P2-L); none holds within recipe and concept type |
| Dalili et al. 2607.10517; Aparin & Gaintseva 2606.06735 | steering geometry; named by the review as dose-calibration work (relevance to verify) | position against C4's "no transferable unit" |
| Steering awareness 2511.21399; IFT 2607.14111 | trained detection rejects Gaussian noise | trained models are a different object; we audit the untrained paradigm |

Every row above is ABS level in `CITATIONS.md`; a specific claim is cited only after a full read
(the ledger's own rule; review item B).

## 4. Contributions and evidence map

Revised after the internal review of 10 Oct (`results/p2_review_analysis.json`,
`results/p2_rule_sensitivity.json`). "Filed" marks a pre-registered endpoint; everything else is
exploratory.

| # | claim | evidence | status | key numbers |
|---|---|---|---|---|
| C1 | **Liveness is not checked, and no health check certifies it.** Pooled across recipes and concept types some checks look useful (distinctness 0.78, 0.82, 0.73; logit-lens accessibility 0.69, 0.77, 0.62), but with recipe and concept type held fixed none is consistent (distinctness 0.05-0.86, logit-lens 0.14-0.75, logit steering 0.36-0.75 across cells of 20 vectors). Raw norm sits at chance once concept type is fixed: the earlier "larger vectors are more often dead" was a confound (the steering rule's associates cover only the 20 concrete concepts; abstract vectors are larger and rarely scored live). At the published point (S-1M, 24 live / 6 dead) no check finds the dead vectors (low power). The identify-stage rule (log P(concept) up >= 2 nats) is not specific: random vectors pass it 21/30 at strength 4, real 27/30; it agrees with the steering gate at kappa 0.10 | S-2, S-1 12B, S-1M, P2-L, P2-R | S-2 primary filed (its "norm near 0.5" prediction held); P2-L filed after labels were known; the within-cell analysis is exploratory | concrete-only norm AUC within recipe 0.27-0.71, all p > 0.1; live concrete 31/60, 31/60, 9/60 against abstract 3/30, 1/30, 0/30 |
| C2 | **Liveness is model- and read-position-specific**, reported as the failed S-1 prediction: tail reads are inert on Qwen, and the released recipe is live on Gemma-27B at its released strength; the KL-calibrated 27B grid stopped below the dose onset | C31, S-2, S-1, S-1M | S-1 criterion as filed does not hold; the 12B direction reverses under a literal-word steering rule | Qwen-7B tail 3/30; S-1M 24/30; 27B grid ceiling 11.9% of the norm against an onset near 14% |
| C3 | **The "significant P(YES) shift" health check is inverted on Qwen**: live vectors shift less | S-2 Qwen; P2-M | filed (S-2 primary statistic); per session not significant after Holm, Fisher-combined p 0.0019 | AUC 0.367, 0.353; concrete only 0.337, 0.329; concept arm 0.38, 0.34; tail arm 0.57 (3B), 0.35 (7B); P2-M explains part of the shift, not the inversion |
| C4 | **Steering dose has no transferable unit.** At the same norm, real vectors move the next token 3-20x (4B) to ~1,000x (27B) less than random directions; fraction of the norm differs about 480x in KL across families at 0.25 and about 4x at 1.0; one vector at one strength gives 0.0006 to 12 nats depending on the prompt; even a vector's own KL does not put models on one curve | P2-D, S-2, S-1, S-1M, `fig_live_kl` | descriptive / manipulation checks | S-1M at strength 4: factual-NO 0.0006, introspective 3.4, story 7.3, neutral 12.1 nats; at 1-2 nats the share steered is 0.43 [0.34, 0.51] on Qwen-3B against 0.10 [0.02, 0.19] on Gemma-4B (vector-clustered intervals); the released strength 4 is past the dose where Gemma-27B stops answering YES/NO (median mass 0.93 at strength 2, 8e-5 at 4; P2-R) |
| C5 | **What the first-token readout measures depends on strength and window.** At alpha 6 (Study 3, older window) content-free vectors produce the whole shift (p 0.33, filed primary); at the published alpha 4 real beats random (0.504 against 0.252, p 0.004). With the released prompt and window (P2-R), real beats impact-matched random only where the model has mostly stopped answering (strength 4: +16.3 vs +10.9 log-odds, p 1.9e-4, content-free share 0.67); at strengths 1-2, where YES/NO keep 0.997 and 0.93 of the mass, they cannot be told apart (p 0.63, 0.22; shares 0.76, 0.80). The released neutral prompt barely responds (AUC 0.56-0.63 at strengths 1-4) | Study 3 (C17-C20), S-4, S-1M, S-2 Qwen, P2-R | filed primary (A2) at alpha 6; P2-R P3 filed; the rest descriptive | content-free share 0.43, 0.50, 0.73 at alpha 2, 4, 6 (Study 3); P2-R shares 0.76, 0.80, 0.67, 0.80 at strengths 1, 2, 4, 8 |
| C6 | **Generated-text detection rates move with dose while first-token P(YES) does not**, largely because generations degrade; descriptive, appendix | Study 3; S-4 | descriptive | generated detection 43% to 0%; first-token 0.42-0.50 |
| C8 | **Injection flattens confident answers; it does not add a global yes-bias. The introspective answer moves more than baseline-matched factual questions predict, but most of that excess is content-free.** P1: factual-YES falls 4.93 log-odds at strength 4 (p 0.011) after +1.07 at strength 1 (p 0.014); impact-matched random flattens more at every strength. P2: introspective excess +4.98 [+2.57, +9.13] at strength 2, +6.90 [+3.22, +10.24] at 4; norm-matched random +3.44 [+1.73, +5.67] and +4.79 [+2.65, +6.76]; impact-matched random +0.90 and +2.76, intervals spanning 0. At strength 4 YES+NO mass is 8e-5 on the introspective prompt, so strength 2 is the only operating point that supports a readout claim. P2-F's raw reading (factual-NO moves more, S-1M prompt) stands for that prompt; with the released prompt the raw ordering reverses (introspective +16.3, factual-NO +11.2) | P2-F, P2-R | P2-R P1 and P2 filed; random-vector P1/P2 are filed secondaries added to the scorer after the primary output was seen | see claim; probability scale: factual-YES median change 0 (p 0.063) |
| C7 | **An audit protocol and its release**: KL-calibrated grid, steering gate per vector, impact-matched / span / shuffled controls, released-recipe and released-prompt cells, first-token log-odds with YES/NO mass, factual-YES/NO/contested controls, every arm pre-registered | the script (v2026-10-10a), the preregs, all runs | method | none |

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
| ~~P2-R~~ | The released protocol on Gemma-3-27B on log-odds. **Done 11 Oct** (C1, C4, C5, C8): flattening, a mostly content-free introspective excess, real > impact-matched random only at strength 4, answer mass collapses at the released strength | | |
| ~~P2-Q~~ | The same design on Qwen2.5-7B. **Ran 11 Oct; manipulation check failed**: with the released prompt Qwen puts at most 1% of first-token mass on YES/NO in every framing, so no endpoint was scored (`results/p2q_concept.json`, `results/p2q_tail.json`). Evidence for C5/C7 (the readout is defined only for models that answer in the format), not for C8 | | |
| ~~P2-Q2~~ | P2-Q with "Start your answer with YES or NO." (Amendment 1). **Ran 11 Oct; mass check passed, check 4 failed**: factual-YES baseline median log-odds -0.39 (15/30 lean YES) against factual-NO -17.7, so the controls are unusable and no forced endpoint was scored. Identify random control: 20/30 at 0.5, 27/30 at 1.0 (concept vectors 29/30, 30/30 in P2-Q). C8 stays Gemma-only; Qwen's two failed checks are C5/C7 evidence (the frame, not the question, drives the answer) | | |
| P2-H | A Llama-3.1-8B cell, Hahami et al.'s model | shows a published conclusion changing (review EIC F5) | one Kaggle session; optional |
| P2-K | A known-YES positive control: a model trained to report injections | validates the readout against a reference that should say YES (review R3) | training on Kaggle; optional |
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

Status 10 Oct, after the internal review (Major Revision): the offline roadmap items are done
(`results/p2_review_analysis.json`) and folded into section 4; C8 is withdrawn as written, C5
corrected, C1 restated at the recipe-by-concept-type level, C3 narrowed. Gate P2 now also needs
P2-R run and scored. ICML if it lands and its outcome supports a coherent readout claim;
otherwise the workshop, with C1, C2, C4 as the core.

Status 11 Oct: P2-R run and scored. Its outcome supports a coherent but narrower readout claim:
the forced-choice readout at the released strength is read where the model no longer answers,
and at the dose where it still answers the introspective excess is mostly content-free. C1, C2,
C4 stay the core; C5 and C8 are now instrument findings about the readout, not about
introspection. Before Gate P2: P2-Q on Qwen-7B, then a second internal review of the claims.

Status 11 Oct, later: P2-Q ran and failed its manipulation check (Qwen does not answer YES/NO to
the released prompt: mass at most 0.01). C8 therefore rests on Gemma-3-27B alone; P2-Q2 (an
answer instruction, filed as an amendment) is the way to get a second family. Next: the second
internal review of the claims.

Status 11 Oct, evening: P2-Q2 also failed a manipulation check (factual baselines not opposite in
sign), so C8 stays Gemma-only. No further Qwen session is planned for C8; the two failures go into
C5 and the checklist as evidence that a forced-choice readout in this frame must be validated per
model with factual controls. Next: the second internal review of the claims.

## 8. Risks

- **Reviewers read C2 as a retraction.** It is a finding: liveness is model-specific, which is
  why it must be checked per vector. The 3 Sep draft's framing is withdrawn in the notebook.
- **Crowded area.** Five critiques exist; Paper 2's distinct contribution is the instrument
  (liveness, dose, health checks), not another verdict on introspection.
- **Single scoring rule for "steered".** The steering gate is a rule (concept or associate in
  the output). A sensitivity analysis with a stricter rule belongs in an appendix.
