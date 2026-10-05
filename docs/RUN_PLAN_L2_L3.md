# Run plan, Levels 2 and 3, with APERTURE merged in

**Plan of record from 6 October 2026.** It covers every run needed for the trait level (L2)
and the self level (L3), the APERTURE runs folded into CALIPER, the order, the compute,
the gates, and who owns what. Level 1's remaining runs (B-14 to B-19) are in
`LAB_NOTEBOOK.md` §7.8 and are only summarised here so the calendar is complete.

Background for every decision below is in the 6 October literature pass
(`projects/reports/CALIPER persona and self levels.md`) and the citation ledger
(`docs/CITATIONS.md` §21-22).

---

## 0. The one design that every level follows

The paper's claim is a measurement-science claim, so every level is the same experiment:

1. **A reference value** that is exact, or planted so that the model itself reads it.
2. **An instrument** that practitioners use at that level.
3. **The ground-truth-free checks** people use to decide whether to trust the instrument.
   These come in two kinds: *agreement* checks (precision: do repeated or alternative runs
   agree?) and *functional* checks (does the answer satisfy a relation the truth must
   satisfy, measured on held-out data?).
4. **A failure label**: the instrument's answer is wrong against the reference.
5. **A difficulty dial** that keeps the failure rate between 20% and 80%. Without one, an
   AUC has nothing to separate.
6. **The primary endpoint**: AUC of each check at predicting failure, with DeLong and a
   paired bootstrap between the best agreement check and the best functional check.
   Calibration and PR-AUC are reported beside it.

**The ordinal prediction across levels:** agreement checks carry some signal but are beaten
by functional checks. At L1 this held (restart 0.78 vs held-out R2 0.94; pooled gap -0.152
over six arms). One known counterexample exists at L3: sample agreement was well calibrated
for answer confidence (Torrielli 2605.26045). The design allows agreement to win, and the
paper says where it does.

**Standing rules carried from L1:**
- Pre-register each run's criterion and failure branch before it runs.
- Save every fitted direction or vector, not only scalars.
- Download every Kaggle artifact before the session ends.
- Run a steering positive control before believing any injection number.
- Report equivalence (TOST) rather than reading a large p-value as "the same".

---

## 1. What APERTURE contributes

APERTURE (`projects/mirror`, package `aperture`) is merged into CALIPER at the paper level.
Its repository stays where it is and keeps its own history; CALIPER uses `aperture` as a
library for L3 runs (`pip install -e ../mirror`) and registers APERTURE runs in its own
notebook under the prefix **A-**.

| APERTURE item | State | Use in the merged programme |
|---|---|---|
| R1-R6 (pipeline, steering, sweeps, detection null at 2B-9B) | R1, R3 data only | Prior: detection is weak in small open models. Cited, not measured |
| R7 PRG 0.83, R8 patching +6.15, R9 naturalistic 0.688, R10 gamma | **raw data lost** | Motivation only. Re-run cheaply (S-11) if any becomes load-bearing |
| R11 framing control, R12 three-framing prereg (falsified) | **raw data lost** | Replaced as evidence by F1's reference cell, which reproduces R11's gammas inside their CIs (P4 holds) |
| **F1 confound hardening** (Gemma-2-2B, 12 OFAT configs x 2 framings) | **11 of 24 files archived and hash-verified** | **A-F1, finished as S-3.** Frozen prereg, scorer exists (`aperture/f1_score.py`) |
| F2-F5 (covariates, PRG CV, audit arm, null robustness) | not run | F4(a), the literature audit of which published claims separate framing from construct, folds into Paper 2's related work. The others are cut |
| **E13 / PLANTED** (verbalizer recovery, attribution, confabulation) | designed, `readouts.py` not built | **S-10**, the verbalizer calibration arm of Paper 3 |
| F9 32B arm | compute-blocked | Superseded: CALIPER already ran Gemma-3-27B |
| F11 human grading + kappa | not started | Needed for any judge-scored number. Assigned in §6 |

**Where the two programmes already overlap, and how that resolves.** The framing control is
measured twice: A-F1 at 2B with forced choice over 16 concepts, and CALIPER C20 at 27B with
a preamble-matched neutral prompt. These are reported together as one result at two scales.
APERTURE's thesis, that output steering confounds self-knowledge claims, becomes one section
of Paper 2. It is not a separate preprint.

---

## 2. Level 2, the trait level

**Instrument:** difference-of-means trait vectors (DiffMean / CAA / the persona-vector
pipeline), with a logistic probe, LDA (whitened) and an SAE-feature route as alternatives.

**Checks to calibrate.** Agreement: split-half cosine, bootstrap cosine, stability across
prompt templates, stability across layers, and method disagreement. Functional: held-out
probe accuracy, held-out projection-predicts-label R2, Braun's activation-difference
consistency and class separation (SNR), and steering effect per unit norm. That last one is
a positive control only in T-2, where it is nearly the reference itself.

**Why P1 and P1b failed, which sets the rules below.** A planted direction is ground truth
only if the model itself reads or writes it. P1's random plants had no reader. P1b's
concept bank was anisotropic (median |cos| 0.42), so the same-bank null (0.187) sat above
recovery (0.156). Rule 1: every L2 reference is a direction the model provably uses. Rule 2:
every cosine is reported raw and whitened, against a null built in the same space.

### T-0 · Trait harness (engineering, CPU, about 1 week)

`caliper/traits.py`:
- token-set trait definitions;
- contrastive prompt generation from templates;
- every extraction route above;
- every check above;
- whitening (activation covariance, and Park et al.'s causal inner product);
- the LayerNorm-null projection;
- direction saving.

Tests:
- a planted linear trait in synthetic activations must be recovered exactly;
- whitening must be idempotent;
- every check must run on a 2-trait toy.

**Gates T-2 and T-4.**

### T-1 · Exact estimand grid, the precision control (CPU, about 2 hours)

A fully enumerable template grid, so the DiffMean over every combination is known exactly,
then subsamples at 8-256 prompts. This scores how well split-half and bootstrap estimate
**precision** (they should do well), as the deliberate contrast with T-2 where the same
checks are asked about **trueness**. The paper's L2 point is this contrast.

*Criterion:* split-half cosine predicts distance-to-exact-estimand with Spearman > 0.8. *Failure
branch:* if precision checks fail even at precision, report that, and T-2 compares against a
weaker baseline.

### T-2 · L2a, the exact readout bench — **the primary L2 run** (CPU, 1-3 days)

**Reference:** for a trait expressed as a choice between token sets A and B, the
final-layer direction that reads it is exact. It is
w = g ⊙ (mean U_A - mean U_B), where U is the unembedding and g the final norm gain, with
the final-LN null direction projected out.

**Models:** GPT-2 small and Pythia-160m first (CPU). Then SmolLM2-360M and Qwen2.5-0.5B.
Qwen and SmolLM2 are preferred over Gemma, whose final-logit soft-capping needs handling.

**Traits:** at least 200, from four families:
1. sentiment word sets;
2. yes/no and agree/disagree;
3. topic vocabularies;
4. language and register pairs (English/French function words, formal/informal).

Each trait has trait-specific answer tokens. A/B multiple choice is avoided, because it
gives every trait the same final-layer direction.

**Instrument run:** contrastive contexts where the model predicts set A versus set B. Extract
at the final layer by every route, and align to w (raw, whitened, LayerNorm-null projected).

**Difficulty dials:**
- prompts per trait (8, 16, 32, 64, 128);
- template heterogeneity (1, 3, 10 templates);
- set size;
- extraction at layers L-1 to L-4. At those layers the reference is approximate (direct path
  plus a tuned-lens or average-gradient estimate), so they are secondary.

**Failure label:** alignment < tau, swept from 0.80 to 0.95. The primary tau is fixed in the
prereg after a 20-trait pilot sets it inside the 20-80% band.

**Primary endpoint:** DeLong, best agreement check (split-half cosine) minus best functional
check (held-out probe accuracy), across about 200 traits x settings, clustered by trait.
Precision at 200 traits: AUC SE about 0.033.

**Gate A (31 Dec 2026):** the failure rate lies in 20-80% at the primary setting, and at
least one check's AUC CI excludes 0.5. If not, Paper 3 drops L2 to a limitations paragraph
and becomes an L3 calibration paper.

**Prereg:** `docs/preregistration-t2-readout-bench.md`, filed after the pilot, before the full
sweep.

### T-3 · Whitened rescore of P1b (Kaggle, about 2 GPU-hours, optional)

P1b archived recoveries but not vectors (C57 could never run). This means re-running P1b
on Gemma-3-27B with the current script (version 2026-09-08e or later saves vectors), then
recomputing recovery and both nulls in whitened space, with the shared bank direction
removed. It answers whether P1b's failure was partly an anisotropy artefact.

*Failure branch:* if whitened recovery still sits at the null, P1b stands as a real negative
on generated-text extraction at late layers, which corroborates 2602.06801 with planted truth.

### T-4 · L2b, trained-in trait directions (Kaggle, 10-20 GPU-hours) — Gate B

**Model:** Qwen2.5-0.5B-Instruct (or SmolLM2-360M-Instruct).

**Implant:** for each trait, choose a direction v, with inter-trait cosines set by a
collinearity dial from 0 to 0.6. Fine-tune with LoRA on three losses:
1. produce the trait;
2. swap the trait when v is swapped (interchange intervention, DAS-style);
3. be invariant to directions orthogonal to v.

Mix in general data and decoy directions so the traits are not unrealistically easy.

**Certify:** a trait is kept only if patching along v flips it AND ablating v removes it.
The kept set is the ground truth.

**Instrument run:** the persona-vector pipeline, end to end. Generate trait-eliciting and
baseline responses, then take the DiffMean over response tokens. Then the alternatives.

**Pilot first:** 4 traits, one collinearity level, about 2 GPU-hours. Then 64-96 traits.

**Gate B (mid-March 2027):** at least 32 certified traits. If not, L2b is reported as a
limitation and L2 rests on T-2.

### T-5 · Toy persona transformers (CPU, optional)

Small transformers trained from scratch on documents written by agents with latent persona
traits and chosen correlations. These give exact posterior labels and a clean collinearity
sweep. Run only if T-4 fails Gate B and the collinearity question is still open.

### T-6 · Model organisms, the realism check (Kaggle, a few GPU-hours, optional)

- **Taboo models:** the secret token's unembedding is an exact read direction.
- **SaTML 2024 trojan models:** give exact trigger labels.

This checks whether T-2's calibration holds on a realistic case. It is expected to be easy,
because defection probes reach AUROC > 0.99, so it is a sanity check, not a headline.

---

## 3. Level 3, the self level

L3 splits in two. **Part A** audits the injection instrument; it goes into Paper 2 and
absorbs APERTURE's thesis. **Part B** calibrates self-report trust checks against exact
truth; it goes into Paper 3 with L2.

### Part A · Auditing the introspection instrument (Paper 2)

**S-0 · Merge the apparatus (engineering, about 1 week).**
- Install `aperture` as a library.
- Make one injection path with three gates built in:
  - vector health checks;
  - the steering positive control as a hard gate per vector (inject on a neutral prompt; the
    concept must appear in output above baseline);
  - impact-matched random vectors (matched on downstream KL, as Ferrara), beside norm-matched
    and shuffled ones.
- Every run saves its vectors.
- Add to the Kaggle bundle and to the bundle parity test.

**S-1 · Dead-vector precision and position ablation (Kaggle, 6-10 GPU-hours).** This is the
reviewer's first demand.
- **Grid:** read position (chat-template tail vs concept token) x precision (4-bit NF4,
  8-bit, bf16).
- **Models:** Gemma-3-12B-it, where bf16 fits 2xT4, and Gemma-3-27B-it at 4- and 8-bit.
- **Layer:** the published 0.6 depth.
- **Setup:** Macar's released configuration where it can be obtained (read their code for the
  token index and dtype first).
- **Outcome:** the steering-control pass rate per cell.

*Criterion:* tail vectors fail the steering control at a rate above concept-position vectors
in every precision cell. *Failure branch:* if tail vectors pass in bf16, the dead-vector
result is a quantisation artefact, and the paper reports it as that. That is still a
warning, but a narrower one.

**S-2 · The instrument audit on small models (Kaggle, about 15 GPU-hours).**
- **Models:** Qwen2.5-3B/7B-Instruct and Gemma-3-4B-it.
- **Vector classes:** live (passes the gate), dead (template tail), random norm-matched,
  random impact-matched, shuffled, span (on-manifold).
- **Readouts on the same trials:** first-token P(YES), generated text scored by rule, and
  generated text scored by a small local judge, validated against human labels (S-12).
- **Primary:** AUC of each standard health check (unit norm, finiteness, distinctness,
  significant P(YES) shift) at telling dead from live vectors.
- **Prediction:** each health check scores about 0.5, and the steering gate about 1.
- **Secondary:**
  - dose-response curves per class;
  - readout disagreement on identical trials;
  - TOST equivalence of random versus live at the published operating point.

**Additions from the APERTURE audit (6 Oct, `docs/APERTURE_INHERITANCE.md`):**
- S-1 and S-2 gain APERTURE's vector recipe as a third extraction arm: whole-sentence
  residual means against a same-category negative.
- S-2 scores APERTURE's three-flag gate (stability = agreement, probe = functional,
  steering = positive control) beside CALIPER's generation-based steering control.
- S-2 records the "As an AI" disclaimer rate against alpha for live and random vectors,
  testing whether the perturbation, not the concept, switches the disclaimer off.
- Forced-choice identification in S-2, S-5 and S-7 is scored with APERTURE's gamma
  prior-null estimator.
- Concept banks are stratified by domain, with emotion concepts reported separately
  (affect confound).
- S-11 (R9 naturalistic re-run) is promoted from optional to planned, since Paper 2's
  reviewers will raise the off-distribution objection.
- S-13 is a new optional run bridging L2 and L3: a dose-response sweep along the
  Assistant Axis.

**S-3 · A-F1, finish APERTURE's confound hardening (Kaggle, about 10 GPU-hours, free tier).**
The 13 remaining files run under the frozen prereg (`mirror/docs/prereg/2026-07-30-f1-confound-hardening.md`),
scored once with `aperture/f1_score.py`. No interim scoring of the 11 files already in.
Together with C20 it gives the framing control at 2B and 27B.

**S-4 · Study 3 reanalysis (offline, CPU, a day).**
- TOST equivalence for every "indistinguishable" claim.
- Exact CIs: 2/30 is about 1-22%, so it does not contradict Macar's 10.8%.
- Dose-response curves from C15-C24 and C45-C52.
- One table of every arm's vector status: dead (C15-C24 tail) or live (C45 onward).

No new data.

**Paper 2 is S-1 + S-2 + S-3 + S-4 plus the existing Study 3 data.** Title direction:
auditing the instrument behind introspection claims.

### Part B · Calibrating self-report trust checks against exact truth (Paper 3)

Every task here has an exact answer computed from the model's own internals or outputs.

**Checks to calibrate:**
- Agreement: mode frequency over 10 samples, paraphrase consistency, seed or temperature
  stability.
- Other: verbalized confidence; forced-choice or logit scoring, the functional check.

**Report beside each task:** a privileged-access index, the model's accuracy minus the best
outside observer's accuracy on the same text.

**Models:** Qwen2.5-0.5B/1.5B/3B/7B-Instruct and Gemma-3-1B/4B-it. Ground truth is always
computed on the same quantised model that reports.

| ID | Task | Exact truth | Cost |
|---|---|---|---|
| **S-5** | Self-prediction of its own answer | its own logits on the object-level prompt | ~4 GPU-h |
| **S-6** | Grammaticality judgement (BLiMP minimal pairs) | its own string probabilities (Song, Hu & Mahowald code) | ~2 GPU-h |
| **S-7** | Which of N inputs was perturbed, and which injection was stronger | the injection log, using S-2's live vectors only | ~4 GPU-h |
| **S-8** | Source: injected vector vs misleading prompt vs nothing | the condition | ~2 GPU-h |
| S-9 | Its own sampling temperature (negative control) | the setting | ~1 GPU-h |
| **S-10** | APERTURE's PLANTED: can a verbalizer (logit lens, Patchscopes-style, SelfIE-style) name a planted concept, for the right reason, without confabulating when nothing is planted? | the plant | ~8 GPU-h; needs `readouts.py` |
| S-11 | Re-run of APERTURE R7/R8/R9 if any becomes load-bearing | as designed | ~2 GPU-h |

**S-12 · Human grading and judge validation (labour, no GPU).**
- About 200 stratified transcripts, two labellers each.
- Cohen's kappa, human-human and judge-human.
- Gates every judge-scored number in Papers 2 and 3.
- **Start in October.** It is pure labour and the item most likely to slip.

**Primary endpoint for Part B:** per task, DeLong of the best agreement check minus the
forced-choice/logit check at predicting a wrong self-report, pooled across tasks with
random effects as at L1.

---

## 4. Order and calendar

GPU and CPU work run in parallel. Kaggle is about 30 GPU-hours a week per account, and is
otherwise idle while L1 runs on the CPU.

| Window | CPU (laptop) | Kaggle GPU | Writing and labour |
|---|---|---|---|
| **Oct, weeks 1-2** | B-14 analysis; L1 re-runs on the fixed estimator; T-0 harness; S-4 reanalysis | **S-3 (A-F1 finish)**; S-0 merge | S-12 labelling starts; Paper 1 skeleton |
| **Oct, weeks 3-4** | T-1; T-2 pilot (20 traits) -> T-2 prereg | S-1 dead-vector ablation | Paper 1 draft |
| **Nov** | T-2 full sweep; L1 transfer run (SAE / transcoder) | S-2 instrument audit | **Paper 1 to arXiv + TMLR (late Nov)** |
| **Dec** | T-2 analysis | T-3 (optional) | Paper 2 draft. **Gate A, 31 Dec** |
| **Jan 2027** | — | T-4 pilot; S-5, S-6 | **Paper 2 to workshop (~1 Feb)** |
| **Feb - mid-Mar** | T-5 only if needed | T-4 full; S-7, S-8, S-9; `readouts.py` then S-10 | **Gate B, mid-March** |
| **Apr - May** | — | Confirmatory reruns only | **Paper 3 (NeurIPS E&D, ~early May; TMLR fallback)** |

APERTURE's binding constraint carries over: **all new science ends by March 2027**, and April
onward is writing and hardening only.

**Compute total (estimates; every run pilots first):**
- L2 Kaggle: 15-25 GPU-hours.
- L3 Part A: 30-35 GPU-hours.
- L3 Part B: 20-25 GPU-hours.

About 70-85 GPU-hours over five months, under a fifth of one account's free quota.

---

## 5. Gates and what each one decides

| Gate | When | Passes if | If it fails |
|---|---|---|---|
| B-14 | this week | primary per its decision table | L1 headline rewritten per that table; Paper 1 scope reassessed |
| S-1 | early Nov | tail vectors fail the steering gate in bf16 too | dead vectors reported as a quantisation artefact; Paper 2 leans on S-2 and A-F1 |
| A-F1 | when finished | frozen P1-P3 | reported as filed; the framing claim narrows to the scales where it holds |
| **Gate A** | 31 Dec | T-2 failure rate 20-80% and some check's CI excludes 0.5 | Paper 3 becomes L3-only |
| S-12 | before any judge number is reported | kappa >= 0.6 human-human | judge numbers drop to secondary; rule-scored and first-token readouts carry the claims |
| **Gate B** | mid-March | at least 32 certified implanted traits | L2b is a limitation; L2 rests on T-2 |

---

## 6. Ownership (proposal, for the team to settle)

| Owner | Lane |
|---|---|
| Santosh | L1 and Paper 1; the B-series; integration and statistics across levels |
| Member 2 | L2: T-0 harness, T-1, T-2, then T-4 |
| Member 3 | L3 Part A: S-0 to S-3 on Kaggle; Paper 2 lead |
| Member 4 | S-12 human grading; citation-ledger verification (the MEM rows the papers lean on); L3 Part B tasks S-5, S-6, S-9 |

---

## 7. What this plan deliberately does not do

- No 32B+ runs beyond what CALIPER already has.
- No training-from-scratch beyond T-5's toy, which is conditional.
- No API judges. Grading uses rules, first-token readouts and a small local judge validated by
  S-12.
- No new L1 science after the transfer run. L1 is in its writing phase.
- APERTURE's F2, F3 and F5 are cut, because the merged programme does not need them.
