# Programme plan, October 2026 to September 2027: three papers and a flagship

**Written 9 October 2026.** Supersedes the venue and paper lines of `PAPER_STRATEGY.md` §2 and
the Paper 3 calendar in `RUN_PLAN_L2_L3.md`. Paper 1's content is unchanged. Everything here
is a plan: no experiment below is filed until its own pre-registration exists.

---

## 1. Why the scope grows now

Paper 1's runs took about two months, and 10 to 12 working months remain. Three things changed
in that time.

- **The evidence went past Paper 1's claim.** Paper 1 says one check is weaker than another.
  The October data say why, give two ground-truth-free signals that see the hardest failures,
  and give a repair. That is a mechanism, a detector and a fix.
- **The field is converging on the question.** 2025-26 papers ask practitioners to report
  stability as a matter of course, and others argue, without ground truth, that stability is
  not correctness (§3). Nobody has calibrated the checks on real trained models.
- **The neighbours are close.** A theory paper on steering-vector non-identifiability and an
  audit of SAE benchmarks that calls for ground truth both appeared in 2026. Paper 1 should be
  out first, and the larger result should follow while the gap is still open.

---

## 2. What we know: the findings, read together

| # | finding | where | status |
|---|---|---|---|
| 1 | Restart agreement predicts failure worse than held-out fit: pooled dAUC -0.134 [-0.189, -0.078], 6 conditions | Paper 1 | confirmatory + replications |
| 2 | The lead holds when the attainable fit varies by unit (dAUC -0.219) | N-1 | pre-registered |
| 3 | Failures split in two. **Under-fitted** units are sparse (firing 2-4% vs 12-21%): a statistical limit. **Converged-wrong** fits look like passing units: a geometric limit | Paper 1, explore 8 Oct | secondary + exploratory |
| 4 | Converged-wrong fits differ from w where the stimulus has almost no variance (81-96% of error energy in the bottom 1%); equivalent on fresh text | Paper 1 §identifiability | exploratory |
| 5 | **Two signals see them**: the fitted direction's low-variance share (AUC 0.94-0.97) and agreement across token samples (0.91-0.92). Restart agreement does not (0.47-0.85) | explore 8 Oct | exploratory; X-1 tests on new units |
| 6 | **A repair**: projecting out the low-variance part fixes 40 of 43 converged-wrong fits | explore 8 Oct | exploratory |
| 7 | Restart agreement is confidently wrong when identifiability collapses (19 of 20 failures at >= 0.99, GPT-Neo L6) and noisy elsewhere (186 false alarms of 254 passes) | explore 8 Oct | exploratory |
| 8 | Estimator pre-processing can manufacture failures: one near-constant coordinate, magnified by standardisation | B-17 | B-17b/c/d test it |
| 9 | Single-unit verdicts are fragile (19-25% flip) but mostly near the bar | B-15 | secondary |
| 10 | Sparse targets (SAE latents) defeat the estimator at our token budget (3/100) | T-SAE | pre-registered, failed |
| 11 | Dead steering vectors steer nothing yet pass health checks; the "significant P(YES) shift" check is inverted on two Qwen models; larger raw vectors are more often dead on all three models | Paper 2, S-2 | pre-registered |
| 12 | A fraction-of-residual-norm dose does not transfer across model families (Gemma saturates at ~45 nats where Qwen sits below 1) | S-2 Gemma | manipulation check; Amendment 3 |
| 13 | **Dead vectors lie more along the direction every concept shares** (live-vs-dead AUC 0.23-0.37, i.e. 0.63-0.77 inverted, on three models) | explore 9 Oct | exploratory; filed for S-1 / S-2-kl |

**The synthesis.** Every failure we have found that the standard checks miss is a failure of
*where in the representation the estimate lives*, not of how well it fits:
- a fitted direction in the stimulus's unexplored directions (4-7);
- a coordinate the data cannot constrain (8);
- a steering vector along the component every concept shares (11, 13);
- a dose measured against a norm that the model's computation largely ignores (12).

Checks that re-run the same computation (restarts, seeds) cannot see this, because the
computation is deterministic given the data. Checks that change the data, or that look at the
geometry of the answer against the geometry of the data, can. That is one claim, and it spans
both papers.

---

## 3. Where the literature stands (9 Oct 2026 pass; ledger in `CITATIONS.md` §26-27)

- **Stability is being recommended as routine practice.** Circuit discovery: report stability
  metrics routinely (Méloux, Portet & Peyrard, arXiv 2510.00845). SAEs: feature consistency
  across seeds (song2025, Paulo & Belrose ICLR 2026; only ~30% of latents shared across seeds).
- **Stability is being doubted without ground truth.** SAEBench audit: a metric can be
  low-noise and still measure the wrong thing; the remedy proposed is ground truth on
  synthetic models (Chanin, arXiv 2605.18229). Unstable SAE latents concentrate in
  reproducible subspaces, i.e. basis ambiguity (arXiv 2606.12138).
- **Non-identifiability is argued in theory.** Steering vectors have large behaviourally
  equivalent classes along the activation covariance's null space (Venkatesh & Kurapath,
  arXiv 2602.06801). No detector, no repair, no ground truth. Circuits are non-identifiable
  (Méloux et al., ICLR 2025).
- **Ground-truth benchmarks plant the truth by construction**: InterpBench, MIB, Pando
  (arXiv 2604.11061), ObserverBench (arXiv 2609.03026), synthetic SAEs. None takes the truth
  from the weights of a pretrained model.
- **Introspection is replicated and disputed on open models**: mechanisms (Macar et al.,
  arXiv 2603.21396, the source of our `macar` recipe), content-agnostic detection (Lederman,
  arXiv 2603.05414), latent introspection (arXiv 2602.20031).
- **Steering dose.** KL-budget calibration exists as an informal write-up (iso-KL) that
  proposes per-model calibration as a hypothesis; optimal strength differs several-fold
  across 7B models (Riemannian steering, arXiv 2607.10517). No cross-model evidence.

**Scoop check.** No work found that calibrates reliability checks against weight-derived
ground truth in pretrained models, or that tests an interventional repair for an
identifiability failure. One search pass cannot prove absence; repeat monthly with the
`sweep-arxiv` routine and before each submission.

---

## 4. The programme

| paper | claim | venue | submit |
|---|---|---|---|
| **Paper 1** (L1) | Restart agreement is a weaker check than held-out fit; the hard case is identifiability | **TMLR** + arXiv | **by 20 Nov 2026** |
| **Paper 2** (L3) | The checks used to trust concept-injection results fail: dead vectors, an inverted P(YES) check, a dose that does not transfer | **ICML 2027** if Gate P2 passes, else an interpretability workshop | late Jan 2027 |
| **Flagship** (L1 broad + L2 exact) | Stability checks are blind to sampling error; data-resampling and geometric checks see it; interventional data repairs it. Shown on six weight-derived substrates and the methods people use | **NeurIPS 2027 main track**; Evaluations & Datasets as fallback | May 2027 |
| **Paper 3** (L2, L3 planted) | Trait- and self-level instruments calibrated with the flagship's checks | ICLR 2028 or NeurIPS 2027 E&D | Sep 2027 |

**Working title for the flagship:** *Stable Is Not True: Calibrating Interpretability's
Reliability Checks Against Exact Ground Truth in Trained Models.* Bench name pending (VOUCH /
WARRANT shortlist).

**Gate P2 (15 Jan 2027).** ICML only if S-1 (both Gemma sizes) and S-2 Gemma-kl have landed,
pass their manipulation checks, and S-3/S-11 are written up. Otherwise the workshop, and the
dose result moves to the flagship.

---

## 5. Flagship experiments (F-series)

Each gets a pre-registration before it runs. Compute: CPU = the laptop queue; Kaggle = 2xT4.

### Theory

**F-0. Why seed checks are blind (CPU, days).**
- Claim: if the fit is a deterministic function of the data up to a basin, restarts sample
  only the basin, and their agreement carries no information about estimation error. Two
  independent data samples give E|v1 - v2|^2 = 2 tr(Sigma) while E|v - w|^2 = tr(Sigma) +
  bias^2, so data-resampling disagreement estimates the variance part of the error and is
  blind to bias. For a linear link, Sigma scales as sigma^2 C^-1 / n, so error lives in the
  low-eigenvalue directions of C, which is what the low-variance share measures.
- Deliverable: a short appendix theorem plus a synthetic single-index simulation with known C
  that reproduces findings 4-6 and predicts which failures remain invisible (bias-type).
- It also defines the identifiable target, the projection of w onto C's well-sampled
  subspace, which the repair of finding 6 recovers.
- **A prediction already visible in Paper 2 data.** Dead steering vectors are a bias-type
  failure: the read position makes them consistently wrong, not noisy. F-0 therefore predicts
  that resampling-based checks are weak on them, and S-2's second-template stability check
  was (AUC 0.53-0.61 pooled over arms on three models; 0.36-0.41 within the concept and
  sentence arms on the two Qwen models), while a geometric check does better (P2-G). The theory note
  states this split explicitly: resampling sees variance, geometry sees where the answer lives,
  nothing ground-truth-free sees a pure bias that lives where the data are rich.
- Grounding: linear causal representation learning proves observational data insufficient and
  one intervention per latent sufficient for identifiability (Seigal, Squires & Uhler, ICML
  2022); F-1 is the measurement of that result on a real trained unit.

### Remedies

**F-1. Interventional data breaks the identifiability ceiling (CPU, about a week of queue).**
- Units: GPT-Neo L10 (B-8b, 20 converged-wrong) and the GPT-2 L6 re-fits (B-15a/b/c).
- Arms, each adding the same number m of extra samples:
  1. **targeted interventions**: stimuli s + d with d drawn in C's bottom-1% eigen-subspace,
     scaled to the typical stimulus norm; the response is the unit's exact function of s;
  2. **random interventions**: d drawn isotropically;
  3. **more natural tokens**: m fresh tokens from disjoint documents.
- Prediction: arm 1 recovers at least 80% of converged-wrong fits and harms no passing unit;
  arm 3 recovers few; arm 2 is in between.
- A realistic variant perturbs the residual before the layer norm, as a practitioner would.
- Failure branch: if arm 3 does as well as arm 1, the limit is sample size, not observation,
  and the flagship's remedy claim becomes "more data", reported as such.

**F-2. Targeted sampling for sparse units (CPU + Kaggle).**
- Under-fitted units are sparse. Add top-activating contexts with importance weights against
  uniform extra tokens of the same count. Re-run T-SAE with it.
- Prediction: the required token count scales as 1 / firing fraction; targeted sampling
  reaches it at a fraction of the cost. Two failure classes, two remedies.

### Substrates (ground truth read from the weights)

| ID | substrate | truth | link | models | compute |
|---|---|---|---|---|---|
| (done) | GELU MLP neurons | W_in column | GELU | GPT-2, GPT-Neo, Pythia | — |
| **F-3** | ReLU MLP neurons | W_in column | ReLU | OPT-125m/350m/1.3b; 6.7b, 13b | CPU; Kaggle |
| **F-4** | SwiGLU / GeGLU neurons | span(gate, up) | product | **Gemma-3 ladder** 270M, 1B (CPU); 4B, 12B (Kaggle); Qwen2.5-0.5B as a second family | CPU; Kaggle |
| **F-5** | MoE routers | router row per expert | softmax / top-k | OLMoE-1B-7B | Kaggle |
| **F-6** | Unembedding rows | W_U column | linear | any model | CPU |
| **F-7** | SAE / transcoder encoders | encoder row | JumpReLU | **Gemma Scope 2** (every layer of every Gemma-3 size, residual SAEs and MLP transcoders, 16k and 262k widths): 270M and 1B on CPU | CPU + Kaggle |
| **F-8** | Exact L2 trait readout | w = g * (mean U_A - mean U_B) | linear | GPT-2 (`caliper/traits.py`) | CPU |

- F-4 needs a two-direction alignment (principal angles between the fitted plane and
  span(gate, up)); the estimator already fits k = 2.
- F-6 isolates identifiability: a linear link has no bad basins, so every failure there is
  geometric.
- Each substrate needs a failure rate between 20% and 80% (the difficulty dial) to score
  checks: token budget, layer and unit sparsity are the dials.

### Methods and checks

**F-9. Methods panel.** On every substrate: our single-index fit; ridge / OLS; a logistic
probe on firing; difference-in-means (the steering-vector recipe); STA; best-matching SAE
latent where an SAE exists. The question is which checks work, not which method wins.

**F-10. The calibration table.** For every substrate x method, AUC at predicting failure of:
seed / restart agreement; data-resampling agreement (document-disjoint halves); held-out
fit; agreement between methods; the low-variance share; control-task selectivity for probes;
an interventional consistency check (does steering along v-hat move the unit as predicted,
including off the data manifold). Primary endpoint per cell: data-resampling agreement vs
seed agreement, DeLong, pooled by random effects.

**F-11. Scale.** Pythia-6.9B and 12B (GELU), OPT-13B (ReLU), Qwen2.5-7B (SwiGLU) on Kaggle,
50-100 units each. Turns "124M to 1.4B" into "up to 13B".

### Application

**F-12. Audit of vectors in use (no ground truth).** Apply the checks F-10 validated to
directions people publish: the refusal direction, persona vectors (Chen et al., released
pipeline), the concept vectors of the introspection-mechanisms release (Macar et al.), and our
own (S-1, S-2). Which pass the checks that work? Bridges the flagship to Papers 2 and 3.

**F-13. Which checks see which failure: the bias-variance split, across papers (offline).**
Classify every failure type found so far (under-fitted, converged-wrong, standardisation
artifact, dead vector, saturated dose) as variance-type or bias-type by F-0's test (does it
move when the data are resampled?), and report each check's AUC by type. Prediction:
resampling checks catch variance-type only; geometric checks catch both where the answer lives
in poorly sampled or shared directions. Uses existing data plus F-1 and P2-G.

---

## 6. Paper 2 additions

- **P2-G. Dead-vector geometry (filed 9 Oct, before data).** The shared-direction share
  predicts dead vectors; tested on S-2 Gemma-kl and S-1. Pre-registration:
  `preregistration-p2g-dead-geometry.md`.
- **P2-D. Dose transfer.** KL-calibrated dosing against fraction-of-norm across the three S-2
  models. Needs the Gemma-kl session; Qwen calibrated re-runs are optional.
- Existing: S-1, S-2, S-3, S-11, S-12 as in `RUN_PLAN_L2_L3.md`.

---

## 7. Calendar

| month | Paper 1 | Paper 2 | Flagship |
|---|---|---|---|
| **Oct 2026** | X-1, B-17b/c/d land; TBDs; author prose; review round 2 | S-2 Gemma-kl, S-1 Gemma-12B and 27B on Kaggle | F-0 simulation; F-1 prereg, then run after queue 6 |
| **Nov** | **Submit TMLR + arXiv by 20 Nov** | analyses; S-3/S-11 write-up | F-1 analysis; F-6 and F-3 (small) on CPU |
| **Dec** | — | draft | F-4 small, F-2; **Gate F-A (15 Dec)** |
| **Jan 2027** | TMLR reviews | **Gate P2 (15 Jan)**; submit ICML or workshop | F-5, F-7 on Kaggle; **Gate F-B (31 Jan)** |
| **Feb-Mar** | revisions | — | F-9 / F-10 full panel; F-11 scale; **Gate F-C (15 Mar)** |
| **Apr** | — | rebuttal / camera-ready | F-12 audit; theory write-up; draft |
| **May** | — | — | **Submit NeurIPS 2027** |
| **Jun-Sep** | — | — | rebuttal (Aug); **Paper 3** runs and draft, submit Sep |

### Gates

- **F-A, 15 Dec.** F-1 result. Pass: targeted interventions recover >= 80% of converged-wrong
  fits and beat natural tokens. Fail: the flagship leads with detection (findings 5-7, F-10),
  and the remedy is reported as tested and negative.
- **F-B, 31 Jan.** At least three new substrates reach a 20-80% failure rate with the
  estimator recovering >= 50% of units. Fail: the flagship keeps the substrates that work and
  says which do not, and why.
- **F-C, 15 Mar.** The calibration table is filled for at least 4 substrates x 3 methods. Fail:
  NeurIPS Evaluations & Datasets instead of the main track.

---

## 8. Compute budget

- **Laptop CPU**: F-0, F-1, F-3 small, F-4 small, F-6, F-8, and the small-model halves of F-9 /
  F-10. About 4 minutes per GPT-Neo unit; keep the queue full overnight with the lid-close fix.
- **Kaggle (~30 GPU-hours a week, about 1,200 hours to September)**: Paper 2 needs ~30-40 h;
  the flagship's F-5, F-7, F-11 and the large halves of F-3 / F-4 need ~80-120 h. Comfortable,
  if sessions are batched by model.

---

## 9. Risks

| risk | likelihood | mitigation |
|---|---|---|
| A competing paper calibrates checks against weight-derived truth first | medium | Paper 1 on arXiv by 20 Nov; monthly scoop check |
| The estimator fails on SwiGLU or sparse substrates, as on SAE latents | medium | GPU fits with 50-100k tokens; F-2's targeted sampling; Gate F-B |
| Interventional repair works only because it adds samples | medium | F-1's natural-token arm is the control |
| Scope creep: six substrates x six methods x seven checks | high | the table needs 4 x 3 to be a paper (Gate F-C); the rest is appendix |
| Paper 2 data late or failing manipulation checks again | medium | Gate P2; the workshop fallback keeps the date |
| Writing load: every paper's prose is the author's own | high | numbers, structure and checks supplied as each result lands; one paper drafting at a time |

---

## 10. What is cut or deferred

- A frontier closed model as a subject: impossible without white-box access.
- Attention QK/OV directions as a substrate: bilinear in two positions, no single exact
  direction per unit. Revisit only if F-4's two-direction machinery generalises.
- Training-dynamics analysis over Pythia checkpoints: interesting, not on the critical path.
- A pip package for the checks: after Gate F-C, if the table is clean.

---

## 11. Document map

- This plan: `docs/flagship-plan.md`.
- Paper 1 state: `paper1/main.tex`, `docs/paper1-hardening-plan.md`.
- L2/L3 runs: `docs/RUN_PLAN_L2_L3.md` (Paper 3 dates now follow §7 here).
- Results and decisions: `docs/LAB_NOTEBOOK.md` (§7.10 points here).
- Citations: `docs/CITATIONS.md` §26-27.

---

## 12. Experiment register for the three remaining papers (second pass, 9 Oct 2026)

A second research pass (sources in `CITATIONS.md` §28) changed several designs. This section
is the register: every experiment still to run for Paper 2, the flagship and Paper 3, with what
changed and why. Where it disagrees with `RUN_PLAN_L2_L3.md`, this section wins.

### 12.1 What the second pass found

1. **The published concept-vector recipe reads the template tail.** The released code for
   Macar et al. (introspection-mechanisms, read 9 Oct) builds each vector as the activation at
   the last token of the chat-templated "Tell me about {concept}" prompt minus the mean over 100
   baseline words, at layer fraction 0.7 by default (the core 27B run uses layer 37 of 62), unnormalised,
   scaled by strength 1-8 (default 8, core run 4), and injected from the token before "Trial"
   through every generated token, in bf16. That read position is our template-tail arm, the
   arm C31 and S-2 found dead-prone. S-1's 27B tail cell therefore tests the published recipe.
2. **"A Reality Check" (Singh, Linzen & Ravfogel 2026)** already shows that models cannot tell
   an injected state from an edited input, and that an input-only classifier matches
   hidden-state self-report. Paper 3's planned S-8 (source discrimination) duplicates the first
   result; the second sets the outside-observer baseline every self-report task must beat.
3. **Gemma Scope 2** ships SAEs and transcoders for every layer of every Gemma-3 size
   (270M-27B). One model family can carry the flagship's GeGLU (F-4) and encoder (F-7)
   substrates on CPU, and its scale axis links to Paper 2's Gemma models.
4. **Introspection adapters** (Anthropic, Apr 2026) and a follow-up train models to report
   fine-tuned behaviours; report rates swing from 1.0 to 0.0 by behaviour and configuration.
   Self-reports of implanted behaviours are a planted-truth task Paper 3 can calibrate.
5. **Persona vectors' own caveat**: monitoring correlations (r 0.75-0.83) shrink when prompt
   type is controlled. "Persona Non Grata" (2604.11120): the imbuing method changes the trait.
6. **MoE routing statistics do not predict causal importance** (ICML 2026 causal audit). The
   router row is still an exact readout; F-5's question is recovery, not importance.
7. **Paper 2's weak stability result fits a bias-type failure.** See F-0 above.

### 12.2 Paper 2 (L3 instrument audit)

| ID | experiment | status | change on 9 Oct |
|---|---|---|---|
| S-1 | Dead vectors: read position x precision, Gemma-3-12B (4-bit, 8-bit, fp16) and 27B (4-bit) | filed; Amendment 2 (KL grid) | **adds S-1M**, below |
| **S-1M** | **The published operating point.** Gemma-3-27B 4-bit, layer 37, template-tail vectors exactly as released (100 baseline words, unnormalised), strength 4 and 8, injected from the token before "Trial" through generation. Outcome: steering-gate pass rate and next-token KL, beside our KL-calibrated cells | **new; prereg addendum to S-1 before the 27B session** | needs `--inject-from trial` and `--vector-recipe macar-release` in the script (S-0b) |
| S-2 | Instrument audit, three small models | Qwen done; Gemma-kl pending | none |
| P2-G | Dead-vector geometry (shared share) | **filed 9 Oct** | none |
| **P2-M** | **The mechanism of the inverted P(YES) check.** Logit-lens projection of each saved vector onto (YES - NO) unembedding through the final norm, against its measured P(YES) shift and its live label | **new; filed before analysis** (`preregistration-p2m-output-steering.md`) | offline: needs only the unembedding and final-norm weights |
| P2-D | Dose transfer: fraction-of-norm vs KL-calibrated, across Qwen-3B/7B, Gemma-4B and S-1's Gemma cells, plus S-1M's published strengths on the same KL scale | planned | uses S-1M |
| S-3 / S-4 / S-11 | A-F1 finish; Study 3 reanalysis; naturalistic re-run | S-3 and S-11 done; S-4 offline | none |
| S-12 | Human grading, one labeller with a blind re-label | not started | none |

Engineering S-0b (before S-1's 27B session): `--inject-from trial` (inject from the token
before "Trial" through generation, as released) and `--vector-recipe macar-release`
(last-token read, 100 baseline words, no normalisation). Tests on the tiny Llama.

**Paper 2's framing after the pass.** Three replications and one critique exist on open
models. None audits the instrument: whether the vectors are live, whether the health checks
can tell, whether the dose transfers. Paper 2's contribution is that audit, with the published
recipe as one of its arms.

### 12.3 Flagship

As §5, with the changes above: Gemma-3 ladder for F-4 and F-7, F-0's bias/variance prediction
and its grounding, F-12's targets, and F-13. **F-1 is filed** (`preregistration-f1-interventional.md`).
Order on the CPU after queue 6: F-1, then F-6 (unembedding, cheapest substrate), then F-3 small
(OPT-125m/350m), then F-4 and F-7 on Gemma-3-270M.

### 12.4 Paper 3 (L2 traits and L3 planted self-report)

| ID | experiment | change on 9 Oct |
|---|---|---|
| T-0 / T-1 | Trait harness; exact-estimand precision control | none |
| **T-2** | Exact readout bench (w = g * (mean U_A - mean U_B)) | **shared with the flagship as F-8**; run once, reported in both (flagship: one substrate row; Paper 3: the L2 result in full) |
| T-3 | Whitened P1b rescore | optional, unchanged |
| **T-4** | Trained-in trait directions (LoRA) | **two imbuing arms**: the LoRA implant and a system-prompt induction of the same trait, since the method changes the trait (Persona Non Grata). Adds the persona-vector monitoring check **within prompt type**, the paper's own caveat |
| T-5 / T-6 | Toy persona transformers; model organisms | unchanged, optional |
| S-5 | Self-prediction of its own answer | **outside observers fixed now**: an input-only classifier (Reality Check), the same-size sibling model, and a model trained on the object-level behaviour (Binder et al.). The privileged-access index is self minus the best of the three |
| S-6 | Grammaticality judgement against its own string probabilities | unchanged |
| S-7 | Which input was perturbed, which injection stronger | live vectors only, from S-2/S-1 (KL-calibrated doses) |
| **S-8** | Source discrimination | **narrowed**: Reality Check showed models cannot tell injection from input edits. S-8 keeps only the calibration question (do agreement checks predict a wrong source report?) with a KL-matched input-edit arm, and cites them for the capability |
| S-9 | Own sampling temperature (negative control) | unchanged |
| S-10 | PLANTED verbalizer calibration | **runs on Gemma-3 with Gemma Scope 2 transcoders** (the features Macar et al. analyse), 1B on CPU first |
| **S-14** | **Self-report of implanted behaviours** | **new**: the T-4 organisms are asked what they were trained to do (direct question, and an introspection-adapter-style probe). Truth = the implant. Calibrates agreement checks (sample mode, paraphrase consistency) against the forced-choice check. Reuses T-4's models, so no extra training |
| S-12 | Human grading | shared with Paper 2 |

**Paper 3's primary endpoint is unchanged**: per task, DeLong of the best agreement check minus
the forced-choice / logit check at predicting a wrong self-report, pooled with random effects.

### 12.5 Compute, revised

| paper | CPU | Kaggle GPU-hours |
|---|---|---|
| Paper 2 (S-1 incl. S-1M, S-2 Gemma-kl, P2-D) | P2-G, P2-M offline | ~30 |
| Flagship (F-1..F-13) | F-0, F-1, F-3 small, F-4/F-7 at 270M-1B, F-6, F-8, F-13 | ~90 |
| Paper 3 (T-2..T-4, S-5..S-10, S-14) | T-0..T-2, S-10 at 1B | ~50 |
| **total** | | **~170 of ~1,200 available** |
