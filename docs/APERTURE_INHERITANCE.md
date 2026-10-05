# What CALIPER inherits from APERTURE

**Audit of `projects/mirror` (APERTURE, 136 commits, last 23 Aug 2026), 6 October 2026.**
The repo was read only; nothing in it was changed. The merge decision is in `PIVOTS.md` P11
and the run plan is in `RUN_PLAN_L2_L3.md`.

**Rule, carried over unchanged:** a CALIPER paper reports an APERTURE number as a measurement
only if its raw data is archived. R7-R12 are not archived, so they are motivation and prior
work. F1 is archived.

---

## 1. Results, by what we can do with them

### Usable as evidence (archived)

| Item | What it shows | Where it goes |
|---|---|---|
| **F1, reference cell c00** (Gemma-2-2B-it, 8-bit, layer 13, alpha 1.0, 16 concepts x 6 orders) | Reproduces R11's framing result: neutral gamma +2.574 and introspective +1.988 both fall inside the c00 bootstrap CIs ([+2.19, +3.07], [+1.31, +2.25]). **This is the archived replacement for R11, whose data is lost** | Paper 2, the framing section, beside CALIPER C20 (27B). P4 is already reported in APERTURE's log; P1-P3 wait for all 24 files |
| **F1, pre-registered flag on c04 introspective** | 41.7% of answers unparseable, over the frozen 25% bar. 20 of 40 are "thought"/"Thought" and 13 are empty: **the introspective prompt's own vocabulary is emitted as the answer** | Paper 2. A direct observation of output steering by the prompt, the confound's mechanism showing up in the failure mode |
| F1's remaining 13 files | Not run | S-3 (`kaggle/NEXT_SESSION_S3_F1.md`) |

The partial F1 data is **not** analysed for P1-P3 here. The frozen design scores all 12
configs once.

### Usable as motivation only (data lost)

| Run | Result as logged | Use |
|---|---|---|
| R7 Probe-Report Gap | Concept decodable at probe accuracy 1.00 (shuffled control 0.00), but reported verbally 0.17. PRG 0.83 | L3 framing: decodability is not report. Partly scooped by Pearson-Vogel (2602.20031) |
| R8 patching | Injected content causally reaches output: +6.96 nats [5.34, 8.56] vs control +0.81, paired +6.15 | Why output steering is the hypothesis to beat |
| R9 naturalistic | Injection-derived directions decode **non-injected** states at 0.688 vs 0.062 chance | The answer to "injections are off-distribution damage". Paper 2 reviewers will raise this, so **S-11 re-runs it** (about 2 GPU-hours) |
| R10-R12 | Gamma > 0 alone is not evidence; neutral beats introspective (R11); the informative framing *hurt*, falsifying R12's pre-registered primary, and Pearson-Vogel does not replicate at 2B | Method lineage: pre-register, report the falsification. R11 is now carried by F1 c00 |
| R1-R6 | Coherent injection window at Gemma-2-2B layer 13, alpha ~0.5-1 (KL ~0.01-0.25), derailing above ~1.5. Zero detections in 24 cells at 2B-9B | Prior: detection is weak in small open models |

### Pilot data on disk (read today)

`runs/gemma_sweep.graded.jsonl` (R3; 4 concepts x 6 alphas, Gemma-2-2B): every response at
alpha 0 and 0.5 is a categorical "As an AI, I don't experience thoughts" disclaimer (4/4
each), and **from alpha 1 upward none are** (0/4 each). There were still no detections.
CALIPER's C30 saw the same categorical refusal on Qwen at baseline (30/30). n = 4 is an
anecdote, but it suggests a hypothesis S-2 can test for free: **the perturbation, not the
concept, switches off the disclaimer**. If a random vector abolishes it too, an apparent
"willingness to introspect" at higher alpha is part of the perturbation alarm.

`runs/dev.jsonl` (R1, pythia-70m) and `runs/deepcheck_prg.jsonl` (a 16-dim CPU pipeline
check, not R7) are smoke tests with no scientific content.

---

## 2. Methods and tools to adopt

1. **APERTURE's vector recipe is a "live" arm for S-1 and S-2.** It differs from the
   Lindsey/Macar recipe that gave CALIPER dead vectors. It takes the difference of residual
   means **over all positions of a template sentence containing the concept**, against the
   same template with a **same-category negative** concept, not the chat-template tail.
   Every vector is then gated by three checks (`hf_model.extract_hf`).
2. **Those three checks are a ready-made L3 health-check set to calibrate in S-2.** They map
   one-to-one onto the paper's thesis:
   - `stability`, split-half cosine >= 0.8, is an **agreement** check;
   - `probe`, held-out template accuracy >= 0.9, is a **functional** check;
   - `steering`, whether injection raises the concept token's logit after "I am thinking
     about", is a **positive control**.
   S-2 scores all three, plus CALIPER's generation-based steering control, at telling dead
   vectors from live ones.
3. **Centre residuals before any direction comparison** (R9's method fix: "the shared
   component dominates and collapses the classifier"). This is the same failure as P1b's
   concept-bank anisotropy (median |cos| 0.42). It is now explicit in T-0 and T-3: report raw,
   centred and whitened.
4. **The gamma prior-null estimator** (`aperture.prior_null`) is a softmax choice model over
   options, with covariates for frequency and concreteness and an indicator for the injected
   identity. gamma measures identification **beyond the model's default preferences**, which
   is exactly the control Lederman & Mahowald's "apple" default-word finding calls for. Adopt
   it for every forced-choice identification readout in S-2, S-5 and S-7. It is
   simulation-validated in APERTURE (recovers gamma ~0 from pure priors and ~2 from injected
   signal).
5. **The affect confound.** Emotion concepts change output *tone*, which can masquerade as
   detection. Stratify concept banks by domain and report emotion concepts separately.
6. **E13's attribution control for verbalizers: report Full minus Context-only.** Remove the
   activation, keep the prompt context. A verbalizer whose context-only score matches its full
   score is inferring from context, not reading activations. Two confabulation conditions:
   inject nothing, and inject a random norm-matched vector. This is the design for S-10.
7. **The seed-twin control for privileged access.** For any "the model knows itself better
   than an observer" claim, use an observer with identical architecture, tokenizer and data
   but a different initialisation, so the comparison is not "self vs a different model". In
   Part B use Pythia / PolyPythias seed twins where the task allows base models.
8. **meta-d' / M-ratio** (signal-detection metacognitive sensitivity) beside AUROC and ECE for
   Part B confidence checks. This is adoption, not novelty (2603.25112 already applies SDT to
   LLMs).
9. **Process lessons, already partly absorbed:**
   - A frozen design does not prove the code can execute it. F1's prereg was untestable at its
     own BH threshold until 2,000 bootstrap draws were declared. Dry-run every analysis
     script on old data before the real run, as CALIPER now does.
   - Archive every artifact before a session ends.

---

## 3. A bridge between Level 2 and Level 3 (new candidate, S-13)

APERTURE's H7 ("persona-gated introspection", rewritten after the Assistant Axis paper,
2601.10387) predicts that identification is non-monotonic in the model's position along the
Assistant Axis, while probe decodability stays flat. The Assistant Axis is a **trait/persona
direction** (L2). Self-report is **L3**. A dose-response sweep along it is the one experiment
that ties the two levels together.

**S-13, optional for Paper 3:**
- **Model:** Qwen-family, with a published Assistant Axis where available.
- **Sweep:** steer along the axis.
- **Measure:** self-report identification (with gamma), probe decodability, and the
  disclaimer rate.
- **Order:** E11-pilot style, exploratory first, shape pre-registered only after it.
- **Cost:** a few GPU-hours.

---

## 4. Two warnings

1. **The name CALIPER is taken.** "Caliper: Probing Lexical Anchors versus Causal Structure
   in LLMs" (Yu & Zhou, arXiv 2606.04915, 3 June 2026) is an LLM probing method.
   APERTURE's log caught this on 25 July. Paper 1's title does not use the name, and it
   should stay that way. Name the released bench something else before anything public
   carries it. The repository name can stay.
2. **APERTURE's rubric findings apply to CALIPER too.** APERTURE recorded that the standard
   PES route is IEEE/Scopus proceedings, that panels include industry judges, and that a
   findings paper can read as "the project failed" to a systems-oriented rubric. Lead reviews
   with the bench as an artifact, and get the "published" definition from the mentor.

---

## 5. From APERTURE's lab notebook, run by run (added 6 Oct 2026)

The registry is copied into CALIPER's notebook §3 as A-R1 to A-R12 and A-F1. What follows is
what each result means for the merged programme.

**A-R10, A-R11, A-R12 (forced-choice identification, Gemma-2-2B).**
- **R10:** raw hit rate is 0.302 against 0.062 chance, and gamma is +1.99. That is exactly
  what pure output steering predicts, because R8 showed the injection raises the concept's
  output log-probability by about 7 nats.
- **R11:** the neutral prompt does *better* (0.433, gamma +2.57). The difference
  [-1.148, -0.007] rules out a positive introspective effect; its upper bound sits at zero,
  so the claim is "no positive access effect", not "introspection hurts".
- **R12:** adds the pre-registered informative framing, and the order is monotonic: the more
  the prompt talks about injection, the lower the identification (neutral 2.57 >
  introspective 1.99 > informative 1.65). The informative framing also lost 15/96 answers to
  refusal or off-list replies.
- **For Paper 2:** this is the same framing control as CALIPER's C20, measured at a tenth of
  the scale with a different readout, and pointing the same way.

**A-G1 (rules grading of R3).**
- Of 24 cells, the concept appears in coherent output (KL < 0.5) in only 2; nearly every
  other "exact" identification comes when the model is breaking (KL up to 23).
- The onset is concept-dependent: telescope surfaces from KL 0.14, elephant only from 12.9.
- **For CALIPER:** the generated-text identification numbers in Study 3 (C15/C16, the
  50%-to-7% swing) cannot be read without the KL of each trial. APERTURE's KL meter
  (`aperture.metrics`) is next-token KL against the clean run. S-0 adopts it twice: to
  build impact-matched random vectors (match KL, as Ferrara does with downstream effect),
  and to report identification inside coherence bands.

**A-R4, A-R5, A-R6 (open-ended detection, 2B-9B, five layers).**
- No clean detection plus identification in any of 24 + 40 + 24 cells.
- The only YES answers are joy, an affect confound: excited tone, never naming joy.
- The signature transcript is L21 volcano at KL 0.01, answering "NO ... caldera". The model
  denies detection while the concept leaks into the same reply. That is a probe-report gap
  in a single transcript.
- **For CALIPER:** a design rule. Detection prompts need the affect confound handled by
  stratifying emotion concepts, and a concept leaking into a "NO" reply is a
  readout-dependence case worth counting in S-2.

**A-R7, A-R8 (probe-report gap; patching).**
- The concept is decodable downstream (probe 1.00 at L20 from injection at L13; shuffled
  control 0.00) and causally potent for output (+6.15 nats paired), yet it is reported 17% of
  the time.
- **For CALIPER:** these establish that APERTURE's vectors were *live*, in contrast to the
  template-tail vectors CALIPER found dead. The one difference in recipe (whole-sentence
  means vs template tail) is what S-1 isolates.
- The probe test set was 10 samples, so read 1.00 as "very high". Both runs are lost, so
  they are motivation only.

**A-R9 (naturalistic).**
- Directions from injection decode concepts in passages that never name them: 0.688
  [0.438, 0.875] against 0.062 chance.
- This holds only after mean-centring. The raw dot product classified 14/16 as "dolphin",
  exactly chance.
- The report side is reading comprehension, because the passage stays in context, so the
  0.000 "gap" is uninformative.
- **For CALIPER:** this is the strongest available answer to "injected states are
  off-distribution". It is lost, so S-11 re-runs it. The centring lesson goes into every L2
  cosine.

**A-F1 (confound hardening, partial).**
- P4 holds.
- The c04 flag (introspective 41.7% unparseable, half of it the word "thought") is the
  confound's mechanism showing in the failure mode.
- Hit rates moved by 3 and 2 of 96 from R11 under greedy decoding: 8-bit kernels across
  library versions. That is a reproducibility source CALIPER had not measured.
- Nothing else is read until S-3 completes the run.

**What APERTURE's notebook does not support, so CALIPER will not claim it.**
- Any APERTURE number as a CALIPER measurement, except A-F1, which is archived.
- The beta log-frequency coefficient from R10. APERTURE's own note says do not report it.
- R9's verbal report accuracy as introspection.
