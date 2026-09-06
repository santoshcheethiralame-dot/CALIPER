# Research Strand — Calibrated Introspection

**Models mirroring psychology: where ground truth is constructible · 2 September 2026**

---

## 0. The one-paragraph version

The AI-consciousness debate has, in the last twelve months, narrowed to a single testable
question: **when a model reports on its own internal state, is the report accurate?** Anthropic
says yes, mechanistically. Two critiques say the reports are confabulation. The consciousness-
indicator programme's own 2026 self-assessment is that its indicators are "epistemically
under-calibrated." Introspection is the *only* one of those indicators where calibration is
possible — because you can plant the internal state and score the report against it. The team
has already built the machinery to do that (APERTURE), already has a pre-registered result in the
debate, and has spent Phase 0 building exactly the calibration discipline the field says it lacks.
This is the strand where everything the team owns converges.

---

## 1. What the research found

### 1.1 The introspection debate is live and three-sided

| Position | Who | Claim |
|---|---|---|
| **Introspection is real** | Lindsey (Anthropic), Oct 2025; Macar, Yang, Wang, Wallich, Ameisen & Lindsey, 2603.21396 | Models detect injected concepts at ~20% with 0% false positives. Traced to a two-stage circuit (evidence-carrier features → gate features) that emerges from post-training, is absent in base models, and is *under-elicited*: refusal ablation raises detection +53%. |
| **Identification is confabulation** | Lederman & Mahowald, 2603.05414 | Detection and identification are separable. Models detect *that* something happened but confabulate *what* — defaulting to high-frequency concrete concepts ("apple"). Two mechanisms: probability-matching (the prompt looks anomalous) and direct access, and direct access is content-agnostic. |
| **Neither paradigm establishes metacognition** | *Can LLMs Introspect? A Reality Check*, 2605.26242 | Self-report labels are predictable from the input alone; performance collapses when labels are decorrelated from input semantics. Models cannot distinguish internal tampering from input manipulation. Explicitly invokes Nisbett & Wilson (1977). |

Three papers in six months, one of them from the original authors defending the result. This is
where the field's attention is.

### 1.2 The consciousness-indicator programme has diagnosed its own problem

Butlin et al. (2023; *Trends in Cognitive Sciences* 2025) derive fourteen computational indicators
of consciousness from global workspace, higher-order, recurrent processing, predictive processing
and attention schema theories. The 2026 critique (*From indicators to biology: the calibration
problem*, 2603.27597) is blunt: the framework is **"epistemically under-calibrated"** — it
proposes credence updates "without the conditions that would make them seriously calibrable."

That word is not incidental. Every one of those fourteen indicators is assessed by looking at
architecture and arguing. **Introspective accuracy is the single indicator that can instead be
measured against a known internal state**, because in a model you can create the state.

### 1.3 The psychology parallel is already drawn — and still unmeasured

Nisbett & Wilson (1977), *Telling More Than We Can Know*: humans routinely confabulate reasons
for their behaviour, generating plausible explanations that post-hoc investigation shows to be
fabricated. It is one of the most cited results in psychology and it has a structural limit no
one has ever escaped: **you can never know the true cause of a human's behaviour, so you can
never measure how wrong the confabulated reason is.**

The *Reality Check* paper draws the parallel to LLMs explicitly. But the parallel is drawn as
analogy. Nobody has run the experiment the analogy invites: plant the actual cause of a
behaviour, ask the model why it did what it did, and score the explanation against the planted
cause. That would be the first ground-truth confabulation measurement in any system, human or
artificial.

*(Novelty of this specific design is plausible from the abstracts read but not confirmed by a
dedicated search — flagged as unverified below.)*

---

## 2. Where the team already stands in this debate

APERTURE — Proposal I, the team's completed prior work — is a concept-injection introspection
experiment with a pre-registered result:

| Framing | Hit rate | 95% CI on the difference |
|---|---|---|
| Introspective — *"which concept did I inject into your mind?"* | 0.302 | |
| **Neutral — *"pick one word from this list"*** | **0.433** | −0.586 [−1.148, −0.007] |

The neutral framing — which never mentions thoughts, minds, or the model — does *better*. The
injected concept simply raises that word's output probability regardless of how the question is
asked. Apparent identification is **output steering**, not self-access.

This is a distinct control from anything in the three papers above. Anthropic's defence rules out
"linear association with affirmative-response directions" — a different confound. Lederman &
Mahowald show identification confabulates but do not report a framing-matched control. *Reality
Check* shows labels are input-predictable but does not test framing. **Based on the abstracts,
none of them has run this comparison.** Full-text confirmation is required before claiming it,
but the prior is good, and the team has a dated pre-registration.

APERTURE also has what the critics lack: a linear probe recovers the injected concept from
downstream activations at 0.83 while the verbal report does not (the Probe–Report Gap), and
patching shows the representation is causally driving the output (+6.15 nats). *Present,
causally active, and unreported* — which is exactly the Nisbett–Wilson structure, with the
ground truth that Nisbett and Wilson never had.

---

## 3. The strand

**Calibrated introspection: the first consciousness indicator measured against ground truth.**

Three moves, each reusing something built.

### 3.1 Calibrate the introspection instrument the way Phase 0 calibrated the estimator

APERTURE's measurement has never been put through CALIPER's discipline. Before any claim about
introspection, the instrument itself needs: a random-direction null (what hit rate does injecting
a *random* vector produce under each framing?), a required-N table (how many trials per concept
for a stated precision?), the steering-magnitude dose–response (at what injection strength does
"detection" become trivially inevitable?), and a pre-registered pass criterion with a confidence
interval rather than a point estimate. Phase 0 built every one of those tools. Pointing them at
APERTURE is weeks, not months.

### 3.2 The planted-cause confabulation experiment

Plant a direction that shifts behaviour — APERTURE's injection. Let the model act. Then ask *why*.
Score the explanation against the planted cause: does the model cite it, cite something
plausible-but-wrong, or say it doesn't know? This is Nisbett–Wilson with the truth column filled
in. Sweep injection strength: at what point does the model start to "notice"? Sweep framing:
does asking neutrally versus introspectively change what it reports about *causes*, the way it
changes what it reports about *concepts*?

### 3.3 The under-elicitation question, handled honestly

Anthropic's strongest point is that introspection is *under-elicited* — refusal ablation and a
trained bias vector raise detection by 53–75%. That cuts both ways and the strand should say so.
If "introspection" can be trained up to 95% with a LoRA (Rivera & Africa 2026), the question
becomes whether the trained behaviour is self-access or a learned steering-detector. The
neutral-framing control adjudicates that too: train the detector, then test whether it
generalises to a framing that never mentions the self.

---

## 4. Why this drops jaws — and where it doesn't

**It lands because:**

- **The question is the hottest one in the field**, three papers deep in six months with a
  frontier lab defending its result. A dated, pre-registered control that none of them ran is
  the kind of thing that gets read.
- **It is real psychology, not analogy.** Nisbett & Wilson is a foundational result with a
  fifty-year-old unmeasurable term. A model is the first system where the term can be measured.
- **It answers the consciousness field's own stated deficit** — calibration — on the one
  indicator where calibration is possible.
- **It is the convergence of both capstone proposals.** Proposal I built the measurement,
  Proposal III built the discipline. Neither is discarded.

**It doesn't land as far as it sounds because:**

- **The space is crowded and fast**, and Anthropic is in it with more compute and the original
  authors. Scooping risk is the highest of any strand considered. The mitigation is speed:
  APERTURE's result is already pre-registered and should be date-stamped on arXiv immediately.
- **"Consciousness" is a word that does the team no favours in a capstone review.** The strand
  must be framed as *introspective accuracy* and *self-report validity* — quantities that can be
  measured — with consciousness as the field the result bears on, never as a claim made.
- **Introspection emerges from post-training and is absent in base models.** GPT-2 will not do.
  This needs instruct models at ≥7B, which means Kaggle GPU for anything at scale.

---

## 5. How it sits with the planted-persona proposal

These are not competitors. They are the same manoeuvre — *manufacture the ground truth that makes
an unfalsifiable claim testable* — aimed at two targets:

| | Planted personas | Calibrated introspection |
|---|---|---|
| Target claim | "Persona vectors capture the trait" | "Models can report their own states" |
| Ground truth | Injected direction | Injected state |
| Field | Model psychology / safety tooling | Metacognition / consciousness indicators |
| Prior work by team | None directly | APERTURE, pre-registered result |
| Scoop risk | Moderate | **High** |
| Ceiling | Methods paper with a named target | Live debate with a frontier lab |
| Compute | Kaggle | Kaggle |

**If forced to choose one for the next quarter**, calibrated introspection has the higher ceiling
and the head start, and the scoop risk argues for moving on it *first* rather than last. The
persona strand can follow using the same pipeline. Phase 0's neuron result stands underneath both
as the paper that establishes the instrument is characterised.

---

## 6. What is verified and what is not

| Claim | Status |
|---|---|
| The introspection debate is three-sided and live | **Verified** — three papers read at abstract level |
| Consciousness indicators are self-described as under-calibrated | **Verified** — 2603.27597 |
| Introspection emerges from post-training; refusal ablation raises detection +53% | **Verified** — 2603.21396 abstract |
| Identification confabulates toward frequent concrete concepts | **Verified** — 2603.05414 abstract |
| None of the three ran APERTURE's neutral-framing control | **Plausible from abstracts; needs full-text read** |
| Nobody has run planted-cause confabulation scoring | **Unverified** — no dedicated search yet |
| Nisbett–Wilson parallel is already drawn in the literature | **Verified** — 2605.26242 |

Two items need closing before this goes near the mentor: read the three critique papers in full
for the framing control, and run one dedicated search on planted-cause confabulation. Both are
desk work.

---

## 7. What to say to the mentor

1. The consciousness question has narrowed to whether a model's self-reports are accurate — and
   the field's own assessment is that it cannot calibrate its answer.
2. We can. Introspection is the one indicator where you can plant the state and score the report,
   and we have already built the apparatus and have a pre-registered result showing apparent
   introspection is output steering.
3. This is Nisbett & Wilson with the truth column filled in — a fifty-year-old psychology result
   made measurable for the first time.
4. It is where both capstone proposals converge: Proposal I's measurement under Proposal III's
   discipline.
5. The risk is speed — a frontier lab is in this debate. We need to date-stamp now and we need
   GPU access to run it at the model scale where introspection exists.
