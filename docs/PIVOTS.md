# The pivot log

**Every change of direction, with its date, its cause, and what it cost.**

Written 8 September 2026, covering July–September 2026.

This file exists because the project has changed direction more than once and the
reasons are scattered across a lab notebook, three pre-registrations and two vault
documents. A reader — a mentor, a reviewer, a teammate joining late, or the author in
six months — should be able to reconstruct why the work looks the way it does without
reading all of it.

**A pivot is not a failure and it is not free.** Each entry states which it was closer
to. Nothing here appears in the paper: a paper is a claim and its evidence. This file is
for us.

---

## Summary — the arc in five lines

1. Started asking whether language models introspect.
2. Found the question was confounded, and pivoted to measuring the confound instead.
3. Mentor selected a different proposal — measuring model units the way neuroscience
   measures neurons.
4. Discovered that the ground truth for that measurement is free, exact, and nobody uses it.
5. Narrowed to: **calibrating the reliability checks the field already relies on.**

Each step narrowed the claim and strengthened the evidence. That is the correct
direction of travel, and it is worth saying so plainly, because living through it felt
like the opposite.

---

## P1 · Introspection → measurement methodology
**30 July 2026 · repositioning · net gain**

**Was:** "Do language models introspect?" — the original APERTURE thesis.

**Became:** "Self-knowledge claims are systematically confounded by *output steering*.
Here is the confound, the control protocol that detects it, and a demonstration that it
overturns a real result — including one of our own."

**Cause.** The introspection question was crowding fast, and empirical claims in a
crowding field depreciate. Methodological assets appreciate.

**Cost.** None in evidence — the same runs support both framings. The write-up changed.

**Kept:** the Gemma null became the *demonstration case* rather than the headline.

---

## P2 · APERTURE is not the capstone
**18 August 2026 · external decision · reset**

The mentor reviewed a seven-proposal set and selected **Proposal III** (population-level
measurement of LM units — tuning curves and pairwise maximum entropy, imported from
systems neuroscience) and **Proposal V** (capture–recapture coverage estimation for
adversarial testing).

APERTURE is Proposal I. It was not selected.

**Cost.** The largest of any entry here. A completed apparatus, 98 passing tests, twelve
logged runs and a publication plan stopped being the main line.

**What survived.** The repo remains valid completed work and is cited as preliminary
evidence. Its strongest single asset — a neutral-framing control that a full-text check
confirms is unreported in the current introspection debate — is still unpublished and
still needs no new compute. **This is the longest-open loose end in the project.**

**Recorded honestly:** the global project index still described APERTURE as "the
capstone" for weeks after this. Documentation lag is itself a cost.

---

## P3 · Proposal III's headline is killed before a line is run
**18 August 2026 · adversarial audit · painful, correct, and cheap**

A six-lens audit of the proposed "efficient coding has never been tested against a known
prior" headline found direct collisions:

- arXiv:2603.20642 has "Efficient Coding" **in its title** and counts the corpus prior.
- arXiv:2604.04469, same author, reports corpus frequency predicting per-magnitude
  variability at rho = 0.84.
- Benjamin et al., *Nature Communications* 13:7972 (2022): gradient descent generically
  produces frequency-tracking sensitivity — the "it is just the training objective"
  objection is a published proof, not a hunch.
- Kim & Paik, *Science Advances* (2021): number tuning curves in **randomly initialised**
  networks.

Consensus odds on the framing as written: 8–15% main track; 25–45% restructured.

**Cost.** Near zero, and this is the point. The claim died at the literature-review stage
rather than after a semester of compute.

**Standing consequences.** Pythia only for anything involving the training prior (GPT-2's
WebText was never released); the stimulus battery cut from six variables to two; the
grid-code probe dropped.

**Lesson that generalised:** an analogy-shaped headline is the most attackable kind.
It came back in P8.

---

## P4 · Three strands become one paper
**2 September 2026 · consolidation · net gain**

Unit readouts (neuron directions), trait readouts (persona vectors) and self readouts
(introspective reports) were three separate efforts. They share one defect: **every one
of them is validated without ground truth.**

**Became one thesis:** construct ground truth at all three levels, under a single
calibration discipline — null, required-N, pre-registered confidence interval,
disagreement flag.

**Cost.** None. This was recognising structure that was already there.

---

## P5 · Study 3 closes, and the result is a negative about someone else's method
**2 September 2026 · pre-registered outcome A2 · the finding is real, the venue is crowded**

At alpha=6 a norm-matched **random** vector drives first-token P(YES) as much as the real
concept vector (0.305 vs 0.417, p=0.33). "Detection" of an injected thought is largely a
report of *having been perturbed*. Content-free share: 36% / 61% / 82% at alpha 2/4/6.

Not self-specific: a neutral prompt that never mentions the model is indistinguishable
from the introspective one at every injected strength.

**Own hypothesis killed and reported as such.** The prediction that free-generation
detection was the model reading its own output was falsified by a first-token test.

**Cost.** Study 3 is a solid result in a fast-crowding area. Later cut from the flagship
(P9).

---

## P6 · The vectors were dead, and every health check passed
**7 September 2026 · C31 · the most consequential find in the project**

Concept vectors read at the chat-template tail carry **no content** — and still perturb
the model, still shift first-token P(YES) monotonically and significantly (p=3.7e-09 on
29/30 concepts), and still pass every health check anyone reports: unit norm, finite,
thirty distinct directions.

**A significance test on the perturbation cannot distinguish a working vector from a dead
one.** Only a steering positive control can — inject on a neutral prompt, check the
concept reaches the output — and it is absent from this entire literature, ours included
until that day.

Confirmed by the repair: refitting at the concept position turned a flat 1/30 into a
monotone 1 → 7 → 14 of 30, with read position the only change.

**Cost.** Every Qwen result void. **Value:** far higher than the cost. This is the origin
of the project's current thesis — that unvalidated readouts fail silently and pass their
own checks.

---

## P7 · Study 2 fails twice, and the second failure is informative
**8 September 2026 · P1 void, P1b substantive negative**

**P1 planted random directions.** Recovery sat *below* null at every strength, flat across
a 4× range, chance at every depth. Recorded **void, not failed**, despite the
pre-registration permitting a finding: a random direction has no natural representation,
so the text it produces carries no signal to recover. The test was mis-specified — the
confound was removed by removing the mechanism the method depends on.

**P1b planted real concept vectors** and added a manipulation check that gates everything.
The plant demonstrably reached the text (median 6/16 at 40% of residual norm). Recovery
0.1557 against a same-bank null of 0.1870 — **the extracted direction lands in concept
space but is no closer to the planted concept than to an unrelated one.**

C58 then measured the bank's own collinearity at median |cos| 0.4216, confirming the null
was the inter-concept floor.

**Independently corroborated:** [Non-Identifiability of Steering Vectors
(2602.06801)](https://arxiv.org/html/2602.06801v4) reaches the same conclusion from
*behavioural* evidence and states explicitly that it does **not** plant known directions
and attempt recovery. P1b is the ground-truth complement to a published behavioural
result — which reframes it from "our failed study" to "the experiment they said they
didn't run".

**Cost.** Two GPU runs and a study. **Recovered value:** moderate, and only because the
literature check was done afterwards.

---

## P8 · From benchmark to calibration bench
**8 September 2026 · scoping · the sharpest single move**

**Was:** build a benchmark of direction-finding methods.

**Became:** build a **calibration bench for the reliability checks themselves.**

**Cause — a four-search literature scout.** The field is crowded on method-ranking
(ObserverBench, MIB, AxBench) and on synthetic ground truth (InterpBench, Tracr, Feature
Recovery Rate). But three literatures leave the same hole:

- Reliability signals are validated against **behaviour or reconstruction**, never against
  a known-correct direction — [Geometric Canary
  (2604.17698)](https://arxiv.org/html/2604.17698v2), [Unstable Features
  (2606.12138)](https://arxiv.org/abs/2606.12138).
- Ground truth for direction recovery is always **planted or compiled**.
- Harnesses rank methods but carry **no ground-truth-free failure detector** —
  [ObserverBench (2609.03026)](https://arxiv.org/abs/2609.03026) says so outright.

**The claim that survives scrutiny:** not "use disagreement" (known, and patented), but
*"ground-truth-free reliability checks are used everywhere and have never been calibrated
— here is the one place they can be, and here is what they are worth."*

**Cost.** None. Same data, sharper claim.

---

## P9 · The bench is Proposal III's instrument paper
**8 September 2026 · repositioning · closes the loop opened in P2**

The bench was being scoped as a standalone artifact. It is not — it is the selected
proposal's Paper 1 and its Phase D, already substantially run.

Proposal III's own method map names **Maximally Informative Dimensions** the crown jewel
and **spike-triggered covariance** "the baseline that fails". E0.5 had already scored both
against free ground truth: STC 0/30, MID-style bottleneck 26/30, with Bussgang's constant
as the stated mechanism. **The proposal's prediction was confirmed, not assumed.**

III's capability table marks *"ground-truth validation of unit characterisation"* as
**Open** and proposes InterpBench for it. The weight column is better: exact *and*
natural, on units that are hard for reasons the model chose in training.

**And the ordering argument:** single-unit calibration is a prerequisite for the
population layer, not a detour. Pythia's median alignment is 0.9994 while that same
population contains a unit recovered at 0.0761 — tuning curves and pairwise maxent fitted
over it would inherit that error invisibly.

**Framing decision, learned from P3.** The mentor responded to "neuroscience and models
are similar / model psychology". That is the hook and it works, but it must not become
the claim — an analogy-shaped headline is the most attackable kind. The committed framing
is: **we do not claim models resemble brains; we claim the measurement problem is
identical, and neuroscience has forty years of method for it.** Same appeal, no attack
surface.

---

## P10 · Venue strategy — optimise for certainty, not prestige
**8 September 2026 · pending mentor input**

The capstone grade depends on publishing. That makes the objective *a real acceptance*,
not the best possible venue.

- **TMLR is the primary target.** It accepts on correctness and clarity and explicitly
  **not** on novelty or significance. Rolling submission, no deadline, respected, indexed.
  A careful methods paper with honest negatives and pre-registration is what it exists for.
- NeurIPS Datasets & Benchmarks — good fit, rewards rigour over scale.
- Interpretability workshops — fast and visible; a backup, not the plan.
- Top-tier main track — low odds. Two small models and a narrow ground-truth substrate.

**Cut from the flagship:** Study 2 (P7) and Study 3 (P5). Both are real work; including
weak studies weakens a paper.

**Two limitations that go in the abstract, not the appendix.** Only 124M and 160M models
so far. Ground truth holds only for a unit reading its own layer, and whether the
calibration transfers is an assumption the work cannot test. Reviewers forgive a stated
limit far more readily than a discovered one.

**⚠ OPEN — the highest-value unresolved question in the project.** Nobody has read the
capstone rubric's definition of "published". Preprint, workshop, or peer-reviewed venue
only? Accepted, or submitted? The entire schedule depends on the answer and it is
currently a guess. **Ask before planning further.**

---

## What the pattern says

Six of the ten entries were caused by **checking before committing** — a literature pass
(P3, P8), a control that had never been run (P6), a manipulation check (P7). Those were
cheap. The expensive entries were the ones driven from outside (P2) or by a framing that
had not been stress-tested.

The project's actual method — file the criterion before the run, report the failure as
filed, run the control nobody runs — is the same discipline the paper is *about*. That is
worth one sentence in the paper and it is not a coincidence.
