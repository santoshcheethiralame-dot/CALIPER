# Sem 5 — Panel Review Plan

**Four reviews, September to November 2026. Written 3 September.**

The department expects sem 5 to cover: team formation, mentor assignment, problem
identification, literature survey, requirements gathering, system design, and an initial
prototype. The project is further along than that sequence assumes — the instrument is
built and characterised, and one full experiment has run. The plan below presents the work
in the panel's order so each review shows a clear step from the last, without hiding
anything and without dumping a finished paper on a panel that is expecting a problem
statement.

Review dates are assumed at roughly four-week spacing. Replace with the actual calendar.

---

## The one-paragraph problem statement (use verbatim, every review)

> Interpretability tools claim to read a model's mind: which direction a neuron responds
> to, which vector encodes a personality trait, whether the model can report on its own
> internal states. None of these tools has ever been checked against a case where the
> right answer was already known. Neuroscience solved this a century ago with the
> *preparation* — the squid axon, the sea slug, the worm — a system simple enough that a
> new instrument could be tested against a known answer before being trusted on an unknown
> one. We build preparations for language models at three levels — unit, trait, self — and
> use them to measure how often each reading tool is right, how often it is silently wrong,
> and what it takes to tell the difference.

Short form for a slide title: **"Does the mind-reading kit work? Building preparations for
language models."**

---

## Review 1 — early September · *Problem, team, mentor, literature*

**Rubric items covered:** team formation, mentor assignment, problem identification,
literature survey (first pass).

| slide | content | source |
|---|---|---|
| 1 | Team, mentor, date mentor selected the problem (18 Aug) | — |
| 2 | The problem statement, verbatim above | — |
| 3 | Why it is a problem: three published tools, three levels, zero ground-truth checks. One example per level | `merged-paper-design.md` |
| 4 | The neuroscience framing: preparation, tuning curve, spike-triggered average, calibration before use | `capstone-III-neuro-method-map.md` |
| 5 | Literature survey, first pass: ~40 papers across three strands; what each strand has and what it is missing | the three deep-research docs |
| 6 | Where prior work collides with ours and how we differ (Cacioli 2026, Distill Circuits, Kim & Paik) | `capstone-III-audit-verdict.md` |
| 7 | Sem 5 plan: what the next three reviews will show | this document |

**Lead with:** the problem statement. Not the method, not the results.

**Expected panel questions**

- *"Isn't this just probing / steering / interpretability?"* — Yes, those are the tools. We
  are not building a new tool; we are building the test bench the existing tools have never
  had.
- *"What is the deliverable?"* — A calibrated instrument at the unit level with a measured
  error rate, a validation protocol at the self level, and the code and data to reproduce
  both.
- *"Why neuroscience?"* — Because it is the one field that has been reading activity from
  units it did not design for a hundred years, and it learned the hard way that every
  reading tool needs a preparation.
- *"Who does what?"* — Have the division ready (see end of document). This is the question
  the panel uses to find out whether four people are working.

**Do not say:** anything about Study 3's results, the arXiv preprint, or the 77/100. Those
belong to the reviews whose rubric they fit.

---

## Review 2 — late September / early October · *Requirements and system design*

**Rubric items covered:** requirements gathering, system design, literature survey
(complete).

For a research capstone, "requirements" are the conditions a measurement must meet before
it can be trusted. We have four, and every one was learned by getting it wrong in August.

| slide | content | source |
|---|---|---|
| 1 | Recap: problem in one line; what Review 1 established | — |
| 2 | **Requirements** — the calibration discipline: (i) a null distribution, (ii) a required-N table, (iii) a pre-registered confidence interval, (iv) a disagreement flag that needs no ground truth | `caliper-phase0-report.md` |
| 3 | Why each requirement exists: the specific August wrong turn that produced it (rank transform halved the pass rate; binned R² returned 0.78 on the model's own pre-activation; the operating point was chosen on the median and the worst case was 0.15) | Phase 0 report |
| 4 | **Compute requirements:** CPU for the unit level; Kaggle 2×T4 at 4-bit for anything ≥27B; float32 compute mandatory for Gemma (peak activation 51,436 vs float16 ceiling 65,504) | `kaggle-run-s3.md` |
| 5 | **System design** — architecture diagram: activations hooks → estimator (rank-K bottleneck, cascade) → batched fitting → checkpointed runtime; separate Kaggle experiment harness for large models | repo layout |
| 6 | The design's key idea: an MLP neuron's true direction is its own weight column, free and exact — so the unit-level preparation costs nothing to build | `estimator.py` docstring |
| 7 | The three-level design as one table: level / what is planted / what is recovered / how it is checked | merged design |
| 8 | Literature survey complete: the matrix, ~60 papers, with the three verified gaps | vault lit notes |

**Lead with:** the four requirements and the mistake that produced each. Panels trust a
team that shows its wrong turns more than one that shows only its results.

**Expected panel questions**

- *"Where is the prototype?"* — Built and characterised; it is the subject of Review 3. Show
  one figure: the estimator recovering a known direction to correlation 1.00000000 on a
  single unit, as proof the design runs.
- *"What are the risks?"* — The estimator fails silently on ~23% of units (say the number).
  The multi-direction case degenerates. The team has one executing member. Name all three.
- *"Why Kaggle?"* — No lab GPU; 30 GPU-hours a week is enough for every experiment in the
  plan, and every run in the project is reproducible on a free account.

---

## Review 3 — late October · *Initial prototype: the instrument*

**Rubric items covered:** development of an initial prototype (unit level).

| slide | content | source |
|---|---|---|
| 1 | Recap: requirements from Review 2 | — |
| 2 | **The prototype:** the calibrated estimator, live or recorded, on one GPT-2 neuron with the answer known | `caliper/estimator.py` |
| 3 | **Pre-registered gate result:** 77 of 100 units recovered, Wilson 95% CI [0.679, 0.842], against a required lower bound of 0.90. **It failed.** | `preregistration-e01-n100.md` |
| 4 | Why that is the result and not a setback: failures are silent (median alignment 0.993), a perfect solution exists in every case (R²=1.000 at the true direction), the cascade recovers most of them, and the disagreement flag predicts failure at p=7.7e-4 with no ground truth needed | Phase 0 report |
| 5 | Required-N: ~200 informative events for one direction; more than one direction degenerates at every N — the single largest open problem | E0.3 results |
| 6 | Classical baseline: spike-triggered estimators recover 1 unit in 30; the fitted method 26 in 30 | E0.5 |
| 7 | What the prototype cannot yet do, honestly: K≥2; anything above GPT-2 scale on CPU | — |
| 8 | Preview: the same discipline applied at the self level, on a 27B model, results at Review 4 | — |

**Lead with:** the failed gate. Say "failed" in the first sentence, then say why it is the
most useful number in the project. A panel that hears "77%" from you will trust "10%" from
you in Review 4.

**Expected panel questions**

- *"Your gate failed. Is the project in trouble?"* — No. The gate was set before the run to
  find out whether the tool could be trusted blind. The answer is no, and we now have a flag
  that says when. That is the deliverable working as designed.
- *"Can you fix the 23%?"* — Partly; the cascade recovers many. The remainder is the
  characterised error rate that goes into every downstream number.
- *"What is K≥2 and why does it matter?"* — Real neurons may respond to more than one
  direction; our estimator currently finds one. Restricting claims to one direction is the
  honest fallback for sem 6.

---

## Review 4 — mid November · *Prototype results and the sem 6 plan*

**Rubric items covered:** prototype demonstrated on a real question; plan for sem 6.

| slide | content | source |
|---|---|---|
| 1 | Recap: three levels; unit level calibrated at Review 3 | — |
| 2 | **The self-level experiment:** a published claim (Macar et al. 2026) that a 27B model can detect when a thought is injected into its activations; 10.8% detection, 0% false positives | `s3-results.md` |
| 3 | We reproduced it on a free Kaggle account: 6.7% (2/30) [1.8, 21.3], 0/30 false positives *(corrected 6 Oct 2026, notebook C40; the 10.0% was a hand-read count)* | §1 |
| 4 | The control nobody had run: an injection with **no content**. At the published operating point it produces the entire effect (real 0.417, random 0.305, p=0.33) | §4 |
| 5 | Asking the model *about itself* adds nothing: a prompt that never mentions the model is indistinguishable once anything is injected (p=0.44–0.75) | §5 |
| 6 | The number depends on the readout: 43% down to 0% from generated text (pre-registered scorer; corrected 6 Oct 2026), flat 0.42-0.50 from the first token | §2 |
| 7 | **Our own hypothesis failed and is reported:** we predicted the effect was the model reading its own output; the first-token test refuted it (p=9e-9). Pre-registered, both branches written before the run | §3 |
| 8 | **Paper A on arXiv** (date) — one slide, the abstract | preprint |
| 9 | Sem 6 plan: flagship paper in January; Phase A (300 real units) with the causal-ablation gate in April; Study 2 go/no-go decided this month | `semester-plan.md` |
| 10 | What each member does in sem 6 | end of this document |

**Lead with:** slide 4. One sentence: "A vector with no content in it produces the same
report as a concept vector. The published control cannot see this."

**Expected panel questions**

- *"So the model can't introspect?"* — We do not claim that. We claim the published protocol
  cannot distinguish introspection from noticing a disturbance, and that the control which
  can distinguish them costs one forward pass.
- *"Is this really a prototype?"* — It is the three-level design running end to end at one
  level with a real result; the unit level ran in Review 3. Sem 6 connects them.
- *"Why should we believe a 30-concept result on a 4-bit model?"* — Two independent
  controls agree at every strength; every criterion was filed before the run; the whole
  thing reproduces in 30 minutes on a free account, and the panel can run it.
- *"What did the other three members do?"* — See below. Have each of them present the
  slide for their track.

---

## Work division to propose at Review 1 and report on at every review after

The panel assesses this. It is also the only lever that gets four people working. Each track
is separable, has its own artefact, and can be presented by its owner.

| member | track | sem 5 artefact | sem 6 continuation |
|---|---|---|---|
| A (currently executing everything) | Unit-level estimator; Study 3; integration | Phase 0 report; Paper A | Paper B; Phase A |
| B | **Literature and framing** — the lit matrix, the neuroscience method map, related-work sections | Review 1 slides 3–6; the ~60-paper matrix maintained | Paper B framing chapter |
| C | **Study 3 replication** — second model (Qwen2.5-32B, ungated, same script), trial-randomised re-run | Review 4 robustness slide | Study 2 if green-lit |
| D | **Phase C calibration** — InterpBench false-discovery rate, planted-latent networks; CPU only | Error-rate table for Paper B | Phase A calibration companion |

If a member does not take a track by Review 2, say so to the mentor then. Do not carry it to
November.

---

## What to bring to every review

- One printed page: problem statement, the three levels, the current number for each, the
  next gate. Panels remember the page, not the slides.
- The repository open and runnable. At least one panel member will ask to see code.
- The pre-registration documents with their dates. They are the single strongest answer to
  "how do we know you didn't tune this?"
