# CALIPER — Semester Plan

**Written 3 September 2026. Supersedes the 19 August timeline where they differ.**

Four semesters remain: sem 5 (now → Nov 2026), sem 6 (Jan–May 2027), sem 7 (Jul–Nov 2027,
placement season), sem 8 (Jan–May 2028). Graduation mid-2028.

## What changed since the August timeline

Study 3 — the injection-detection experiment — is complete, pre-registered, and drafted as a
standalone preprint. It was not on the August plan at all. It now leads the publication
order, because it is finished and because it carries the highest scoop risk of anything in
the programme.

Study 1 (the unit-level estimator) missed its own pre-registered gate: 77 of 100 units
recovered against a required 90. That is not a failure of the programme — the failures are
characterised, a cascade recovers most of them, and a disagreement flag predicts them — but
it changes Study 1 from "a working instrument" to "a characterised instrument with a known
23% silent-failure rate". That is the honest headline and it is what the paper says.

## The papers

| | content | when | target |
|---|---|---|---|
| **A** | Study 3 alone: content-free control for injection detection | Sep 2026 | arXiv preprint; workshop later |
| **B** | The capstone flagship: Studies 1 + 3 under the "preparation" framing, Study 2 if it lands | Jan–Feb 2027 | ICML / ACL 2027 main track; BlackboxNLP fallback |
| **C** | Phase A: what the calibrated instrument measures in unknown units, at scale | Sep 2027 | ICLR 2028; conditional on Phase A clearing its gate |

Two papers are committed (A, B). The third exists only if Phase A clears the causal-ablation
gate in sem 6. It is written here so nobody is surprised by it, not as a promise.

---

## Sem 5 — now to November 2026

**Theme: ship what is done, characterise what is not.**

| when | what | done when |
|---|---|---|
| **Sep, week 1–2** | Mentor meeting. Three items: Paper A authorship for a 4-person team; work division for sem 5–6; the GPU ask | Decisions recorded |
| **Sep, week 2–3** | Paper A: fill references, mentor read, fix what she flags. Optional: trial-randomised replication of the four Kaggle conditions (~2 h GPU) so the limitations section can say the confound was closed | On arXiv |
| **Sep–Oct** | Study 1 residual: characterise the 23% failure class. Is it predictable from the disagreement flag alone? Generalise `fit_cascade` to K>1 or restrict every multi-dimensional claim to K=1 | E0.1 re-run with the answer known in advance; required-N table (E0.3) indexed by events |
| **Oct** | Paper B skeleton: framing chapter (unit / trait / self as three century-old questions with no preparation), Study 1 chapter from the Phase 0 report, Study 3 chapter from Paper A | Internal draft, figures roughed |
| **Oct–Nov** | Optional robustness for Paper B: Study 3 on a second model (Qwen2.5-32B, ungated, same script). One result on one model is a finding; on two it is a pattern | If time permits |
| **Nov** | Exams. Mentor gate review on Phase 0 + Study 3 | — |

**Sem 5 deliverables:** Paper A on arXiv. Study 1 failure class characterised. Paper B
skeleton. Study 2 go/no-go decision made (see below).

**The Study 2 decision, by end November.** Study 2 (planted personas) needs the estimator to
recover more than one direction at once, and Phase 0 showed K≥2 degenerates to K=1 at
every N. If the cascade generalises by November, Study 2 runs in sem 6 and joins Paper B.
If it does not, Study 2 is cut from the flagship and Paper B ships as Studies 1 + 3. Decide
in November; do not let it drift into sem 6 undecided.

---

## Sem 6 — January to May 2027

**Theme: the flagship, then point the instrument at something unknown.**

| when | what | gate |
|---|---|---|
| **Jan** | Paper B complete. Every number carries a measured error rate from Phase 0 | Submit ICML 2027 (~late Jan) |
| **Jan–Mar** | Study 2, only if green-lit in November: plant a known persona direction, test recovery (S2.1), then the multitrait-multimethod matrix (S2.2) | Joins Paper B camera-ready or becomes its own workshop paper |
| **Feb–Mar** | Phase A begins: E1.1 stimulus-depth sweep, 300 units × 4 depths (~22 GPU-h on Kaggle) | K=1 at distance 1 must hold on real data |
| **Mar** | Paper B reviews / rebuttal | — |
| **Apr–May** | **E1.4 causal validation** — ablate the recovered subspace against matched random and top-K PC | **Hard gate.** If ablation does not collapse the response, every K is a curve fit and Phase A stops here |
| **May** | Exams | — |

**Sem 6 deliverables:** Paper B submitted. Phase A gate result, either way.

---

## Sem 7 — July to November 2027

**Theme: placement season. Plan for 40% productivity and protect the writing.**

| when | what | note |
|---|---|---|
| **Jul–Aug** | Paper C draft from Phase A. Writing survives interruption; experiments do not | Only if E1.4 cleared |
| **Sep** | ICLR 2028 deadline — Paper C | If Phase A did not clear, Paper C is a workshop paper in sem 8 instead. Say so in July, not September |
| **Sep–Nov** | Paper A to a workshop with reviews (BlackboxNLP, ICML MechInterp, NeurIPS ATTRIB — whichever is open) | Low effort; the preprint already exists |
| **Oct–Nov** | Phase B (communication subspaces) only if ahead of schedule | **Default: cut.** It is the one component the thesis can lose without damage |

**Decision point, July 2027:** if Phase A is behind, cut Phase B outright and move Phase D
into sem 8. Do not run two half-finished phases through placement season.

---

## Sem 8 — January to May 2028

**Theme: consolidate, release, defend.**

| when | what |
|---|---|
| **Jan–Feb** | Paper C revisions; ICML 2028 as fallback venue |
| **Feb–Mar** | Phase D (planted-prior manipulation) only if the delta against Cacioli's 2026 papers still holds. Re-check the literature first; it moved fast in 2026 |
| **Mar–Apr** | Thesis assembly. Artefact release: estimator library, Kaggle scripts, every raw JSONL, every pre-registration with its filing date |
| **Apr–May** | Defence |

---

## The risk that is not technical

The August timeline said it and it is still true: this is a 4-person capstone and one person
is executing. The plan above is sized to roughly 1.0× one person's capacity with no slack.
It survives if the other three take *any* of the following, each of which is separable:

- **Study 2** — self-contained, needs the estimator as a black box, needs a GPU
- **Phase C calibration** — InterpBench FDR, planted-latent networks; CPU only; no dependency on anyone else
- **Second-model replication of Study 3** — the script exists; it is a Kaggle session and a table
- **Literature and writing** — Paper B's framing chapter, the related-work sections, reference management

Raise this in the September mentor meeting as a scheduling fact, not a complaint. If nobody
takes a track, cut Study 2 and Phase B now rather than in May.

## What is fixed and what is not

Fixed: Paper A this month. Paper B in January. The E1.4 gate as the go/no-go for everything
in year two. The Study 2 decision in November.

Not fixed: whether there are two papers or three; whether Study 2 exists; whether Phase B
or Phase D runs. Each of those has a named decision point above. The plan is built so that
missing any of them costs a chapter, not the thesis.
