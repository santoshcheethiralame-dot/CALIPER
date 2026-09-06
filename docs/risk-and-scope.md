# Risk and Scope — How Good Will This Paper Actually Be

**Honest assessment of the merged design · 2 September 2026**

---

## 0. The verdict first

**As it stands today, this is a solid methods paper and it is not jaw-dropping.** The route to
jaw-dropping exists, it runs through exactly one experiment, and I put the probability of
reaching it at roughly **25–30%**.

| Version | Main-track odds | Jaw-drop | Status |
|---|---|---|---|
| Study 1 alone | 25–35% | Low–medium | Done |
| Studies 1 + 3 **as currently held** (Gemma 2B–9B) | 20–30% | Low — and the L3 result invites a one-line rebuttal | Done |
| Studies 1 + 3 **re-run at scale with a detection control, result goes our way** | **40–55%** | **High** | Not done; one hard experiment |
| Same, result goes Anthropic's way | 30–40% | Medium — a confirmation | Not done |
| All three studies | 15–25% | Low — breadth punishes, Study 2 thin | Not buildable at capacity |

Everything below is the reasoning behind that table. The audit's earlier numbers (8–15% as
originally framed, 25–45% restructured) still hold for Study 1 and I have not moved them.

---

## 1. Study 1 — unit readout

**What we have.** 77% [68–84%] at n = 100, pre-registered. Silent failure. Cascade fix.
Disagreement detector at p < 0.001. Classical estimators 1/30. Tight null. Required-N of ~200
events. K ≥ 2 degeneracy characterised.

**Killer objection.** *"It is your estimator."* Rank-one bottleneck regression on MLP neurons has
no users; a failure rate for a method nobody runs is a curiosity. The defence is that it is MID,
the field-standard receptive-field estimator, and that the classical baselines fail worse — but
that defence only becomes a *claim about the world* if the Poisson transfer (R1) shows the
failure is estimator-intrinsic rather than GELU-specific. R1 is not done.

**Second objection, now verified as real.** The disagreement detector is a between-estimator
variant of **jackknife reliability**, which is standard practice in receptive-field estimation —
a reliability index from jackknifed fits is in the literature. Our version is different in
specifics (two different estimators, not splits of one) but a neuroscience reviewer will say "you
rediscovered split-half." The honest claim is *quantification*, not invention: we measured how
well a reliability flag predicts failure against ground truth, which the neuroscience version
never could.

**Third.** GELU units are not neurons. Every level inherits this. The only defence is to never
claim they are: we test whether *procedures* fail when their assumptions break.

**Fourth.** K ≥ 2 degenerates to K = 1 at every N. Reviewers will ask why it is not fixed.

**Verdict.** A genuinely solid, well-powered, pre-registered contribution. Not jaw-dropping on its
own — the estimator's user base is too small for "it fails 23% of the time" to land. **Its
ceiling depends entirely on R1.**

---

## 2. Study 3 — self readout, and the fact that changes everything

**What we have.** APERTURE's three-framing battery (R12, pre-registered, commit 4d40941):
neutral framing beats introspective on identification, 0.433 vs 0.302, CI excludes zero.
Verified at full text: none of the three current debate papers runs this control.

**What I found in APERTURE's lab notebook today, and it matters more than anything else in this
document.** The models were **gemma-2-2b-it and gemma-2-9b-it**. In APERTURE's own detection
runs (R4, R5, R6): **zero detections across 24 cells, confabulation dominant, 0/4 false alarms.**
The notebook's own conclusion: "neither depth nor scale (to 9B) yields introspective
identification."

So at 2B–9B, **there is no introspection to debunk.** Anthropic's position is precisely that the
capability emerges from post-training at scale and is absent in smaller models. Their rebuttal to
our current L3 result is one sentence: *"Your models do not introspect. Of course identification
is steering there. Run it on a model that does."* That rebuttal is correct, and as currently held
our L3 result **strengthens** their story rather than challenging it.

**Second problem.** Our control attacks *identification*. Lederman & Mahowald already attack
identification (confabulation). Anthropic's mechanism paper concedes identification is weaker and
defends *detection* — the two-stage circuit, 0% false positives, +53% under refusal ablation.
Our control does not touch detection. We would be piling onto the conceded claim.

**What would make L3 jaw-dropping — the one experiment.** A **neutral-framing detection control**
on a model where detection is replicated (Qwen2.5-32B-class, per Vogel and per Macar et al.).
Ask *"Is there anything unusual about this text?"* — no mention of the model, its mind, or
injection — and compare the hit rate to the introspective detection prompt. If neutral anomaly
detection matches or beats introspective detection, the *defended* claim is steering too, shown
with ground truth and a control nobody ran, in a debate with a frontier lab. That is the
headline the whole design has been circling.

**Its risks, honestly.**
- **Compute.** 32B in 4-bit is ~16 GB; a single Kaggle T4 is 16 GB. It may not fit alongside
  activations. The 2×T4 configuration or a 14B-class model where detection is also reported are
  the fallbacks — but this needs checking before it is promised.
- **The result may go the other way.** Neutral detection below introspective detection at 32B
  would *confirm* Anthropic. Still publishable as calibration; not a headline; and we would be
  confirming a frontier lab's result as undergraduates.
- **Scoop.** Highest of any strand. Anthropic could add this control in a revision.

**Verdict.** As held: dismissable, and arguably counterproductive to publish as a challenge. With
the detection control at scale: the highest-ceiling result available to the project. Probability
of reaching that ceiling — needing compute to fit, the result to go our way, and no scoop —
**~25–30%.**

---

## 3. Study 2 — trait readout

**What we have.** Nothing. Novelty verified open (40 papers, zero inject-and-recover).

**Problems.**
- **S2.2 may be obvious.** "A direction extracted downstream of where it was planted drifts from
  the planted direction" — a reviewer may say: extract at the right layer. Whether the *rate* of
  drift is interesting is unknown until measured.
- **A planted direction may not behave like a persona.** If injection yields incoherent text
  rather than a trait, the setup is disanalogous to real persona vectors and the comparison fails.
- **K ≥ 2 degeneracy blocks S2.4**, the multitrait matrix, which was the psychologically
  interesting part.
- **Capacity.** The audit costed the original plan at 2× capacity. Three studies plus an arm is
  3–4×. Study 2 is the one with no data.

**Verdict.** Cut from the flagship. Keep as the second paper. Including a thin Study 2 in the
flagship *lowers* the odds — reviewers punish incomplete studies more than absent ones.

---

## 4. The reciprocity arm

**R1 — Poisson transfer.** The risk is specific: MID's local-minima problem under correlated
natural stimuli is *discussed in Sharpee's own papers*, and jackknife reliability indices exist.
R1 could return "neuroscience already knows this and already has a flag for it." That would
collapse the export. **Before running R1, read what Rowekamp & Sharpee and the Kaardal papers
already report about MID convergence failures and stability indices.** If they report it, the
honest contribution shrinks to a quantified failure rate against ground truth — real, but not an
export.

**R2 — real-data audit.** Stretch. Cannot be validated, which is the point, but also means a
reviewer can dismiss the flagged fraction as meaningless.

---

## 5. Cross-cutting risks, ranked by how much they move the odds

| # | Risk | Effect on odds | Mitigation |
|---|---|---|---|
| 1 | **L3 run only at 2B–9B, where introspection is absent** | Turns the headline strand into a rebuttable footnote | Re-run detection + framing control at 32B-class, or drop the challenge framing |
| 2 | **Breadth at 3–4× capacity** | Thin studies drag the whole paper | Flagship = Studies 1 + 3 + R1 only |
| 3 | **One person executing** | Every timeline in every document assumes four | Resolve in weeks |
| 4 | **Disagreement flag ≈ jackknife reliability** | Detector loses its "novel" label | Claim the quantification, cite the lineage |
| 5 | **R1 finds neuroscience already knows** | Export collapses to a footnote | Read Sharpee-group papers first |
| 6 | **Scoop on L3** | Confirmation instead of finding | Date-stamp now; run the detection control fast |
| 7 | **GELU ≠ neuron, planted ≠ trait** | Every level | State first, never concede later |
| 8 | **My own error rate** — 5 proxy errors, 4 novelty overclaims in two weeks | Every probability here should be read as optimistic | Pre-register everything; full-text every neighbour before claiming |

Item 8 is not a rhetorical gesture. The person who built this design has been wrong about
what is novel four times and about what a measurement meant five times. Discount accordingly.

---

## 6. What "jaw-dropping" actually requires

Working backwards from the sentence that would land:

> *Apparent introspection in language models — the claim at the centre of the AI-consciousness
> debate — is output steering. We show it with a control no one ran, on the models where the
> effect is claimed, against ground truth no one else has.*

Every clause has a precondition:

| Clause | Requires | Have it? |
|---|---|---|
| "is output steering" | Neutral detection ≥ introspective detection | **No** — untested |
| "control no one ran" | Verified absent in all three papers | **Yes** |
| "on the models where the effect is claimed" | 32B-class run | **No** — 2B–9B only |
| "against ground truth no one else has" | Planted state, known | **Yes** |

Two of four. The two missing ones are the same experiment.

---

## 7. Recommendations, in order

1. **Cut Study 2 from the flagship.** Second paper. Non-negotiable at current capacity.
2. **Check whether Qwen2.5-32B fits on Kaggle in 4-bit with activation capture.** If not,
   identify the smallest model where Macar et al. or Vogel report detection. This is a
   one-afternoon check and it gates the only jaw-dropping path.
3. **Design and pre-register the neutral-framing detection control** before running it — with
   the "result goes Anthropic's way" branch written down as publishable calibration, so the
   experiment is worth running regardless.
4. **Date-stamp APERTURE's identification result on arXiv now**, framed honestly as a
   calibration finding at 2B–9B, not as a challenge to Anthropic. It is still a verified-absent
   control and it protects priority.
5. **Read the Sharpee-group MID convergence literature before R1.** Half a day; decides whether
   the export exists.
6. **Reposition the detector** as a quantified between-estimator reliability index with lineage
   to jackknife RF reliability. It survives that framing; it does not survive being called new.
7. **Fix K ≥ 2 or restrict every multi-dimensional claim.** Do not leave it as a limitation row.

---

## 8. The honest bet

If I had to bet: **Study 1 + a properly scaled Study 3 + a pre-read R1, with Study 2 deferred,
lands main-track about 40% of the time and is jaw-dropping about a quarter of the time.** The
guaranteed version — Study 1 alone — is a good workshop-to-main-track methods paper at 25–35%
and drops no jaws.

The gap between those is one experiment on one model class, and whether it can be run is a
compute question answerable this week. That is the decision to bring to the mentor: not
*"which framing,"* but *"do we spend the next month on the one experiment that could make this
the paper, knowing it may confirm the other side."*
