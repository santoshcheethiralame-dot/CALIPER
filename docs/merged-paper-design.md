# The Merged Paper — Design

## The Language Model as a Preparation: Three Questions Neuroscience and Psychology Could Never Check, Checked

**Design document · 2 September 2026 · supersedes the three separate strand documents**

---

## 0. The thesis in one paragraph

In biology a *preparation* is the experimental system you choose because it makes an otherwise
impossible measurement tractable — the squid giant axon for the action potential, *Aplysia* for
memory, *C. elegans* for a complete wiring diagram. Three questions have sat at the centre of
neuroscience and psychology for a century without a preparation that could check them. **What
does a single neuron encode?** — asked since Hubel and Wiesel, never verifiable, because you
cannot read a cell's weights. **Does a personality trait exist, or does the instrument
manufacture it?** — the construct-validity problem psychometrics built its entire apparatus
around, never resolvable, because you cannot see the trait. **Is introspection accurate?** —
Nisbett and Wilson's 1977 demonstration that people confabulate the reasons for their behaviour,
never measurable, because you cannot see the state being reported. A language model is the
first preparation where all three have ground truth: a unit's true direction can be read from
its weights, and a trait or an internal state can be *planted*. We import the measurement
discipline neuroscience developed for cells — response characterisation, null models,
calibration before trust — and apply it at all three levels. Where we have measured, the
readout is silently wrong a quarter of the time, and a test that needs no ground truth catches
it. That finding then travels back: the same estimator is neuroscience's own receptive-field
tool, and the same confabulation is psychology's oldest unmeasured result.

---

## 1. What the research established

Every claim below was checked against full text or a dedicated search, and each is marked.

| Level | Readout under test | Ground truth | Prior validation in the field | Our status |
|---|---|---|---|---|
| **L1 — Unit** | Subspace estimator (maximally informative dimensions) | **Free** — a neuron's input weights *are* its direction | None; neuroscience suspects MID errors are large but cannot locate them | **Measured**: 77% [68–84%], n = 100 |
| **L2 — Trait** | Difference-of-means persona vectors | **Planted** — inject a known direction | By steering effect only; validity checks exist (2606.30449) but no ground truth | **Open** — 40 papers, none inject-and-recover |
| **L3 — Self** | Concept-injection introspective report | **Planted** — inject a known state | Three-sided live debate; **none of the three runs a neutral-framing control** (all checked at full text) | **APERTURE**: neutral framing beats introspective, 0.433 vs 0.302, pre-registered |

Two further verified gaps that belong to L3:

- **Planted-cause confabulation is open.** NeuroFaith (2506.09277) measures self-explanation
  faithfulness by alignment with internal representations — no planted cause. *Training LLMs for
  Self-Explanation Faithfulness* (2607.21090) states plainly that faithfulness "has no static
  ground truth." The Nisbett & Wilson parallel is drawn in *Reality Check* (2605.26242) as analogy
  only.
- **The consciousness-indicator programme calls itself under-calibrated** (2603.27597). Of Butlin
  et al.'s fourteen indicators, introspective accuracy is the only one where the internal state
  can be constructed and the report scored against it.

And the unifying gap: **manufactured ground truth exists in the field only as circuit-discovery
benchmarks** (Tracr, InterpBench). Nobody has articulated it as a programme, and nobody has
applied it to direction-finding at the trait level or to self-report.

---

## 2. Why one paper rather than three

Because the three studies are one manoeuvre — *choose a preparation where the answer is
knowable, then check the instrument against it* — applied to the three questions biology and
psychology most wanted answered and could never verify. The reciprocity is the structure, not a
subsection:

| Direction | What crosses |
|---|---|
| **Neuroscience → model** | The estimator (maximally informative dimensions), the classical baselines (spike-triggered average and covariance), and the calibration habit itself |
| **Model → neuroscience** | The finding that the estimator fails silently, and a detector — exportable to the receptive-field literature that has used it for twenty years without a way to audit it |
| **Psychology → model** | Construct validity and the multitrait–multimethod matrix at the trait level; the Nisbett–Wilson confabulation paradigm at the self level |
| **Model → psychology** | The first measurement, in any system, of *how far* a confabulated self-report departs from the true cause — the quantity Nisbett and Wilson could name but never measure |

What makes each study convincing is identical across them, and it is the neuroscientist's habit:

**The calibration discipline** — built in Phase 0, portable to every level:

1. A **random-direction / random-state null** — what does the readout return when there is
   nothing to find? (L1: R² ≤ 0.044 for random directions.)
2. A **required-N table** — how much data for a stated precision, indexed by informative events?
   (L1: ~200 events; more data adds nothing.)
3. A **pre-registered pass criterion with a confidence interval**, filed before the run.
4. A **disagreement flag** — two estimation routes; where they diverge, the readout is unreliable.
   (L1: 87% vs 55% success, p = 0.0008, no ground truth needed.)

Every study reports the same four quantities. That is what stops it reading as three papers
stapled together: the reader sees one instrument characterised at three depths, and the
disagreement flag in particular is a *deliverable* — a thing practitioners can run on their own
readouts where no truth exists.


---

## 3. The three studies

### Study 1 — Unit readout *(complete)*

Rank-one subspace estimation on GPT-2 MLP neurons, where the true direction is the weight vector.

| Result | Value |
|---|---|
| Recovery rate | **77 / 100**, Wilson 95% CI [0.679, 0.842] |
| Median alignment | 0.993 — the failures are invisible in aggregate |
| Restart agreement on failures | 0.85–0.99 — confidently wrong |
| Oracle at the true direction | R² = 1.000 — a perfect solution exists |
| Cause | Optimisation landscape; the k=2 fit contains the true direction at 0.997 |
| Partial fix | Cascade from k+1; n2977 0.452 → 0.999 |
| Detector | Method disagreement: 87% vs 55% pass, p = 7.7×10⁻⁴ |
| Classical baselines | STA 0/30, decorrelated STA 1/30, STC 0/30 vs 26/30 fitted |
| Null | Random direction explains ≤ 4.4%; true direction 70–98% |
| Required N | ~200 informative events; plateau thereafter |
| **Limitation** | **Joint estimation at K ≥ 2 degenerates to K = 1**: median 0.50 at K=2, 0.36 at K=3 — the signature of recovering exactly one direction. Not reached at any N. |

That last row is new (E0.3 additive, completed today) and it matters for Study 2: any
multi-dimensional readout needs the cascade generalised to K > 1 first, or must be restricted to
one direction at a time.

### Study 2 — Trait readout *(to build)*

Plant a known direction `v` at layer L via APERTURE's injection. Run the persona-vector extraction
pipeline — Anthropic's public code — on injected-versus-baseline generations. Score
`|extracted · v|`.

| Experiment | What it answers |
|---|---|
| **S2.1** Recovery at the plant layer | Positive control; should be near-perfect; if not, stop |
| **S2.2** Recovery vs depth | How fast does the extracted "persona vector" drift from the true cause as computation intervenes? The headline figure |
| **S2.3** Strength × N sweep | Required-N for trait extraction |
| **S2.4** Multitrait–multimethod matrix | Difference-of-means vs probe vs subspace estimator, several planted traits: convergent, discriminant, and the ground-truth column |
| **S2.5** Stimulus set | AIPsy-Affect (480 items, keyword-free, MIT) — validated, released, kills the stimulus-design objection |

Requires an instruct model with expressible personas (≥ 7B) — Kaggle.

### Study 3 — Self readout *(APERTURE exists; calibrate and extend)*

| Experiment | What it answers |
|---|---|
| **S3.1** APERTURE under the discipline | Random-state null, required-N, dose–response on injection strength, pre-registered CI. Turns a result into a calibrated measurement |
| **S3.2** The framing control, extended | Neutral vs introspective identification across concepts, strengths, and models. **Verified unreported in all three current papers.** |
| **S3.3** Planted-cause confabulation | Inject a direction that shifts behaviour; ask *why*; score the explanation against the planted cause. First ground-truth Nisbett–Wilson in any system |
| **S3.4** The under-elicitation test | Anthropic shows detection trains up to 95% with a LoRA. Does the trained detector generalise to a framing that never mentions the self? Adjudicates self-access vs learned steering-detector |

Requires ≥ 7B instruct models — introspection is absent in base models — Kaggle.

### Reciprocity arm — *(runs alongside, CPU)*

| Experiment | What it answers |
|---|---|
| **R1** Poisson transfer | Linear–nonlinear–Poisson model neurons under natural stimuli: does the 23% silent failure appear, and does disagreement flag it? Decides whether L1 exports to neuroscience |
| **R2** Real-data audit *(stretch)* | Disagreement flag on public natural-stimulus recordings; report the flagged fraction |

---

## 4. The paper, written out

**Title candidates**
1. *The Language Model as a Preparation: Three Questions Neuroscience and Psychology Could Never Check, Checked*
2. *What a Neuron Encodes, Whether a Trait Exists, Whether Introspection Is Accurate — Measured Where the Answer Is Known*
3. *Ground Truth at Last: Calibrating the Readout of Cells, Traits, and Self-Report in a System Built to Be Read*

**Abstract (draft)**

> Three questions have organised neuroscience and psychology for a century and have never had a
> preparation that could check them: what a single neuron encodes, whether a personality trait
> exists independently of the instrument that measures it, and whether introspective report is
> accurate. Each is unverifiable in a brain because the ground truth — the cell's weights, the
> trait, the internal state — cannot be observed. A language model is the first system where all
> three can be: a unit's true direction is its weight vector, and a trait or a state can be
> planted. We import the measurement discipline systems neuroscience built for single cells and
> apply it at each level. At the unit level, the field's own receptive-field estimator recovers
> the known direction for 77% of units [95% CI 68–84%] and fails silently on the rest —
> aggregate alignment 0.993, restart agreement 0.85–0.99 — while a perfect solution exists for
> every failure and disagreement between estimation routes flags them at p < 0.001 without
> ground truth. The classical spike-triggered estimators recover 1 unit in 30. At the self level
> we plant a state and find that a question which never mentions the model identifies it better
> than the introspective one — apparent self-knowledge is output steering — a control absent
> from every current paper in the debate. At the trait level we plant a direction and test
> whether persona-vector extraction recovers it. The same failure structure recurs at each level
> and the same detector catches it; we then test whether it travels back to the neurons the
> estimator was built for.

**The figure that carries it.** A single panel per level, same axes: x = readout confidence
(R², steering effect, report confidence); y = actual correctness against ground truth. Three
clouds, each with a dense mass near (high, high) and a tail at (high, *low*) — confident and
wrong — with the disagreement-flagged points marked. The reader sees the same shape three times.

---

## 5. Honest risks

| Risk | Severity | Response |
|---|---|---|
| **Breadth reads as three papers stapled** | High | Identical four-quantity reporting in every study; the figure above; the detector as the through-line deliverable |
| **L3 scoop — a frontier lab is in this debate with the original authors** | **High** | Date-stamp APERTURE's framing result on arXiv **now**, before anything else |
| **K ≥ 2 degeneracy blocks S2.4** | High | Generalise the cascade (find one, project out, find next) or restrict S2.4 to one direction per trait |
| **L2 is entirely prospective** | Medium | It is the least-built study; see shipping order |
| **"Consciousness" in a capstone review** | Medium | Never claimed. The words are *introspective accuracy* and *self-report validity* |
| **Compute** — L2 and L3 need ≥ 7B instruct models | Medium | Kaggle, single account, resumable pipeline already built |
| **Cheap proxies misleading us** — five times this fortnight | Standing | A screen is a screen; decide with the real objective; verify synthetic conclusions transfer |

---

## 6. Shipping order — the honest version

The full three-study paper is the target. But **L1 and L3 have data now; L2 has none.** So:

| | Content | When |
|---|---|---|
| **Preprint now** | APERTURE's framing result + Study 1 — date-stamps both results in a live debate | **This month** |
| **Flagship** | Studies 1 + 3 + R1, the calibration discipline as through-line | Submit Jan–Feb 2027 |
| **Follow-up or Study 2 in flagship** | Planted personas, if S2.1–S2.2 land by December | Same submission if ready; else the second paper |

This ordering spends the scoop risk where it is highest (L3) and defers the build that has no
result yet (L2) without dropping it.

---

## 7. What to say to the mentor

1. Every way we read a model is validated without ground truth. We build the ground truth — free
   at the neuron level, planted at the trait and self levels — and check.
2. Where we have checked, readouts fail silently a quarter of the time and a simple disagreement
   test catches it. That is a deliverable practitioners can use today.
3. The self-report level is the live consciousness debate, and we hold a pre-registered control
   that none of the three current papers ran — verified at full text.
4. It is genuine reciprocity in both directions: neuroscience's method audited in a model where
   truth is knowable; psychology's confabulation result measured against a planted cause.
5. Both capstone proposals converge here — Proposal I's measurement under Proposal III's
   discipline — and the mentor's interest in psychology and consciousness is the third study, not
   a detour.
6. Two asks: GPU time on Kaggle for the ≥ 7B work, and permission to date-stamp the APERTURE
   result this month before we are scooped.
