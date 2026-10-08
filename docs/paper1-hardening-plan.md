# Paper 1: hardening plan after review round 1 (8 Oct 2026)

Status: proposals. Nothing here is filed or run unless a line says so. Ordered by value for
cost. Each experiment gets its own pre-registration before it runs.

## Running

- **N-1, response noise** (`docs/preregistration-n1-response-noise.md`, queue 4). Answers
  the "noiseless ceiling" objection.

## Proposed

### 1. Cross-sample agreement as a check (highest value)

**Exploratory lead** (`experiments/explore_cross_sample.py`,
`results/cross_sample_agreement.json`, notebook §4, 8 Oct):
- Agreement between two fits of the same unit on different token samples (B-15a vs B-15b)
  predicts failure with AUC 0.922. Held-out R2 scores 0.896 and restart agreement 0.726.
- On the converged-wrong class, the one neither filed check detects, it scores 0.911,
  against 0.713 and 0.685.

**Mechanism.** Directions the stimulus barely excites are poorly determined. They move when
the token sample changes, but not when only the initialisation changes. This is PCS's "perturb
the data" (Yu & Kumbier), measured against ground truth.

**Weak points of the lead.**
- Only 6 converged-wrong units.
- The two fits differ in restart count (5 vs 2) as well as token sample.
- Not pre-registered.

**Proposed filed test.** New units (not B-14's), each fitted twice on document-disjoint
samples with identical settings.
- Primary: AUC(cross-sample agreement) minus AUC(held-out R2), overall and on the
  converged-wrong class.
- Prediction: cross-sample agreement leads on converged-wrong.
- Cost: about 2 x 4 h of local CPU.

**If confirmed.** A positive result to sit beside the headline. Agreement is not useless:
the kind of perturbation decides what it measures.

### 2. Confirmatory replication of the identifiability finding

Turns the exploratory section into a confirmatory one. It shares runs with item 1: the
second fit on new units provides the data.

**Prediction, to file.**
- At least 80% of converged-wrong fits reach stimulus-weighted alignment of 0.99 or more
  on fresh text.
- Their median share of error energy in the bottom-1%-variance directions is at least
  0.70, against under 0.50 for under-fitted fits.

**Candidate arm.** GPT-Neo-125m layer 10, which has the most converged-wrong units, with
units not used before.

### 3. Stimulus diversity (tests the mechanism directly)

Re-fit with a mixed corpus: Gutenberg plus wikitext, and code if a corpus can be cached.

**Prediction.** The converged-wrong rate falls, because the low-variance directions get
excited. Error energy also moves out of the bottom-variance directions.

**Cost.** A corpus loader change plus about 4 h.

### 4. Out-of-distribution and interventional check on saved directions (cheap)

**OOD.** Score converged-wrong directions on code-domain text. The prediction is that
predictive equivalence degrades there.

**Interventional.** Steer along v-hat versus w and compare the unit's response and
downstream logits.

**Why.** It answers "predictive, not interventional" (internal review: DA M4, R2 W4)
without new fits.

### 5. A second estimator configuration

Change the link-network width or depth on 100 units, so "one estimator" becomes a stated
sensitivity.

## Framing and literature to fold in

Leads in `docs/CITATIONS.md` §26. Verify each before citing.
- **Steering-vector non-identifiability** (2602.06801, 2505.22637): the same phenomenon
  without a ground truth. Paper 1 gives the unit-level measurement and the mechanism.
- **MetaQuantus** (TMLR 2023): the closest framing, evaluating evaluation metrics, and a
  venue precedent.
- **Simulation-based calibration** (Talts et al.): the methodological parallel for the
  bench.
- **Statistical vs structural multiplicity** (2601.06730): vocabulary for under-fitted vs
  converged-wrong.
- **Imperfect-reference-standard statistics** (latent-class models): grounding for treating
  the Euclidean label as imperfect.
- **Massive activations and residual sinks:** why the stimulus covariance is ill-conditioned.

## Beyond Paper 1

- **Paper 2.** Steering non-identifiability plus S-2 (dead tail vectors give the largest
  P(YES) shifts) raises a sharp question. Is the introspection signal a property of the
  direction, or only of the perturbation's size?
- **Paper 3.** Calibrate SAE cross-seed agreement against planted features (SynthSAEBench).
  That is where the transfer question can be answered.
