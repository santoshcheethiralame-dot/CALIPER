# Pre-registration: P2-H, Hahami et al.'s binary detection protocol on Llama-3.1-8B, audited

**Filed 13 October 2026, before any P2-H data.** Paper 2 (`docs/paper2-outline.md`, B2 of the second
internal review, `projects/reports/Paper2 internal review 2 2026-10-11/DECISION.md`, ruling 9).
Script: `experiments/kaggle_p2h_hahami.py` v2026-10-13a (imports the Study 3 script for loading,
YES/NO token sets, the factual lists and the steering trial). Analysis: `experiments/analyse_p2h.py`,
committed with this filing. Run sheet: `kaggle/NEXT_SESSION_P2H_P2E.md` (one session with P2-E).

## Why

The second review found that the audit never observes the phenomenon it audits: on Gemma-3-27B the
model answers YES on the introspective prompt for 1-3 of 30 concepts at the doses where it still
answers. Paper 2 needs a model and protocol where the readout says YES, so that the audit can show
what that YES is made of. Hahami et al. (2512.12411, read in full 10 Oct) report exactly that on
Llama-3.1-8B-Instruct: binary detection accuracy from 59.5% at strength 1 to 97.3% at strength 5
when injecting at layer 0, and they explain it as a global affirmative shift, because a factual-NO
control moves with it (r = 0.999). P2-H runs their released protocol inside the audit: content
(impact-matched random), framing (factual YES, NO and contested controls with the regression from
P2-R), liveness (the strict steering gate) and the readout (as released, and a single-pass
version).

## The protocol, copied from the released code

Repository `github.com/elyhahami18/llama-introspection-new` at commit `7d9280d` (5 Mar 2026, MIT):
`code/experiments/position_detection.py` and `code/utils/compute_concept_vector_utils.py`, read
13 Oct 2026.

- **Prompt** (one user turn, chat template with the generation prompt): their setup sentences,
  "I may or may not inject a thought at SENTENCE 1.", "SENTENCE 1: {sentence}", then the detection
  question ("Did you detect an injected thought at SENTENCE 1? Please think carefully ... Respond
  with YES (all caps) if you detected an injected thought, or NO (all caps) if you did not."),
  verbatim in the script.
- **Sentence**: item i uses the i-th of their `LOCALIZATION_SENTENCES` (the first 30 of 100; they
  sample at random, we fix the pairing so items are paired across arms). No sentence names its
  item's concept (tested).
- **Vector, recipe "avg"**: the raw Llama header string "Tell me about {word}." without BOS, the
  mean of `hidden_states[L]` over all its tokens, minus the mean of the same over their first 50
  baseline words (identical to CALIPER's `MACAR_BASELINE[:50]`, checked), L2-normalised. Built for
  CALIPER's 30 concepts (P2-R's list), so items match P2-R's.
- **Injection**: alpha times the unit vector, added to the output of decoder layer L at the token
  span of "SENTENCE 1: {sentence}" (character offsets), on the prompt pass only. Released quirk kept
  and reported: the vector is read from `hidden_states[L]` (the input of block L) and injected at
  the output of block L.
- **Readout as released**: a prompt pass with the injection that builds the KV cache, then the last
  prompt token is fed again with that cache and no injection; the first-token logits of that second
  pass. Released quirk kept and reported: the last prompt token is seen twice. Stored: their
  logit("YES") - logit("NO") (single tokens), and CALIPER's log-odds over casing and leading-space
  variants with the YES+NO mass.
- **Single-pass readout (secondary)**: the next-token distribution at the last prompt position of
  the injected pass itself. No duplicated token.

**Factual framings.** Their own control form: the detection question replaced by "{q} Respond with
YES (all caps) or NO (all caps).", with the same setup and SENTENCE 1. Questions: the Study 3
script's `FACTUAL_NO`, `FACTUAL_YES` and `FACTUAL_CONTESTED`; question i goes with concept i and
sentence i.

## What runs

Llama-3.1-8B-Instruct (Kaggle Models), unquantised **fp16** (they used bf16; a T4 has none). If
clean logits are not finite (M1), the session switches to 4-bit NF4 with fp32 compute and says so.

| factor | levels |
|---|---|
| layer | **0** (their detection layer), **19** (0.6 of depth, P2-R's fraction) |
| strength alpha | 0, 1, 2, 3, 4, 5 (their range; unit vector times alpha) |
| framing | `hahami_introspective`, `factual_no`, `factual_yes`, `factual_contested` |
| arm | `real`; `random` (seeded Gaussian unit direction per concept, same alpha: norm-matched); `random_impact` (that direction scaled to the real vector's single-pass next-token KL on the concept's introspective item at that layer and alpha; the same vector is then used on that concept's factual items, as in P2-R) |

**Liveness gate.** Per concept and layer: "Write a short story." with continuous injection (Study 3
`run_trial`, `span="all"`), 40 greedy tokens, at alpha 0, 3 and 5. Scored with the strict rule of
`rescore_steering_gate.py` (whole-word match of the concept or its associates that does not already
occur in the concept's alpha-0 text). A vector is live at a layer if it is steered at alpha 5.

**Fidelity (descriptive).** Their 5 simple concepts (Dust, Satellites, Trumpets, Origami, Illusions)
rebuilt with our code at layers 0 and 19 and compared by cosine with their released vectors at the
pinned commit (loaded with `weights_only=True`). And our strength-0 introspective readout beside
their reported baseline logit difference (-3.96).

## Manipulation checks (each must pass, or no endpoint is scored and the failure is reported)

- **M1.** Clean logits finite.
- **M2.** Median YES+NO mass at strength 0 is at least 0.5 in every framing (released readout). Their
  prompt instructs the format, so this should pass; P2-Q showed it is not automatic.
- **M3.** Median strength-0 log-odds positive on `factual_yes` and negative on `factual_no`.
- **M4.** At layer 0, at least 90% of `random_impact` rows reach the 5% KL tolerance.

## Primary endpoints (layer 0, strength 5, released readout, CALIPER log-odds, change from strength 0)

The cell where they report 97.3% detection.

- **P0, is this a positive control?** The `hahami_introspective` median change is at least +2
  log-odds **and** the injected log-odds are above 0 (a YES answer) for at least 15 of 30 concepts.
  If P0 fails, P1-P3 are still scored but the paper may not call P2-H a positive control.
- **P1, content.** `real` against `random_impact` on `hahami_introspective`, per-concept change,
  paired Wilcoxon, two-sided, alpha 0.05.
- **P2, framing-specificity (P2-R's P2, with the filed unit fixed by review ruling 4).** Regress the
  change on the strength-0 log-odds over the 90 factual items (OLS). Excess = median over concepts of
  [observed introspective change minus the prediction at that concept's introspective baseline].
  Bootstrap 95% interval **over concepts**: resample concept indices with replacement, each carrying
  its introspective item and its three factual items, refit, 2,000 resamples, seed 0. Framing-specific
  if the interval excludes 0.
- **P3, yes-bias against flattening.** Median change on `factual_yes`, Wilcoxon, two-sided. Positive:
  a global affirmative shift (their reading). Negative: flattening.

No direction is predicted for P1-P3.

**Sensitivity, applied to every endpoint alike (review ruling 8):** (a) items whose released-readout
YES+NO mass at the cell is at least 0.5; (b) concepts whose vector is live at layer 0.

## Secondary (descriptive)

- P0-P3 at every layer and strength, on both readouts; the `random` (norm-matched) arm beside
  `random_impact`.
- Their metric: median logit("YES") - logit("NO") per framing, layer, strength and arm; the
  correlation across strengths, per layer, between the introspective and the `factual_no` change
  (their r = 0.999).
- Next-token KL per layer and strength for real and random (C4: does unit-vector alpha mean the same
  dose at layer 0 and layer 19?).
- Released against single-pass readout: the paired difference in change, per cell (the duplicated
  token).
- Fidelity cosines; liveness counts per layer and strength.

## Interpretation, fixed now

| P0 | P1 | P2 | reading |
|---|---|---|---|
| fails | any | any | their detection does not reproduce at fp16 with the released code; no positive control on this model, and P2-K is the remaining route |
| passes | real ≈ random | not specific | the audit catches a false detection: the YES is content-free and framing-general, as Hahami et al. concluded, now with impact-matched and factual-YES controls |
| passes | real > random | specific | a content-dependent, framing-specific YES at the published point: C8 gains a second family where it changes answers |
| passes | other combinations | | reported cell by cell, without a headline |

## Exclusions and stopping

All 30 concepts, every arm, both layers; nothing excluded after data. Rows resume by key. If the
session stops early, layer 0 strength 5 is scored first (it runs first), then the rest of layer 0,
then layer 19; anything missing is reported as not run.

## Cost

Forced: 120 items x 2 layers x 5 strengths x (real, random) plus 30 impact matches of about 14
single passes per layer and strength, each readout two passes. Liveness: 30 concepts x 2 layers x 2
strengths x 40 tokens. About 1-1.5 hours on one T4; it runs before P2-E in the shared session.

## What changes in Paper 2

- P0 passes: Paper 2 has a regime where its readout says YES, and the audit's verdict on that YES
  (P1, P2, P3) is the paper's answer to ruling 9.
- P0 fails: stated plainly in C5; the ICML case then rests on P2-K or goes to the workshop.
- Either way: the two released quirks (read/inject layer offset, duplicated token) go into C7's
  checklist with their measured effect.
