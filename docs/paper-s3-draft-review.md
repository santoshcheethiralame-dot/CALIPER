# Paper draft: self-review and claim-evidence map

**Companion to `paper-s3-draft.md`, 3 September 2026.** Produced during the humanizer and
research-paper-writing pass. Not part of the paper.

## Section outline

1. Abstract: one paragraph, claim then evidence then scope.
2. Introduction: the claim, why the no-injection contrast cannot separate three
   explanations, the two controls we add, five contributions, what we do not claim.
3. Method: model, vectors, injection, two prompts, two readouts, two controls, analysis,
   pre-registration.
4. Results, seven parts in the order the experiments ran: reproduction, readout
   dependence, first-token shift and refuted hypothesis, content-free control,
   framing control, preamble inflation, what detections contain.
5. Discussion: what the no-injection control measures, what remains, the word
   "introspective", readouts, relation to the original claim.
6. Limitations, reproducibility, references.

## Paragraph roles, Results section

| section | role | one-line message |
|---|---|---|
| 3.1 | evidence | the effect is present in our setup |
| 3.2 | challenge | the reported number depends on the readout |
| 3.3 | evidence, limitation | the shift is real; our mechanism hypothesis was wrong |
| 3.4 | evidence | content-free vectors reproduce most or all of the shift |
| 3.5 | evidence | self-directed framing adds nothing once injected |
| 3.6 | advantage | the neutral prompt is the cleaner instrument |
| 3.7 | evidence | affirmations at low strength contain no concept |

## What the humanizer pass changed

- Removed every em and en dash (the source draft had 41). Replaced with commas, periods,
  colons, or a rewritten sentence.
- Removed bold lead-ins from the limitation bullets and rewrote them as plain paragraphs.
- Converted the three-item "reporting / reporting / answering" list in the introduction
  from a numbered list to prose, since it is an argument rather than an enumeration.
- Replaced "3.5%–25.6%" and similar ranges with "3.5% to 25.6%".
- Stabilised terminology: "generated-text readout" and "first-token readout" throughout;
  previously the draft alternated with "free generation" and "P(YES) at the first token".
- Left the contributions bullet list in place. Bullet contributions are the convention in
  ML papers and the items are not bold-labelled.
- Left one deliberate short sentence ("It moved.") as the single emphatic fragment in the
  paper.

No number, name, date, p-value, quotation, or citation was added or removed. The
reference placeholders remain bracketed on purpose.

## Claim-evidence map

| claim | evidence | status |
|---|---|---|
| The detection effect reproduces in 4-bit on free hardware | 2/30 at α=6 under the pre-registered scorer, CI 1.8% to 21.3%, vs 10.8%; 0/30 at α=0 *(corrected 6 Oct 2026, notebook C40; was a hand-read 3/30)* | supported, weakly: the interval is wide |
| The generated-text rate is a property of the readout | 43% / 17% / 7% / 0% at α=2/4/6/8 (pre-registered scorer; corrected 6 Oct 2026, the old 27% and 7% matched no rule) vs flat 0.43 to 0.50 first-token | supported |
| Injection shifts the first token before any output exists | 0.00003 to 0.417, p=9.3e-9, 28/30 | supported |
| Our post-hoc-inference hypothesis was wrong | same test; pre-registered as outcome B, observed A | supported |
| At α=6 a content-free vector reproduces the whole effect | real 0.417 vs random 0.305, p=0.33, 17/30 | supported (pre-registered primary) |
| A concept-specific component exists at α=2 and 4 | real vs random p=0.0011, 0.0040; real vs shuffle p=0.0099, 0.13 | supported at α=2; α=4 supported vs random only |
| Content-free share is 36% / 61% / 82% | computed from mean rise of random and shuffle vs real | supported (descriptive) |
| The two controls agree | p=0.50, 0.27, 0.60 | supported |
| Self-directed framing adds nothing once injected | intro vs neutral p=0.44, 0.75, 0.70 | supported |
| Concept-specific signal survives the neutral prompt | real vs random under neutral p=0.0087 at α=2 and 4 | supported |
| Introspective prompt inflates response to content-free vectors eightfold | +0.184 vs +0.023 at α=2, random | supported (ratio ≈ 8.0; descriptive, no test) |
| Affirmations at α=2 contain no injected concept | 0 of 15 named it; 7 preamble echo; 3 "red apple" | supported (hand-coded categories, listed in `rescore_s3.py`) |
| Identification, where it occurs, is far above chance | 145× at α=6, p=1.7e-7 | supported |
| "The model detects an injection" rather than "an injected thought" | all of the above | interpretation; scoped in Discussion |
| On-manifold vs off-manifold disruption could explain the residual | none | stated as an open alternative, not a claim |

## Five-dimension self-review

**Contribution.** Clear and bounded. The paper does not claim introspection is absent,
which a reviewer would attack. The contribution is a missing control and what it shows.
Risk: a reviewer says "obvious control, small result". Answer in text: the control had not
been run, it costs one forward pass, and it changes the interpretation of a defended claim.

**Writing clarity.** Every results subsection opens with its message. Terminology is now
stable. The one place a reader may stumble is the "outcome A / A2" labels in 3.3 and 3.4,
which refer to the pre-registration document. Consider replacing with plain words in a
later pass, or adding one sentence that maps the labels.

**Experimental strength.** One model, one layer, 30 concepts. Wilcoxon paired tests are
appropriate for n=30 with non-normal P(YES). The primary test is under-powered to detect a
small real-vs-random gap at α=6 (17/30 is close to chance), and the paper should say
"indistinguishable at this n" rather than imply equality. Currently says
"indistinguishable"; acceptable, but a reviewer may push.

**Evaluation completeness.** Missing: a second model, a second layer, a prompt-sensitivity
sweep for the neutral framing, and the trial-randomised re-run. All four are named in
Limitations. The second model (Qwen2.5-32B) is the cheapest and most persuasive addition.

**Method design soundness.** The two content-free controls are the strongest part. The
readout comparison is sound. The framing comparison is weakened by the different
no-injection floors (0.000 vs 0.188), which the paper reports but does not fully resolve;
the "rise over baseline" framing in 3.6 partly addresses it.

## Unresolved items for the authors

1. Fill the four reference placeholders from the arXiv pages.
2. Decide whether to run the trial-randomised replication before posting. Two hours of
   GPU; would let Limitations say the confound was closed.
3. Decide whether to add Qwen2.5-32B. One session; turns "a finding" into "a pattern".
4. Consider softening "indistinguishable" to "not distinguishable at n=30" in 3.4.
5. Authorship for a four-person capstone.
