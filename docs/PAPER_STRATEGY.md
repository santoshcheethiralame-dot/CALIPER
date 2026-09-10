# Paper strategy

**Written 9 September 2026, after B-1b landed and the second scout closed.**
Companion to `BENCH_SPEC.md` (what we built), `CITATIONS.md` (what we cite), and
`PIVOTS.md` (how we got here).

This is a drafting plan, not a draft. Its most important section is the claim-evidence map,
because that is where a paper is usually lost.

---

## 1. The story, in one paragraph

Interpretability runs on reliability heuristics — restart agreement, held-out fit, method
disagreement — that decide whether a result is trusted. None has been calibrated against a
known-correct answer, because calibrating one requires exactly the ground truth whose
absence motivated it. There is one place that ground truth is free and exact: an MLP
neuron's input weight column *is* the direction it reads. We calibrate the heuristics
there, and find that the check the field uses most is significantly worse than one the fit
already computes for free.

**One sentence for the abstract's first line:** *The checks interpretability uses to decide
what to trust have never themselves been checked; here is the one place they can be.*

---

## 2. Target venues, in order

| venue | fit | timing | why |
|---|---|---|---|
| **TMLR** | **primary** | rolling, no deadline | Accepts on **correctness and clarity**, explicitly not novelty or significance. A careful methods paper with honest negatives is what it exists for. **This is the answer to the grade risk** |
| **NeurIPS 2027, Evaluations & Datasets** | strong | abstract ~May 2027 | Scope explicitly covers *"work that analyzes strengths, limitations, or failure modes of existing benchmarks or evaluation practices"* and *"negative results are welcome"*. That is a description of this paper. **NeurIPS 2026 closed in May 2026** |
| Interpretability workshops | backup | rolling | Fast, visible, lower stakes |
| ICLR/ICML main track | not the plan | — | Two small models and a narrow substrate |

**arXiv first, and soon.** It date-stamps the weight-column calibration and costs nothing.

---

## 3. Section outline

1. **Introduction** — the heuristics, the circularity, the one escape, the result.
2. **The substrate** — why `s·w` is exact ground truth, the verification, and the honest
   limit stated immediately rather than in a limitations paragraph.
3. **What we calibrate** — the four estimators (E0.5) and the four signals.
4. **Calibration results** — the primary endpoint, operating characteristics, reliability
   diagram, transfer to a second family.
5. **The error budget** — the variance decomposition. Device, restarts, scale, steps,
   layer, family.
6. **Limitations** — substrate narrowness, model scale, what the calibration cannot claim.
7. **The bench** — adapter interface and spec-sheet format, as a released artifact.

**Section 5 is the differentiator.** Sections 1–4 are a good calibration paper; section 5
is a question nobody has asked.

---

## 4. Claim–evidence map

The hard constraint: **every claim in the abstract and introduction must be supported by
evidence already on disk.** Anything below marked *needs evidence* stays out of the
abstract until its run lands.

### Supported today

| claim | evidence | status |
|---|---|---|
| An MLP neuron's input weight column is exact ground truth for the direction it reads | residual 3.3e-06 (GPT-2), correlation 1.0000000000 (Pythia-160m) | **supported** |
| Classical spike-triggered estimators fail almost completely on LM activations | E0.5, n=30: STA 0/30, decorrelated STA 1/30, STC 0/30, fitted bottleneck 26/30 | **supported** |
| The mechanism is non-Gaussian stimulus and non-monotone GELU driving Bussgang's constant toward zero | E0.5 plus the analytic argument | **supported, but Bussgang is an ABS citation — read before asserting** |
| The working estimator fails silently on a substantial minority of real units | B-1b 52/300 (17%); C13 23/100; Pythia 7/100 | **supported** |
| **Restart agreement is significantly worse than held-out R2 at predicting failure** | B-1b, n=300, 52 failures: AUC 0.778 vs 0.942, DeLong z = -5.22, p < 0.0001 | **supported — on GPT-2, layer 6** |
| The gap is practical, not just statistical | At 50% catch, held-out R2 discards 0 of 248 good units; restart agreement discards 40 | **supported** |
| Aggregate quality statistics hide these failures | Pythia median alignment 0.9994 in a set containing a unit recovered at 0.0761 | **supported** |
| A stable-and-wrong class exists | 6 of 52 failing units have restart agreement > 0.95 at 2 restarts | **supported** |
| Restart count is a lever on recovery, not only a diagnostic | 77/100 at 2 restarts, 91/100 at 5 | **supported** |

### Needs evidence — currently running or queued

| claim | what would support it | run |
|---|---|---|
| ~~The calibration transfers across model families~~ | **PARTIAL, 10 Sep.** Rank order of all four signals is identical on both families (R2 > restart > disagreement > spread). But DeLong is significant only on GPT-2 (p=1.8e-07); Pythia gives p=0.114 with **13 failures**. Per Addendum 1's pre-commitment, the Pythia arm is reported **underpowered and inconclusive** and is NOT re-run at larger n | **done** |
| ~~The scale trend is real, or is an optimiser artefact~~ | **RESOLVED 10 Sep: artefact. Scale claim withdrawn.** 0 of 19 failing 1.4b units converged (R2 > 0.99); held-out R2 detects them at AUC 0.980/0.996. They are under-fitting, not silent failure | **done** |
| Design choice contributes materially to the reported failure rate | Layer and family components | **B-7, B-8, queued** |
| Failure classification depends on the **batch**, not only the device | **B-0 n=50, 10 Sep.** 4/50 flip CPU vs CUDA, Wilson [0.032, 0.188]. But the bigger result: holding units, seeds and device fixed and changing only batch size moves **5 of 16** across the bar on CPU and 2 on CUDA. Mechanism is `batched.py:30` - one `randn(n,d,k)` draw, so a unit's init depends on how many units share the batch | **supported, and stronger than the device claim** |

### Claims to weaken or cut

| tempting claim | why it must not be made |
|---|---|
| "Restart agreement does not work" | **False.** It reaches AUC 0.778 and catches most failures. The supportable claim is *dominated*, not *useless*. An early framing in this project said useless; that was wrong |
| "The failure rate rises with model scale" | Pre-registration forbids it until the steps check runs, and the artefact reading currently looks more likely |
| "This calibration transfers to SAE latents / persona vectors" | The substrate does not extend there. **Untestable by us**, and must be labelled an assumption |
| "Interpretability results are unreliable" | Far broader than the evidence. We measured one estimator family on two small models |

**B-0 resolved, and it changed the claim.** Extending to n=50 gave 4/50 rather than 5/16 —
and since the draw is nested, that discrepancy was itself the finding. The two runs used
different batch sizes, and **batch size alone moves units across the pass bar on CPU as well
as GPU**. The mechanism is deterministic: `batched.py:30` draws one `randn(n, d, k)` from a
single generator, so a unit's initialisation depends on how many units share its batch.

**The claim to make is therefore about batching, not hardware:** *the direction an estimator
recovers for a unit depends on which other units were fitted alongside it.* That is
reproducible, mechanistically explained, and affects anyone who batches for speed — which is
everyone. The device result stands as a secondary observation, valid within a fixed batch.

---

## 5. Rejection risks, and what answers each

Mapped against the standard rejection dimensions.

| risk | severity | answer |
|---|---|---|
| **"Only 124M and 160M models"** | **high** | The B-11 ladder reaches 1.4B. The substrate needs white-box weight access, and the claim is about estimator behaviour, not scale. **State it in the abstract** |
| **"Only MLP units reading their own layer"** | **high, unfixable** | It is the only non-special case anyone has. Goes in the abstract, not the appendix. Reviewers forgive a stated limit far more readily than a discovered one |
| "Where is the method? This only measures things" | **high** | This is why the venue matters. TMLR and the E&D track both accept analysis of evaluation practice as a contribution. Do not submit to a main track that expects a method |
| "Disagreement-as-a-signal is already known" | medium | True, and cited. **Our claim is the calibration, not the idea.** The novelty is the operating characteristic against a known-correct direction |
| "The improvement is marginal" | low | ΔAUC 0.164 at p < 0.0001, and 0 versus 40 discarded units at equal catch. Not marginal |
| "Underpowered" | **answered in advance** | 52 failures at n=300, inside the 36–142 the power literature requires. Addendum 1 documents the interim look that prompted the extension |
| "Non-standard statistics" | low | DeLong for correlated AUCs, Wilson intervals, McNemar for paired designs, Benjamini–Hochberg for multiplicity. **Adopt NIST AI 800-3 explicitly** |

---

## 6. Rigour upgrades to make before submission

1. **Reliability diagram per signal.** We report AUC and operating points but no calibration
   curve — does signal value *X* correspond to failure probability *Y*? Standard in the
   failure-prediction literature and about 20 lines offline. **Cheapest credibility gain
   available.**
2. **NIST AI 800-3 compliance**, named in the paper. Adopt the **benchmark accuracy vs
   generalized accuracy** distinction: our 300 units estimate a 3,072-unit population.
3. **Design-sensitivity-corrected intervals** beside the naive Wilson ones, per
   `2604.11581` — naive intervals run 40–60% too narrow when design variance is ignored.
4. **Release the bench.** The E&D track requires code and data accessible at submission,
   and non-compliance is grounds for desk rejection.
5. **Pre-registrations as an appendix.** Fourteen filings, including the void runs and the
   two withdrawn claims. Most papers cannot show this and it is unusually strong evidence
   of process.

---

## 7. Drafting order

Write in the order that fails fastest.

1. **Claim-evidence map** — done, section 4 above. Any claim without evidence is cut now,
   not at review.
2. **Figures and tables first.** The operating-characteristic table and the ladder table are
   the paper. If they do not carry the argument alone, prose will not save them.
3. **Methods and results** — mechanical, since the pre-registrations already contain them.
4. **Introduction last**, written against the finished results so no claim outruns them.
5. **Abstract very last.**

**One paragraph, one message. State the message in the first sentence.** After each
section, reverse-outline it: write the thesis, then each topic sentence, and check every
topic sentence maps to the thesis. Cut what does not map.

---

## 8. Self-review before submission

Answer all of these in writing and fix what fails.

**Contribution.** What does a reader know afterwards that they did not before? Is the
failure case real rather than contrived? Is the calibration non-obvious given that the
signals themselves are known?

**Clarity.** Could a competent reader reproduce the bench from the paper? Is every term
defined before reuse? Does each paragraph carry one message?

**Empirical strength.** Are the gaps meaningful rather than statistically tiny? Do results
hold across both model families? Are failures reported as honestly as successes?

**Evaluation completeness.** Are all four signals reported, including the one that fails
(k2 gain, AUC 0.367)? Are baselines fair? Is the substrate challenging enough to be
convincing?

**Design soundness.** Is the setting realistic? Does the ground truth have hidden defects?
Do the limitations outweigh the contribution?

---

## 9. What would make this genuinely strong rather than merely sound

Three things, in order of value.

1. **The error budget lands (section 5).** *What fraction of a reported interpretability
   number is design choice rather than signal?* Nobody has asked. Five of six components
   are already measured.
2. **The steps check resolves cleanly.** Either outcome is good: a scale effect that
   survived its own strongest challenge, or a methodological finding that a fixed step
   budget silently under-fits wider models and would have been read as a scale effect.
3. **The hardware result holds at n=50.** *Reproducing an interpretability result on
   different hardware may reproduce the aggregate and not reproduce which units failed* is
   a memorable, checkable, useful claim — if the sample supports it.

None requires a new idea. All three are runs already scheduled or nearly so.
