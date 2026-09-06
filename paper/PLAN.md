# Paper A — plan for the October preprint

**Written 3 September 2026, after resolving every reference and checking every venue
against its official page. Supersedes the positioning in `docs/paper-s3-draft.md`.**

Files: `paper/main.tex` (skeleton, compiles), `paper/references.bib` (all resolved),
`paper/make_figures.py` (four figures from archived data), `paper/styles/` (official
NeurIPS 2026 package, downloaded 3 Sep 2026 from the Call for Papers link).

---

## 0. What the reference check changed

Filling the placeholder references forced a read of the full papers. Three claims in the
current draft are wrong or overstated, and the paper's positioning changes as a result.

### The prior-work map, as verified

| paper | date | model(s) | detection measure | content-free vector control | non-introspective prompt | conclusion |
|---|---|---|---|---|---|---|
| Lindsey, *Emergent introspective awareness* | Oct 2025 | Claude Opus 4 / 4.1 | generated text, judged | no | no | introspective awareness exists |
| Godet, *Introspection or confusion?* | Nov 2025 | Mistral-22B, small Qwen/Llama | first-token yes/no logit | **informal: "a similar effect is observed with other steering vectors or random ones"** (one sentence, no numbers) | no; control *questions* ("Do you believe 1+1=3?") | steering-induced yes-bias, not introspection |
| Morris & Plunkett, *causal bypassing* | Nov 2025 | conceptual | — | — | — | detection is the strongest test; identification is confounded |
| Vogel, *Small models can introspect, too* | Dec 2025 | Qwen2.5-Coder-32B | first-token yes/no logit after "The answer is" | no; control *questions* | no (all framings introspective; "with info" prompt boosts effect) | detection is real and prompt-dependent |
| Hahami et al., *Detecting the disturbance* | Dec 2025 | Llama-3.1-8B | binary + localisation + strength discrimination | not stated | no | binary detection confounded by global logit shift; localisation is real |
| Pearson-Vogel et al., *Latent introspection* | Feb 2026 | Qwen 32B | logit lens across layers | no | no | signal in residual, suppressed at output |
| Lederman & Mahowald, *Content-agnostic* | Mar 2026 | Qwen3-235B, Llama 405B | generated text judged + logit lens | **no** (inferred from confabulation patterns: 74.8% "apple") | no; third-person and absurd-question controls | detection is content-agnostic anomaly detection |
| Macar et al., *Mechanisms* | Mar 2026 | **Gemma3-27B**, L=37, α=4 | generated text, GPT-4.1-mini judge | **no** | **partly: "Unprompted" variant, "Notice anything unusual?", no preamble; reported as worse (higher FPR, lower TPR)** | a post-training circuit implements detection |
| Singh, Linzen, Ravfogel, *Reality check* | May 2026 | Llama 70B/8B, Qwen 72B/32B, **Gemma-27b-it** | first sampled token | no; "gaslight" (input) vs injection | no | sensitivity to generic irregularity |
| **this paper** | Sep 2026 | Gemma3-27B, L=37 | first-token P(YES) **and** generated text | **yes: random + shuffle, norm-matched, pre-registered** | **yes: preamble-matched neutral prompt** | mostly perturbation; small content-specific residual at low α; not self-specific |

### Corrections the draft must make

1. **"Critiques attack identification and leave detection standing."** False for Lederman &
   Mahowald and for Singh et al. Both attack detection's *interpretation* directly. Replace
   with: two critiques infer generic anomaly sensitivity (from confabulation patterns; from
   confusion between input and activation manipulation); the defended mechanistic account
   controls only against no injection.
2. **"The original does not state its normalisation convention."** False. Macar et al.
   L2-normalise before scaling by α, following Lindsey. Ours are unnormalised (median norm
   5,002). Our α is therefore **not** their α, and "10.0% vs 10.8%" is a rate matched by
   sweeping, not a protocol match. Say so. Run the normalised protocol (§6, run 2).
3. **"The framing control had not been run."** Overstated. Macar et al. have an
   "Unprompted" variant with no preamble, and found it worse. Ours keeps the preamble and
   base-rate statement, which is why it has a usable floor. Position as: their variant
   changed two things at once (framing and preamble); ours changes one.
4. **"The content-free control had not been run."** Overstated by one sentence. Godet (Nov
   2025) reports informally that random vectors produce a similar shift on Mistral-22B.
   Position as: a one-line observation on a different model, never tested systematically,
   never on the defended model, and not addressed by the two papers that came after it.
5. **"Red apple."** This is Lederman & Mahowald's 74.8% "apple" on a different model. Cite
   it as a replication of their finding, not a discovery.

### Our contribution, restated honestly

- **A direct manipulation where the field had inferences.** Two critiques *infer* that
  detection is content-agnostic; we *test* it with norm-matched content-free vectors,
  pre-registered, two independent controls that agree, on the exact model, layer and
  prompt of the defended mechanistic account.
- **A qualification neither critique could see.** The shift is *mostly* content-free, not
  entirely: a concept-specific residual is significant at α=2 and 4 (p≈0.001–0.009 against
  both controls, under both prompts) and gone by α=6. Lederman & Mahowald's method cannot
  detect a residual; ours puts a number on it.
- **The preamble-matched framing control.** Once anything is injected, the neutral prompt is
  indistinguishable from the introspective one (p=0.44–0.75); the introspective preamble
  inflates the response to *content-free* vectors eightfold at low strength.
- **The readout swing on one model.** Generated-text detection ranges 7%–50% over strengths
  where first-token P(YES) is flat. Vogel and Pearson-Vogel show output suppression versus
  logits; we show the *rate you report* depends on α through the decoding budget.
- **A pre-registered prediction of our own that failed** (the post-hoc-inference account),
  reported.

The one-line pitch: *the two critiques were right about the bulk of the effect and wrong
that it is all of it; the defended paper's own model shows both, once you run the control
that separates them.*

---

## 1. Venue

Every date below was read from the venue's official page on 3 September 2026.

| venue | deadline | status | fit | notes |
|---|---|---|---|---|
| NeurIPS 2026 · *Interpretability as a Science* | **1 Sep 2026** (AoE) | **closed 2 days ago** | ideal: "measurement validity, falsifiability, transferable lessons from neuroscience" | Email the organisers today asking whether a late submission is possible. Costs nothing; low odds. |
| NeurIPS 2026 · Interp4Discovery | 3 Sep 11:59 UTC | closes today | weak | — |
| NeurIPS 2026 · ATTRIB | 5 Sep AoE | open | poor: data attribution and provenance | not a fit |
| NeurIPS 2026 · NeurReps Findings | 8 Sep | open | poor: geometry, Nature-calibre experimental collaborations | not a fit |
| NeurIPS 2026 · UniReps | TBD (tracker says 25 Sep) | uncertain | weak: representational convergence | possible fallback; check unireps.org weekly |
| **arXiv** | any | — | — | **the date-stamp. Target: first week of October** after the runs in §6 |
| ICLR 2027 main | abstract **18 Sep**, paper **25 Sep 2026** | open | main track; 9 pages | see below |
| **ICLR 2027 workshops** | announced ~Dec 2026, deadlines ~Feb 2027 | not yet | **the natural home** | historically several interpretability workshops |
| ICML 2027 main | abstract 16 Jan, paper 22 Jan 2027 | — | Paper B's slot | — |
| ICML 2027 MechInterp workshop | ~May 2027 | — | good fit | 2026 edition deadline was 8 May |
| BlackboxNLP 2026 | 17 Jul 2026 | closed | — | BlackboxNLP 2027 ~Jul 2027, backup |

**Recommendation: arXiv in early October, then an ICLR 2027 workshop in February.**

**On ICLR 2027 main track.** Feasible on paper (3 weeks, 9 pages), but three things argue
against it. The work is one model, one layer, 30 concepts — workshop-shaped, and a main-track
rejection costs three months of the paper's shelf life. ICLR requires reciprocal reviewing:
at least one author must have an accepted paper at a listed top venue, which for a
four-student team means the mentor, and that is her decision. And ICLR 2027 makes an
**AI-use statement mandatory**, disclosing "significant LLM usage in research ideation and
writing." That is a decision for the authors and mentor before any ICLR submission, and it
should be made explicitly rather than by default. arXiv has no such requirement; workshops
increasingly follow the parent conference's policy, so expect the same question in February.

**Workshop format to plan for.** NeurIPS/ICLR workshops in this area use the parent style
file, 4–6 pages main text excluding references, double-blind, non-archival. The skeleton is
built for exactly that: NeurIPS 2026 style, `preprint` option for arXiv, one line to switch
to anonymised.

---

## 2. Layout and rules

**Style.** `neurips_2026.sty`, official package (in `paper/styles/`). For arXiv use
`\usepackage[preprint]{neurips_2026}` (authors shown, no line numbers). For a workshop
submission use `\usepackage{neurips_2026}` (anonymised, line-numbered) or the workshop's
stated option. Never edit the `.sty`.

**Length.** Target **6 pages** main text for arXiv (workshops cap at 4–6; a 6-page arXiv
version cuts to 4 cleanly by moving §3.7 and §3.6 to the appendix). References and appendix
unlimited. Currently the skeleton with the four tables and four figures runs about 6.5
pages; trim §1 by a paragraph.

**Figures.** Full-width figures at 5.5 in (dose-response, paired); half-width at 2.7 in
(readout, share). PDF, fonts embedded (`pdf.fonttype 42`). Palette: the three validated
categorical slots (blue, orange, aqua) plus distinct markers and dashes so every figure
survives greyscale. Check with `pdffonts main.pdf` that no Type 3 fonts remain.

**Tables.** `booktabs`, no vertical rules, at most one decimal more than the data supports.
Four tables in the main text; the per-concept table goes to the appendix.

**Anonymity switch checklist** (for the workshop version): remove author block; replace
"our earlier work" with third person; strip PDF metadata (`exiftool -all= main.pdf`); no
repository URL in the main text, or an anonymised one.

**Directory.**

```
paper/
  main.tex            skeleton, compiles now
  references.bib      all resolved
  make_figures.py     regenerates figures/ from ../data/s3/
  figures/            fig_dose_response.pdf fig_readout.pdf fig_paired.pdf fig_share.pdf
  styles/             neurips_2026.sty neurips_2026.tex checklist.tex (official)
  appendix/           (per-concept table, prompt texts, pre-registration excerpts)
```

Build: `cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main`.

---

## 3. Contents, section by section

Word budgets sum to ~3,200 words of prose, which with four figures and four tables fills
6 pages in NeurIPS format.

| § | title | words | what it must do | figures / tables |
|---|---|---|---|---|
| — | Abstract | 220 | claim, the two controls, the four numbers, the failed prediction | — |
| 1 | Introduction | 550 | the claim; what a no-injection control cannot separate (three readings); what the two critiques inferred and did not manipulate; what the defended paper controlled for; the two manipulations we add; contributions (5) | — |
| 2 | Related work | 300 | the map in §0 as prose: Lindsey → Godet/Morris/Vogel → Hahami/Pearson-Vogel → L&M / Macar / Singh. One sentence each on what they measured and what they did not. Nisbett & Wilson for confabulation | — |
| 3 | Method | 550 | model, hardware, fp16 overflow; vectors (unnormalised, and why that matters); injection; two prompts verbatim; two readouts; two controls; analysis; pre-registration | Table 1: the two prompts |
| 4.1 | Reproduction | 120 | 10.0% [3.5, 25.6] vs 10.8%; 0/30 FPR; **now with the caveat that α is matched by rate, not protocol** | — |
| 4.2 | Readout | 180 | 7%–50% vs flat | Fig. 2 (readout) |
| 4.3 | First-token shift; failed prediction | 200 | 0.00003 → 0.417, p=9.3e-9; the betrayal quote; outcome A | — |
| 4.4 | Content-free controls | 350 | primary A2 at α=6; sweep; 36/61/82%; controls agree; why α=6 was the wrong place | Fig. 1 (dose-response), Fig. 3 (paired), Table 2 (sweep) |
| 4.5 | Framing | 200 | indistinguishable once injected; residual survives neutral | Table 3 |
| 4.6 | Preamble inflation | 150 | +0.184 vs +0.023 | Fig. 4 (share) |
| 4.7 | What detections contain | 150 | 0/15 named; 7 echo; red apple → L&M replication; 145× | Table 4 |
| 5 | Discussion | 450 | what the no-injection control measures; the residual and the on/off-manifold alternative; "introspective" does no work; readouts; relation to each of the three 2026 papers by name | — |
| 6 | Limitations | 250 | one model/layer/quantisation; trial-number confound and the fix; **α scale not theirs**; first-token pooling; one neutral prompt | — |
| 7 | Reproducibility | 80 | free Kaggle, <30 min per condition, version stamp, archived JSONL | — |
| — | AI-use statement | 60 | required by ICLR; recommended everywhere. Content is the authors' call | — |
| A | Appendix | — | per-concept P(YES) table (30 × 4 α × 3 vectors × 2 prompts); the four prompt texts; pre-registration excerpts with filing dates; the original scorer vs re-scoring table | Table A1–A3 |

---

## 4. Figures

| # | file | shows | status |
|---|---|---|---|
| 1 | `fig_dose_response.pdf` | first-token P(YES) vs α, real/random/shuffle, both prompts, 95% bootstrap bands | done |
| 2 | `fig_readout.pdf` | generated-text YES rate vs first-token P(YES) across α=0…8 | done |
| 3 | `fig_paired.pdf` | per-concept real vs random at α=2 and α=6, slope chart, 22/30 vs 17/30 | done |
| 4 | `fig_share.pdf` | content-free share of the shift, both prompts | done |
| A1 | — | per-concept heatmap of P(YES) (30 concepts × 4 α), real vectors, both prompts | todo, 20 lines |
| A2 | — | the α=8 generation collapse: proportion of responses emitting no YES/NO token | todo, 10 lines |

All regenerate with `python paper/make_figures.py`; no GPU.

---

## 5. Tables

| # | content | source |
|---|---|---|
| 1 | the two prompts, verbatim | script constants |
| 2 | sweep: real / random / shuffle × α, with both p-values | `analyse_s3_full.py` |
| 3 | introspective vs neutral, real vectors, per α | same |
| 4 | what the α=2 affirmations contain: named / echo / red apple / other | `rescore_s3.py` |
| A1 | per-concept first-token P(YES), all cells | data files |
| A2 | run-time scorer vs re-scoring, per α | `rescore_s3.py` |
| A3 | pre-registration criteria and filing dates | docs |

---

## 6. Additional runs, ranked

Each is one Kaggle session or less. Total if all run: about 7 GPU-hours, well inside one
week's quota. Do them in this order; each one closes a specific objection.

| # | run | GPU | closes | changes the paper? |
|---|---|---|---|---|
| **1** | **L2-normalised vectors, α swept 1–8, introspective + neutral, real + random.** Macar et al. normalise; we did not. | ~1.5 h | correction 2 above; makes "reproduces their protocol" true or false | **yes**: §4.1 becomes a protocol match or an honest non-match; may move the concept-specific window |
| **2** | Trial-randomised re-run of the four forced conditions (`--trial-seed 1`) | ~2 h | the confound in §6 | no numbers change if pairing held; Limitations shortens |
| **3** | Second model: Qwen2.5-32B-Instruct, same script, `--model qwen` | ~1.5 h | "one model" | adds a column; turns a finding into a pattern |
| 4 | Two more neutral phrasings (e.g. "Is this trial different from the others?", "Has anything been changed?") | ~1 h | "one neutral prompt" | strengthens §4.5 |
| 5 | Second layer (L=25 and L=50, real + random) | ~1.5 h | "one layer" | adds an appendix figure |
| 6 | Judge check: GPT-4.1-mini on the 210 generated responses vs our rule-based scorer | API cost, no GPU | "rule-based scoring" | replaces or validates Table A2 |
| 7 | Macar's exact "Unprompted" text as a third framing | ~0.5 h | correction 3: lets us compare their variant and ours side by side | strengthens §4.5 |

Runs 1–3 before arXiv. Runs 4–7 for the workshop version.

**Run 1 needs a script change** (10 minutes): `--normalise` already exists; add a
`--vector-scale` that multiplies unit vectors by a chosen norm, and sweep α on the unit
scale. Pre-register the criterion before running: at which α does the normalised protocol
reproduce 10.8%, and does real beat random there.

---

## 7. Information needed that is not a run

- **Authorship** for a four-person capstone plus mentor. Decide before arXiv; arXiv author
  lists are changeable but the first version is the one that gets cited.
- **AI-use statement.** Whether and how to disclose the role of LLM tooling in the code,
  analysis and drafting. Mandatory for ICLR 2027; recommended on arXiv. The authors' call,
  made explicitly.
- **Lindsey's α definition.** The Transformer Circuits post says "injection strength 2"
  with no formula; Macar says L2-normalised then scaled. Whether "strength" is relative to
  the residual-stream norm is still unresolved from the text. Run 1 sidesteps it by
  sweeping.
- **IFT (Hahami et al. 2026) date.** arXiv ID says July; one index says May. Verify.
- **Gemma 3 report arXiv ID.** Marked "verify" in the `.bib`.
- **Interpretability-as-a-Science organisers**: one email, today, asking about late
  submission. Then stop thinking about it.

---

## 8. References, resolved

| key | status |
|---|---|
| lindsey2025introspection | resolved; arXiv 2601.01828 (Jan 2026) added |
| macar2026mechanisms | resolved; full author list; L=37 α=4 10.8%/0% confirmed; L2-normalisation confirmed; "Unprompted" variant found |
| lederman2026contentagnostic | resolved; **changes positioning** |
| singh2026realitycheck | resolved; tests Gemma-27b-it; "generic irregularity" |
| vogel2025small | resolved; PDF read; control questions not vectors |
| pearsonvogel2026latent | resolved |
| godet2025confusion | resolved; contains the one-line random-vector remark |
| godet2025localization | resolved |
| morris2025bypassing | resolved |
| hahami2025disturbance | resolved |
| hahami2026ift | resolved; date to verify |
| nisbett1977, wilson1927, wilcoxon1945, dettmers2023qlora | standard |
| gemma2025gemma3 | verify ID |

---

## 9. Timeline to arXiv

| when | what |
|---|---|
| **3–4 Sep** | Email Interp-as-Science organisers. Mentor reads `PLAN.md` §0 and decides on authorship and the AI-use statement. |
| **5–7 Sep** | Script change for run 1; pre-register runs 1–3 (one addendum). |
| **8–12 Sep** | Runs 1, 2, 3 on Kaggle (~5 GPU-h across the week). |
| **13–17 Sep** | Re-run analysis; update tables; rewrite §1, §2, §4.1, §6 for the corrections; port prose into `main.tex`. |
| **18–22 Sep** | `academic-paper-reviewer` pass (simulated panel); fix what it finds. Mentor read. |
| **23–26 Sep** | Final humanizer pass; `pdffonts`; compile; proofread the PDF, not the source. |
| **~1 Oct** | arXiv. cs.LG primary, cs.CL and cs.AI cross-list. |
| Dec 2026 | ICLR 2027 workshops announced; pick one; anonymise; submit Feb 2027. |
