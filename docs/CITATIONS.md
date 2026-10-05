# Citation ledger

**Every paper this project has leaned on, with what it is for and how far it was verified.**
Started 9 September 2026. `paper/references.bib` holds 16 formal entries; this file is the
superset and the working record.

Updated 5 Oct 2026 from three literature passes (improvement scout, related-work broadening, rigor and grounding). 403 rows added; 170 marked MEM. The level table gains a MEM row. 25 existing rows carry a dated "5 Oct:" note instead of an edit; new material is in §11–§20. Where two passes gave the same work different levels, the lower level is used and the row says so. 6 Oct 2026: 63 rows added from the persona and self-level pass (§21–§22; 36 ABS, 27 MEM, none FULL). 10 existing rows and the §8 list carry a "6 Oct:" note, including corrections to Ferrara 2608.20569 and Zou 2609.35108 (§11b).

**Verify before submission.** The introduction leans on these, and none is FULL yet:

- Song et al. 2025, [2505.20254](https://arxiv.org/abs/2505.20254) (ABS, §11): the direct foil. Read in full; confirm our per-configuration vs per-unit reading and the author list. **6 Oct: CHECKED in the arXiv HTML.** Authors: Xiangchen Song, Aashiq Muhamed, Yujia Zheng, Lingjing Kong, Zeyu Tang, Mona T. Diab, Virginia Smith, Kun Zhang. The proxy claim is that PW-MCC "follow[s] the same trend as GT-MCC" over training on synthetic data (m=8, d_gt=16, k=3, five seeds), which is an aggregate trend, not a per-feature check. On Pythia-160M and Gemma-2-2B there is no ground truth; they validate with LLM-judged semantic similarity. Call: "routinely report quantitative consistency scores (e.g., PW-MCC)". Our aggregate-vs-per-unit reading holds.
- Ben-David, von Luxburg & Pál 2006 (ABS, §13): pull the exact statement from the full text before quoting. **6 Oct:** bibliographic record confirmed ("A Sober Look at Clustering Stability", COLT 2006, DOI 10.1007/11776420_4). The Springer text is behind a login. Still need library or author-copy access before quoting it.
- Knight & Leveson 1986 (ABS, §13): confirm the quote and counts in the IEEE TSE text. **6 Oct:** bibliographic record confirmed (IEEE TSE 1986, DOI 10.1109/TSE.1986.6312924). The author copy at sunnyday.mit.edu refused the connection. The 27-version, 1M-test counts are still unverified.
- Yu & Kumbier 2020, PCS (ABS, §16): quotes come from arXiv v5; check against the PNAS text. "Necessary but not sufficient" is not their phrase. **6 Oct: CHECKED in the PNAS PDF text.** Verbatim: "The PCS workflow uses predictability as a reality check"; "random initial values in gradient descent and stochastic gradient descent. These random components provide natural model perturbations that can be used to assess the stability"; "Stability is a common sense principle and a prerequisite for knowledge"; PCS principles "serve as minimum requirements"; step 2, "Prediction screening: ... screen out models that do not fit the data"; stability selection "is similar, but without the prediction error screening".
- JCGM 200:2012 §2.14 Note 2 and §2.15 Note 4 (MEM, §15): quote verbatim from the BIPM PDF. **6 Oct: CHECKED on jcgm.bipm.org/vim/en/2.14.** Trueness: "closeness of agreement between the average of an infinite number of replicate measured quantity values and a reference quantity value". Note 2: "Measurement trueness is inversely related to systematic measurement error, but is not related to random measurement error." §2.15 not yet fetched.
- Ferrara 2026, [2608.20569](https://arxiv.org/abs/2608.20569) (ABS, §11): Study 3 scoop; read in full. **6 Oct: abstract checked.** Title: "Open-Weight Masked Introspection: Measuring What Language Models Can Report About Their Own Computation". Single author. Eight open-weight models from seven families; none beats chance at detecting real versus sham interventions (AUROC ~0.5007, 78,000 measurements); controls include impact-matched random perturbations. Gemma-3-27B is not named in the abstract page. The 7-of-8 "random beats targeted" figure in §11b is from a summary and is not in the abstract, so recheck it in the body. **6 Oct, persona and self pass:** the sentence "Random-direction rate exceeds the intervention rate in seven of the eight models" is in the HTML body, not claimed as significant. Impact-matched random covers Qwen2.5-7B and Mistral-7B only; the other six are unit-norm matched (§11b row corrected). Still not read end to end.
- Gerasimov et al. 2026, [2606.12138](https://arxiv.org/abs/2606.12138) (ABS, §3): read the subspace result before writing the reconciliation. **6 Oct: abstract checked.** Six authors (Gerasimov, Rusalev, Balagansky, Laptev, Kurochkin, Gavrilov). Stable features "carry most of the reconstruction- and prediction-relevant signal". Unstable ones are "reproducible low-dimensional structure" resolved differently per seed, and a synthetic model shows subspace-level recovery. They do not equate stability with correctness, so frame them as an ally on subspaces, not a foil.
- Tan et al. 2025 and Garg et al. 2022 (ABS, §13); Kirsch & Gal 2022 and D'Amour et al. 2022 (MEM, §13).
- Jacobs & Wallach 2021 and Wallach et al. 2025 (ABS, §15). **6 Oct:** Jacobs & Wallach abstract checked (arXiv 1912.05511). It contributes "conceptualizations of construct reliability and construct validity"; the definitions sit in the body, so read §2-3 before paraphrasing.
- CAST 1991 and Fleming & DeMets 1996 (ABS, §14): confirm 63/755 vs 26/743. **6 Oct:** CAST record confirmed (Echt et al., NEJM 1991, DOI 10.1056/NEJM199103213241201). The abstract is not exposed by the API, so the 63/755 vs 26/743 counts are still unverified.
- Shah & Nagaraja 2020, ICISS (ABS, §20): carries the mentor link; full text unread.
- Bhattarai & Alhanai 2026, [2609.14151](https://arxiv.org/abs/2609.14151) (ABS, §21): closest L2 prior art. Read in full; record the title; confirm its only ground-truth check is the synthetic toy and that it never plants a direction in a real model, before calling the L2 slot open.
- Macar et al. 2026, 2603.21396 (BIB, §8): confirm the vector read position (last token of the chat-templated prompt) and the precision used, from the released code config. The dead-vector finding stands or falls on whether our 4-bit template-tail extraction matches theirs.
- Singh, Linzen & Ravfogel 2026, 2605.26242 (BIB, §8): check whether it already compares first-token and generated-text readouts on the same trials before claiming readout dependence as new. It includes Gemma-3-27B-It.
- Hahami et al. 2512.12411 (BIB, §8): settle the author list and the version cited. The two titles carry different numbers (10-way localisation 88% vs 4-bin strength up to 70%) and opposite depth trends (early layers only vs rising with depth).

## Verification levels — this column is load-bearing

| level | meaning |
|---|---|
| **FULL** | main text read, claims checked against it |
| **ABS** | abstract or a search summary only — **do not cite a specific claim from these without reading first** |
| **BIB** | already formalised in `paper/references.bib` |
| **MEM** | recalled from memory during a literature pass, not checked against any source — **must be verified before use** |

The project has been burned by abstract-level reading before: two novelty claims in the
Proposal III set were killed by an audit that found direct collisions the abstracts did not
reveal. **Anything marked ABS is a lead, not a citation.**

---

## 1. The direct competitors — what the bench is positioned against

| id | title | level | why it matters |
|---|---|---|---|
| [2609.03026](https://arxiv.org/abs/2609.03026) | ObserverBench | **FULL** | Scores "observers" by downstream action quality on held-out cases. **States outright it has no ground-truth-free failure detector.** The single most important positioning target — different question (actions, not directions), and the gap we fill is one they name. 5 Oct: author V. Erramilli (2 Sep 2026); scores estimators against task loss, not checks against ground truth; scoop risk low |
| — | [MIB / Benchmarking Interpretability](https://benchmarking-interpretability.csail.mit.edu/) (MIT CSAIL) | ABS | Head-to-head method comparison. Cite as the crowded lane we are *not* entering. 5 Oct: arXiv [2504.13151](https://arxiv.org/abs/2504.13151) (Mueller, Geiger, Wiegreffe et al., ICML 2025); finds SAE features no better than neurons |
| — | AxBench | ABS | Steering evaluation suite. Same role. 5 Oct: arXiv [2501.17148](https://arxiv.org/abs/2501.17148) (Wu, Arora, Geiger et al. 2025); prompting beats SAEs for steering. 6 Oct: abstract numbers on Gemma-2-2B/9B. Steering: prompting 0.894, DiffMean 0.239, SAEs 0.165. Detection AUROC: DiffMean 0.942, linear probe 0.940, ReFT-r1 0.938, SAEs below. Scores against behaviour and labels, not a known direction. SAE rebuttals are in §21 |
| [2608.19338](https://arxiv.org/html/2608.19338v1) | Mechanistic Tomography | ABS | Surfaced in the first harness scout; not yet read |
| [2309.03886](https://arxiv.org/html/2309.03886v3) | FIND: Function Description Benchmark | ABS | Synthetic-function ground truth. Contrast with weight-derived |

## 2. Ground truth for direction recovery — all planted or compiled, none free

| id | title | level | why it matters |
|---|---|---|---|
| [2410.08417](https://arxiv.org/pdf/2410.08417) | Bilinear MLPs enable weight-based interpretability | ABS | Weight space as an *analysis target*. **We use it as a scoring key** — that is the distinction to draw. 5 Oct: Pearce, Dooms, Rigg, Oramas, Sharkey, ICLR 2025 Spotlight; exact interaction tensors would give K≥2 targets but need a bilinear-trained model; whether pretrained bilinear LMs are downloadable is unchecked |
| [2604.06005](https://www.opentrain.ai/papers/disentangling-mlp-neuron-weights-in-vocabulary-space--arxiv-2604.06005/) | ROTATE: disentangling MLP neuron weights | ABS | Same neighbourhood, same distinction. 5 Oct: Avrahamy, Gur-Arieh, Geva; [abs page](https://arxiv.org/abs/2604.06005). Rotates weights to maximise vocabulary-space kurtosis; interprets weights, does not score estimators |
| [2407.13594](https://arxiv.org/abs/2407.13594) | Validating Mechanistic Interpretations (axiomatic) | ABS | Axiomatic validation, not empirical calibration |
| [2506.14002](https://arxiv.org/pdf/2506.14002) | Taming Polysemanticity: Provable Feature Recovery | ABS | Source of the **Feature Recovery Rate** metric; synthetic ground truth. 5 Oct: Chen, Sheen, Xiong, Wang, Yang. **Conflict: the abstract does not define a Feature Recovery Rate.** The best-match decoder-cosine metric is used in 2602.14111 and 2602.14687 (§11); check this paper's full text before attributing FRR to it |
| [2512.05534](https://arxiv.org/pdf/2512.05534) | Unified Theory of Sparse Dictionary Learning | ABS | Spurious minima in dictionary learning — relevant to the basin argument |
| — | InterpBench, Tracr | ABS | Compiled ground truth. Named in Proposal III's own capability table as the ground-truth option; **the weight column is better because it is natural as well as exact**. 5 Oct: InterpBench = [2407.14494](https://arxiv.org/abs/2407.14494) (Gupta, Arcuschin, Kwa, Garriga-Alonso; NeurIPS 2024 D&B), 86 SIIT-trained models, 85 from tracr. Its own limitations: models "very small", one algorithm each. Use that against the realism objection; circuits, not directions. 6 Oct: Tracr = [2301.05062](https://arxiv.org/abs/2301.05062) (Lindner et al. 2023), exact but unrealistic compiled weights. SIIT is the known recipe for forcing a latent onto chosen directions; no single-trait-direction SIIT recipe for a pretrained LM was found (L2 design #2, §21) |
| — | CLEVR-XAI; A Benchmark for Interpretability Methods in DNNs (NeurIPS) | ABS | Older ground-truth-evaluation lineage. 5 Oct: the NeurIPS benchmark is Hooker, Erhan, Kindermans, Kim (2019), ROAR, [1806.10758](https://arxiv.org/abs/1806.10758), cited from memory (MEM-level): under remove-and-retrain many attribution methods were no better than random |
| [2506.10920](https://arxiv.org/html/2506.10920) | Interpretable Features from Compositional Neuron Groups | ABS | Unit-level structure. 5 Oct: semi-NMF of GPT-2/Llama/Gemma MLP activations; does not estimate single-neuron input directions against weights |
| [2605.29358](https://arxiv.org/pdf/2605.29358) | Scaling Monosemanticity | ABS | The SAE mainline |

## 3. Reliability signals — validated against behaviour or reconstruction, never a known-correct direction

**This table is the argument.** Every row proposes or studies a reliability signal, and not
one calibrates it against a known-correct answer.

| id | title | level | what it validates against |
|---|---|---|---|
| [2604.17698](https://arxiv.org/html/2604.17698v2) | The Geometric Canary: predicting steerability via representational stability | **FULL** | **Behavioural** — accuracy drop under steering, task degradation under drift. AUC 0.990 on LoRA drift, 69 embedding models. Closest competitor on "ground-truth-free signal", and it never scores against a true direction. 5 Oct: author P. C. Raju; supervised stability predicts steerability at ρ 0.89–0.97 across 35–69 embedding models |
| [2606.12138](https://arxiv.org/abs/2606.12138) | Unstable Features, Reproducible Subspaces: seed dependence in SAEs | ABS | **Reconstruction and downstream prediction.** Proposes feature stability as a quality signal. 5 Oct: Gerasimov, Rusalev, Balagansky, Laptev, Kurochkin, Gavrilov. A p8 quote was checked in the full text during the usage census (upgrade candidate once the subspace result is read). Unstable features sit in reproducible low-rank subspaces. Reply: stability tracks importance, not correctness; test whether our failing units land in a seed-reproducible wrong subspace. Must be reconciled in related work |
| [2607.19386](https://arxiv.org/html/2607.19386) | Building Fast, Evaluating Slow | ABS | Finds **methodological variance exceeds architectural variance** across autointerp metrics. Strong support for the variance-decomposition framing |
| [2505.10399](https://arxiv.org/html/2505.10399v1) | Evaluating Model Explanations without Ground Truth (FAccT 2025) | ABS | The ground-truth-free XAI evaluation line. 5 Oct: DOI [10.1145/3715275.3732219](https://doi.org/10.1145/3715275.3732219) (existence only) |
| [2602.13450](https://arxiv.org/abs/2602.13450) | Inference From Random Restarts | **FULL** | **Theoretical, econometric solver, not neural nets.** Says restart heuristics "lack a formal inferential foundation despite widespread use". Proves uniqueness concentrates *polynomially* while basin size concentrates *exponentially* — **a mechanism for stable-and-wrong: restart agreement measures basin size, not correctness**. 5 Oct: Nehzati & Cussen, revised 21 Jun 2026. Prop. 5.1–5.2 (dominance, exponential), Prop. 5.5 and Thm 5.7 (uniqueness, polynomial) and the "solver-problem mismatch" / "limited visibility" failure modes re-checked through the HTML; table values not transcribed. Assumes finitely many discrete outcomes (thresholded cosine gives that). Precursor: Boender & Rinnooy Kan 1987 (§16). Two agreeing restarts happen with probability ≈ Σ p_i², so a wrong basin of mass 0.8 agrees ≥64% of the time |
| [2501.17727](https://arxiv.org/html/2501.17727) | Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers | ABS | A metric that fails its own sanity check. 5 Oct: Heap et al. 2025 ("SAEs Can Interpret Randomly Initialized Transformers"); cited from memory in the metric-precedents pass |
| [2603.04198](https://arxiv.org/html/2603.04198v2) | Stable and Steerable SAEs with Weight Regularization | ABS | Improving stability, treating it as desirable. 5 Oct: Jedryszek & Crook; L2 penalty raises cross-seed consistency (3 seeds, Pythia-70M) and roughly doubles steering success. Usage-census instance (stability as a training target) |
| [2605.18629](https://arxiv.org/pdf/2605.18629) | Aligned Training for SAE stability | ABS | Same. 5 Oct: Brzozowski & Chung (2026); usage-census instance |
| [2606.02061](https://arxiv.org/pdf/2606.02061) | Ablating Archetypes: stability as an artifact of initialization | ABS | **Counterweight** — stability can be an artifact of init, not a quality signal |
| [2603.20101](https://arxiv.org/pdf/2603.20101) | Pitfalls in Evaluating Interpretability Agents | ABS | **Read this next.** Closest neighbour on the thesis: evaluations overestimate interpretability reliability and obscure failure modes |

## 4. Steering-vector reliability and non-identifiability — Study 2's context

| id | title | level | why it matters |
|---|---|---|---|
| [2602.06801](https://arxiv.org/html/2602.06801v4) | On the Non-Identifiability of Steering Vectors | **FULL** | Reaches P1b's conclusion from **behavioural evidence only**, and states explicitly it does **not** plant known directions and attempt recovery. **P1b is the ground-truth complement to a published behavioural result**. 5 Oct: Venkatesh & Kurapath (6 Feb 2026). No one found plants a persona direction and tests recovery against a same-bank null; Study 2 is corroboration plus a control, low novelty |
| [2407.12404](https://arxiv.org/html/2407.12404v1) | Analysing Generalisation and Reliability of Steering Vectors | ABS | Steerability varies wildly, sometimes negative; prior work reports no error bars. 6 Oct: Tan et al., NeurIPS 2024 ([PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/fb3ad59a84799bfb8d700e56d19c231b-Paper-Conference.pdf)). Steerability is "mostly a property of the dataset rather than the model"; a significant fraction of inputs are anti-steerable; spurious biases drive per-input effects; OOD generalisation brittle. Persona-landscape pass checked the abstract and proceedings page; the other two passes cite it from memory |
| [2505.22637](https://arxiv.org/abs/2505.22637) | Understanding (Un)Reliability of Steering Vectors | ABS | Companion result. 6 Oct: Braun, Eickhoff, Krueger, Bahrainian, Krasheninnikov (28 May 2025); ICLR 2025 Building Trust workshop per [OpenReview](https://openreview.net/pdf?id=JZiKuvIK1t). Cosine of training-set activation differences and positive/negative separation predict steering; vectors from seven prompt types all steer net positive yet sit at cosine 0.07–0.86; effects often opposite to intended. Validated against steering success, not a known direction. Thesis restatement: 2602.17881 (§11a) |

## 5. Statistics and method — the rigour spine

| source | level | used for |
|---|---|---|
| DeLong, DeLong & Clarke-Pearson (1988) | ABS | **Primary endpoint test.** Comparing two correlated AUCs on the same units. Implemented in `b1_signal_calibration.py`. 5 Oct: *Biometrics* 44:837–845, doi:[10.2307/2531595](https://doi.org/10.2307/2531595) (both new passes cite it from memory). Practitioner guidance calls the covariance unstable below ~25 per class (heuristic, not a published threshold): add a permutation or stratified-bootstrap check there. Miscalibrated for nested models (Demler et al. 2012, §18). Pair only within identical units; never across layers |
| Obuchowski–McClish; Hanley–McNeil | ABS | Sample size for AUC differences. **Source of the 36–142 figure** that drove Addendum 1. 5 Oct: Hanley & McNeil 1982 doi:[10.1148/radiology.143.1.7063747](https://doi.org/10.1148/radiology.143.1.7063747) and 1983 doi:[10.1148/radiology.148.3.6878708](https://doi.org/10.1148/radiology.148.3.6878708); Obuchowski 1998 sample size doi:[10.1177/096228029800700405](https://doi.org/10.1177/096228029800700405); McClish 1989 partial AUC doi:[10.1177/0272989X8900900307](https://doi.org/10.1177/0272989X8900900307). All cited from memory in the new passes. Recomputed SE at AUC 0.8: ≈0.085 with 10 failures, ≈0.038 with 52 |
| Benjamini–Hochberg | BIB-pending | Multiplicity across the signal set. 5 Oct: *JRSS-B* 57:289–300, doi:[10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x) (from memory). Exploratory grid only; use Benjamini–Yekutieli if positive dependence is doubtful, since tests share units |
| Wilson (1927) | **BIB** | Interval on a proportion near 1.0 |
| Wilcoxon (1945) | **BIB** | Paired signed-rank, Study 3 |
| McNemar exact | ABS | Paired pass/fail comparison — the steps check. 5 Oct: exact or mid-p with few discordant pairs (Fagerland et al. 2013, §18); report the discordant count and the flip rate with a Wilson CI |
| **NIST AI 800-3** (Feb 2026) | ABS | **Official standard for statistical rigour in AI benchmarks.** Benchmark accuracy vs generalized accuracy; intervals over point scores; GLMMs. Cheap to comply with, strong reviewer signal. 5 Oct: [news page](https://www.nist.gov/news-events/news/2026/02/new-report-expanding-ai-evaluation-toolbox-statistical-models), [PDF](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=961314); PDF still unread. Siblings 800-2 and 800-4 in §15 |
| [2604.11581](https://arxiv.org/abs/2604.11581) | ABS | **Hidden Measurement Error in LLM Pipelines.** Naive SEs are **40–60% too small** because they ignore design-choice variance. **The basis for the variance-decomposition reframe**. 5 Oct: author S. Messing (13 Apr 2026). Tier 1 (random components, shrink with n) vs Tier 2 (design choices, do not shrink; naive CIs under-cover more as n grows); REML variance components plus a G-theory D-study. The 40–60%, 56→32 Elo and coverage figures came through a page summary; recheck before quoting |

## 6. Hardware nondeterminism — B-0's positioning

**B-0 must be positioned carefully here.** These establish that *numbers* vary across
hardware. B-0's claim is narrower and sharper: **a qualitative classification flips** —
which units count as failures — with median agreement 0.00068 and a tail of 0.66.

| id | title | level |
|---|---|---|
| [2408.05148](https://arxiv.org/pdf/2408.05148) | Impacts of floating-point non-associativity on reproducibility for HPC and deep learning | ABS |
| [2511.00025](https://arxiv.org/pdf/2511.00025) | Structure of Floating-Point Noise in Batch-Invariant GPU Matrix Multiplication | ABS |
| [2104.07651](https://arxiv.org/pdf/2104.07651) | mlf-core: deterministic machine learning | ABS |
| [2001.11396](https://arxiv.org/pdf/1911.11396) | Non-Determinism in TensorFlow ResNets | ABS |

## 7. Failure prediction and calibration — the reliability-diagram gap

| id | title | level | why |
|---|---|---|---|
| [2303.02970](https://arxiv.org/pdf/2303.02970) | Rethinking Confidence Calibration for Failure Prediction | ABS | Standard expectations for a failure detector — **includes reliability diagrams, which we do not yet produce** |
| [2403.02886](https://arxiv.org/html/2403.02886v1) | Revisiting Confidence Estimation: Towards Reliable Failure Prediction | ABS | Same |
| [2608.29705](https://arxiv.org/abs/2608.29705v1) | A Calibration Audit of Confidence in Feed-Forward 3D Reconstruction | ABS | A confidence audit in another field — useful template |
| [2604.19444](https://arxiv.org/html/2604.19444v1) | Unsupervised Confidence Calibration for Reasoning LLMs | ABS | Ground-truth-free calibration, different domain |
| [2601.03042](https://arxiv.org/html/2601.03042v1) | BaseCal | ABS | Same |
| [2402.15610](https://arxiv.org/pdf/2402.15610) | Selective "Selective Prediction" | ABS | Abstention framing |
| [2607.04430](https://arxiv.org/pdf/2607.04430) | Uncertainty-Aware Abstention with Provable Guarantees | ABS | Same |
| [2510.22224](https://arxiv.org/abs/2510.22224) | Taming Silent Failures: Verifiable AI Reliability | ABS | The phrase "silent failure" in a safety framing |
| [2404.09932](https://arxiv.org/pdf/2404.09932) | Foundational Challenges in Assuring Alignment and Safety of LLMs | ABS | Broad framing, useful for the intro |

## 8. The introspection line — Paper A / Study 3, now cut from the flagship

All **BIB** and verified at full text during the 3 September reference check. Retained
because Study 3 remains a separate output.

`lindsey2025introspection` (2601.01828) · `macar2026mechanisms` (2603.21396) ·
`lederman2026contentagnostic` (2603.05414) · `singh2026realitycheck` (2605.26242) ·
`vogel2025small` · `pearsonvogel2026latent` (2602.20031) · `godet2025confusion` ·
`godet2025localization` · `morris2025bypassing` · `hahami2025disturbance` (2512.12411) ·
`hahami2026ift` (2607.14111) · `nisbett1977telling` · `dettmers2023qlora` ·
`gemma2025gemma3` · [2607.21090](https://arxiv.org/abs/2607.21090) (faithfulness has no
static ground truth)

## 9. Neuroscience lineage — the methods being transferred

| source | level | why |
|---|---|---|
| Sharpee, Rust & Bialek (2004), Maximally Informative Dimensions | ABS | **The crown jewel of Proposal III.** Only 5 arXiv papers use MID, all neuroscience, none on a modern deep network. **Read end to end before writing the method section**. 5 Oct: arXiv [physics/0208057](https://arxiv.org/abs/physics/0208057); NeurIPS 2002 / Neural Comput. 2004. Consistent under arbitrary stimulus distributions but non-convex; the annealing detail is from memory. MID equals LNP maximum likelihood (Williamson, Sahani & Pillow 2015, §17) |
| Bussgang's theorem | ABS | **The mechanism for E0.5's result** — non-Gaussian stimulus and non-monotone GELU drive Bussgang's constant toward zero, which is why STA/STC fail |
| Spike-triggered average / covariance (classical) | ABS | The failing baselines, 0/30 and 0/30. 5 Oct: Paninski 2003 (§17) shows STA needs elliptical and STC Gaussian stimuli. Closest ANN precedent: Borji & Lin [1912.12106](https://arxiv.org/abs/1912.12106), white-noise probes on CNN units |
| Distill *Circuits* thread (2020–21) | ABS | **Tuning curves already exist for vision models.** Corrects the "never done" claim to "never done for language models" |

## 10. Proposal III collisions — do not restate the killed framing

Found by the August adversarial audit. **These killed a headline before a line was run.**

| id | finding |
|---|---|
| [2603.20642](https://arxiv.org/abs/2603.20642) | Cacioli — "Efficient Coding" **in the title**, counts the corpus prior (6,046,515 integer mentions, alpha=0.773) |
| [2604.04469](https://arxiv.org/abs/2604.04469) | Same author — corpus frequency predicts per-magnitude variability at rho = .84 |
| Benjamin et al., *Nature Communications* 13:7972 (2022) | Gradient descent generically produces frequency-tracking sensitivity — the "it's just the objective" objection is a **published proof** |
| Kim & Paik, *Science Advances* (2021) | Number tuning curves in **randomly initialised** networks |

**Competitor to watch:** Jon-Paul Cacioli, ~20 pre-registered papers Mar–May 2026 in this lane.

---

## 11. Closest concurrent work and direct foils (added 5 Oct)

Scoop ranking as of 5 Oct: weight-column calibration bench LOW (no collision found); "stability is not validity" framing MEDIUM; Study 3 introspection HIGH. Our repo is public and indexed, so an early arXiv post is the cheapest priority lock.

### 11a. Interpretability: foils, allies, neighbours

| id | title | level | why it matters |
|---|---|---|---|
| [2505.20254](https://arxiv.org/abs/2505.20254) | Song, Muhamed, Zheng, Kong, Tang, Diab, Smith, Zhang (2025). Position: Mechanistic Interpretability Should Prioritize Feature Consistency in SAEs. ACL 2026 long, [2026.acl-long.99](https://aclanthology.org/2026.acl-long.99/) | ABS | **The direct foil.** Proposes PW-MCC (run-to-run consistency) as "a reliable proxy for ground-truth recovery", validated on synthetic data only; GT-MCC ≈0.97 TopK vs ≈0.63 standard, PW-MCC ≈0.80 on Pythia-160M. Engage as "informative but dominated", not wrong. Census pass checked p2/p4 quotes in the PDF; metric-precedents pass confirmed existence only, not authors. Per-configuration vs per-unit reading is our inference |
| [2501.16615](https://arxiv.org/abs/2501.16615) | Paulo & Belrose (2025). Sparse Autoencoders Trained on the Same Data Learn Different Features | ABS | Seed-varied SAEs share only ~30% of features (131K latents, Llama-3-8B MLP); TopK more seed-dependent than ReLU/L1. The motivating seed-instability measurement, made without ground truth. Peer-reviewed venue unchecked |
| [2602.14111](https://arxiv.org/abs/2602.14111) | Korznikov et al. (15 Feb 2026). Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines? | ABS | **Strongest external ally.** ~9% of planted features recovered at 0.71 explained variance (JumpReLU 225/3200, BatchTopK 297/3200 at cos > 0.8); random directions match trained SAEs on interpretability, probing and editing. Synthetic version of "good fit, wrong answer" |
| [2607.12166](https://arxiv.org/abs/2607.12166) | Bal (13 Jul 2026). From Geometric Recovery to Causal Validation | ABS | Up to 77% of SAE features passing cos ≥ 0.90 are causally inert, even at cos ≈ 1.000. Source of the "typically ≥0.90" recovery convention. **Conflict:** one pass reports 14% inert on a production SAE, another 9% on a well-trained one; recheck. Single author, not peer-reviewed |
| [2607.05355](https://arxiv.org/abs/2607.05355) | Eswar, Seth, Avaiya, Sankarapu (6 Jul 2026). Faithfulness to Refusal: A Causal Audit of Neuron Selectors | ABS | "Rank-stable selectors can be among the least causally valid"; causal ground truth over 5 LMs with random controls. Closest conceptual match to our claim |
| [2608.13754](https://arxiv.org/abs/2608.13754) | Mahale (13 Aug 2026). Explanation Multiplicity: Circuit-Level Interpretability Evidence Does Not Survive Defensible Analytic Variation | ABS | 73.2% of circuit claims flip over 15,840 pre-registered specifications; median circuit Jaccard 4%. Framing scoop risk medium; no ground truth |
| [2608.05670](https://arxiv.org/abs/2608.05670) | Khanbayov & Kurban (6 Aug 2026). When Does Consensus Mean Correctness? | ABS | Exact-ground-truth VLM study: agreement certifies correctness only in some regimes; training on consensus lowers accuracy. Analogue outside interpretability |
| [2512.18092](https://arxiv.org/abs/2512.18092) | Yan, Oikarinen, Weng (2025). Faithful and Stable Neuron Explanations | ABS | Treats bootstrap stability as reliability of neuron explanations, no exact truth. Differentiate |
| [2602.14687](https://arxiv.org/abs/2602.14687) | Chanin & Garriga-Alonso (16 Feb 2026). SynthSAEBench | ABS | Synthetic SAE benchmark; scores recovery with continuous Hungarian-matched MCC, no cosine cut-off (definition read in the HTML); calls itself a lower-bound test. The 16,384-direction / 128-tree figures come from a secondary review; recheck. Substrate for running our checks |
| [2605.18229](https://arxiv.org/abs/2605.18229) | Chanin (18 May 2026). Are Sparse Autoencoder Benchmarks Reliable? | ABS | Reseed-noise, ground-truth and discriminability lenses; two SAEBench metrics "should not be used". Legitimises reseed noise as a lens; restart agreement is its per-estimator version |
| [2503.09532](https://arxiv.org/abs/2503.09532) | Karvonen, Rager, Lin et al. (2025). SAEBench, ICML 2025 ([PMLR](https://proceedings.mlr.press/v267/karvonen25a.html)) | ABS | Proxy-metric gains "do not reliably translate" to practical performance; Gemma-2-2B advantages absent on Pythia-160M (the "small models" objection). **No seed-consistency metric found; do not cite it for one** |
| [2510.14936](https://arxiv.org/abs/2510.14936) | Golimblevskaia … Samek, Lapuschkin (2025, rev. 2026). WeightLens / Circuit Insights | ABS | Reads features from weights with no data. Weights as object, not as ruler |
| [GitHub](https://github.com/openai/automated-interpretability) | OpenAI automated-interpretability repo | ABS | Defines GPT-2 neuron input weights with the LN gain folded (`wte @ diag(ln_2.g) @ c_fc`): precedent for our folding, used there only as an explanation aid |
| [2607.02964](https://arxiv.org/pdf/2607.02964) | Individual Parameters in Weight-Sparse Transformers Appear Interpretable (Jul 2026) | ABS | Title from search only. Weight-level neighbour |
| [2301.12608](https://arxiv.org/pdf/2301.12608) | Evaluating Neuron Interpretation Methods of NLP Models (2023) | MEM | Ground-truth-free evaluation by cross-method voting, i.e. the disagreement check we calibrate. Abstract not re-fetched |
| [2309.08600](https://arxiv.org/abs/2309.08600) | Cunningham et al. (2023), sparse dictionary features in LMs; title not recorded | MEM | Not re-checked for any recovery threshold; do not cite for one |
| [1909.03368](https://arxiv.org/abs/1909.03368) | Hewitt & Liang (2019). Control tasks for probes | ABS | Canonical ground-truth-free probe validation. Background |
| [2602.22968](https://arxiv.org/abs/2602.22968) | Anani, Lorenz, Schiele, Fritz, Fischer (2026). Certified Circuits: Stability Guarantees for Mechanistic Circuits | ABS | Data subsampling certifies circuit inclusion decisions. Usage-census instance (stability as validation) |
| [2606.16920](https://arxiv.org/abs/2606.16920) | Wu, Tonin, Cevher (15 Jun 2026). Demystifying Variance in Circuit Discovery | ABS | EAP-IG variance traced to prompt templates; proposes CEAP |
| [2603.00523](https://arxiv.org/html/2603.00523v2) | Parekh (2026). CIRCUS: Circuit Consensus under Uncertainty via Stability Ensembles | ABS | Keeps edges by inclusion frequency across pruning configurations. Census instance |
| [2510.00845](https://arxiv.org/abs/2510.00845) | Méloux, Portet, Peyrard (2025). Mechanistic Interpretability as Statistical Estimation: A Variance Analysis | ABS | Circuit scores have high intrinsic variance; recommends routine stability reporting |
| [2605.31245](https://arxiv.org/abs/2605.31245) | Nelson, Karaletsos, Locatello (29 May 2026). Toward Identifiable SAEs | ABS | SAE identifiability context |
| [2609.10299](https://arxiv.org/abs/2609.10299) | Plascencia (9 Sep 2026). A Dominant Diffuse Phase in the SAE Phase Diagram | ABS | SAE phase behaviour context |
| [2608.27754](https://arxiv.org/abs/2608.27754) | Jedryszek & Crook (27 Aug 2026). Efficient Auto-Interpretability of AI Models in Biology | ABS | Uses cross-seed SAE stability to rank latents on Boltz-1. ρ = 0.51 with concept-recovery F1 is from a search snippet, not on the abs page; recheck. Stability in use, weakly calibrated |
| [2505.15728](https://arxiv.org/pdf/2505.15728) | Are machine learning interpretations reliable? A stability study on global interpretations (2025) | ABS | Defines reliability (closeness to truth) separately from stability; definition from a snippet |
| [2602.17881](https://arxiv.org/abs/2602.17881) | Braun (Feb 2026). Understanding Unreliability of Steering Vectors | ABS | Ground-truth-free steering-reliability heuristics (consistency of activation differences, class separability). Calibrate them against exact logit-difference directions. Models used not in the abstract. Title is close to 2505.22637 (§4) under a different ID; check whether one supersedes the other. 6 Oct: settled. This is Braun's thesis (19 Feb 2026), which restates the 2505.22637 consistency result as "a practical diagnostic for steering unreliability"; vectors from different prompt variations "are directionally distinct, yet perform similarly well". Cite 2505.22637 for the finding, this for the diagnostic framing |
| [2604.15557](https://arxiv.org/abs/2604.15557) | Billa (16 Apr 2026). Predicting Where Steering Vectors Succeed | ABS | Linear Accessibility Profile predicts steering efficacy and layer without training |

### 11b. Introspection line, new since 9 Sep (Study 3)

| id | title | level | why it matters |
|---|---|---|---|
| [2608.20569](https://arxiv.org/abs/2608.20569) | Ferrara (20 Aug 2026). Open-Weight Masked Introspection | ABS | **Scoop risk HIGH for Study 3.** 8 open-weight models (not Gemma-3-27B); in 7 of 8 an impact-matched random perturbation triggers "change" reports more often than the targeted one (Phi-4 0.580 vs 0.295); no report beats sham. Argues norm-matching, our control, understates random damage. What stays ours: 27B scale, shuffled-vector control, steering positive control. Quotes taken from the HTML; read in full. 6 Oct: **correction.** Impact-matched random was run for Qwen2.5-7B and Mistral-7B only; the other six models use unit-norm matching, so "in 7 of 8 an impact-matched random" is wrong. The checked sentence is "Random-direction rate exceeds the intervention rate in seven of the eight models", not claimed as significant; the headline is chance-level discrimination (AUROC ~0.5007 over ~78k measurements; pooled d′ 0.0039 over 11,216 pairs). Models: Qwen2.5-0.5B-Instruct, Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Gemma-2-9B-IT, GLM-4-9B, Phi-4, DeepSeek-R1-Distill-Qwen-14B; nothing above 15B. Interventions at residual sites, attention heads and SAE features; readout is JSON-parsed text. Its positive control is a LoRA-trained detector (d′ 5.15), which validates the instrument, not vector content. Probes recover intervention presence at 75–95.8%. Stated limits: one site and one dose per model; no ground-truth label for the altered concept. Llama-3.1-8B 0.711 vs 0.500 and the norm-matching argument are carried from the earlier pass, not rechecked |
| [2609.35108](https://arxiv.org/abs/2609.35108) | Zou, Sun, Kong, Wang (28 Sep 2026). A Mechanistic Study of Language Model Introspection | ABS | Gate heads decide whether a change is reported, router heads select position. Snippet numbers for norm-matched random localisation (0.84% / 6.71% / 0%) may belong to 2603.21396 instead; unconfirmed. Scope our claim to detection, not localisation. 6 Oct: **correction.** The 0.84% / 6.71% / 0% figures are Zou's, checked in the HTML: norm-matched Gaussian random directions localise at those pooled rates on Qwen3-4B-IT, LLaMA-3.1-8B-IT and Gemma-3-12B-IT, 180k trials per model. 2603.21396 has no random control. Injection at L3 / L0 / L0, strength 3 / 6 / 5; vectors are the unit-normalised final-token residual difference for "Tell me about [concept]"; readout is first-position logits over {0–9, none}, no judge. Label shuffle drops accuracy from 48–84% to 27–62%. No steering validation. Gate heads are mid-layer, router heads later |
| [2511.21399](https://arxiv.org/abs/2511.21399) | Fonseca Rivera & Africa (26 Nov 2025). Steering Awareness | ABS | Models can be fine-tuned to detect activation steering |
| [2607.15495](https://arxiv.org/abs/2607.15495) | Gurnee, Sofroniew, Pearce et al. (16 Jul 2026). Verbalizable Representations Form a Global Workspace | ABS | Jacobian lens ("J-space"); no injection controls. Context |
| [2607.18553](https://arxiv.org/abs/2607.18553) | Kirin (20 Jul 2026). Operational Proto-Introspection in Looped LMs | ABS | Hidden-state readout predicts success; tangential |

5 Oct notes on §8 entries (no rows changed): `hahami2026ift` (2607.14111) date unresolved, the arXiv record lists 8 May 2026 against a July ID. `hahami2025disturbance` (2512.12411) is titled "Detecting the Disturbance" in one pass and "Feeling the Strength but Not the Source: Partial Introspection in LLMs" in another; check. `godet2025localization`: Gemma3-27B-IT localises injections at 98% at layer 18, our Study 3 model, so localisation is not detection. `macar2026mechanisms`: evidence-carrier features respond to diverse directions (supports the perturbation-alarm reading); detection absent in base models, DPO-dependent. `lindsey2025introspection`: the original norm-matched random control (9/100) to report against.

6 Oct notes on §8 entries, persona and self pass (no level changes):
- `hahami2025disturbance` (2512.12411): title question settled. v1 was "Feeling the Strength but Not the Source: Partial Introspection in LLMs"; the current version (rev. 1 Mar 2026) is "Detecting the Disturbance: A Nuanced View of Introspective Abilities in LLMs". Cite the current title and note the change. Still open: the author list (Hahami, Sinha, Jain, Kaplan, Hahami in one pass; Hahami, Jain, Sinha in the other), and numbers that differ by version. Current: 10-way sentence localisation 88% (chance 10%), strength discrimination 83% (chance 50%), early-layer injection only, "collapse to chance thereafter". v1 HTML: strength in 4 bins up to 70% (chance 25%) rising with depth, naming ~20%, 10-way concept choice ~20%, two simultaneous injections 0%. Quote only from the version cited.
- `hahami2026ift` (2607.14111): date settled from the PDF (p.1–2): 8 May 2026; authors Hahami, Sinha, Jain. Yes/no detection in small models is confounded by yes-bias (r = 0.999 with a factual-no control, Llama-3.1-8B). Proposes sentence localisation and strength comparison; IFT lifts Llama-1B localisation 9.6% → 60.6%, strength 30.2% → 52.2%. The model list reads "Llama-3.2 1B/3B/8B; Gemma-4 2B/4B/26B", which looks garbled; recheck before quoting.
- `lindsey2025introspection`: the concept vector is read on the last token before the model's turn (the chat-template tail), minus the mean over other words. Judge-graded text; 0/100 false positives without injection; ~20% awareness at the best layer and strength for the strongest model. Liveness evidence is qualitative only (high strength leaves the model "consumed by the injected concept"). The 9/100 random figure still rests on the earlier pass.
- `macar2026mechanisms` (2603.21396): Gemma3-27B (62 layers), injection at L37, α = 4: our Study 3 operating point. Also Qwen3-235B (L75) and OLMo-3.1-32B. Vector read at the last token of the chat-templated "Tell me about {concept}", minus the mean over 100 baseline words; mean norm 4,664 ± 982 at d = 5,376; 500 concepts × 100 trials; judge-graded text. **No random, norm-matched or orthogonal control**, and no neutral-prompt steering check (two fetches agree). TPR ~10.8% baseline, 63.8% with refusal-direction ablation; base models ~42% false positives; 23.3% of pairs detected both ways in the A−B / B−A test. Precision not stated; code at `safety-research/introspection-mechanisms`. Our dead template-tail vectors bear directly on this recipe unless they are a 4-bit artefact.
- `singh2026realitycheck` (2605.26242): models Llama-3.1-70B/8B-Instruct, Qwen-2.5-72B, Qwen-3-32B, Gemma-3-27B-It (HTML). The "gaslight" prompt condition is often classed as injection; recommends input-only probes and 3-way designs. Whether it compares first-token and generated readouts is unchecked.
- `lederman2026contentagnostic` (2603.05414): PDF p.1–3: Qwen3-235B-A22B and Llama-3.1-405B, 821 concepts. Wrong guesses track default probability; identification comes from steering and drops when steering stops before the response. A secondary summary of this PDF invented a "confidence calibration across domains" claim; use the PDF only.
- `pearsonvogel2026latent` (2602.20031): Pearson-Vogel, Vanek, Douglas, Kulveit (23 Feb 2026). Qwen-32B denies injection in text while the logit lens detects it; sensitivity 0.3% → 39.9% when the prompt explains the mechanism (+0.6% FPR); concept MI 0.61 → 1.05 bits. P("yes") near 100% around layers 58–62, attenuated in the last 2–3 layers, is from a snippet.
- `godet2025localization`: no norm-matched random control; a "which sentence do you prefer?" control sits at chance. `godet2025confusion`: the equal-norm random comparison rests on the earlier pass.

## 12. Usage census: who uses restart or seed agreement as a reliability check

34 instances across 8 sub-areas. Counts by use: validate or trust 16, select K or model 12, robustness report 9, argues stability ≠ correctness 4, documents instability 6. 13 are 2025–26 interpretability papers. Already elsewhere: Song, Paulo & Belrose, Certified Circuits, CIRCUS, Méloux, Mahale (§11); Gerasimov, 2603.04198, 2605.18629 (§3); Ben-David (§13); Yu & Kumbier (§16). Model-seed studies are a weaker match than estimator restarts; the row says which.

| id | title | level | why it matters |
|---|---|---|---|
| [2502.12892](https://arxiv.org/abs/2502.12892) | Fel, Lubana, Prince, Kowal, Boutin, Papadimitriou, Wang, Wattenberg, Ba, Konkle (2025). Archetypal SAE, ICML 2025 | ABS | "Severe instability ... undermining their reliability as an interpretability tool." Estimator restarts. Quotable |
| [AF post](https://www.alignmentforum.org/posts/z6QQJbtpkEAX3Aojj/interim-research-report-taking-features-out-of-superposition) | Sharkey, Braun, Millidge (13 Dec 2022). Interim report: taking features out of superposition with sparse autoencoders | FULL | MMCS between independently trained dictionaries used as a ground-truth-free size selector; authors admit firm conclusions are hard. **Sets no 0.9 "recovered" threshold; do not cite it for one** |
| [Transformer Circuits](https://transformer-circuits.pub/2023/monosemantic-features/index.html) | Bricken et al. (2023). Towards Monosemanticity | FULL | Universality section uses cross-run recurrence (median corr 0.72) as corroboration. Anecdote: a replication note reads a cross-seed direction as "a true thing about the model", while the main text calls the same ultralow-density cluster an autoencoder artifact. Mixes transformer seed and SAE run |
| [2401.12181](https://arxiv.org/abs/2401.12181) | Gurnee, Horsley, Guo, Kheirkhah, Sun, Hathaway, Nanda, Bertsimas (2024). Universal Neurons in GPT2 Language Models | ABS | Cross-seed study "motivated by the hypothesis that universal neurons are likely to be interpretable". Model seeds |
| [1511.07543](https://arxiv.org/abs/1511.07543) | Li, Yosinski, Clune, Lipson, Hopcroft (2016). Convergent Learning, ICLR 2016 | MEM | Some features learned reliably across nets, others not; also the Hungarian-matching recipe if cross-layer correspondence is ever needed. Census checked the abstract; two other passes cite from memory |
| [1706.05806](https://arxiv.org/abs/1706.05806) | Raghu, Gilmer, Yosinski, Sohl-Dickstein (2017). SVCCA, NeurIPS 2017 | MEM | Compares representations across random inits; also the CCA basis for Σ-metric subspace angles. Census checked a p3 quote; the validity pass did not fetch it |
| [1905.00414](https://arxiv.org/abs/1905.00414) | Kornblith, Norouzi, Lee, Hinton (2019). Similarity of Neural Network Representations Revisited (CKA), ICML 2019 | MEM | §6.1 sanity check: an index should match layers across seeds. Cross-seed match as a desideratum of a similarity measure, not of correctness. Census checked §6.1; agreement pass cites from memory |
| [1905.12614](https://arxiv.org/abs/1905.12614) | Duan, Matthey, Saraiva, Watters, Burgess, Lerchner, Higgins (2020). Unsupervised Model Selection for Disentangled Representations (UDR), ICLR 2020 | MEM | Most direct published use of seed agreement as label-free model selection ("happy families are all alike"). Census checked abstract and Fig. 1, not the defining sentence; other passes from memory. Verify the premise before naming it a counterexample target |
| [1811.12359](https://arxiv.org/abs/1811.12359) | Locatello, Bauer, Lucic, Rätsch, Gelly, Schölkopf, Bachem (2019). Challenging Common Assumptions in Unsupervised Disentanglement, ICML 2019 | MEM | Seeds matter more than models; good runs cannot be identified without labels. Census checked p2/p5; agreement pass from memory. JMLR follow-up evaluating UDR unchecked |
| [1711.11279](https://arxiv.org/abs/1711.11279) | Kim et al. (2018). TCAV, ICML 2018 | FULL | p4: "A meaningful concept should lead to TCAV scores that behave consistently across training runs" (500 runs, t-test). Quotable |
| [2605.06303](https://arxiv.org/abs/2605.06303) | Molecules Meet Language: Confound-Aware Representation Learning and Chemical Property Steering in Transformer-VAE Latent Spaces (2026); authors not recorded | FULL | App. C.1 bootstrap agreement of a probe direction read as "a reproducible feature". Census instance |
| [2608.05732](https://arxiv.org/abs/2608.05732) | CircuitSteer: Geometrically Aligned Multi-Layer Steering via SAE Circuits (2026); authors not recorded | FULL | Seed stability of circuits (Jaccard 0.58–0.77) and vectors (cos 0.90–0.97) reported as validation (p9, App. J) |
| [2609.07037](https://arxiv.org/abs/2609.07037) | Hiramatsu, Atarashi, Takeuchi, Kashima (7 Sep 2026). Disentangling Steering Vectors | ABS | App. B: recovered concepts "broadly similar across seeds, though not identical"; no planted-recovery test. Census read App. B; concurrent-work pass had the abstract only |
| [K19-1087](https://aclanthology.org/K19-1087/) | Madhyastha & Jain (2019). On Model Stability as a Function of Random Seed, CoNLL | ABS | Seeds change attention, gradient and LIME interpretations; proposes ASWA |
| [2605.21492](https://arxiv.org/abs/2605.21492) | The Attribution Impossibility: No Feature Ranking Is Faithful, Stable, and Complete Under Collinearity (2026); authors not recorded | FULL | p5–6: a reseed can invert the top feature; 68% of 77 datasets show it. Proposes seed-ensemble consensus |
| [doi:10.1016/j.neuroimage.2004.03.027](https://doi.org/10.1016/j.neuroimage.2004.03.027) | Himberg, Hyvärinen, Esposito (2004). Validating the independent components of neuroimaging time series (Icasso), NeuroImage 22:1214 | ABS | Canonical origin: a single ICA run "should be interpreted with some reserve"; reliability from many runs |
| [doi:10.1073/pnas.0308531101](https://doi.org/10.1073/pnas.0308531101) | Brunet, Tamayo, Golub, Mesirov (2004). Metagenes and molecular pattern discovery using matrix factorization, PNAS | FULL | Methods: a strong clustering should "vary little from run to run"; cophenetic coefficient picks k |
| [doi:10.1073/pnas.1521171113](https://doi.org/10.1073/pnas.1521171113) | Wu, Joseph, Hammonds, Celniker, Yu, Frise (2016). staNMF, PNAS | FULL | Optimal dictionary should be "reproducibly independent of the initialization values". Cleanest statement of the norm |
| [doi:10.7554/eLife.43803](https://doi.org/10.7554/eLife.43803) | Kotliar et al. (2019). cNMF, eLife | FULL | Maximum stability "would have led to the incorrect choice for the simulated data". Counterexample |
| [doi:10.1023/A:1023949509487](https://doi.org/10.1023/A:1023949509487) | Monti, Tamayo, Mesirov, Golub (2003). Consensus Clustering, Machine Learning 52:91 | ABS | Resampling consensus to assess cluster stability. Search summary only |
| [doi:10.1038/srep06207](https://doi.org/10.1038/srep06207) | Şenbabaoğlu, Michailidis, Li (2014). Critical limitations of consensus clustering in class discovery, Sci Rep 4:6207 | ABS | "Apparently stable clusters" on cluster-less data. Counterexample, pairs with Kotliar |
| [1404.4606](https://arxiv.org/abs/1404.4606) | Greene, O'Callaghan, Cunningham (2014). How Many Topics? Stability Analysis for Topic Models | ABS | Stability chooses topic count |
| [Q18-1008](https://aclanthology.org/Q18-1008/) | Antoniak & Mimno (2018). Evaluating the Stability of Embedding-based Word Similarities, TACL | ABS | Recommends bootstrap samples over a single embedding |
| [2302.03025](https://arxiv.org/abs/2302.03025) | Chughtai, Chan, Nanda (2023). A Toy Model of Universality | ABS | Circuits learned are arbitrary per seed. Model seeds |
| [Nat Commun](https://www.nature.com/articles/s41467-020-19632-w) | Mehrer, Spoerer, Kriegeskorte (2020). Individual differences among deep neural network models, Nat Commun 11:5725 | ABS | Weight seed alone changes representations at equal accuracy. Search summary |
| [2406.04093](https://arxiv.org/abs/2406.04093) | Gao et al. (2024), OpenAI SAEs | ABS | No cross-seed consistency analysis found in a noisy text extraction. Treat as "not found"; do not cite either way |

## 13. Agreement versus correctness precedents

Rule for the paper: disagreement is evidence of error; agreement is not evidence of correctness; held-out R² is a necessary-relation check, not a certificate.

| id | title | level | why it matters |
|---|---|---|---|
| [doi:10.1007/11776420_4](https://doi.org/10.1007/11776420_4) | Ben-David, von Luxburg, Pál (2006). A Sober Look at Clustering Stability, COLT 2006, LNCS 4005:5–19 | ABS | **Closest formal precedent.** For large samples stability "is fully determined by the behavior of the objective function", not by correctness. Full text unread; no counterexample quoted yet |
| [1007.1075](https://arxiv.org/abs/1007.1075) | von Luxburg (2010). Clustering Stability: An Overview | ABS | Survey restating the caveats; FnT ML 2(3):235–274 venue from memory |
| — | Lange, Roth, Braun, Buhmann (2004). Stability-Based Validation of Clustering Solutions, Neural Computation 16(6):1299–1323 | MEM | Identifier not recorded. The main positive proposal for resampling stability as validation; the practice being critiqued |
| [1609.00978](https://arxiv.org/abs/1609.00978) | Jin, Zhang, Balakrishnan, Wainwright, Jordan (2016). Local Maxima in the Likelihood of Gaussian Mixture Models, NeurIPS 2016 | ABS | Random-init EM reaches bad critical points with probability ≥ 1 − e^(−Ω(M)); bad maxima arbitrarily worse. Mechanism: the wrong basin can be the typical one, and it carries a fit penalty |
| — | Arthur & Vassilvitskii (2007). k-means++, SODA 2007 | MEM | Identifier not recorded. Lloyd's can be arbitrarily bad; seeding matters |
| — | Hyvärinen & Pajunen (1999). Nonlinear ICA: Existence and Uniqueness Results, Neural Networks 12(3):429–439 | MEM | Identifier not recorded. Non-identifiability route: restarts agree on the wrong member of an equivalence class |
| [Semantic Scholar](https://www.semanticscholar.org/paper/An-experimental-evaluation-of-the-assumption-of-in-Knight-Leveson/990ea46ace7c5cc96441bd3d5f318eeaa1855f8c) | Knight & Leveson (1986). An Experimental Evaluation of the Assumption of Independence in Multiversion Programming, IEEE TSE SE-12(1):96–109 | ABS | 27 independent versions, a million tests; coincident failures "substantially more than expected". Shared specification gives correlated failure, as shared objective and data do for restarts. Strongest non-ML citation |
| — | Avizienis (1985). The N-Version Approach to Fault-Tolerant Software, IEEE TSE SE-11(12):1491–1501 | MEM | Identifier not recorded. The practice Knight & Leveson tested |
| — | Eckhardt & Lee (1985). A Theoretical Basis for the Analysis of Multiversion Software Subject to Coincident Errors, IEEE TSE SE-11(12):1511–1517 | MEM | Identifier not recorded. Input difficulty alone correlates failures: "hard units" fail every route without a shared bug |
| [doi:10.1145/3143561](https://doi.org/10.1145/3143561) | Chen, Kuo, Liu, Poon, Towey, Tse, Zhou (2018). Metamorphic Testing: A Review of Challenges and Opportunities, ACM CSUR 51(1) | ABS | Label-free checks of necessary relations, not agreement. Held-out R² is structurally one |
| [doi:10.1109/TSE.2014.2372785](https://doi.org/10.1109/TSE.2014.2372785) | Barr, Harman, McMinn, Shahbaz, Yoo (2015). The Oracle Problem in Software Testing: A Survey, IEEE TSE 41(5) | MEM | Weight column = specified oracle; restart agreement and method disagreement = pseudo-oracles |
| — | McKeeman (1998). Differential Testing for Software, Digital Technical Journal 10(1):100–107 | MEM | Identifier not recorded. Disagreement flags a bug; agreement proves nothing. Our method-disagreement score |
| [2606.20158](https://arxiv.org/html/2606.20158v1) | N-Version Programming with Coding Agents (2026) | ABS | Search result only. A "still true today" line if read |
| [2607.02808](https://arxiv.org/html/2607.02808) | A Systematic Methodology for Evaluating Failure Independence in LLM-Generated Code (2026) | ABS | Search result only. Same use |
| [GitHub](https://github.com/ASSERT-KTH/Knight-Leveson-Redux) | ASSERT-KTH, Knight-Leveson-Redux repo | ABS | Search result only |
| [2011.03395](https://arxiv.org/abs/2011.03395) | D'Amour et al. (2022). Underspecification Presents Challenges for Credibility in Modern ML, JMLR 23(226) | MEM | The converse of our failure: equal held-out scores, different solutions. Frame R² as a necessary-condition check. Levels differ: agreement pass checked the JMLR page (ABS), metric-precedents pass from memory |
| [Project Euclid](https://projecteuclid.org/journals/statistical-science/volume-16/issue-3/Statistical-Modeling--The-Two-Cultures-with-comments-and-a/10.1214/ss/1009213726.full) | Breiman (2001). Statistical Modeling: The Two Cultures, Statistical Science 16(3):199–231 | MEM | The Rashomon effect: prediction alone does not identify mechanism |
| [1909.06677](https://arxiv.org/abs/1909.06677) | Marx, Calmon, Ustun (2020). Predictive Multiplicity in Classification, ICML 2020 | MEM | Low multiplicity among restarts does not mean the set contains the truth |
| [OpenReview jESY2WTZCe](https://openreview.net/forum?id=jESY2WTZCe) | Krishna et al. (2024). The Disagreement Problem in Explainable ML: A Practitioner's Perspective, TMLR; also [2202.01602](https://arxiv.org/abs/2202.01602) | MEM | Closest XAI precedent for method disagreement, without ground truth. **Conflict:** passes give different author lists (…Gu, Wu, Jabbari… vs …Gu, Pombra, Jabbari, Wu…) and IDs; cite the TMLR version after checking OpenReview. TMLR record checked in one pass, arXiv from memory in the other |
| [PMLR 48](https://proceedings.mlr.press/v48/platanios16.html) | Platanios, Blum, Mitchell (2014, UAI) "Estimating Accuracy from Unlabeled Data"; Platanios, Dubey, Mitchell (2016, ICML); Platanios, Poon, Mitchell, Horvitz (2017, [NeurIPS](https://proceedings.neurips.cc/paper/2017/file/95f8d9901ca8878e291552f001f67692-Paper.pdf)) | ABS | Agreement as a label-free accuracy estimate; needs weak error dependence, which seeds on the same data violate. 2014 identifier not recorded |
| [1407.7644](https://arxiv.org/pdf/1407.7644) | Jaffe, Nadler, Kluger (2015). Estimating the Accuracies of Multiple Classifiers Without Labeled Data, AISTATS | MEM | Spectral method under conditional independence |
| — | Dawid & Skene (1979). Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm, Applied Statistics 28(1):20–28 | MEM | Identifier not recorded. Origin of accuracy-from-agreement under conditional independence |
| [2106.13799](https://arxiv.org/abs/2106.13799) | Jiang, Nagarajan, Baek, Kolter (2022). Assessing Generalization of SGD via Disagreement, ICLR 2022 | MEM | Test error ≈ disagreement of two SGD runs, given ensemble calibration. About predictive error, not parameter recovery. Stability pass checked the abstract; agreement pass did not |
| [2202.01851](https://arxiv.org/abs/2202.01851) | Kirsch & Gal (2022). A Note on "Assessing Generalization of SGD via Disagreement", TMLR | MEM | Calibration deteriorates as disagreement grows, and checking it needs labels. Mirrors our argument. Agreement pass checked the abstract; stability pass recalled it |
| [2206.13089](https://arxiv.org/abs/2206.13089) | Baek, Jiang, Raghunathan, Kolter (2022). Agreement-on-the-Line, NeurIPS 2022 | ABS | **Do not cite as a failure case.** The abstract shows it working even off the line; the documented limit is scope (neural classifiers only). Body unread |
| [2201.04234](https://arxiv.org/abs/2201.04234) | Garg, Balakrishnan, Lipton, Neyshabur, Sedghi (2022). Leveraging Unlabeled Data to Predict OOD Performance (ATC), ICLR 2022 | ABS | Label-free accuracy estimation is "just as hard as identifying the optimal predictor". Covers both our signals; forbids calling R² a certificate. Applying it in-distribution is an analogy |
| [2505.17656](https://arxiv.org/abs/2505.17656) | Tan et al. (2025). Too Consistent to Detect: A Study of Self-Consistent Errors in LLMs, EMNLP 2025 | ABS | **Closest modern analogue.** Self-consistent errors stay stable or grow with scale and evade all four detector families; the fix is an external verifier |
| [2203.11171](https://arxiv.org/abs/2203.11171) | Wang et al. (2023). Self-Consistency Improves Chain of Thought Reasoning, ICLR 2023 | MEM | The agreement-as-confidence method Tan et al. stress-test |
| — | Farquhar, Kossen, Kuhn, Gal (2024). Detecting Hallucinations in LLMs Using Semantic Entropy, Nature 630:625–630 | MEM | Identifier not recorded. Low entropy read as reliable; blind to repeated errors |
| [1612.01474](https://arxiv.org/abs/1612.01474) | Lakshminarayanan, Pritzel, Blundell (2017). Deep Ensembles, NeurIPS 2017 | MEM | Member disagreement as uncertainty |
| [1912.02757](https://arxiv.org/abs/1912.02757) | Fort, Hu, Lakshminarayanan (2019). Deep Ensembles: A Loss Landscape Perspective | MEM | Inits explore different modes. "Same-mode ensembles understate error" is our inference, not their finding |
| [1906.02530](https://arxiv.org/abs/1906.02530) | Ovadia et al. (2019). Can You Trust Your Model's Uncertainty?, NeurIPS 2019 | MEM | Uncertainty quality degrades under shift |
| [1810.11953](https://arxiv.org/abs/1810.11953) | Rabanser, Günnemann, Lipton (2019). Failing Loudly, NeurIPS 2019 | MEM | Label-free shift detectors scored on injected shifts with known truth. Methodological twin |
| — | Gama et al. (2014). Concept-drift survey, ACM CSUR | MEM | Identifier not recorded. Drift detection without labels |
| [2010.09470](https://arxiv.org/abs/2010.09470) | Arp et al. (2022). Dos and Don'ts of Machine Learning in Computer Security, USENIX Security 2022; CACM 2024 doi:[10.1145/3643456](https://doi.org/10.1145/3643456) | ABS | Ten pitfalls across 30 top-venue papers, including label inaccuracy, that "can affect the validity of research". Pairs with Gardiner & Nagaraja (§20). **Conflict:** 7 authors in one pass, 8 (adds Rieck) in another; check the USENIX PDF |
| [doi:10.1145/2342356.2342394](https://doi.org/10.1145/2342356.2342394) | Dave, Guha, Zhang (2012). Measuring and Fingerprinting Click-Spam in Ad Networks, SIGCOMM 2012 | ABS | Independent measurement vs the ad network's self-report: the part of click-fraud work that maps onto ours |
| [doi:10.1145/3460120.3484546](https://doi.org/10.1145/3460120.3484546) | Dissecting Click Fraud Autonomy in the Wild, CCS 2021; authors not verified | ABS | "Humanoid" clicks that look normal evade consistency-based detectors |
| [ACM DL](https://dl.acm.org/doi/10.5555/2627435.2627438) | Oentaryo et al. (2014). Detecting Click Fraud in Online Advertising: A Data Mining Approach, JMLR 15(1) | ABS | FDMA 2012 competition results. Background only |

## 14. Misleading-metric precedents

Hooker et al. ROAR is noted on the §2 CLEVR-XAI row; Heap et al. on the §3 2501.17727 row; SAEBench is in §11a.

| id | title | level | why it matters |
|---|---|---|---|
| [1810.03292](https://arxiv.org/abs/1810.03292) | Adebayo, Gilmer, Muelly, Goodfellow, Hardt, Kim (2018). Sanity Checks for Saliency Maps, NeurIPS 2018 | MEM | Plausible maps unchanged under model randomisation. Canonical "check passes for the wrong reason" |
| [1711.00867](https://arxiv.org/abs/1711.00867) | Kindermans et al. (2017/2019). The (Un)reliability of Saliency Methods, LNCS 11700 | MEM | A function-preserving input shift changes attributions |
| [1912.01451](https://arxiv.org/abs/1912.01451) | Tomsett, Harborne, Chakraborty, Gurram, Preece (2020). Sanity Checks for Saliency Metrics, AAAI 2020 | ABS | Faithfulness metrics "statistically unreliable and inconsistent". Closest XAI precedent for meta-evaluating the metric |
| [2104.14403](https://arxiv.org/abs/2104.14403) | Zhou, Booth, Ribeiro, Shah (2022). Do Feature Attribution Methods Correctly Attribute Features?, AAAI 2022 | ABS | Plants ground truth by dataset edits; methods "mostly can't" find it. Same move as ours |
| [2212.11870](https://arxiv.org/abs/2212.11870) | Bilodeau, Jaques, Koh, Kim (2024). Impossibility Theorems for Feature Attribution, PNAS 121(2) | MEM | Complete, linear attributions can fail to beat random guessing |
| [2108.01661](https://arxiv.org/abs/2108.01661) | Ding, Denain, Steinhardt (2021). Grounding Representation Similarity with Statistical Testing, NeurIPS 2021 | ABS | **Methodological twin.** Scores seed-comparison measures by sensitivity/specificity against functional ground truth; they disagree on whether reseeded nets match |
| [2210.16156](https://arxiv.org/abs/2210.16156) | Davari et al. (2023). Reliability of CKA as a Similarity Measure, ICLR 2023 | ABS | Existence only. Companion to Ding |
| [2311.17030](https://arxiv.org/abs/2311.17030) | Makelov, Lange, Geiger, Nanda (2024). An Interpretability Illusion for Subspace Activation Patching, ICLR 2024 | ABS | A working intervention can act through a dormant pathway. Positive check, wrong mechanism |
| [PMLR 80](https://proceedings.mlr.press/v80/athalye18a/athalye18a.pdf) | Athalye, Carlini, Wagner (2018). Obfuscated Gradients Give a False Sense of Security, ICML 2018; [1802.00420](https://arxiv.org/abs/1802.00420) | ABS | 7 of 9 ICLR 2018 defences relied on obfuscated gradients. The check measured the attack's failure, not the defence's success |
| [1902.06705](https://arxiv.org/abs/1902.06705) | Carlini et al. (2019). On Evaluating Adversarial Robustness | MEM | A methods paper whose deliverable is an evaluation protocol |
| [2002.08347](https://arxiv.org/abs/2002.08347) | Tramèr, Carlini, Brendel, Madry (2020). On Adaptive Attacks to Adversarial Example Defenses, NeurIPS 2020 | MEM | Broke 13 defences that already claimed adaptive evaluation. Count from memory |
| [2003.01690](https://arxiv.org/abs/2003.01690) | Croce & Hein (2020). AutoAttack, ICML 2020 | MEM | Lower robust accuracy than reported for most of ~50 defences. Counts from memory; do not quote |
| [doi:10.1109/SP.2010.25](https://doi.org/10.1109/SP.2010.25) | Sommer & Paxson (2010). Outside the Closed World, IEEE S&P 2010 | MEM | Closed-world metrics vs operational failure in intrusion detection |
| [1709.06560](https://arxiv.org/abs/1709.06560) | Henderson et al. (2018). Deep Reinforcement Learning that Matters, AAAI 2018 | MEM | Seed variance produces spurious "significant" differences |
| [2103.03098](https://arxiv.org/abs/2103.03098) | Bouthillier et al. (2021). Accounting for Variance in Machine Learning Benchmarks, MLSys 2021 | MEM | Randomise many variance sources jointly, cheaper and closer to the truth. Statistics pass saw it via search (ABS); two passes from memory |
| [ECCV PDF](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700681.pdf) | Musgrave, Belongie, Lim (2020). A Metric Learning Reality Check, ECCV 2020; [2003.08505](https://arxiv.org/abs/2003.08505) | ABS | Claimed gains "often more than doubling" shrank to "marginal at best" under fair protocol |
| [1902.10811](https://arxiv.org/abs/1902.10811) | Recht, Roelofs, Schmidt, Shankar (2019). Do ImageNet Classifiers Generalize to ImageNet?, ICML 2019 | ABS | Accuracy dropped but rankings held. Contrast: restart agreement fails at ranking failures |
| [NeurIPS D&B](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/757b505cfd34c64c85ca5b5690ee5293-Abstract-round2.html) | Liao, Taori, Raji, Schmidt (2021). Are We Learning Yet? A Meta Review of Evaluation Failures Across ML, NeurIPS 2021 D&B | MEM | 107 surveys sorted into internal vs external validity failures. Metric-precedents pass checked the proceedings page (ABS); rigor pass from memory with an unverified OpenReview URL |
| [2107.07002](https://arxiv.org/abs/2107.07002) | Dehghani et al. (2021). The Benchmark Lottery | MEM | Benchmark choice flips which method looks best |
| [2207.07048](https://arxiv.org/abs/2207.07048) | Kapoor & Narayanan (2023). Leakage and the Reproducibility Crisis in ML-based Science, Patterns 4(9):100804 | MEM | Leakage inflated results across fields (≈294 papers, 17 fields, from memory) |
| [2407.01502](https://arxiv.org/abs/2407.01502) | Kapoor, Stroebl, Siegel, Nadgir, Narayanan (2024). AI Agents That Matter | MEM | Agentic-benchmark flaws; venue unverified |
| [2507.02825](https://arxiv.org/pdf/2507.02825) | Zhu et al. (2025). Establishing Best Practices for Building Rigorous Agentic Benchmarks | ABS | Benchmark flaws misestimate capability by up to 100% relative (TAU-bench counts empty responses); task vs outcome validity. **Carries the claim the mentor attributes to Dholakia et al. (§20)** |
| [1807.03341](https://arxiv.org/abs/1807.03341) | Lipton & Steinhardt (2018). Troubling Trends in Machine Learning Scholarship | MEM | Background |
| [2109.08203](https://arxiv.org/abs/2109.08203) | Picard (2021). Seed scan of up to 10^4 seeds on CIFAR-10; title not recorded | ABS | Easy to find an outlier seed; 0.5% gaps explainable by seed alone |
| [2103.04514](https://arxiv.org/abs/2103.04514) | Summers & Dinneen (2021). Nondeterminism and Instability in Neural Network Optimization, ICML 2021 | MEM | One-bit init change ≈ full reseed: all nondeterminism acts through optimisation instability. Explains flips that survive fp64 |
| [2106.11872](https://arxiv.org/abs/2106.11872) | Zhuang et al. (2022). Randomness in Neural Network Training: Characterizing the Impact of Tooling, MLSys | MEM | Tooling nondeterminism alone ≈ seed variance for some metrics |
| [2203.06498](https://arxiv.org/abs/2203.06498) | Hullman, Kapoor, Nanayakkara, Gelman, Narayanan (2022). The Worst of Both Worlds, AIES 2022 | MEM | Errors in learning from data, psychology vs ML |
| [E06-1032](https://aclanthology.org/E06-1032/) | Callison-Burch, Osborne, Koehn (2006). Re-evaluating the Role of Bleu in MT Research, EACL 2006 | ABS | Better BLEU neither necessary nor sufficient for better translation. Take wording from the PDF |
| [J18-3002](https://aclanthology.org/J18-3002/) | Reiter (2018). A Structured Review of the Validity of BLEU, Computational Linguistics 44(3) | ABS | 284 correlations from 34 papers: valid for diagnostic MT only. **Best template for our verdict**: restart agreement is valid for flagging gross instability, not for certifying correctness |
| [D16-1230](https://aclanthology.org/D16-1230/) | Liu, Lowe, Serban, Noseworthy, Charlin, Pineau (2016). How NOT To Evaluate Your Dialogue System, EMNLP 2016 | ABS | Word-overlap metrics barely track human judgement |
| [1801.01973](https://arxiv.org/abs/1801.01973) | Barratt & Sharma (2018). A Note on the Inception Score | MEM | Score sensitive to irrelevant factors, gameable |
| [2203.06026](https://arxiv.org/abs/2203.06026) | Kynkäänniemi, Karras, Aittala, Aila, Lehtinen (2023). The Role of ImageNet Classes in FID, ICLR 2023 | ABS | FID matched across models that differ in human evaluation |
| [2401.09603](https://arxiv.org/abs/2401.09603) | Jayasumana et al. (2024). Rethinking FID, CVPR 2024 | MEM | FID assumptions fail; proposes CMMD |
| [1902.01007](https://arxiv.org/abs/1902.01007) | McCoy, Pavlick, Linzen (2019). Right for the Wrong Reasons, ACL 2019 | MEM | High score, wrong mechanism (HANS) |
| [doi:10.7326/0003-4819-125-7-199610010-00011](https://doi.org/10.7326/0003-4819-125-7-199610010-00011) | Fleming & DeMets (1996). Surrogate End Points in Clinical Trials: Are We Being Misled?, Ann Intern Med 125(7):605–613 | ABS | Correlated surrogates can still miss the true effect; CAST central |
| [doi:10.1056/NEJM199103213241201](https://doi.org/10.1056/NEJM199103213241201) | CAST Investigators / Echt et al. (1991). Cardiac Arrhythmia Suppression Trial, NEJM 324(12):781–788 | ABS | Drugs meeting the surrogate killed more patients: 63/755 vs 26/743. A surrogate above chance (like AUC 0.778) can still be the wrong criterion |
| — | Prentice (1989). Surrogate Endpoints in Clinical Trials: Definition and Operational Criteria, Stat Med 8(4):431–440 | MEM | Identifier not recorded. Formal surrogate criteria; restart agreement as a candidate surrogate for estimation error |
| [1803.04585](https://arxiv.org/abs/1803.04585) | Manheim & Garrabrant (2018). Categorizing Variants of Goodhart's Law; with Goodhart 1975, Campbell 1979, Strathern 1997 (European Review 5(3):305–321) | MEM | If architectures are chosen for consistency (as Song et al. urge), regressional and extremal Goodhart predict the proxy degrades. Originals' identifiers not recorded |
| [doi:10.1371/journal.pone.0118432](https://doi.org/10.1371/journal.pone.0118432) | Saito & Rehmsmeier (2015). The Precision-Recall Plot Is More Informative than the ROC Plot on Imbalanced Datasets, PLoS ONE 10(3) | MEM | Self-check: failures are 17–23% of units, so report PR-AUC beside ROC-AUC |

## 15. Measurement science and validity

| id | title | level | why it matters |
|---|---|---|---|
| [BIPM PDF](https://www.bipm.org/documents/20126/2071204/JCGM_200_2012.pdf); [clause pages](https://jcgm.bipm.org/vim/en/) | JCGM 200:2012, International Vocabulary of Metrology (VIM, 3rd ed.): §2.13 accuracy, 2.14 trueness, 2.15 precision, 2.20–2.25 repeatability / intermediate / reproducibility conditions, 2.39 calibration, 2.41 traceability, 2.45 validation | MEM | **Spine of the framing.** 2.14 Note 2: trueness "is not related to random measurement error". 2.15 Note 4: precision "erroneously used to mean measurement accuracy". 2.39 Note 2: adjustment "mistakenly called self-calibration". Restart agreement = precision under repeatability; weight-column cosine = trueness; our protocol = calibration in the 2.39 sense. **Do not claim traceability (2.41): one link, not a chain.** 2.45 makes validation measurand-specific (scope limit). Metrology pass read and quoted the clause pages; mentor-works pass flagged the same reference as unverified, so kept at MEM. 2.41 Note 5 came from a page summary. Cite the 2012 edition |
| [VIM 5.13](https://jcgm.bipm.org/vim/en/5.13.html) | VIM 5.1 measurement standard, 5.13 reference material, 5.14 certified reference material; ISO Guide 30 / ISO 17034 | MEM | Weight column as the "certified reference material". Clause pages not fetched; ISO identifiers not recorded |
| [ISO](https://www.iso.org/standard/69418.html) | ISO 5725-1:2023, Accuracy (trueness and precision) of measurement methods and results, Part 1 | ABS | Accuracy = trueness + precision. Catalogue page only, text paywalled; quote the VIM instead |
| [GUM PDF](https://www.bipm.org/documents/20126/2071204/JCGM_100_2008_E.pdf) | JCGM 100:2008 (GUM); also JCGM 104:2009 and JCGM GUM-1:2023 | ABS | Type A vs Type B uncertainty. URLs checked; text not read |
| [Eurachem PDF](https://www.eurachem.org/images/stories/Guides/pdf/MV_guide_2nd_ed_EN.pdf) | Magnusson & Örnemark (eds.) (2014). Eurachem: The Fitness for Purpose of Analytical Methods, 2nd ed., ISBN 978-91-87461-59-0 | ABS | Trueness and precision validated as separate parameters on known-value materials before unknowns. A 2025 3rd edition reportedly supersedes it; editors and ISBN unchecked |
| [Eurachem supplement](https://www.eurachem.org/images/stories/Guides/pdf/MV_Guide_planning_supplement_EN.pdf) | Eurachem, Planning and Reporting Method Validation Studies | ABS | Existence only |
| [ICH PDF](https://database.ich.org/sites/default/files/ICH_Q2(R2)_Guideline_2023_1130.pdf) | ICH Q2(R2), Validation of Analytical Procedures (Step 4, 1 Nov 2023) | ABS | Intermediate precision must cover days, analysts and equipment (our batch/device facets); re-validation on method transfer (recalibrate per model family). Details from secondary summaries |
| [NIST PDF](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.800-2.ipd.pdf) | NIST AI 800-2 (IPD, Jan 2026). Practices for Automated Benchmark Evaluations | ABS | Existence verified. Sibling of 800-3 (§5) |
| [NIST PDF](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.800-4.pdf) | NIST AI 800-4 | ABS | Exists; title and content not checked |
| [doi:10.6028/NIST.AI.100-1](https://doi.org/10.6028/NIST.AI.100-1) | NIST AI 100-1, AI Risk Management Framework 1.0 (Jan 2023) | MEM | MEASURE function |
| [doi:10.6028/NIST.SP.1270](https://doi.org/10.6028/NIST.SP.1270) | Schwartz et al. (2022). NIST SP 1270, Towards a Standard for Identifying and Managing Bias in AI | MEM | Background |
| [NPL](https://www.npl.co.uk/research/data-science-and-ai/national-metrology-institutes-in-artificial-intelligence-machine-learning) | NPL, role of NMIs in AI/ML; [NPL Report MS 34](https://eprintspublications.npl.co.uk/9306/1/MS34.pdf), Uncertainty evaluation for machine learning | ABS | Traceability and uncertainty for ML predictions. Search summary |
| [PTB](https://www.ptb.de/cms/en/ptb/competence-centers/ai-and-metrology.html) | PTB AI and Metrology centre (founded 17 Jan 2025), M4AIM programme; related [ScienceDirect paper](https://www.sciencedirect.com/science/article/pii/S2665917424007657) | ABS | Closest institutional precedent for "metrology for explainability". Paper author/year unverified. One signalling sentence at most |
| [2608.27463](https://arxiv.org/pdf/2608.27463) | Rating the Raters: Rasch Measurement Theory for LLM Evaluation (2026) | ABS | Listing only. Psychometric models entering LLM evaluation |
| [doi:10.1145/3442188.3445901](https://doi.org/10.1145/3442188.3445901) | Jacobs & Wallach (2021). Measurement and Fairness, FAccT '21, pp. 375–385; [1912.05511](https://arxiv.org/abs/1912.05511) | ABS | Construct reliability (≈ precision) vs construct validity. "Stable but wrong" = reliable but not valid. Pair with VIM 2.14 |
| [PMLR 267](https://proceedings.mlr.press/v267/wallach25a.html) | Wallach et al. (2025). Position: Evaluating Generative AI Systems Is a Social Science Measurement Challenge, ICML 2025; [2502.00561](https://arxiv.org/abs/2502.00561) | ABS | Most persuasive single citation for an ML reviewer: asks for validity interrogation of measurement instruments |
| [2111.15366](https://arxiv.org/abs/2111.15366) | Raji, Bender, Paullada, Denton, Hanna (2021). AI and the Everything in the Whole Wide World Benchmark, NeurIPS 2021 D&B | MEM | Construct-validity critique of general benchmarks. Metrology pass checked it (ABS); metric-precedents pass from memory |
| [doi:10.1037/h0040957](https://doi.org/10.1037/h0040957) | Cronbach & Meehl (1955). Construct validity in psychological tests, Psych Bull 52(4):281–302 | MEM | Foundational construct validity |
| [doi:10.1037/h0046016](https://doi.org/10.1037/h0046016) | Campbell & Fiske (1959). Convergent and discriminant validation by the multitrait-multimethod matrix, Psych Bull 56(2):81–105 | MEM | Shared method variance inflates agreement between methods: our two-method check |
| — | Messick (1995). Validity of psychological assessment, American Psychologist 50(9):741–749 | MEM | Identifier not recorded. Unified validity |
| — | AERA, APA, NCME (2014). Standards for Educational and Psychological Testing | MEM | Identifier not recorded. Reliability necessary, not sufficient, for valid interpretation |
| — | Bland & Altman (1986). Statistical methods for assessing agreement between two methods of clinical measurement, Lancet 327(8476):307–310 | MEM | Identifier not recorded. Agreement between two methods does not show either is correct |
| — | Shadish, Cook, Campbell (2002). Experimental and Quasi-Experimental Designs for Generalized Causal Inference (ISBN 0-395-61556-9, unverified) | MEM | Four validity types: frame for the threats-to-validity table |
| [Semantic Scholar](https://www.semanticscholar.org/paper/A-Guideline-of-Selecting-and-Reporting-Intraclass-Koo-Li/33122fc2f544888a0f46dd221a783ad1ecc1ddad) | Koo & Li (2016). A Guideline of Selecting and Reporting ICCs, J Chiropr Med 15:155–163, doi:10.1016/j.jcm.2016.02.012 | ABS | ICC(A,1) for test-retest of alignment and signals; state the ICC form. Search snippets |
| [doi:10.1037/0033-2909.86.2.420](https://doi.org/10.1037/0033-2909.86.2.420) | Shrout & Fleiss (1979). Original ICC taxonomy, Psych Bull 86:420–428 | MEM | Original ICC taxonomy |
| — | Brennan (2001). Generalizability Theory; Cronbach et al. (1972) | MEM | Identifier not recorded. G-study / D-study over seed, batch, device, token facets |
| — | Spearman (1904). Attenuation | MEM | Identifier not recorded. Achievable correlation capped by √(ICC_signal × ICC_outcome): weak vs noisy signal |

## 16. Stability versus predictability theory

Jin et al., Garg et al. and Jiang et al. are in §13; 2602.13450 is in §3.

| id | title | level | why it matters |
|---|---|---|---|
| [1901.08152](https://arxiv.org/abs/1901.08152) | Yu & Kumbier (2020). Veridical Data Science, PNAS 117(8):3920–3929 ([PNAS](https://www.pnas.org/doi/10.1073/pnas.1901326117)) | ABS | **PCS puts prediction screening before stability** and names random initial values as a stability perturbation (Sec. D.3); stability is a "prerequisite", the three principles "minimum requirements". The stability pass quoted arXiv v5 full text (FULL there); the census checked the abstract only, so ABS. PNAS text not fetched. **"Necessary but not sufficient" is not their phrase** |
| [1310.0150](https://arxiv.org/abs/1310.0150) | Yu (2013). Stability, Bernoulli 19(4):1484–1500 | ABS | Reproducibility "at a minimum" as stability; ES-CV traded 60% fewer predictors for 1.3% accuracy. Stability not the end goal |
| [MIT Press](https://mitpress.ublish.com/book/veridical-data-science-the-practice-of-responsible-data-analysis-and-decision-making) | Yu & Barter (2024). Veridical Data Science (book, ISBN 9780262049191; vdsbook.com) | ABS | Publisher summary only. Stability and PCS chapters may hold the "necessary but not sufficient" wording; unchecked |
| [0809.2932](https://arxiv.org/abs/0809.2932) | Meinshausen & Bühlmann (2010). Stability Selection, JRSS-B 72(4):417–473 | ABS | Error control needs exchangeability and a better-than-random base procedure. Stable-and-wrong units break the second |
| [PDF](http://www.statslab.cam.ac.uk/~rjs57/SSFinal.pdf) | Shah & Samworth (2013). Restatement of stability selection, JRSS-B 75:55–80; title not recorded | ABS | Restates the stability-selection assumptions; calls exchangeability "perhaps stronger than desired" |
| [JMLR PDF](https://www.jmlr.org/papers/volume2/bousquet02a/bousquet02a.pdf) | Bousquet & Elisseeff (2002). Stability and Generalization, JMLR 2:499–526 | ABS | Stability bounds the generalisation gap, not distance to the true parameter |
| [JMLR PDF](https://jmlr.org/papers/volume6/elisseeff05a/elisseeff05a.pdf) | Elisseeff, Evgeniou, Pontil (2005). Stability of Randomized Learning Algorithms, JMLR 6 | ABS | Extends to random init. Same limit |
| [1509.01240](https://arxiv.org/abs/1509.01240) | Hardt, Recht, Singer (2016). Train faster, generalize better, ICML 2016 | MEM | Few SGD steps are uniformly stable. Hypothesis: early stopping makes restarts look alike before resolving |
| [Springer](https://link.springer.com/article/10.1007/BF02591684) | Boender & Rinnooy Kan (1987). Bayesian stopping rules for multistart global optimization methods, Math Programming 37:59–80 | ABS | Local minima as multinomial draws; posteriors on number and basin sizes. 40-year-old form of "agreement measures basin size" |
| — | Zieliński (1981). Multistart precursor cited by 2602.13450 | MEM | Identifier not recorded |
| [Springer](https://link.springer.com/article/10.1007/BF01581094) | Optimal and sub-optimal stopping rules for the Multistart algorithm in global optimization, Math Programming | ABS | Title only; authors and year not recorded |
| [2001.05205](https://arxiv.org/abs/2001.05205) | Yehudai & Shamir (2020). Learning a Single Neuron with Gradient Methods, COLT 2020 | MEM | Gradient methods need assumptions on distribution and activation. Stability pass checked the abstract; estimator pass from memory |
| [2106.01101](https://arxiv.org/abs/2106.01101) | Vardi, Yehudai, Shamir (2021). Learning a Single Neuron with Bias Using Gradient Descent, NeurIPS 2021 | ABS | With bias, GD can fail with probability near 1/2 over init, even on a ball. Our failures are strongly biased sparse units |
| [1712.08968](https://arxiv.org/abs/1712.08968) | Safran & Shamir (2018). Spurious Local Minima are Common in Two-Layer ReLU Networks, ICML 2018 | ABS | Spurious minima hit with high probability; over-parameterisation mitigates (cf. `fit_cascade` rescues). Venue from memory |
| [1510.06096](https://arxiv.org/abs/1510.06096) | Sun, Qu, Wright (2015). When Are Nonconvex Problems Not Scary? | ABS | Benign class: all local minima global plus negative curvature at saddles. Only there does agreement imply correctness |
| [PMLR 70](https://proceedings.mlr.press/v70/ge17a.html) | Ge, Jin, Zheng (2017). No Spurious Local Minima in Nonconvex Low Rank Problems, ICML 2017 | ABS | Benign landscapes for sensing, completion, robust PCA |
| [2003.10409](https://arxiv.org/abs/2003.10409) | Ben Arous, Gheissari, Jagannath (2021). Online SGD on Non-convex Losses from High-dimensional Inference, JMLR 22(106) | MEM | Information exponent; "almost all of the data is used simply in the initial search phase". Bussgang ≈ 0 means exponent ≥ 2. Stability and validity passes checked it; estimator pass from memory |
| [1607.06534](https://arxiv.org/abs/1607.06534) | Mei, Bai, Montanari (2018). The Landscape of Empirical Risk for Non-convex Losses | ABS | Population critical points survive in finite samples; more data cannot remove a spurious population minimum. Ann Stat venue from memory |
| [1104.2018](https://arxiv.org/abs/1104.2018) | Kakade, Kanade, Shamir, Kalai (2011). Efficient Learning of GLMs and SIMs with Isotonic Regression (GLMtron, L-Isotron), NIPS 2011 | ABS | Guarantees need a monotone, Lipschitz link; GELU's lobe breaks this. Isotron (Kalai & Sastry, COLT 2009) from the same abstract |
| [2006.06997](https://arxiv.org/abs/2006.06997) | Sarao Mannelli, Biroli, Cammarota, Krzakala, Urbani, Zdeborová (2020). Complex Dynamics in Simple Neural Networks: Gradient Flow in Phase Retrieval, NeurIPS 2020 | ABS | Spurious minima with large basins at low n/d, destabilised by a BBP Hessian outlier above a threshold. Basis for a Hessian-outlier diagnostic. Follow-up on over-parameterised students lowering the threshold is from memory |
| [2305.18502](https://arxiv.org/abs/2305.18502) | Arnaboldi, Krzakala, Loureiro, Stephan (2023). Escaping mediocrity: two-layer networks learn hard GLMs | ABS | Over-parameterisation helps online SGD only by a constant factor; the cascade's large rescues are unexplained by current theory |

## 17. Estimators and subspace recovery

### 17a. Single-index estimation

| id | title | level | why it matters |
|---|---|---|---|
| — | Kalai & Sastry (2009). Isotron, COLT 2009 | MEM | Identifier not recorded. First efficient SIM learner; monotone link |
| [NeurIPS 2023 PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/2f46ef5725a8eca24f7f24a17955ad1a-Paper-Conference.pdf) | Omnipredictors for single-index models (NeurIPS 2023) | ABS | Robust SIM learning, mostly monotone links (scope from memory) |
| — | Härdle & Stoker (1989), JASA; Powell, Stock & Stoker (1989), Econometrica. Average derivative estimation | MEM | Identifier not recorded. Needs the input score; hopeless in d = 768 |
| — | Hristache, Juditsky, Spokoiny (2001). Direct estimation of the index coefficient, Ann Stat | MEM | Identifier not recorded |
| — | Brillinger (1982); Li & Duan (1989). Regression analysis under link violation, Ann Stat | MEM | Identifier not recorded. OLS is direction-consistent under linear conditional means, scale Cov(g(z), z) ≈ 0 for non-monotone links: our STA/OLS failures |
| [ACM DL](https://dl.acm.org/doi/10.5555/2969442.2969621) | Thrampoulidis, Abbasi, Hassibi (2015). LASSO with non-linear measurements is equivalent to one with linear measurements, NIPS 2015 | ABS | Effective signal μ = E[g(z)z]. Title checked; details from memory |
| [PMLR 70](https://proceedings.mlr.press/v70/yang17a.html) | Yang, Balasubramanian, Liu (2017). High-dimensional Non-Gaussian Single Index Models via Thresholded Score Function Estimation, ICML 2017 | ABS | Stein score estimators with heavy-tail truncation |
| [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2017/file/4db0f8b0fc895da263fd77fc8aecabe4-Paper.pdf) | Yang, Balasubramanian, Wang, Liu (2017). Learning Non-Gaussian Multi-Index Model via Second-Order Stein's Method; companion [1709.08795](https://arxiv.org/pdf/1709.08795) | ABS | Works when E[g′] = 0 but E[g″] ≠ 0. Needs a second-order score in 768-d: weak initialiser at best |
| [2305.10633](https://arxiv.org/abs/2305.10633) | Damian, Nichani, Ge, Lee (2023). Smoothing the Landscape Boosts the Signal for SGD, NeurIPS 2023 | ABS | Smoothed loss reaches the CSQ bound. Motivates continuation schedules |
| [PMLR 247](https://proceedings.mlr.press/v247/damian24a.html) | Damian, Pillaud-Vivien, Lee, Bruna (2024). Computational-Statistical Gaps in Gaussian Single-Index Models, COLT 2024 | MEM | Generative exponent: a label transform can restore low-order signal; basis for the 1[y > 0] initialiser. Estimator pass checked it; validity pass flagged it unfetched |
| [2405.15459](https://arxiv.org/abs/2405.15459) | Dandi et al. (2024). Repetita Iuvant: data repetition allows SGD to learn multi-index functions | ABS | Multi-pass reuse beats one-pass limits. Body unread; authors unconfirmed in one pass |
| [2602.02431](https://arxiv.org/pdf/2602.02431) | Full-Batch Gradient Descent Outperforms One-Pass SGD: Sample Complexity Separation in Single-Index Learning (2026) | ABS | Exists; contents unverified |
| [2402.03220](https://arxiv.org/abs/2402.03220) | Dandi et al. (2024). Benefits of reusing batches | MEM | Data reuse |
| [2406.01581](https://arxiv.org/abs/2406.01581) | Lee, Oko, Suzuki, Wu (2024) | MEM | Data reuse; title not recorded |
| [2510.21020](https://arxiv.org/html/2510.21020v1) | From Information to Generative Exponent: Learning Rate Induces Phase Transitions in SGD (2025) | ABS | Exists; contents unverified |
| [2210.15651](https://arxiv.org/abs/2210.15651) | Bietti, Bruna, Sanford, Song (2022). Learning single-index models with shallow neural networks, NeurIPS 2022 | MEM | Unknown-link two-layer net, our setup |
| [2307.15804](https://arxiv.org/abs/2307.15804) | Bruna, Pillaud-Vivien, Zweig (2023). On Single Index Models beyond Gaussian Data, NeurIPS 2023 | ABS | Closest theory to our non-Gaussian inputs; direction must not align with special input structure |
| [2506.09887](https://arxiv.org/abs/2506.09887) | Joshi, Koubbi, Misiakiewicz, Srebro (2025). Learning Single-Index Models via Harmonic Decomposition, NeurIPS 2025 | ABS | Spherically symmetric inputs; tensor unfolding vs online SGD trade-off |
| [2503.23642](https://arxiv.org/abs/2503.23642) | Braun, Quang, Imaizumi (2025). Learning a Single Index Model from Anisotropic Data with vanilla SGD, AISTATS 2025 | ABS | Complexity set by an effective dimension of Σ: report Σ-whitened cosine beside Euclidean |
| [2602.09959](https://arxiv.org/pdf/2602.09959) | Statistical-Computational Trade-offs in Learning Multi-Index Models via Harmonic Analysis (2026) | ABS | Exists; contents unverified |
| [2603.17896](https://arxiv.org/html/2603.17896) | A Noise Sensitivity Exponent Controls Large Statistical-to-Computational Gaps (2026) | ABS | Exists; contents unverified |
| [1705.04591](https://arxiv.org/abs/1705.04591) | Soltanolkotabi (2017). ReLU learning via gradient descent, NeurIPS 2017; title not recorded | MEM | No-bias single neuron baseline |
| [1606.08415](https://arxiv.org/abs/1606.08415) | Hendrycks & Gimpel (2016). GELU | MEM | GELU = xΦ(x). Minimum ≈ −0.17 near −0.75 (SiLU ≈ −0.28 near −1.28) are unverified numerics |

### 17b. Multi-index and sufficient dimension reduction

| id | title | level | why it matters |
|---|---|---|---|
| [doi:10.1080/01621459.1991.10475035](https://doi.org/10.1080/01621459.1991.10475035) | Li (1991). Sliced Inverse Regression, JASA 86:316–327 | MEM | Needs linearity condition; misses symmetric links |
| — | Cook & Weisberg (1991). SAVE (discussion of Li 1991) | MEM | Identifier not recorded. Sees symmetric links; needs constant variance |
| [doi:10.1080/01621459.1992.10476258](https://doi.org/10.1080/01621459.1992.10476258) | Li (1992). Principal Hessian Directions, JASA 87:1025–1039 | MEM | Blind to tanh(z₂) in the additive unit; sort eigenvalues by absolute value or lose a direction in multiplicative units |
| [RePEc](https://ideas.repec.org/a/bes/jnlasa/v102y2007mseptemberp997-1008.html) | Li & Wang (2007). Directional Regression, JASA 102:997–1008 | MEM | Most accurate first-two-moment SDR under its conditions. Multi-index pass checked RePEc; single-index pass from memory |
| [Wiley](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/1467-9868.03411) | Xia, Tong, Li, Zhu (2002). An adaptive estimation of dimension reduction space (MAVE, OPG), JRSS-B 64:363–410 | ABS | No linearity condition; ill-conditioned smoothing in d = 768, so screen first. R `MAVE` package |
| [doi:10.1080/01621459.2013.838167](https://doi.org/10.1080/01621459.2013.838167) | Fukumizu & Leng (2014). Kernel/gradient SDR (gKDR), JASA 109(505):359–370 | ABS | RKHS gradient estimate, proven SDR recovery |
| [ACM DL](https://dl.acm.org/doi/10.5555/3020751.3020836) | Trivedi, Wang, Kpotufe, Shakhnarovich (2014). EGOP, UAI 2014 | ABS | EGOP span = central mean subspace for any x distribution |
| [2212.13881](https://arxiv.org/pdf/2212.13881) | Radhakrishnan, Beaglehole, Pandit, Belkin. AGOP / Recursive Feature Machines | ABS | Iterated AGOP reweighting; baseline |
| [2605.15082](https://arxiv.org/abs/2605.15082) | Zhu, Davis, Drusvyatskiy, Fazel (May 2026). KRR + AGOP provable multi-index recovery | ABS | **Recommended K≥2 warm start.** AGOP must come from an unconstrained model. Whether x must be Gaussian is not stated in the abstract |
| [1304.2070](https://arxiv.org/abs/1304.2070) | Constantine, Dow, Wang (2014). Active subspaces, SIAM J Sci Comput | MEM | EGOP with exact gradients; usable downstream when y is differentiable |
| [Project Euclid](https://projecteuclid.org/journals/annals-of-statistics/volume-37/issue-3/Dimension-reduction-for-nonelliptically-distributed-predictors/10.1214/08-AOS598.full) | Li & Dong (2009). Dimension reduction for non-elliptically distributed predictors, Ann Stat 37(3):1272–1298 | ABS | Removes the elliptical requirement |
| [OUP](https://academic.oup.com/biomet/article-abstract/97/2/279/219362) | Dong & Li (2010), Biometrika 97(2):279 | ABS | Second-order version |
| [R Journal](https://journal.r-project.org/articles/RJ-2019-006/) | Ma & Zhu (2012). Semiparametric SDR, JASA 107(497):168–179; R `orthoDr` | ABS | Linearity and constant variance unnecessary; orthogonality-constrained fit |
| [doi:10.1214/18-EJS1428](https://doi.org/10.1214/18-EJS1428) | Babichev & Bach (2018). Slice inverse regression with score functions, EJS 12(1):1507–1543 | MEM | Score-based SIR for non-Gaussian x. Multi-index pass checked it; single-index pass from memory |
| — | Hall & Li (1993), Ann Stat; Diaconis & Freedman (1984) | MEM | Identifier not recorded. Most low-dimensional projections near-Gaussian, not adversarial ones aligned with outlier dims |
| [Statistics Surveys](https://projecteuclid.org/journals/statistics-surveys/volume-16/issue-none/Central-subspaces-review-methods-and-applications/10.1214/22-SS138.pdf) | Central subspaces review (2022) | ABS | Review; exists |
| — | Li (2018). Sufficient Dimension Reduction (CRC); Ma & Zhu (2013), Int Stat Review; Python `sliced` package | MEM | Identifier not recorded. Reviews and reference implementation |
| [doi:10.1080/01621459.1981.10477729](https://doi.org/10.1080/01621459.1981.10477729) | Friedman & Stuetzle (1981). Projection Pursuit Regression, JASA 76:817–823 | MEM | Principled version of our greedy deflation, with backfitting |
| [1412.2863](https://arxiv.org/abs/1412.2863) | Janzamin, Sedghi, Anandkumar (2014). Score function features | MEM | Tensor recovery with score functions. Multi-index pass checked it; single-index pass from memory |
| [1506.08473](https://arxiv.org/abs/1506.08473) | Janzamin, Sedghi, Anandkumar (2015). Beating the perils of non-convexity | MEM | Tensor decomposition for first-layer weights |
| [2504.05426](https://arxiv.org/abs/2504.05426) | Bruna & Hsu (2025). Survey on algorithms for multi-index models | ABS | Review of efficient estimators (Gaussian x) |
| [2506.05500](https://arxiv.org/abs/2506.05500) | Damian, Lee, Bruna (2025). Generative leap, NeurIPS 2025 | ABS | Optimal estimators are sequential and condition on found directions instead of deleting them: correct deflation design |
| [2608.12183](https://arxiv.org/abs/2608.12183) | Spectral phase transitions in Gaussian multi-index models (2026) | ABS | Exists; content not read |
| [2302.11055](https://arxiv.org/pdf/2302.11055) | Abbe, Boix-Adserà, Misiakiewicz (2023). Leap complexity / saddle-to-saddle, COLT 2023 | ABS | Supports learned sequentially |
| [2202.08658](https://arxiv.org/abs/2202.08658) | Abbe, Boix-Adserà, Misiakiewicz (2022). Merged staircase, COLT 2022 | MEM | Precursor |
| [2305.18270](https://arxiv.org/abs/2305.18270) | Dandi, Krzakala, Loureiro, Pesce, Stephan (2024). One giant step, JMLR 25 | ABS | One step with n = O(d) learns only one direction: explains the 0.52 / 0.36 collapse |
| [2310.19793](https://arxiv.org/abs/2310.19793) | Bietti, Bruna, Pillaud-Vivien. Grassmannian gradient flow for multi-index regression | ABS | Global convergence with orthonormal V: argues for a Stiefel constraint |
| [2411.08798](https://arxiv.org/abs/2411.08798) | Simsek et al. Learning Gaussian Multi-Index Models with Gradient Flow | MEM | Exists per a ResearchGate listing; the arXiv ID itself is unverified |
| [2609.10879](https://arxiv.org/abs/2609.10879) | Zhou, Xu, Du, Fazel (9 Sep 2026). Beyond small initialisation | ABS | Incremental learning by Hermite order; competitive dynamics (symmetry trap) |
| [2412.19033](https://arxiv.org/abs/2412.19033) | Xu & Yu (2025). NNs do SDR with rank regularisation, AAAI 2025 | ABS | First layer consistent for the central mean subspace |
| [2104.10009](https://arxiv.org/abs/2104.10009) | Kapla, Fertl, Bura (2021/22). NN-based MAVE, CSDA | ABS | Literally our estimator class (MLP link, low-rank B). Baseline |
| [2412.08961](https://arxiv.org/pdf/2412.08961) | Belted and ensembled NN for SDR | ABS | Exists; not read |
| [2606.01346](https://arxiv.org/pdf/2606.01346) | FlowSDR (2026) | ABS | Exists; not read |
| [2303.00055](https://arxiv.org/abs/2303.00055) | Berthier, Montanari, Zhou (2023). Two-timescale analysis | MEM | Fast g / slow V order |
| [2109.04404](https://arxiv.org/abs/2109.04404) | Timkey & van Schijndel (2021). Rogue dimensions in GPT-2 | MEM | Anisotropy biases E[y x] toward rogue coordinates: whiten first |

### 17c. Phase-retrieval structure and optimisation fixes

| id | title | level | why it matters |
|---|---|---|---|
| [2509.23527](https://arxiv.org/pdf/2509.23527) | Chen & Shen (Sep 2025). Spectral initialization and precise asymptotics for SIMs | ABS | Spectral init + GD converges when it lands in the benign region |
| [1708.05932](https://arxiv.org/abs/1708.05932) | Mondelli & Montanari (2019). Fundamental limits of weak recovery, FoCM | MEM | Optimal spectral preprocessing (Gaussian design) |
| [1702.06435](https://arxiv.org/abs/1702.06435) | Lu & Li. Phase transitions of spectral initialization | MEM | Same |
| [1811.04420](https://arxiv.org/abs/1811.04420) | Luo, Alghamdi, Lu (2019). Optimal spectral initialization, IEEE TSP | MEM | Same |
| [2010.03460](https://arxiv.org/pdf/2010.03460) | AMP with spectral initialisation for GLMs | ABS | Exists |
| [1306.0160](https://arxiv.org/pdf/1306.0160) | Netrapalli, Jain, Sanghavi (2013). Phase Retrieval using Alternating Minimization, NeurIPS 2013 | ABS | AltMin over the hidden branch label: report branch-aware inversion as an oracle ceiling, not a competitor |
| [1812.01255](https://arxiv.org/pdf/1812.01255) | Random-init AltMin for phase retrieval | ABS | Exists |
| [1310.3745](https://arxiv.org/abs/1310.3745) | Yi, Caramanis, Sanghavi (2014). Mixed linear regression, ICML 2014 | MEM | Analogue for hidden labels |
| [1407.1065](https://arxiv.org/abs/1407.1065) | Candès, Li, Soltanolkotabi (2015). Wirtinger flow | MEM | Phase-retrieval baseline |
| [1602.06664](https://arxiv.org/abs/1602.06664) | Sun, Qu, Wright (2018). Phase-retrieval landscape, FoCM; title not recorded | MEM | No spurious minima for Gaussian phase retrieval at n ≳ d log³ d |
| — | Tan & Vershynin. Online SGD for non-smooth phase retrieval | MEM | Identifier not recorded |
| [2311.15221](https://arxiv.org/pdf/2311.15221) | The Local Landscape of Phase Retrieval Under Limited Samples | ABS | Exists |
| [2602.17779](https://arxiv.org/pdf/2602.17779) | Topological Exploration of High-Dimensional Empirical Risk Landscapes (phase retrieval) | ABS | Exists |
| — | Absil, Mahony, Sepulchre (2008). Optimization Algorithms on Matrix Manifolds | MEM | Identifier not recorded. Riemannian methods escape strict saddles, not spurious minima |

### 17d. Neuroscience receptive-field estimators

| id | title | level | why it matters |
|---|---|---|---|
| [PLoS CB](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1004141) | Williamson, Sahani, Pillow (2015). MID ≡ LNP maximum likelihood, PLoS CB 11(4); [1308.3542](https://arxiv.org/abs/1308.3542); [2019 correction](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1007139) | ABS | MID = LNP maximum likelihood; suboptimal for non-Poisson responses. **Author is Sahani, not "Savin"** |
| [0801.0311](https://arxiv.org/pdf/0801.0311) | Sharpee (2007/08). Comparison of objective functions for estimating linear-nonlinear models | ABS | Rényi-2 = least squares; KL (MID) has smallest asymptotic error. Changing the objective changes efficiency, not local optima |
| — | Kouh & Sharpee (2009). Estimating LN models using Rényi divergences, Neural Comput | MEM | Identifier not recorded |
| [PubMed](https://pubmed.ncbi.nlm.nih.gov/12938766/) | Paninski (2003). Convergence properties of three spike-triggered analysis techniques, Network 14(3):437–464 | ABS | STA needs elliptical, STC Gaussian stimuli; gives a consistent information-theoretic estimator. One pass relied on a Wikipedia summary |
| [doi:10.1167/6.4.9](https://doi.org/10.1167/6.4.9) | Pillow & Simoncelli (2006). iSTAC, J Vision 6(4):414–428 | MEM | Exact only for Gaussian stimuli: do not invest |
| [J Vis](https://jov.arvojournals.org/article.aspx?articleid=2192881) | Schwartz, Pillow, Rust, Simoncelli (2006). Spike-triggered neural characterization | ABS | Review; STA/STC valid under white or Gaussian stimuli |
| — | Park & Pillow (2011) Bayesian STC; Park, Archer, Priebe, Pillow (2013) spectral GQM; Ramirez & Paninski (2014) expected log-likelihood | MEM | Identifier not recorded. Moment forms exact only under Gaussian stimuli |
| [1201.0321](https://arxiv.org/pdf/1201.0321) | Rajan & Bialek (2013). Maximally Informative "Stimulus Energies", PLoS ONE | ABS | Quadratic MID extension |
| [1209.0121](https://arxiv.org/pdf/1209.0121) | Learning quadratic receptive fields from neural responses to natural stimuli | ABS | Exists |
| — | Paninski (2004). Maximum likelihood estimation of cascade point-process neural encoding models, Network | MEM | Identifier not recorded. Concavity needs convex, log-concave link; not GELU |
| [Springer](https://link.springer.com/article/10.1007/s10827-012-0411-y) | Samengo & Gollisch (2013). Spike-triggered covariance: geometric proof, symmetry properties, and extension beyond Gaussian stimuli, J Comput Neurosci | MEM | Non-Gaussian STC. Estimator pass from memory; validity pass linked it without a content check |
| — | Rowekamp & Sharpee (2011), Network; Fitzgerald, Rowekamp, Sincich, Sharpee (2011), PLoS CB | MEM | Identifier not recorded. Multi-component natural-stimulus MID |
| [PLoS CB](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1005113) | MID in auditory cortex vs inferior colliculus | ABS | A1 needs several dimensions, IC one: neuroscience analogue of rank-1 failing on some units |
| [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3974709/) | Discriminative learning of receptive fields from non-Gaussian stimuli (CbRF) | ABS | Classifier weights as RF filter; candidate baseline |
| [1912.12106](https://arxiv.org/abs/1912.12106) | Borji & Lin (2019). White Noise Analysis of Neural Networks, ICLR 2020 | ABS | Closest ANN precedent; Gaussian probes, no exact-truth validation. Authors and venue from memory in one pass |
| [NIPS PDF](https://papers.nips.cc/paper/5962-convolutional-spike-triggered-covariance-analysis-for-neural-subunit-models.pdf) | Wu, Park, Pillow (2015). Convolutional Spike-triggered Covariance Analysis, NIPS 2015 | ABS | STA/STC efficient for quadratic subunit models under Gaussian stimuli |
| [Sci Rep](https://www.nature.com/articles/s41598-019-40535-4) | CNNs as models of biological receptive fields (Sci Rep 2019); [1711.02837](https://ar5iv.labs.arxiv.org/html/1711.02837) | ABS | Reverse direction from ours |

### 17e. Geometry and recovery metrics

| id | title | level | why it matters |
|---|---|---|---|
| [2311.03658](https://arxiv.org/abs/2311.03658) | Park, Choe, Veitch (2024). Causal inner product on LLM representation spaces, ICML 2024 ([PMLR](https://proceedings.mlr.press/v235/park24c.html)) | ABS | Euclidean inner product not canonical: support for the "wrong geometry" objection and for our Σ-cosine answer |
| [Colab](https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Main_Demo.ipynb) | TransformerLens Main Demo | ABS | Folds γ into reading weights and centres them because LN removes the all-ones direction. Precedent for treating unexcited components as null; our ln_2 output has an exactly unexcited direction |
| [physics/9806030](https://arxiv.org/abs/physics/9806030) | Edelman, Arias, Smith (1998). Grassmann geometry; title not recorded | MEM | Grassmann distances for K ≥ 2 labels |
| — | Ye & Lim (2016). Subspace distances, SIAM J Matrix Anal Appl; title not recorded | MEM | Identifier not recorded |
| — | Björck & Golub (1973). Principal angles, Math Comp 27:579–594; `scipy.linalg.subspace_angles` | MEM | Identifier not recorded; function name unverified |
| — | Hooper (1959); Hotelling (1936). Trace and vector correlation | MEM | Identifier not recorded. Standard SDR accuracy measures |
| [Springer](https://link.springer.com/chapter/10.1007/978-3-642-15995-4_29) | Amari, Cichocki, Yang (1996). Amari performance index, NIPS 8; later "A New Performance Index for ICA" (linked); [survey of ICA performance indexes](https://www.researchgate.net/publication/228737660_A_survey_of_the_performance_indexes_of_ICA_algorithms) | MEM | ICA reports a continuous, permutation-invariant index with no threshold. 1996 paper not fetched |

## 18. Statistics and reproducibility

### 18a. AUC inference, calibration, guaranteed trust rules

| id | title | level | why it matters |
|---|---|---|---|
| [doi:10.2307/2533497](https://doi.org/10.2307/2533497) | Obuchowski (1997). Nonparametric analysis of clustered ROC curve data, Biometrics 53(2):567–578 | MEM | Clustered DeLong: units nested in layers and models. Fixes the cross-layer pairing problem. **Conflict: DOI given as 10.2307/2533497 in one pass and 10.2307/2533958 in another;** check |
| — | Venkatraman & Begg (1996). Distribution-free comparison of paired ROC curves, Biometrika 83:835–848 | MEM | Identifier not recorded. Permutation test of curve equality |
| [doi:10.1186/1471-2105-12-77](https://doi.org/10.1186/1471-2105-12-77) | Robin et al. (2011). pROC, BMC Bioinformatics 12:77 | ABS | DeLong, bootstrap and Venkatraman tests; manual not read in full |
| [doi:10.1109/LSP.2014.2337313](https://doi.org/10.1109/LSP.2014.2337313) | Sun & Xu (2014). Fast DeLong, IEEE SPL 21(11):1389–1393 | MEM | O(n log n) implementation |
| [PyPI](https://pypi.org/project/MLstatkit/) | MLstatkit | ABS | Python DeLong |
| [doi:10.1002/sim.5328](https://doi.org/10.1002/sim.5328) | Demler, Pencina, D'Agostino (2012). Misuse of DeLong test to compare AUCs for nested models, Stat Med 31:2577–2587 | MEM | Never DeLong a combined signal against its component |
| [doi:10.1002/sim.5727](https://doi.org/10.1002/sim.5727) | Pepe et al. (2013), Stat Med | MEM | Test improvement through the coefficient instead |
| [doi:10.1186/s12874-015-0025-y](https://doi.org/10.1186/s12874-015-0025-y) | BMC Med Res Methodol (2015). Small-sample AUC comparison for biomarker selection | ABS | Permutation best at small n; asymptotic liberal. Snippet only; numbers unverified |
| [metricgate](https://metricgate.com/docs/roc-paired-comparison-delong/) | metricgate DeLong and bootstrap documentation | ABS | Practitioner source for the "<25 per class" DeLong caution. Heuristic, not a published threshold |
| — | Benavoli, Corani, Demšar, Zaffalon (2017). Time for a change: Bayesian comparison of classifiers, JMLR 18(77) | MEM | Identifier not recorded. Posterior P(A > B) with a ROPE |
| — | Gu, Ghosal, Roy (2008). Bayesian bootstrap for ROC, Stat Med 27 | MEM | Identifier not recorded |
| [2108.13264](https://arxiv.org/abs/2108.13264) | Agarwal et al. (2021). Deep RL at the Edge of the Statistical Precipice, NeurIPS 2021 | MEM | Interval estimates over point verdicts with few runs |
| [lme4](https://doi.org/10.18637/jss.v067.i01) | Bates, Mächler, Bolker, Walker (2015). lme4, J Stat Softw 67(1) | MEM | Crossed random effects for variance components and GLMM signal scoring |
| — | Niculescu-Mizil & Caruana (2005). Predicting good probabilities with supervised learning, ICML | MEM | Identifier not recorded. Platt over isotonic with few failures |
| [1909.10155](https://arxiv.org/abs/1909.10155) | Kumar, Liang, Ma (2019). Verified Uncertainty Calibration, NeurIPS | MEM | Debiased ECE |
| [2012.08668](https://arxiv.org/abs/2012.08668) | Roelofs et al. (2022). Mitigating bias in calibration error estimation, AISTATS | MEM | Equal-mass binning |
| — | Brier (1950), Monthly Weather Review 78:1–3 | MEM | Identifier not recorded. Brier decomposition |
| [doi:10.1186/s12916-019-1466-7](https://doi.org/10.1186/s12916-019-1466-7) | Van Calster et al. (2019). Calibration: the Achilles heel of predictive analytics, BMC Med 17:230 | MEM | Report calibration, not only discrimination |
| [doi:10.1136/bmj.332.7549.1080](https://doi.org/10.1136/bmj.332.7549.1080) | Altman & Royston (2006). The cost of dichotomising continuous variables, BMJ 332:1080 | MEM | Report continuous alignment beside the < 0.95 label |
| [doi:10.1136/bmj.m441](https://doi.org/10.1136/bmj.m441) | Riley et al. (2020). Sample size for developing a clinical prediction model, BMJ 368:m441 | MEM | Sample-size justification template |
| [doi:10.1177/1948550617697177](https://doi.org/10.1177/1948550617697177) | Lakens (2017). Equivalence tests: a practical primer, SPPS 8(4):355–362 | MEM | TOST for any "no better than baseline" claim |
| — | Harrell et al. (1982). Concordance index, JAMA | MEM | Identifier not recorded. Threshold-free C for continuous alignment |
| [2110.01052](https://arxiv.org/pdf/2110.01052) | Angelopoulos, Bates, Candès, Jordan, Lei. Learn then Test, Ann Appl Stat 19(2) (2025) | ABS | "Trust this unit" rule with failure rate among trusted units ≤ α w.p. ≥ 1 − δ. With ~200 calibration units, 5% is certifiable only with ≤ 5 trusted failures; state per layer and family |
| [2208.02814](https://arxiv.org/pdf/2208.02814) | Conformal Risk Control | ABS | Controls expected monotone loss |
| [2107.07511](https://arxiv.org/html/2107.07511v6) | Angelopoulos & Bates. A Gentle Introduction to Conformal Prediction | ABS | Tutorial |
| [2101.02703](https://arxiv.org/abs/2101.02703) | Bates et al. (2021). Distribution-free, risk-controlling prediction sets, JACM | MEM | RCPS precursor |
| [2603.24704](https://arxiv.org/pdf/2603.24704) | Conformal selective prediction with general risk control (2026) | ABS | Search result only; selection with guarantees is our problem exactly |
| — | Vovk, Gammerman, Shafer (2005). Algorithmic Learning in a Random World | MEM | Identifier not recorded. Mondrian (per-stratum) conformal |

### 18b. Multiplicity

| id | title | level | why it matters |
|---|---|---|---|
| [doi:10.1002/sim.3495](https://doi.org/10.1002/sim.3495) | Bretz, Maurer, Brannath, Posch (2009). Graphical approach to multiple testing (gatekeeping, fixed-sequence, fallback), Stat Med 28:586–604 | ABS | One primary at full α; secondaries get recycled α. Fixes the B-12 summed-flips verdict structure |
| [1812.00250](https://arxiv.org/pdf/1812.00250) | Hierarchically structured families in a graphical framework | ABS | Search result only |
| [1211.3313](https://arxiv.org/pdf/1211.3313) | Goeman & Solari. Sequential rejection principle | ABS | Search result only |
| — | R `gMCP` package | MEM | Identifier not recorded; CRAN status unchecked |
| — | Holm (1979). A simple sequentially rejective multiple test procedure, Scand J Stat 6:65–70 | MEM | Identifier not recorded. Confirmatory family |
| — | Benjamini & Yekutieli (2001). FDR under arbitrary dependence, Ann Stat 29:1165–1188 | MEM | Identifier not recorded |
| [doi:10.1080/01621459.1955.10501294](https://doi.org/10.1080/01621459.1955.10501294) | Dunnett (1955). Many-to-one comparisons against a common control, JASA 50:1096–1121 | MEM | Shared-control comparisons are correlated: never sum them |
| — | Westfall & Young (1993). Resampling-Based Multiple Testing | MEM | Identifier not recorded. Bootstrap max-T |
| — | Dror et al. (2018). The Hitchhiker's Guide to Testing Statistical Significance in NLP, ACL | MEM | Identifier not recorded |
| [doi:10.1186/1471-2288-13-91](https://doi.org/10.1186/1471-2288-13-91) | Fagerland, Lydersen, Laake (2013). Exact and mid-p McNemar, BMC Med Res Methodol 13:91; title not recorded | MEM | Exact or mid-p McNemar |

### 18c. Reporting guidelines and diagnostic-accuracy bias

| id | title | level | why it matters |
|---|---|---|---|
| [doi:10.1136/bmj-2023-078378](https://doi.org/10.1136/bmj-2023-078378) | Collins, Moons, Dhiman, Riley, Beam et al. (2024). TRIPOD+AI statement, BMJ 385:e078378 ([full text](https://epub.ub.uni-muenchen.de/116670/1/bmj-2023-078378.full.pdf)) | FULL | Item text read: 8a/8c outcome definition and blinding, 10 sample size, 12e measures, 15 thresholds, 18c–f protocol/registration/data/code, 23a CIs, 23b cluster heterogeneity, 26 limitations. Our signal is a prediction model for a binary outcome with an exact reference |
| [doi:10.1136/bmj-2024-082505](https://doi.org/10.1136/bmj-2024-082505) | Moons et al. (2025). PROBAST+AI, BMJ 388 | ABS | Run its signalling questions as a self-audit; ship the table as a supplement. Question wording not fetched |
| [doi:10.7326/M18-1376](https://doi.org/10.7326/M18-1376) | Wolff, Moons, Riley et al. (2019). PROBAST, Ann Intern Med 170(1):51–58 | MEM | Outcome assessed without predictor knowledge |
| [doi:10.1038/s41591-025-03953-8](https://doi.org/10.1038/s41591-025-03953-8) | STARD-AI reporting guideline, Nat Med 31:3283–3289 (2025) | ABS | 2×2 flagged × failed table and a unit flow diagram. Item list not fetched |
| [doi:10.1136/bmj.h5527](https://doi.org/10.1136/bmj.h5527) | Bossuyt et al. (2015). STARD 2015, BMJ 351:h5527 | MEM | Base list |
| [doi:10.1038/s41591-022-01772-9](https://doi.org/10.1038/s41591-022-01772-9) | Vasey et al. (2022). DECIDE-AI, Nat Med 28:924–933 | MEM | Weak transfer; say why in one sentence |
| [doi:10.1111/acem.12255](https://doi.org/10.1111/acem.12255) | Kohn, Carpenter, Newman (2013). Direction of bias in diagnostic accuracy studies, Acad Emerg Med 20(11) | ABS | Strict incorporation bias is absent (truth does not use R²), but shared-cause coupling via SNR is real: report incremental value over a nuisance baseline. Pages from memory |
| [Semantic Scholar](https://www.semanticscholar.org/paper/Incorporation-bias-in-studies-of-diagnostic-tests:-Worster-Carpenter/c37e785965d1705d3b5182c51c22b5890ed3b10f) | Worster & Carpenter (2008). Incorporation bias in studies of diagnostic tests, CJEM | MEM | Same point |
| [doi:10.1056/NEJM197810262991705](https://doi.org/10.1056/NEJM197810262991705) | Ransohoff & Feinstein (1978). Problems of spectrum and bias, NEJM 299:926–930 | MEM | Spectrum bias: report AUC within difficulty strata |

### 18d. Multiverse, blinding, controls, pre-registration, artefacts

| id | title | level | why it matters |
|---|---|---|---|
| [doi:10.1177/1745691616658637](https://doi.org/10.1177/1745691616658637) | Steegen, Tuerlinckx, Gelman, Vanpaemel (2016). Multiverse analysis, Perspect Psychol Sci 11(5) | MEM | Headline as a distribution over defensible specifications |
| [doi:10.1038/s41562-020-0912-z](https://doi.org/10.1038/s41562-020-0912-z) | Simonsohn, Simmons, Nelson (2020). Specification curve analysis, Nat Hum Behav 4:1208–1214 (correction exists) | ABS | Joint inference under a permutation null. The exact joint statistic (Stouffer vs Fisher) came through a garbled summary; check the PDF |
| [specr](https://masurp.github.io/specr/index.html) | `specr` R package | ABS | Implementation |
| [2609.28177](https://arxiv.org/html/2609.28177v1) | How Sensitive Are LLM Leaderboard Claims to Hidden Model Selection? (2026) | ABS | Title only |
| [doi:10.1177/0956797611417632](https://doi.org/10.1177/0956797611417632) | Simmons, Nelson, Simonsohn (2011). False-positive psychology, Psychol Sci 22(11) | MEM | Researcher degrees of freedom: our criterion drift |
| [2412.03491](https://arxiv.org/html/2412.03491v1) | Beyond algorithm hyperparameters: preprocessing hyperparameters and pitfalls | ABS | Search result only |
| [doi:10.1038/526187a](https://doi.org/10.1038/526187a) | MacCoun & Perlmutter (2015). Blind analysis: Hide results to seek the truth, Nature 526:187–189 | ABS | Blind the confirmatory run (label scramble or hidden AUC offset); unblind once. Paywalled, not read in full |
| [doi:10.1146/annurev.nucl.55.090704.151521](https://doi.org/10.1146/annurev.nucl.55.090704.151521) | Klein & Roodman (2005). Blind analysis in nuclear and particle physics, Annu Rev Nucl Part Sci 55 | MEM | Source for the specific blinding techniques |
| [doi:10.1007/s11229-019-02456-7](https://doi.org/10.1007/s11229-019-02456-7) | Dutilh, Sarafoglou, Wagenmakers (2019). Flexible yet fair: blinding analyses, Synthese | ABS | Metadata only |
| [Semantic Scholar](https://www.semanticscholar.org/paper/Negative-Controls:-A-Tool-for-Detecting-Confounding-Lipsitch-Tchetgen/730b1c3d7476589e4028f8d6ef367dc858a57264) | Lipsitch, Tchetgen Tchetgen, Cohen (2010). Negative controls, Epidemiology 21(3):383–388 (erratum exists) | ABS | NC/PC rows in every headline table; PC2 + NC1 would have caught the summed-vs-single bug |
| [2311.18807](https://arxiv.org/abs/2311.18807) | Hofman, Chatzimparmpas, Sharma, Watts, Hullman (2023). Pre-registration for Predictive Modeling | ABS | Template built for predictive modelling. Abstract level |
| [PMLR v148](https://proceedings.mlr.press/v148/) | NeurIPS 2020 and 2021 pre-registration workshops (proceedings PMLR v148, [v181](https://proceedings.mlr.press/v181/)) | MEM | ML precedent for protocol-first review. Statistics pass saw the pages via search; rigor pass from memory; volume numbers to check |
| [2407.12220](https://arxiv.org/html/2407.12220v1) | Questionable practices in machine learning (2024) | ABS | Search result only |
| [doi:10.1038/s41562-022-01497-2](https://doi.org/10.1038/s41562-022-01497-2) | Reducing bias, increasing transparency and calibrating confidence with preregistration, Nat Hum Behav | ABS | Metadata only |
| — | Willroth & Atherton (2024). Best laid plans: reporting preregistration deviations, AMPPS 7(1); Lakens (2024). When and how to deviate from a preregistration, Collabra 10(1) | MEM | Identifier not recorded. Deviation table: planned, actual, date, reason, data-dependent?, effect |
| [doi:10.1016/j.cortex.2012.12.016](https://doi.org/10.1016/j.cortex.2012.12.016) | Chambers (2013). Registered Reports at Cortex, Cortex 49(3):609–610 | MEM | Registered-report model |
| [doi:10.1073/pnas.1708274114](https://doi.org/10.1073/pnas.1708274114) | Nosek et al. (2018). The preregistration revolution, PNAS 115:2600–2606 | MEM | General rationale |
| [2405.02200](https://arxiv.org/pdf/2405.02200) | Herrmann et al. (2024). Position: rethinking empirical ML research (exploratory vs confirmatory), ICML | ABS | Authorship from memory |
| [1904.10922](https://arxiv.org/pdf/1904.10922) | Scientific-method framing for ML | ABS | Search result only |
| [2003.12206](https://arxiv.org/abs/2003.12206) | Pineau et al. (2021). Improving Reproducibility in ML Research, JMLR 22(164) ([JMLR](https://www.jmlr.org/papers/v22/20-303.html)) | MEM | ML Reproducibility Checklist. Rigor pass checked authors and volume (ABS); statistics pass from memory |
| [NeurIPS](https://neurips.cc/public/guides/PaperChecklist) | NeurIPS Paper Checklist | MEM | Error-bar item forces saying what variability CIs capture. Page not fetched |
| [ACM](https://www.acm.org/publications/policies/artifact-review-and-badging-current) | ACM Artifact Review and Badging | MEM | Available + Functional badges as a self-declared target. Page not fetched |
| [1803.09010](https://arxiv.org/abs/1803.09010) | Gebru et al. (2021). Datasheets for Datasets, CACM 64(12):86–92 | MEM | Datasheet for the bench, including non-uses |
| [doi:10.1145/3287560.3287596](https://doi.org/10.1145/3287560.3287596) | Mitchell et al. (2019). Model Cards, FAT* 2019 | MEM | Same family |
| [2601.07189](https://arxiv.org/html/2601.07189v1) | Standardization of Post-Publication Code Verification by Journals (2026) | ABS | Search result only |
| [1909.03004](https://arxiv.org/abs/1909.03004) | Dodge et al. (2019). Show Your Work | MEM | Expected validation performance reporting |
| [2204.07610](https://arxiv.org/abs/2204.07610) | Gundersen et al. (2022). Sources of Irreproducibility in ML: A Review | MEM | Review |

### 18e. Numerical reproducibility and nondeterminism (extends §6)

| id | title | level | why it matters |
|---|---|---|---|
| [blog](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) | He / Thinking Machines Lab (10 Sep 2025). Defeating Nondeterminism in LLM Inference; repo `thinking-machines-lab/batch_invariant_ops` | ABS | Batch-size variance, not float atomics, is the main cause; batch-invariant RMSNorm/matmul/attention gave 1000/1000 identical completions. Basis for a training-side invariance test. Statistics pass read the blog; concurrent-work pass only secondary summaries |
| [vLLM docs](https://docs.vllm.ai/en/latest/features/batch_invariance/) | vLLM batch invariance mode | ABS | `VLLM_BATCH_INVARIANT=1`; beta, compute capability ≥ 8.0 |
| [Spheron blog](https://www.spheron.network/blog/vllm-batch-invariance-why-temperature-0-isnt-deterministic/) | SGLang deterministic mode (third-party summary) | ABS | ~34% throughput cost; not checked against SGLang's own post |
| [2506.09501](https://arxiv.org/abs/2506.09501) | Yuan et al. (2025). Give Me FP32 or Give Me Death? Reproducible Reasoning | ABS | bf16 accuracy swings up to 9% with GPU count, type and batch size; narrower in fp32. Case for fp64 fits |
| [2601.17768](https://arxiv.org/html/2601.17768v2) | LLM-42 (2026), deterministic inference | ABS | Search result only |
| [2609.38981](https://arxiv.org/abs/2609.38981) | Vosti (2026), deterministic inference | ABS | Search result only |
| [2606.21023](https://arxiv.org/html/2606.21023) | HEAL: Demystifying Numerical Instability in LLM Inference (2026) | ABS | Title only |
| [PyTorch docs](https://docs.pytorch.org/docs/stable/notes/randomness.html) | PyTorch reproducibility notes | MEM | No cross-device or cross-release guarantee; `use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Page would not load; check wording and defaults |
| [tutorial](https://docs.pytorch.org/tutorials/intermediate/ensembling.html) | PyTorch model ensembling tutorial (`torch.func`, `vmap`) | ABS | Stacked-parameter pattern behind `fit_batch` |

### 18f. Transportability and decision value

| id | title | level | why it matters |
|---|---|---|---|
| [doi:10.1214/14-STS486](https://doi.org/10.1214/14-STS486) | Pearl & Bareinboim (2014). External Validity: From Do-Calculus to Transportability, Stat Sci 29(4):579–595; [1503.01603](https://arxiv.org/pdf/1503.01603) | MEM | Argue transfer to SAE latents by transportability, not assertion. Validity pass cited the Project Euclid page; rigor pass from memory |
| [Biometrics](https://academic.oup.com/biometrics/article/79/3/2382/7513886) | Li, Gatsonis, Dahabreh, Steingrimsson (2023). Target-population AUC estimators, Biometrics 79(3):2382 | ABS | Inverse-odds-weighted target AUC: predict SAE-latent AUC from MLP neurons, then check |
| [AJE](https://academic.oup.com/aje/article/192/2/296/6648776) | Steingrimsson et al. (2023), AJE 192(2):296; [2101.11182](https://arxiv.org/pdf/2101.11182) | ABS | Companion |
| [2304.03779](https://arxiv.org/pdf/2304.03779) | A roadmap to prediction model validation and transportability in healthcare | ABS | Search result only |
| [doi:10.1177/0272989X06295361](https://doi.org/10.1177/0272989X06295361) | Vickers & Elkin (2006). Decision curve analysis, Med Decis Making 26(6):565–574 | MEM | Threshold as an explicit harm ratio: answers "0.95 is arbitrary". Validity pass cited it via the guide below; rigor pass from memory |
| [PMC6777022](https://pmc.ncbi.nlm.nih.gov/articles/PMC6777022/) | Vickers et al. Step-by-step guide to decision curve analysis | ABS | "The ratio of harms is the odds at the probability threshold" |
| [BMC](https://link.springer.com/article/10.1186/1472-6947-8-53) | Extensions to decision curve analysis, BMC Med Inform Decis Mak 8:53 (2008) | ABS | Same |

## 19. Ground-truth substrates, models and SAE resources

Exact readouts beyond GELU neurons: SwiGLU gate and up rows (K = 2), SAE or transcoder encoder rows (decoder row as distractor), unembedding logit differences, attention W_Q/W_V row spaces. Derivations are ours; resources below.

| id | title | level | why it matters |
|---|---|---|---|
| [config.json](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/raw/main/config.json) | HuggingFaceTB/SmolLM2-135M config | FULL | SwiGLU (silu), d 576, 1536 units, 30 layers, RMSNorm, tied embeddings. First SwiGLU target; no exactly null direction (RMSNorm does not centre). Note `_mlp_in()` returns `up_proj` only, wrong for SwiGLU |
| [config.json](https://huggingface.co/Qwen/Qwen2.5-0.5B/raw/main/config.json) | Qwen/Qwen2.5-0.5B config | FULL | SwiGLU, d 896, 4864 units, 24 layers. Scale step on a T4 |
| [config.json](https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json) | Qwen/Qwen3-0.6B config | FULL | SwiGLU, d 1024, 3072 units, 28 layers |
| [HF docs](https://huggingface.co/docs/transformers/model_doc/gemma) | Gemma family (GeGLU, RMSNorm, RoPE) | ABS | GeGLU substrate |
| [HF card](https://huggingface.co/google/gemma-3-270m) | google/gemma-3-270m | ABS | GeGLU; gated licence must be accepted by hand; dims not on the card |
| [2503.09543](https://arxiv.org/abs/2503.09543) | PolyPythias (ICLR 2025) | ABS | 10 seeds × 5 sizes (14M–410M), ~7,000 checkpoints; replicate calibration curves with error bars. Neurons do not correspond across seeds |
| [2304.01373](https://arxiv.org/abs/2304.01373) | Biderman et al. (2023). Pythia | MEM | 154 checkpoints per model (from memory): training-time axis, separates the under-fit confound from size |
| [2106.16163](https://arxiv.org/abs/2106.16163) | Sellam et al. (2021). MultiBERTs | MEM | 25 BERT-base seeds; encoder-only GELU option |
| [2509.26643](https://arxiv.org/pdf/2509.26643) | Convergence and Divergence of Language Models under Different Random Seeds | ABS | Not read in detail |
| [2406.11944](https://arxiv.org/abs/2406.11944) | Dunefsky, Chlenski, Nanda (2024). Transcoders Find Interpretable LLM Feature Circuits; weights HF `pchlenski/gpt2-transcoders` | MEM | Cleanest transfer test: same model, layer and probably the same ln_2 stimulus, ReLU link. Substrate pass recalled the ID; validity pass linked it, but the input hook is unverified |
| [2501.18823](https://arxiv.org/pdf/2501.18823) | Paulo, Shabalin, Belrose (2025). Transcoders beat SAEs; skip-transcoders | ABS | Transcoder substrate |
| [2409.14507](https://arxiv.org/html/2409.14507) | Chanin et al. A is for Absorption | ABS | Encoder/decoder asymmetry under absorption: decoder row is a documented near-miss distractor |
| [LessWrong](https://www.lesswrong.com/posts/kcg58WhRxFA9hv9vN/toy-models-of-feature-absorption-in-saes) | Toy models of feature absorption in SAEs | ABS | Tied weights remove absorption; asymmetry may detect it |
| [2503.17547](https://arxiv.org/pdf/2503.17547) | Matryoshka SAEs | ABS | Recover hierarchy where vanilla SAEs absorb |
| [2505.11756](https://arxiv.org/html/2505.11756) | Feature hedging: correlated features break narrow SAEs | ABS | Failure mode for SAE substrates |
| [2408.05147](https://arxiv.org/abs/2408.05147) | Gemma Scope | MEM | Pretrained SAEs for Gemma-2-2B; not fetched; T4-fitting widths unknown |
| [SAELens table](https://decoderesearch.github.io/SAELens/sae_table/) | SAELens pretrained SAEs: `gpt2-small-res-jb` (resid_pre, d_sae 24576, ReLU), `pythia-70m-deduped-res-sm` / `-mlp-sm` / `-att-sm` | ABS | ReLU encoder rows are exact truth for what a latent reads. **Conflict:** a second extraction gave `jbloom/GPT2-Small-SAEs`, `blocks.12.hook_resid_post`, d_sae 16384; check `pretrained_saes.yaml` |
| [HF card](https://huggingface.co/jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs) | OpenAI v5 GPT-2 SAEs (Gao et al. 2024), 32k and [128k](https://huggingface.co/jbloom/GPT2-Small-OAI-v5-128k-resid-post-SAEs) | ABS | TopK: firing depends on other latents, so use ReLU(pre-activation) as a planted link. k and input normalisation unverified |
| [HF tree](https://huggingface.co/EleutherAI/sae-pythia-160m-32k/tree/cab7c56a1c4eed5381f449face106b617cbf2a37/layers.9.mlp) | EleutherAI/sae-pythia-160m-32k; [sparsify](https://github.com/EleutherAI/sparsify) | ABS | TopK; same caveat. Whether b_dec is subtracted unconfirmed |
| — | Elhage et al. (2022). Toy Models of Superposition (transformer-circuits) | MEM | Identifier not recorded. Exact features in tiny models; unit-test substrate |
| [2202.05262](https://arxiv.org/abs/2202.05262) | Meng et al. (2022). ROME | MEM | Rank-one edit plants a known key direction in a real model |
| — | Anthropic crosscoders (transformer-circuits 2024) | MEM | Identifier not recorded. Exact encoder directions across layers |
| [2507.21509](https://arxiv.org/abs/2507.21509) | Chen et al. (2025). Persona vectors | ABS | ID taken from a citing paper ([2510.10157](https://arxiv.org/html/2510.10157v1)); primary not fetched. 6 Oct: primary [HTML v3](https://arxiv.org/html/2507.21509v3) fetched. Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct; traits evil, sycophancy, hallucination (more in the appendix). Vector = difference in mean response-token residual activations between judge-filtered trait and non-trait responses (> 50 / < 50); layer chosen by steering efficacy, so steering both selects and validates. Projection vs later trait expression r 0.75–0.83; fine-tune shift vs trait r 0.76–0.97, cross-trait baselines 0.34–0.86 (weak specificity). No split-half, bootstrap or CIs reported. Same estimator family as our L2 re-extraction. Upgrade candidate once each cited claim is checked in the text |
| [2601.10387](https://arxiv.org/html/2601.10387v1) | The Assistant Axis (2026) | ABS | Persona-direction context |
| [2608.24335](https://arxiv.org/pdf/2608.24335) | SteerCheck: attribution specificity and alignment leakage in activation-steering audits (2026) | ABS | Not read. 6 Oct: Luo, Liang, Xuan (25 Aug 2026); abstract checked. 960 interventions on Qwen3-14B; sign-randomised control directions often keep substantial target alignment; effect tracks cosine to target (ρ = .94); 25.3% of draws exceed cosine .5 ("alignment leakage"). The only paper found that reports bank-aware controls; supports our same-bank null |
| [2603.24543](https://arxiv.org/html/2603.24543v1) | Analysing the Safety Pitfalls of Steering Vectors (2026) | ABS | Not read |
| [2502.02716](https://arxiv.org/html/2502.02716v2) | A Unified Understanding and Evaluation of Steering Methods (2025) | ABS | Diff-of-means analysis; search only. 6 Oct: Im & Li, v2 Jan 2026; abstract checked. Unified framework plus evaluation on multiple-choice and open-ended generation "demonstrating the superiority of certain methods"; the abstract does not name them, so "DiffMean wins" is unverified |
| [2604.08524](https://arxiv.org/html/2604.08524v1) | What Drives Representation Steering? A refusal case study (2026) | ABS | Not read |

## 20. Mentor-suggested works (Nagaraja et al.)

All four exist and Shishir Nagaraja is an author on each; confirm with him that "Shishir" means Nagaraja. Link strength: item 1 strong, item 4 moderate, item 2 weak to moderate, item 3 weak. Four off-field citations from one author read as padding unless each sentence makes a checkable point; if he co-authors, self-citations go in the third person.

| id | title | level | why it matters |
|---|---|---|---|
| [doi:10.1007/978-3-030-65610-2_1](https://doi.org/10.1007/978-3-030-65610-2_1) | Shah & Nagaraja (2020). A Unified Access Control Model for Calibration Traceability in Safety-Critical IoT, ICISS 2020, LNCS pp. 3–22 (item 1, primary) | ABS | **Strong.** The peer-reviewed calibration-traceability paper; full text unread. Credit with operationalising and securing traceability, not formalising it; cite beside the VIM |
| [1908.00740](https://arxiv.org/abs/1908.00740) | Shah, McIntee, Nagaraja, Bhandary, Arote, Kuri (2019). Secure Calibration in High-Assurance IoT: Traceability for Safety Resilience (item 1) | FULL | Preprint only. Traceability against a reference sensor; Ethereum contracts verify status back to a national measurement institute; takes the unbroken-chain definition from metrology. No PUFs or acoustics anywhere in the text |
| [doi:10.1145/3488306](https://doi.org/10.1145/3488306) | Vaidya, Prabhakar, Gnani, Shah, Nagaraja (2022/23). Sensor Identification via Acoustic Physically Unclonable Function, ACM DTRAP 4(2) (item 1) | ABS | Device identity for supply chains (99% over thousands of devices), not calibration. Cite only for a device-integrity example. **No paper combines acoustic PUFs with calibration; never write "acoustic-PUF-based calibration integrity verification"**; ask the mentor if an in-press paper exists |
| [doi:10.1145/3003816](https://doi.org/10.1145/3003816) | Gardiner & Nagaraja (2016). On the Security of Machine Learning in Malware C&C Detection: A Survey, ACM CSUR 49(3), Art. 59 ([author PDF](https://strathprints.strath.ac.uk/66222/1/Gardiner_Nagaraja_ACM_CS_2016_On_the_security_of_machine_learning_in_malware.pdf)) (item 4) | FULL | **Moderate.** §7.1: no clear metric for evasion resilience; §7.4: testing on known malware inflates detection. Convenient metric ≠ property. Pair with Arp et al. **Title in the mentor email is wrong:** not "...Command-and-Control Infrastructure Detection" |
| [doi:10.1007/978-3-032-18070-4_6](https://doi.org/10.1007/978-3-032-18070-4_6) | Dholakia, Wani, Ellison, Hodak, Dutta, Nagaraja, Ranjan. Benchmarking Considerations for Agentic AI Systems, TPCTC 2025, LNCS pp. 89–98 (online Apr 2026) (item 2) | ABS | **Weak to moderate.** Panel-derived systems-benchmarking chapter; abstract withheld; "necessary but not sufficient" seen in search snippets only. Do not attribute "misses failure modes" until read; let Zhu et al. 2507.02825 (§14) carry that claim. Year 2025 or 2026 per bib style |
| [doi:10.1145/3317549.3323407](https://doi.org/10.1145/3317549.3323407) | Nagaraja & Shah (2019). Clicktok: Click Fraud Detection using Traffic Analysis, WiSec 2019, pp. 105–116; [1903.00733](https://arxiv.org/abs/1903.00733) (item 3) | ABS | **Weak; discussion only, or drop.** Adversarial fraud, not "unreliable system behaviour"; never calls labels sparse; evaluation fully labelled (~81% mimicry, 95% bait-click). Passes disagree: one reads bait clicks as planted ground truth, the mentor pass rates the link weak; we keep the cautious reading (bait clicks are a detection mechanism). Mentor pass read the PDF, agreement pass the abstract |

## 21. Level 2: trait and persona directions

Added 6 Oct from the persona-level pass. Where two passes gave a work different levels, the lower level is used and the row says so. Already elsewhere, with 6 Oct notes where the pass went deeper: AxBench (§1); InterpBench and Tracr (§2); Gerasimov 2606.12138 (§3); Venkatesh & Kurapath 2602.06801, Tan et al. 2407.12404, Braun et al. 2505.22637 (§4); Korznikov 2602.14111, SynthSAEBench 2602.14687, Chanin 2605.18229, Braun 2602.17881 (§11a); Hiramatsu 2609.07037 (§12); Park, Choe, Veitch 2311.03658 (§17e; the causal inner product is the whitening metric for every L2 cosine); ROME 2202.05262, persona vectors 2507.21509, SteerCheck 2608.24335, Im & Li 2502.02716 (§19). Injection papers the pass cites are in §8 and §22.

Gap the pass found: no paper plants a known direction in a real LM and scores the field's ground-truth-free checks (SNR, pair-difference consistency, prompt-set cosine, split-half, method disagreement) against recovery. Absence after about 25 and 15 searches in the two passes, not a systematic review; run a targeted check before claiming it.

| id | title | level | why it matters to CALIPER |
|---|---|---|---|
| [2609.14151](https://arxiv.org/html/2609.14151) | Bhattarai & Alhanai (12 Sep 2026); title not recorded | ABS | **Closest L2 prior art.** Formalises SNR = ‖μ‖² / (Tr Σ₊ + Tr Σ₋), mean alignment of the vector with pair differences, manifold capacity and Two-NN intrinsic dimension. Over 6 models (Qwen3 4B/14B, Gemma3 4B/12B, Llama 3.1 8B, 3.2 3B) and 38 datasets (26 persona), SNR vs steerability Pearson 0.77–0.82; intrinsic dimension weakest (0.16–0.39). Its only ground truth is a synthetic superposition toy (recovery = cos(μ, v₀)): SNR ρ 0.91, alignment 0.90. Stated limits: CAA only, correlational, MC-logit steerability only. L2 lifts this into real activations. HTML fetched; not read end to end |
| [2312.06681](https://arxiv.org/abs/2312.06681) | Rimsky et al. (2024). Contrastive Activation Addition (CAA), ACL 2024; full title not recorded | MEM | DiffMean at the answer-letter token over A/B pairs, added at every post-prompt position. The estimator family L2 calibrates. Caution: in A/B formats the final-layer readout is the same A−B direction for every trait, so use trait-specific answer tokens. Two passes cite from memory |
| [2310.01405](https://arxiv.org/abs/2310.01405) | Zou et al. (2023). Representation Engineering (RepE / LAT); full title not recorded | MEM | PCA on stimulus-pair differences, validated by reading accuracy and control. A method-disagreement comparator |
| [2306.03341](https://arxiv.org/abs/2306.03341) | Li et al. (2023). Inference-Time Intervention (ITI), NeurIPS 2023; full title not recorded | MEM | Heads chosen by probe accuracy, then shifted along the mass-mean direction. Probe accuracy as a selection criterion is one of the checks to calibrate |
| [2310.06824](https://arxiv.org/abs/2310.06824) | Marks & Tegmark. The Geometry of Truth ([OpenReview PDF](https://openreview.net/pdf/0ae055dff16c072ac024f0560d5484ddc2cb4aa8.pdf)) | MEM | Mass-mean probing "is as accurate for classification as logistic regression" but more causally implicated, ahead in 7 of 8 intervention conditions; the iid variant applies a covariance correction. High probe accuracy does not certify the direction. Persona-landscape pass checked the abstract and OpenReview page (ABS); ground-truth pass from memory; lower level kept |
| [2306.03819](https://arxiv.org/abs/2306.03819) | Belrose et al. (2023). LEACE, NeurIPS 2023; full title not recorded | MEM | Erased subspace spanned by whitened class-mean differences: the theory that DiffMean holds all linearly available binary-concept information. Basis for the LDA (Σ⁻¹ DoM) comparator |
| [2506.19823](https://arxiv.org/abs/2506.19823) | Wang et al. (OpenAI). SAE model-diffing of emergent misalignment; title not recorded; [OpenAI post (2025)](https://openai.com/index/emergent-misalignment/) | MEM | "Toxic persona" latent most strongly controls emergent misalignment and predicts whether a model shows it. Behavioural (H3-style) validation, no known direction. Persona-landscape pass checked the abstract (ABS); ground-truth pass from memory; lower level kept |
| [2607.21356](https://arxiv.org/abs/2607.21356) | Nadaf (23 Jul 2026). Persona subspaces by contrastive teacher forcing on Qwen2.5-14B-Instruct; title not recorded | ABS | Shared low-rank core at 657× a random-subspace null; projecting it out during fine-tuning cuts misalignment 27.7% → 0.0%; injecting it induces misalignment dose-proportionally. Causal necessity and sufficiency, but the "truth" is itself estimated. Single author, unreplicated |
| [2505.20063](https://arxiv.org/pdf/2505.20063) | SAEs Are Good for Steering -- If You Select the Right Features | ABS | AxBench rebuttal. Search listing only; authors and year not recorded |
| [2605.31183](https://arxiv.org/pdf/2605.31183) | Steering LLMs? Actually, Sparse Autoencoders can outperform simple baselines | ABS | AxBench rebuttal. Search listing only; authors and year not recorded |
| [2502.17420](https://arxiv.org/abs/2502.17420) | Wollschläger et al. (2025), ICML 2025 ([PMLR](https://proceedings.mlr.press/v267/wollschlager25a.html)); title not recorded | MEM | Refusal is mediated by multiple independent directions and multi-dimensional concept cones; "representational independence" is stricter than orthogonality. Truth for a trait may be a cone, not a line. Untagged in the note; provenance not stated |
| [2504.04635](https://arxiv.org/abs/2504.04635) | Da Silva et al. (2025). Steering off Course, ACL 2025 ([Anthology](https://aclanthology.org/2025.acl-long.974/)) | ABS | 36 models, 14 families (1.5B–70B): function vectors recover 5-shot performance in only 52% of model-task combinations even with hyperparameter search, task vectors 35%. Model and layer dependence at scale |
| [2508.16560](https://arxiv.org/pdf/2508.16560) | Sparse but Wrong | ABS | Synthetic-feature SAE test. Search listing only |
| [2512.15712](https://arxiv.org/pdf/2512.15712) | Predictive Concept Decoders | ABS | Injection-adjacent decoding. Search listing only |
| [2503.10965](https://arxiv.org/pdf/2503.10965) | Marks et al. Auditing language models for hidden objectives | MEM | Red team trains a known hidden RM-sycophancy objective; 3 of 4 blind teams find it (SAEs, behavioural attacks, training-data analysis). Truth is an objective, not a direction. Untagged in the note; provenance not stated |
| [2602.22755](https://arxiv.org/pdf/2602.22755) | AuditBench | ABS | Implanted-behaviour audit benchmark. Search listing only |
| [2509.06608](https://arxiv.org/html/2509.06608) | Small Vectors, Big Effects | ABS | RL-induced reasoning studied through trained steering vectors: a model whose whole change is one learned additive vector is a realistic object (L2 design #2). Search listing only |
| [2303.08112](https://arxiv.org/abs/2303.08112) | Belrose et al. (2023). Tuned lens | ABS | Affine per-block translator A_L; A_Lᵀw approximates the total-effect reader at earlier layers. The logit lens is "often brittle". Fidelity decay with depth in small models has to be measured |
| [2406.01506](https://arxiv.org/abs/2406.01506) | Park et al. (2024; ICLR 2025). Categorical and hierarchical concepts in LLM representations; full title not recorded; [code](https://github.com/KihoPark/LLM_Categorical_Hierarchical_Representations) | ABS | Categorical concepts are simplices, hierarchical ones orthogonal under the causal inner product (957 WordNet concepts, Gemma). A ready bank of lexical traits with a second exact structural prediction (child − parent ⟂ parent) |
| [OpenReview PDF](https://openreview.net/pdf/1168e73c1bc3eca4d4e890f40240a9124c64f382.pdf) | Cywiński et al. (2025). Taboo secret-word models; title and arXiv ID not recorded | ABS | Gemma-2-9B fine-tuned on 20 single-token secret words. The secret token's unembedding row is an exact read direction, linking model organisms back to the lexical-readout design |
| [2506.11618](https://arxiv.org/abs/2506.11618) | Soligo et al. (2025) | ABS | 9 rank-1 LoRA adapters misalign Qwen2.5-14B-Instruct; scalar states interpretable; one fine-tune's misalignment direction ablates misalignment in others. Sufficiency, not uniqueness: certify necessity per trait |
| [2510.13900](https://arxiv.org/abs/2510.13900) | Minder et al. (ICLR 2026). Activation Difference Lens | ABS | Base vs narrow-fine-tune activation differences recover the fine-tuning data (Gemma, LLaMA, Qwen, 1B–32B). Warns narrow fine-tunes are "very salient": implanted traits must include harder variants or every check looks good |
| [2404.03592](https://arxiv.org/abs/2404.03592) | Wu et al. (2024). LoReFT; `pyreft` | ABS | Low-rank subspace intervention on a frozen model, built on DAS; 15–65× fewer parameters than LoRA. Plant optimiser for the repaired P1 protocol (design #6) |
| [2502.18862](https://arxiv.org/abs/2502.18862) | Dunefsky & Cohan (2025). One-shot optimised steering vectors; title not recorded; [code](https://github.com/jacobdunefsky/one-shot-steering-repro), [llm-steering-opt](https://github.com/jacobdunefsky/llm-steering-opt) | ABS | Single-example optimised vectors mediate safety behaviour and transfer (96.9% HarmBench ASR, refusal suppression). Whether they align with DoM for the same behaviour is untested |
| [2404.14461](https://arxiv.org/abs/2404.14461) | SaTML 2024 RLHF trojan competition report; [repo](https://github.com/ethz-spylab/rlhf_trojan_competition), [model](https://huggingface.co/ethz-spylab/poisoned_generation_trojan4), [dataset](https://huggingface.co/datasets/ethz-spylab/rlhf_trojan_dataset) | ABS | 5 Llama-2-7B models RLHF-poisoned with known 5–15-token suffix triggers. Exact trait-state labels, not directions: external validity only |
| [blog](https://www.anthropic.com/research/probes-catch-sleeper-agents) | Anthropic (Apr 2024). Defection probes on backdoored "sleeper" models; title not recorded | ABS | Generic contrast-pair probes reach AUROC > 99% across triggers. Defection sits at the easy end: confirms checks pass, rarely produces failures |
| [2401.05566](https://arxiv.org/abs/2401.05566) | Sleeper Agents | MEM | Model organism with known triggers. ID from memory |
| — | TDC 2023, NeurIPS Trojan Detection Challenge, known-trigger LLM track | MEM | Identifier not recorded. Not verified |
| [2310.18168](https://arxiv.org/abs/2310.18168) | Joshi et al. (EMNLP 2024). Truthful-persona inference in a synthetic arithmetic world; title not recorded | ABS | Generated speakers hold true or false operator semantics; models infer a "truthful persona"; data structure decides whether it is inferred. Template for a CPU toy with correlated persona traits and an exact Bayes-posterior label |
| [2608.13329](https://arxiv.org/abs/2608.13329) | Noël (2026); title not recorded | ABS | A single-prompt design produces study-to-study contradictions: probe scores follow the prompt, not the model. Basis for the prompt-template stability check |

## 22. Level 3: self-reports and introspection

Added 6 Oct from the self-level pass. Lower level used where passes differ. Already elsewhere: the injection line (Lindsey, Macar, Lederman & Mahowald, Singh et al., Pearson-Vogel et al., Godet, Hahami 2512.12411 and 2607.14111) in §8, with 6 Oct notes; Ferrara 2608.20569, Zou 2609.35108 and Fonseca Rivera & Africa 2511.21399 in §11b; stability-as-reliability analogues 2607.05355, 2608.05670, 2608.13754 in §11a.

Cautions carried from the pass. Lindsey and Macar both read concept vectors at the chat-template tail, the position where our Gemma-3-27B vectors carried no content; Macar runs our exact operating point (L37) with no random control. Ferrara's impact-matched random covers 2 of 8 models; random above targeted in 7 of 8 is not claimed as significant. Zou owns the random-localisation numbers and shows random fails at localisation, so our claim stays at detection. Random ≈ real detection alone is no longer novel (Godet, Hahami, Lederman & Mahowald, Singh, Ferrara). No paper found runs a per-vector steering positive control or documents dead vectors.

| id | title | level | why it matters to CALIPER |
|---|---|---|---|
| [2410.13787](https://arxiv.org/abs/2410.13787) | Binder, Chua, Korbak, Sleight, Hughes, Long, Perez, Turpin, Evans (2024). Looking Inward: Language Models Can Learn About Themselves by Introspection; ICLR 2025 per the [proceedings listing](https://proceedings.iclr.cc/paper_files/paper/2025/hash/0a6059857ae5c82ea9726ee9282a7145-Abstract-Conference.html) | ABS | Cross-prediction control: M1 fine-tuned to predict itself beats M2 fine-tuned on M1's behaviour (Llama-70B 48.5% vs 31.8%; GPT-4o 49.4% vs 36.6%). Only after fine-tuning, simple tasks only, fails OOD. Numbers from search snippets; the abs page states neither the gap nor the venue |
| [2503.07513](https://arxiv.org/abs/2503.07513) | Song, Hu, Mahowald (2025). Language Models Fail to Introspect About Their Knowledge of Language, COLM 2025; [code](https://github.com/SiyuanSong2004/language-introspection) | ABS | 21 open models; metalinguistic reports vs own string probability, controlled by a near-twin model: no privileged self-access. Ready L3 task with exact truth and public code; they did not score trust signals |
| [2508.14802](https://arxiv.org/abs/2508.14802) | Song, Lederman, Hu, Mahowald (20 Aug 2025). Privileged Self-Access Matters for Introspection in AI | MEM | Definition: introspection beats any third-party process of equal or lower cost. Temperature self-reports follow the prompt framing ("crazy" → HIGH) whatever T is; self accuracy equals across-model prediction. The L3 negative control. Ground-truth pass checked the HTML (ABS); landscape pass had the title only; lower level kept |
| [2506.05068](https://arxiv.org/pdf/2506.05068) | Comsa & Shanahan. Does It Make Sense to Speak of Introspection in LLMs? | ABS | Temperature self-report as "lightweight" introspection mediated by the model's own outputs. Search snippet only |
| [2509.13316](https://arxiv.org/abs/2509.13316) | Li, Ceballos Arroyo, Rogers, Saphra, Wallace. Do Activation Verbalization Methods Convey Privileged Information?, ICML 2026 | ABS | Verbaliser benchmarks solvable without target internals: zero-shot prompting ~0.64 matches LIT/Patchscopes; text inverted from activations gives the same 79%; fabricated-fact PersonaQA-Fantasy 0%. A benchmark-validity failure of our type |
| [2505.13763](https://arxiv.org/abs/2505.13763) | Ji-An, Xiong, Wilson, Mattar, Benna. Language Models Are Capable of Metacognitive Monitoring and Control of Their Internal Activations | ABS | Neurofeedback: report a median-binarised LR-axis or PC projection after N in-context examples; LR axes beat PCs, early PCs beat late. Exact truth. Caution: the HTML lists Qwen2.5 "1B/3B/7B", but Qwen2.5 ships 0.5B/1.5B; check sizes |
| [2609.00904](https://arxiv.org/abs/2609.00904) | Aoki, Takatsuki, Minegishi, Haruki, Kawahara (1 Sep 2026). In-Context Neurofeedback: Can LLMs Control Their Internal Representations through Privileged Access? | ABS | Holds output text fixed so the target is not inferable from the prompt (a gap they name in Ji-An et al.). Significant in 45/120 settings, all d < 0.5, inconsistent across models |
| [2606.23671](https://arxiv.org/html/2606.23671v2) | Nguyen, Ahmed, Kim (29 Jun 2026). Can LLMs Reliably Self-Report Adversarial Prefills, and How? | MEM | 10 models incl. Gemma-3 4B/12B/27B; claim rate 27.3% on prefilled responses (ideal ~0), 49.3% on controls (ideal ~100%). Refusal-direction ablation closes the gap to ~3%, but random-direction ablation also closes most of it on small models. Ground-truth pass checked the HTML (ABS); landscape pass title only; lower level kept |
| [2605.26045](https://arxiv.org/html/2605.26045) | Torrielli, Schneider-Kamp, Galke Poech (v2 3 Aug 2026). Confidence and Calibration of Activation Oracles; full title not recorded | ABS | **The one scoring of confidence signals against exact truth** (fine-tuned secret word, 6,000 samples per method-oracle pair). Spoken numeric confidence AUROC 0.40–0.53; bootstrap mode frequency the only label-free calibrated method (ECE 0.04–0.12); forced choice AUROC 0.92–0.96. Sets the L3 bar. Oracle-based, not self-report |
| [2512.15674](https://arxiv.org/pdf/2512.15674) | Karvonen et al. Activation Oracles; [MATS page](https://www.matsprogram.org/research/activation-oracles-training-and-evaluating-llms-as-general-purpose-activation-explainers) | ABS | Trained on LatentQA plus classification and context prediction; poorly calibrated, answers at low confidence. Listing only |
| [2412.08686](https://arxiv.org/abs/2412.08686) | Pan, Chen, Steinhardt. LatentQA | ABS | Verbaliser lineage. IDs and authors from search listings |
| [2403.10949](https://arxiv.org/pdf/2403.10949) | Chen, Vondrick, Mao. SelfIE | ABS | Training-free verbaliser. Search listing |
| [2401.06102](https://arxiv.org/abs/2401.06102) | Ghandeharioun, Caciularu, Pearce, Dixon, Geva. Patchscopes | ABS | Training-free verbaliser. Search listing |
| [2511.08579](https://arxiv.org/abs/2511.08579) | Transluce / MIT. Training Language Models to Explain Their Own Computations; [project page](https://transluce.org/self-explanations) | ABS | Self-explainers beat other-model explainers "even if the explainer model is significantly more capable". Abs listing |
| [2605.25052](https://arxiv.org/abs/2605.25052) | Gur-Arieh, Marasović, Geva (24 May 2026). Faithfulness Metrics Don't Measure Faithfulness: A Meta-Evaluation with Ground Truth | ABS | BonaFide: 3,066 labelled CoTs, 13 tasks, 10 models. Most metrics near chance; best AUROC 0.70 (CoT) and 0.59 (step); biased, degrade on long chains, do not transfer. Closest methodological template for "calibrate the signal against planted truth" |
| [2608.30980](https://arxiv.org/abs/2608.30980) | Zeng, Assis, Wang (31 Aug 2026). Evaluating and Improving LLM Self-Modeling, EMNLP 2026 | ABS | Behavioural truth (would a prompt edit change the model's own answer). Self-modelling "non-trivial but limited"; RL gains "may not arise from privileged access" |
| [Anthropic](https://alignment.anthropic.com/2026/introspection-adapters/) | Shenoy, Yang, Sheshadri, Mindermann, Lindsey, Marks, Wang (28 Apr 2026). Introspection Adapters | ABS | One LoRA over 582 models with implanted behaviours: 89% raw verbalisation, 37.7% (0.6B) to 77.3% (14B). **High false-positive rate on unmodified models**; pre-existing behaviours verbalise at 8–26%. Implanted behaviour as ground truth |
| [SPAR](https://sparai.org/projects/f26/recNKpeygLfUGyGiz/) | SPAR project listing. Introspection Training for Verbalizing Activations | ABS | Proposes probe readouts, feature activations and patching effects as labels for verbal reports. Listing, no results |
| [2207.05221](https://arxiv.org/abs/2207.05221) | Kadavath et al. (2022). Language Models (Mostly) Know What They Know | MEM | P(True) and P(IK) reasonably calibrated in large models, worse OOD. Title and ID confirmed by search; content from memory in both passes |
| [2306.13063](https://arxiv.org/abs/2306.13063) | Xiong et al. (ICLR 2024). Can LLMs Express Their Uncertainty? | MEM | Verbalised confidence overconfident, clustered at multiples of 5. From memory |
| [2305.14975](https://arxiv.org/abs/2305.14975) | Tian et al. (2023). Just Ask for Calibration | MEM | Verbalised confidence can beat token probabilities in RLHF models. From memory |
| — | Unidentified 2026 survey on verbalised confidence | MEM | Identifier not recorded. Snippet: verbalised confidence misaligns with token probabilities; nominal 99% intervals cover ~65%. Do not cite until the source is found |
| [2605.27752](https://arxiv.org/html/2605.27752v1) | Asking Is Not Enough: Protocol Sensitivity in LLM Confidence Calibration | MEM | Title only |
| [2609.32470](https://arxiv.org/html/2609.32470) | On the Pitfalls of Verbalized Confidence Priors (full title truncated in the note) | MEM | Title only |
| [2406.06140](https://arxiv.org/pdf/2406.06140) | Can I understand what I create? Self-Knowledge Evaluation of LLMs | MEM | Only hit for own-tokenisation self-report; content unverified. Token-count self-report remains an untested proposal |
| [2505.17120](https://arxiv.org/abs/2505.17120) | Plunkett et al. (2025). Self-interpretability of fine-tuned attribute weights; title not recorded | MEM | Cited inside Lederman & Mahowald; ID from memory |
| [2501.11120](https://arxiv.org/abs/2501.11120) | Betley et al. (2025). "Tell me about yourself": behavioural self-awareness | MEM | Cited inside Lederman & Mahowald; ID from memory |
| [2609.33702](https://arxiv.org/html/2609.33702v1) | Understanding Confabulation and Rethinking Reconstruction in Activation Explanations | MEM | Verbalisation faithfulness. Title only |
| [2609.34033](https://arxiv.org/html/2609.34033) | Faithful Activation Verbalization | MEM | Title only |
| [2607.20379](https://arxiv.org/html/2607.20379v1) | Train the Model, Not the Reader: Decodability Supervision (full title truncated in the note) | MEM | Title only |
| [2603.18893](https://arxiv.org/pdf/2603.18893) | Quantitative Introspection in Language Models: Tracking Emotive States Across Conversation | MEM | Title only |
| [2607.04277](https://arxiv.org/pdf/2607.04277) | Self-Reference in LLMs: The Introspection Threshold (full title truncated in the note) | MEM | Title only |
| [2602.00333](https://arxiv.org/abs/2602.00333) | Efficient and accurate steering via attention-guided feature learning (title truncated in the note) | MEM | A snippet credits a 2026 steering paper with 49.3% of concepts steerable when extraction is fixed at the template token `end_header_id` vs 78.2% with dynamic selection; this abstract says only "nearly doubling". Numbers and attribution unverified. Would support the template-tail dead-vector reading |

---

## Maintenance rule

A paper enters this file **the session it is first mentioned**, at whatever level it was
actually read. When one is promoted ABS → FULL, update the level in the same session and
say what the full text changed. Formalise into `paper/references.bib` only at FULL.

**Never cite a specific claim from an ABS row.** That is the rule the August audit exists
to enforce.
