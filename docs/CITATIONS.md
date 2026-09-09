# Citation ledger

**Every paper this project has leaned on, with what it is for and how far it was verified.**
Started 9 September 2026. `paper/references.bib` holds 16 formal entries; this file is the
superset and the working record.

## Verification levels — this column is load-bearing

| level | meaning |
|---|---|
| **FULL** | main text read, claims checked against it |
| **ABS** | abstract or a search summary only — **do not cite a specific claim from these without reading first** |
| **BIB** | already formalised in `paper/references.bib` |

The project has been burned by abstract-level reading before: two novelty claims in the
Proposal III set were killed by an audit that found direct collisions the abstracts did not
reveal. **Anything marked ABS is a lead, not a citation.**

---

## 1. The direct competitors — what the bench is positioned against

| id | title | level | why it matters |
|---|---|---|---|
| [2609.03026](https://arxiv.org/abs/2609.03026) | ObserverBench | **FULL** | Scores "observers" by downstream action quality on held-out cases. **States outright it has no ground-truth-free failure detector.** The single most important positioning target — different question (actions, not directions), and the gap we fill is one they name |
| — | [MIB / Benchmarking Interpretability](https://benchmarking-interpretability.csail.mit.edu/) (MIT CSAIL) | ABS | Head-to-head method comparison. Cite as the crowded lane we are *not* entering |
| — | AxBench | ABS | Steering evaluation suite. Same role |
| [2608.19338](https://arxiv.org/html/2608.19338v1) | Mechanistic Tomography | ABS | Surfaced in the first harness scout; not yet read |
| [2309.03886](https://arxiv.org/html/2309.03886v3) | FIND: Function Description Benchmark | ABS | Synthetic-function ground truth. Contrast with weight-derived |

## 2. Ground truth for direction recovery — all planted or compiled, none free

| id | title | level | why it matters |
|---|---|---|---|
| [2410.08417](https://arxiv.org/pdf/2410.08417) | Bilinear MLPs enable weight-based interpretability | ABS | Weight space as an *analysis target*. **We use it as a scoring key** — that is the distinction to draw |
| [2604.06005](https://www.opentrain.ai/papers/disentangling-mlp-neuron-weights-in-vocabulary-space--arxiv-2604.06005/) | ROTATE: disentangling MLP neuron weights | ABS | Same neighbourhood, same distinction |
| [2407.13594](https://arxiv.org/abs/2407.13594) | Validating Mechanistic Interpretations (axiomatic) | ABS | Axiomatic validation, not empirical calibration |
| [2506.14002](https://arxiv.org/pdf/2506.14002) | Taming Polysemanticity: Provable Feature Recovery | ABS | Source of the **Feature Recovery Rate** metric; synthetic ground truth |
| [2512.05534](https://arxiv.org/pdf/2512.05534) | Unified Theory of Sparse Dictionary Learning | ABS | Spurious minima in dictionary learning — relevant to the basin argument |
| — | InterpBench, Tracr | ABS | Compiled ground truth. Named in Proposal III's own capability table as the ground-truth option; **the weight column is better because it is natural as well as exact** |
| — | CLEVR-XAI; A Benchmark for Interpretability Methods in DNNs (NeurIPS) | ABS | Older ground-truth-evaluation lineage |
| [2506.10920](https://arxiv.org/html/2506.10920) | Interpretable Features from Compositional Neuron Groups | ABS | Unit-level structure |
| [2605.29358](https://arxiv.org/pdf/2605.29358) | Scaling Monosemanticity | ABS | The SAE mainline |

## 3. Reliability signals — validated against behaviour or reconstruction, never a known-correct direction

**This table is the argument.** Every row proposes or studies a reliability signal, and not
one calibrates it against a known-correct answer.

| id | title | level | what it validates against |
|---|---|---|---|
| [2604.17698](https://arxiv.org/html/2604.17698v2) | The Geometric Canary: predicting steerability via representational stability | **FULL** | **Behavioural** — accuracy drop under steering, task degradation under drift. AUC 0.990 on LoRA drift, 69 embedding models. Closest competitor on "ground-truth-free signal", and it never scores against a true direction |
| [2606.12138](https://arxiv.org/abs/2606.12138) | Unstable Features, Reproducible Subspaces: seed dependence in SAEs | ABS | **Reconstruction and downstream prediction.** Proposes feature stability as a quality signal |
| [2607.19386](https://arxiv.org/html/2607.19386) | Building Fast, Evaluating Slow | ABS | Finds **methodological variance exceeds architectural variance** across autointerp metrics. Strong support for the variance-decomposition framing |
| [2505.10399](https://arxiv.org/html/2505.10399v1) | Evaluating Model Explanations without Ground Truth (FAccT 2025) | ABS | The ground-truth-free XAI evaluation line |
| [2602.13450](https://arxiv.org/abs/2602.13450) | Inference From Random Restarts | **FULL** | **Theoretical, econometric solver, not neural nets.** Says restart heuristics "lack a formal inferential foundation despite widespread use". Proves uniqueness concentrates *polynomially* while basin size concentrates *exponentially* — **a mechanism for stable-and-wrong: restart agreement measures basin size, not correctness** |
| [2501.17727](https://arxiv.org/html/2501.17727) | Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers | ABS | A metric that fails its own sanity check |
| [2603.04198](https://arxiv.org/html/2603.04198v2) | Stable and Steerable SAEs with Weight Regularization | ABS | Improving stability, treating it as desirable |
| [2605.18629](https://arxiv.org/pdf/2605.18629) | Aligned Training for SAE stability | ABS | Same |
| [2606.02061](https://arxiv.org/pdf/2606.02061) | Ablating Archetypes: stability as an artifact of initialization | ABS | **Counterweight** — stability can be an artifact of init, not a quality signal |
| [2603.20101](https://arxiv.org/pdf/2603.20101) | Pitfalls in Evaluating Interpretability Agents | ABS | **Read this next.** Closest neighbour on the thesis: evaluations overestimate interpretability reliability and obscure failure modes |

## 4. Steering-vector reliability and non-identifiability — Study 2's context

| id | title | level | why it matters |
|---|---|---|---|
| [2602.06801](https://arxiv.org/html/2602.06801v4) | On the Non-Identifiability of Steering Vectors | **FULL** | Reaches P1b's conclusion from **behavioural evidence only**, and states explicitly it does **not** plant known directions and attempt recovery. **P1b is the ground-truth complement to a published behavioural result** |
| [2407.12404](https://arxiv.org/html/2407.12404v1) | Analysing Generalisation and Reliability of Steering Vectors | ABS | Steerability varies wildly, sometimes negative; prior work reports no error bars |
| [2505.22637](https://arxiv.org/abs/2505.22637) | Understanding (Un)Reliability of Steering Vectors | ABS | Companion result |

## 5. Statistics and method — the rigour spine

| source | level | used for |
|---|---|---|
| DeLong, DeLong & Clarke-Pearson (1988) | ABS | **Primary endpoint test.** Comparing two correlated AUCs on the same units. Implemented in `b1_signal_calibration.py` |
| Obuchowski–McClish; Hanley–McNeil | ABS | Sample size for AUC differences. **Source of the 36–142 figure** that drove Addendum 1 |
| Benjamini–Hochberg | BIB-pending | Multiplicity across the signal set |
| Wilson (1927) | **BIB** | Interval on a proportion near 1.0 |
| Wilcoxon (1945) | **BIB** | Paired signed-rank, Study 3 |
| McNemar exact | ABS | Paired pass/fail comparison — the steps check |
| **NIST AI 800-3** (Feb 2026) | ABS | **Official standard for statistical rigour in AI benchmarks.** Benchmark accuracy vs generalized accuracy; intervals over point scores; GLMMs. Cheap to comply with, strong reviewer signal |
| [2604.11581](https://arxiv.org/abs/2604.11581) | ABS | **Hidden Measurement Error in LLM Pipelines.** Naive SEs are **40–60% too small** because they ignore design-choice variance. **The basis for the variance-decomposition reframe** |

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
| Sharpee, Rust & Bialek (2004), Maximally Informative Dimensions | ABS | **The crown jewel of Proposal III.** Only 5 arXiv papers use MID, all neuroscience, none on a modern deep network. **Read end to end before writing the method section** |
| Bussgang's theorem | ABS | **The mechanism for E0.5's result** — non-Gaussian stimulus and non-monotone GELU drive Bussgang's constant toward zero, which is why STA/STC fail |
| Spike-triggered average / covariance (classical) | ABS | The failing baselines, 0/30 and 0/30 |
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

## Maintenance rule

A paper enters this file **the session it is first mentioned**, at whatever level it was
actually read. When one is promoted ABS → FULL, update the level in the same session and
say what the full text changed. Formalise into `paper/references.bib` only at FULL.

**Never cite a specific claim from an ABS row.** That is the rule the August audit exists
to enforce.
