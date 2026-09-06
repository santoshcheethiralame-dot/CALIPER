# Project CALIPER: Comprehensive Technical Master Notes & System Blueprint
**Calibrated Measurement Instruments for Language-Model Internals, Adapted from Systems Neuroscience**

---

# TABLE OF CONTENTS
1. **SECTION 1: THE CORE MEASUREMENT PROBLEM & CONCEPTUAL FRAMING**
   - 1.1 The Uncalibrated Instrument Problem in AI Interpretability
   - 1.2 The Thermometer Analogy (Physical Calibration Standards)
   - 1.3 Why Current Interpretability Methods Rely on Unchecked Assumptions
   - 1.4 The Systems Neuroscience Discipline: Biological "Preparations"
   - 1.5 The CALIPER Solution: Ground-Truth Preparations for Language Models
2. **SECTION 2: MATHEMATICAL GROUND TRUTH INSIDE DEEP NETWORKS**
   - 2.1 Transformer Architecture Overview (Layers, MLP Units, and Residual Stream)
   - 2.2 Mathematical Formulation of MLP Neuron Activation ($y = f(\mathbf{w} \cdot \mathbf{s})$)
   - 2.3 Proof of Exact 100% Ground Truth ($\text{Correlation} = 1.000000$)
   - 2.4 The CALIPER Protocol: Calibration Before Unconstrained Estimation
3. **SECTION 3: PHASE 0 — INSTRUMENT CALIBRATION & FAILURE DIAGNOSTICS**
   - 3.1 Failure Mode 1: Silent Estimation Failures (~20%–23% Deceptive Convergence)
   - 3.2 Optimization Remedy: `fit_cascade` (2D Subspace Search Unlocks 1D Ground Truth)
   - 3.3 Failure Mode 2: Insufficiency of Held-Out $R^2$ (Fit Quality vs. Directional Accuracy)
   - 3.4 Failure Mode 3: Breakdown of Classical Closed-Form Estimators (STA/Bussgang)
   - 3.5 Failure Mode 4: Empirical Detection Threshold ($R^2 = 0.0443$ Null Baseline)
   - 3.6 Optimization Budget vs. Information Limits ($8\text{k tokens} \times 1,600\text{ steps} \times 2\text{ restarts}$)
   - 3.7 Joint Estimator Degeneracy in Multi-Dimensional Subspaces ($K \ge 2$)
   - 3.8 Five Methodological Corrections Forced by Empirical Measurement
4. **SECTION 4: TRACK 1 — UNIT & TRAIT LEVEL RESEARCH (PLANTED PERSONAS)**
   - 4.1 Persona Vectors & Trait Representation Formulations
   - 4.2 The Flaw in Behavioral Steering Validation (The Effect-Validation Trap)
   - 4.3 The Planted Persona Preparation (Synthetic Injection $|\hat{\mathbf{v}} \cdot \mathbf{v}|$)
   - 4.4 Computational Distance & Subspace Degradation Across Layers
   - 4.5 Multitrait-Multimethod (MTMM) Matrices for Language Models
5. **SECTION 5: TRACK 2 — STUDY 3 / PAPER A (AI INTROSPECTION & THOUGHT DETECTION AUDIT)**
   - 5.1 Analysis of Published Introspection Claims (Macar et al., 2026)
   - 5.2 Replication Methodology on Gemma 3 27B (4-Bit NF4 Quantization)
   - 5.3 Primary Finding: The Perturbation Alarm (Norm-Matched Content-Free Controls)
   - 5.4 Introspective vs. Neutral Prompt Framing (Preamble Priming Inflation)
   - 5.5 Readout Volatility (First-Token Logit Probabilities vs. Generated Text Decoding)
   - 5.6 Content Qualitative Analysis ("Red Apple" & Prompt Echoes)
   - 5.7 Refutation of the Post-Hoc Text Output Reading Hypothesis
6. **SECTION 6: SYSTEM ARCHITECTURE & CODEBASE SPECIFICATION**
   - 6.1 Repository Structure & Module Design
   - 6.2 `caliper/activations.py` (PyTorch Forward Hooks & State Extraction)
   - 6.3 `caliper/estimator.py` (Subspace Bottleneck Module & Cascade Optimization)
   - 6.4 `caliper/batched.py` (Shared-Stimulus Grouped Projection Matrix Math)
   - 6.5 `caliper/runtime.py` (Float32 Precision Safeguards & Checkpointing)
   - 6.6 Hardware Setup & Kaggle Execution Pipeline
7. **SECTION 7: COMPREHENSIVE LITERATURE SURVEY (40+ PAPERS IN 3 STRANDS)**
   - 7.1 Strand 1: Systems Neuroscience & Subspace Dimensionality Reduction
   - 7.2 Strand 2: Representation Probing, Steering Vectors & Persona Extraction
   - 7.3 Strand 3: AI Introspection, Self-Reports & Anomaly Detection
   - 7.4 Comparative Methodological Matrix
8. **SECTION 8: PROJECT ROADMAP, WORK DIVISION & RISK MANAGEMENT**
   - 8.1 Semester 5 & 6 Milestone Timeline
   - 8.2 Work Division Across 4 Independent Ownership Tracks
   - 8.3 Hardware & Compute Feasibility (CPU & Kaggle Free Tier)
   - 8.4 Technical Risk Matrix & Fallback Strategies
   - 8.5 Publication Roadmap (arXiv, BlackboxNLP, ICLR Workshops)
9. **SECTION 9: FORMAL TECHNICAL GLOSSARY & PANEL DEFENSE GUIDE**
   - 9.1 40+ Formal Technical Terms Defined
   - 9.2 Comprehensive Panel Q&A Defense Guide

---

# SECTION 1: THE CORE MEASUREMENT PROBLEM & CONCEPTUAL FRAMING

### 1.1 The Uncalibrated Instrument Problem in AI Interpretability
Mechanistic interpretability aims to reverse-engineer the internal representations and computations of Large Language Models (LLMs). Researchers employ diagnostic instruments—such as linear probes, difference-of-means activation steering vectors, sparse autoencoders (SAEs), and bottleneck subspace estimators—to claim that specific internal activation directions correspond to high-level concepts (e.g., "honesty", "sycophancy", or "introspective awareness").

However, the field currently lacks **standardized calibration protocols**. Methods are proposed, adopted, and deployed in safety audits before anyone establishes:
1. What these instruments measure in absolute terms.
2. How reliably they recover internal directions.
3. How frequently they report structured features that do not exist (false discovery rate).
4. How to detect when an estimation algorithm has converged to an incorrect solution.

---

### 1.2 The Thermometer Analogy (Physical Calibration Standards)
In physical measurement science, an instrument (such as a thermometer) is validated by checking its response against physical calibration standards—such as the freezing point (0°C) and boiling point (100°C) of water under standard pressure. 

- Testing an instrument against known reference points determines its systematic error, operational range, and noise floor.
- Once calibrated against known reference states, the instrument can be trusted when measuring unknown physical systems.

In AI interpretability, current diagnostic tools are regularly applied directly to unknown, complex internal representations without prior validation on reference states where ground truth is mathematically guaranteed.

---

### 1.3 Why Current Interpretability Methods Rely on Unchecked Assumptions
Without calibration against ground truth, interpretability research suffers from three main failure modes:

1. **Silent Estimation Failures:** An estimator converges to an incorrect direction while displaying high optimization stability and restart agreement, providing no internal warning of failure.
2. **The Effect-Validation Trap:** A proposed feature direction $\mathbf{v}$ is validated solely by demonstrating that intervention along $\mathbf{v}$ alters downstream model behavior. Because intervention along *any* direction with non-zero alignment to the target feature induces behavioral shifts, effect-validation cannot separate the true feature vector $\mathbf{v}^*$ from a correlated incorrect direction $\mathbf{v}'$.
3. **Perturbation Confounders:** Behavioral responses to activation manipulations are evaluated against zero-intervention baselines ($\alpha=0$). This fails to separate feature-specific processing from a generic computational response to activation magnitude perturbations.

---

### 1.4 The Systems Neuroscience Discipline: Biological "Preparations"
Systems neuroscience encountered identical measurement challenges when recording neural population responses in biological brains. To validate receptive-field estimation algorithms (such as Spike-Triggered Averages and Maximally Informative Dimensions), neuroscientists established simple biological **preparations**—such as the squid giant axon, the sea slug (*Aplysia*), or the *C. elegans* nervous system.

- In a preparation, the physical wiring, synaptic input, or anatomical ground truth is known independently of the recording device.
- Algorithms were evaluated, calibrated, and stress-tested on these preparations before being applied to unmapped cortical areas.

---

### 1.5 The CALIPER Solution: Ground-Truth Preparations for Language Models
**Project CALIPER** ports this measurement discipline to artificial neural networks:

> *"CALIPER constructs ground-truth preparations inside language models—environments where the true internal representation vector is known with 100% mathematical certainty—and uses them to score diagnostic instruments, characterize their sample complexity, remedy their failure modes, and build reliability flags for uncalibrated settings."*

---

# SECTION 2: MATHEMATICAL GROUND TRUTH INSIDE DEEP NETWORKS

### 2.1 Transformer Architecture Overview (Layers, MLP Units, and Residual Stream)
A standard Transformer architecture processes input token sequences through a sequence of layers connected by a high-dimensional **residual stream** $\mathbf{s} \in \mathbb{R}^D$:

$$\mathbf{s}^{(l)} = \mathbf{s}^{(l-1)} + \text{Attention}^{(l)}(\text{LN}(\mathbf{s}^{(l-1)})) + \text{MLP}^{(l)}(\text{LN}(\mathbf{s}^{(l-1)}))$$

- **Residual Stream ($\mathbf{s}$):** The main representation vector space ($D = 768$ in GPT-2 Small; $D = 3,584$ in Gemma 3 27B).
- **MLP Layer:** Contains an up-projection matrix $W_{\text{in}} \in \mathbb{R}^{D \times d_{\text{mlp}}}$, a non-linear activation function $f(\cdot)$, and a down-projection matrix $W_{\text{out}} \in \mathbb{R}^{d_{\text{mlp}} \times D}$.

---

### 2.2 Mathematical Formulation of MLP Neuron Activation ($y = f(\mathbf{w} \cdot \mathbf{s})$)
Consider a single MLP neuron $j$ within layer $l$. Let $\mathbf{s} \in \mathbb{R}^D$ denote the post-LayerNorm residual stream input to the MLP block.

The pre-activation scalar $z_j$ and output activation $y_j$ of neuron $j$ are defined by:

$$z_j = \mathbf{w}_j^{\top} \mathbf{s} + b_j$$

$$y_j = f(z_j) = f(\mathbf{w}_j^{\top} \mathbf{s} + b_j)$$

where $\mathbf{w}_j \in \mathbb{R}^D$ is the $j$-th column of the input weight matrix $W_{\text{in}}$, and $f(\cdot)$ is the element-wise non-linearity (e.g., GELU).

---

### 2.3 Proof of Exact 100% Ground Truth ($\text{Correlation} = 1.000000$)
With respect to the stimulus space $\mathbf{s}$ defined by its own layer's post-LayerNorm residual stream:

1. The neuron's functional dependence on $\mathbf{s}$ is strictly 1-dimensional.
2. The exact direction vector defining this 1D subspace is precisely the weight column $\mathbf{w}_j$.
3. The scalar stimulus value driving the neuron is $z_j = \mathbf{w}_j \cdot \mathbf{s}$.

Empirical verification confirms that the Pearson correlation between $z_j$ and $\mathbf{w}_j^{\top} \mathbf{s}$ is **1.00000000**, with numerical precision errors $< 3.3 \times 10^{-6}$.

**Significance:** This provides **exact, free ground truth inside real, fully-trained neural networks**. We do not need synthetic toy models; the weight matrices of real models provide exact target directions $\mathbf{w}_j$ for every MLP neuron.

---

### 2.4 The CALIPER Protocol: Calibration Before Unconstrained Estimation
CALIPER establishes a strict operational gate:

> **THE CALIPER GATE PROTOCOL:**  
> *An estimator must achieve direction recovery accuracy $|\hat{\mathbf{w}} \cdot \mathbf{w}| \ge 0.95$ on this free ground truth before it can be deployed on earlier layers or deeper latent representations where ground truth is unavailable.*

---

# SECTION 3: PHASE 0 — INSTRUMENT CALIBRATION & FAILURE DIAGNOSTICS

Phase 0 evaluated rank-1 bottleneck subspace search algorithms across 100 real MLP neurons in GPT-2 Small layer 6 under natural text stimulus driving.

---

### 3.1 Failure Mode 1: Silent Estimation Failures (~20%–23% Deceptive Convergence)
When fitting a rank-1 subspace estimator $\hat{\mathbf{w}}$ directly:
- **77/100 neurons** passed ($|\hat{\mathbf{w}} \cdot \mathbf{w}| \ge 0.95$).
- **23/100 neurons failed**, converging to directions with alignment as low as $|\hat{\mathbf{w}} \cdot \mathbf{w}| = 0.452$.

**Empirical Diagnostic (Experiment `E0.1g`):**
For failing neuron `n2977`:
- $R^2$ at the true weight direction $\mathbf{w} = 1.000$ (The global optimum exists).
- $R^2$ at the optimizer's solution $\hat{\mathbf{w}} = 0.809$ (Trapped in a local minimum).
- Optimization restart agreement across independent seeds = **0.85 to 0.99**.

**Conclusion:** Optimization restart stability is deceptive. The estimator converged with high confidence to an incorrect direction without issuing any diagnostic error.

---

### 3.2 Optimization Remedy: `fit_cascade` (2D Subspace Search Unlocks 1D Ground Truth)
Experiment `E0.1g` revealed that a 2-dimensional fit ($k=2$) contained the true 1D direction $\mathbf{w}$ at an alignment of **0.997**. The 1D optimization landscape contains local traps, whereas the 2D landscape is smooth.

**Algorithm (`fit_cascade`):**
1. Fit a 2D subspace matrix $V_{\text{2D}} \in \mathbb{R}^{D \times 2}$ using joint bottleneck optimization.
2. Perform a 1D line-search parameterised by angle $\theta \in [0, \pi)$ within the span of $V_{\text{2D}}$:
   $$\mathbf{v}(\theta) = V_{\text{2D}}[:, 0] \cos\theta + V_{\text{2D}}[:, 1] \sin\theta$$
3. Select $\hat{\mathbf{w}} = \mathbf{v}(\theta^*)$ maximizing held-out fit quality.

**Results:**
- Neuron `n2977`: Direction recovery jumped from **0.452 $\rightarrow$ 0.999**.
- Neuron `n230`: Direction recovery jumped from **0.866 $\rightarrow$ 0.999**.
- Total pass rate increased to $> 95\%$.

---

### 3.3 Failure Mode 2: Insufficiency of Held-Out $R^2$ (Fit Quality vs. Directional Accuracy)
Evaluating model selection between direct 1D search and `fit_cascade` using held-out $R^2$:

```
Neuron n1989:
  Direct Fit Direction Alignment  = 0.964 | Held-Out R² = 0.973531
  Cascade Fit Direction Alignment = 0.922 | Held-Out R² = 0.988305  <-- Selected by R²!
```

Selecting by held-out $R^2$ chose the **less accurate direction** (0.922 vs 0.964) by a margin of 0.015 in $R^2$. 

**Takeaway:** The flexible MLP non-linearity absorbs directional error, decoupling fit quality ($R^2$) from directional accuracy ($|\hat{\mathbf{w}} \cdot \mathbf{w}|$). In uncalibrated settings (where ground truth is absent), held-out $R^2$ cannot guarantee directional correctness. CALIPER requires reporting **estimator disagreement** as a per-unit reliability flag.

---

### 3.4 Failure Mode 3: Breakdown of Classical Closed-Form Estimators (STA/Bussgang)
Bussgang's theorem proves that for Gaussian inputs $\mathbf{s}$, Spike-Triggered Averaging (STA) recovers $\mathbf{w}$ up to scale without non-linear optimization. 

Scoring classical closed-form estimators against CALIPER's fitted bottleneck estimator across 30 neurons:

| Estimation Method | Median Alignment | Fraction $> 0.95$ |
|---|---|---|
| Spike-Triggered Average (STA) | 0.287 | 0.00 (0 / 30) |
| Decorrelated STA (Bussgang) | 0.528 | 0.03 (1 / 30) |
| STC Top Eigenvector | 0.322 | 0.00 (0 / 30) |
| **CALIPER Fitted Bottleneck** | **0.995** | **0.87 (26 / 30)** |

**Mechanistic Causes:**
1. Residual stream representations $\mathbf{s}$ are non-Gaussian (heavy-tailed distributions).
2. The GELU non-linearity is non-monotone for $x < 0$. Neurons occupy the sub-zero regime 90%–99% of the time, driving the Bussgang expectation $E[f'(z)] \rightarrow 0$ and destroying STA signal.

---

### 3.5 Failure Mode 4: Empirical Detection Threshold ($R^2 = 0.0443$ Null Baseline)
Testing null projections across 24,000 random directions over 12 real neurons:

| Parameter | Value |
|---|---|
| **Detection Threshold (Max Null $p_{99}$)** | **$R^2 = 0.0443$** |
| True-Direction $R^2$ Range | $0.700 - 0.985$ |
| Alignment Null Distribution (Median / $p_{99}$ / Max) | 0.024 / 0.091 / 0.144 |

**Result:** Random directions explain at most 4.4% of variance. $R^2 = 0.0443$ serves as CALIPER's strict detection threshold for establishing feature presence.

---

### 3.6 Optimization Budget vs. Information Limits ($8\text{k tokens} \times 1,600\text{ steps} \times 2\text{ restarts}$)
Sweeping optimization parameters across synthetic sparse-firing units:

```
Tokens | Steps | Restarts | Median Recovery | Min Recovery (Worst-Case)
 8,000 | 1,600 |    2     |     0.9995      |      0.9621  <-- CALIPER Operating Point
16,000 |   800 |    1     |     0.9994      |      0.0077  (Worst-case failure!)
```

Doubling data size while reducing optimization steps degraded worst-case recovery. Optimization budget, rather than data volume, is the primary constraint. Operating point: **8,000 tokens $\times$ 1,600 steps $\times$ 2 restarts**.

---

### 3.7 Joint Estimator Degeneracy in Multi-Dimensional Subspaces ($K \ge 2$)
Sweeping joint subspace estimators for $K \in \{2, 3\}$:
- $K=2 \rightarrow$ Measured alignment $= 0.502 \approx 1/2$.
- $K=3 \rightarrow$ Measured alignment $= 0.356 \approx 1/3$.

Joint estimators recover one direction perfectly and miss remaining orthogonal dimensions. Multi-dimensional subspace recovery must be performed **iteratively** (recovering 1 direction, deflating/projecting out, then recovering subsequent directions).

---

### 3.8 Five Methodological Corrections Forced by Empirical Measurement
1. **PCA Truncation Removed:** MLP weight vectors retain only 0.352 norm fraction in top-64 PCA space (chance $= 0.267$). Retaining 80% norm requires $D \approx 512$ (no dimensionality reduction).
2. **Whitening Removed:** Whitening covariance matrices near $D=768$ are near-singular, amplifying noise and reducing real-stream alignment to 0.02.
3. **Event-Indexed Sample Complexity:** Sample requirements must be indexed by active firing events ($z > 0$), not total token count.
4. **Response Transforms Reverted:** Rank-gaussianizing responses degraded pass rates from 0.75 to 0.50.
5. **Shared-Stimulus Batching:** Grouping projection operations across neurons yields a **4.5x to 20x speedup**.

---

# SECTION 4: TRACK 1 — UNIT & TRAIT LEVEL RESEARCH (PLANTED PERSONAS)

### 4.1 Persona Vectors & Trait Representation Formulations
Persona vector literature (e.g., Anthropic, 2025) extracts trait representations by computing difference-of-means vectors across trait-eliciting vs. baseline responses:

$$\mathbf{v}_{\text{persona}} = \mathbb{E}[\mathbf{s}_{\text{trait}}] - \mathbb{E}[\mathbf{s}_{\text{baseline}}]$$

---

### 4.2 The Flaw in Behavioral Steering Validation (The Effect-Validation Trap)
Published persona vector validations rely on behavioral steering: injecting $\mathbf{v}_{\text{persona}}$ into residual activations and verifying that the model outputs trait-aligned text.

**The Effect-Validation Flaw:**  
Intervention along *any* vector $\mathbf{v}'$ satisfying $\mathbf{v}'^{\top} \mathbf{v}^* > 0$ alters model output toward the target trait. Behavioral steering cannot separate the ground-truth trait vector $\mathbf{v}^*$ from an inaccurate vector $\mathbf{v}'$.

---

### 4.3 The Planted Persona Preparation (Synthetic Injection $|\hat{\mathbf{v}} \cdot \mathbf{v}|$)
CALIPER constructs a ground-truth preparation for persona extraction:

1. Inject a synthetic vector $\mathbf{v}$ into layer $L$ during forward execution.
2. Sample model text generations under intervention.
3. Run published difference-of-means persona extraction pipelines to yield $\hat{\mathbf{v}}$.
4. Calculate directional recovery accuracy $|\hat{\mathbf{v}} \cdot \mathbf{v}|$.

---

### 4.4 Computational Distance & Subspace Degradation Across Layers
By extracting persona vectors at downstream layers ($L+1, L+2, \dots, L+k$), CALIPER measures the rate at which feature recovery degrades as a function of intervening non-linear computation.

---

### 4.5 Multitrait-Multimethod (MTMM) Matrices for Language Models
CALIPER applies psychological construct validation (Campbell & Fiske, 1959) to feature extraction:

| Planted Vector | Extracted via Difference-of-Means | Extracted via Linear Probe | Extracted via CALIPER Bottleneck |
|---|---|---|---|
| **Planted Vector A (Sycophancy)** | Convergent Score vs $\mathbf{v}_A$ | Convergent Score vs $\mathbf{v}_A$ | Convergent Score vs $\mathbf{v}_A$ |
| **Planted Vector B (Evasiveness)** | Discriminant Score | Discriminant Score | Discriminant Score |

Evaluates both **convergent validity** (cross-tool agreement on identical planted features) and **discriminant validity** (separation of distinct planted features).

---

# SECTION 5: TRACK 2 — STUDY 3 / PAPER A (AI INTROSPECTION & THOUGHT DETECTION AUDIT)

### 5.1 Analysis of Published Introspection Claims (Macar et al., 2026)
Macar et al. (2026) injected concept vectors into Gemma 3 27B at Layer 37 and evaluated free-generation responses to:

> *"Do you detect an injected thought? Answer starting with YES or NO."*

They reported a 10.8% detection rate with 0% false positives (against zero injection $\alpha=0$), attributing the response to a post-training introspective circuit.

---

### 5.2 Replication Methodology on Gemma 3 27B (4-Bit NF4 Quantization)
CALIPER execution environment:
- Model: `google/gemma-3-27b-it` in 4-bit NormalFloat (NF4) quantization via `bitsandbytes`, float32 compute, 2x Tesla T4 GPUs (Kaggle free tier).
- 30 concept vectors injected at Layer 37 prompt positions.
- **Replication Result:** Free-generation detection rate $= \mathbf{10.0\% \text{ [95\% CI 3.5\%--25.6\%]}}$ with $\mathbf{0/30}$ false positives at $\alpha=0$.

---

### 5.3 Primary Finding: The Perturbation Alarm (Norm-Matched Content-Free Controls)
Macar et al. evaluated false positives solely against zero intervention ($\alpha=0$). CALIPER introduced two magnitude-matched, content-free control vectors:
1. **Random Control:** Gaussian vector rescaled to match the exact $L_2$ norm of the real concept vector.
2. **Shuffle Control:** Real concept vector with coordinate positions randomly permuted.

**Pre-Registered Primary Test ($\alpha=6$, Introspective Prompt):**

| Vector Condition | Mean First-Token P(YES) | Statistical Comparison |
|---|---|---|
| **Real Concept Vector** | **0.417** | Baseline |
| **Random Control Vector** | **0.305** | $p = 0.33$ (Not Significant) |
| **Shuffle Control Vector** | **0.374** | $p = 0.76$ (Not Significant) |

**Conclusion:** At the operating point matching the published detection rate, magnitude-matched content-free noise vectors produce reports statistically indistinguishable from real concept vectors. The model detects **activation perturbation magnitude**, not concept content.

---

### 5.4 Introspective vs. Neutral Prompt Framing (Preamble Priming Inflation)
CALIPER evaluated prompt framing sensitivity by comparing:
- **Introspective Prompt:** Contains preamble detailing activation manipulation and asking about "injected thoughts".
- **Neutral Prompt:** Preamble-matched control asking *"Is anything unusual present?"* (Never mentions activations, mind, or injection).

**First-Token Logit Results Across Injected Strengths ($\alpha$):**

| Strength $\alpha$ | Introspective Prompt P(YES) | Neutral Prompt P(YES) | $p$-value |
|---|---|---|---|
| $\alpha = 0$ | 0.000 | 0.188 | $< 0.0001$ |
| $\alpha = 2$ | 0.429 | 0.370 | $p = 0.44$ |
| $\alpha = 4$ | 0.504 | 0.481 | $p = 0.75$ |
| $\alpha = 6$ | 0.417 | 0.395 | $p = 0.70$ |

Once intervention is applied, introspective and neutral prompts yield **statistically indistinguishable detection probabilities** ($p = 0.44 - 0.75$). Self-directed framing adds no unique detection sensitivity, but inflates low-strength noise responses by **8-fold (+0.184 vs +0.023)**.

---

### 5.5 Readout Volatility (First-Token Logit Probabilities vs. Generated Text Decoding)
Comparing two readout methods:
1. **Generated-Text Readout:** 60-token greedy sampling, scored for affirmative prefix and response coherence.
2. **First-Token Logit Readout:** Normalized probability $P(\text{YES}) / [P(\text{YES}) + P(\text{NO})]$ at the initial decoding position.

```
Strength alpha | Generated-Text Detection Rate | First-Token Logit P(YES)
     alpha=2   |             50%               |          0.429
     alpha=4   |             23%               |          0.504
     alpha=6   |             27%               |          0.417
     alpha=8   |              7%               |          0.430
```

While first-token logit probabilities remain flat ($\approx 0.42$), generated-text detection rates swing from **7% to 50%**. At high injection strengths ($\alpha=8$), activation steering induces repetitive generation loops, preventing the model from emitting coherent affirmative prefixes within token budgets. Reported detection rates reflect decoding dynamics rather than internal model sensitivity.

---

### 5.6 Content Qualitative Analysis ("Red Apple" & Prompt Echoes)
Analysis of 210 free-generation text responses:
- At $\alpha=2$, among 15 affirmative detections, **zero responses identified the injected concept**.
- 7 responses echoed preamble text ("a researcher studying my activations").
- 3 responses reported "red apple" (un-injected concept; reproducing Lederman & Mahowald's confabulation finding).

Affirmative detection reports and correct concept identification peak at opposite ends of the injection strength spectrum and rarely coincide.

---

### 5.7 Refutation of the Post-Hoc Text Output Reading Hypothesis
CALIPER pre-registered a hypothesis that affirmative detection reports resulted from the model generating concept-laden text and reading its own output.

First-token logit evaluation (prior to text generation) demonstrated immediate probability shifts ($0.00003 \rightarrow 0.417$, $p = 9.3 \times 10^{-9}$). The hypothesis was formally refuted and reported under pre-registered protocol discipline.

---

# SECTION 6: SYSTEM ARCHITECTURE & CODEBASE SPECIFICATION

### 6.1 Repository Structure & Module Design
```
caliper/
  ├── activations.py   # Forward hook registration & residual state capture
  ├── estimator.py     # Subspace bottleneck PyTorch modules & fit_cascade search
  ├── batched.py       # Shared-stimulus grouped einsum projection math
  └── runtime.py       # Device selection, float32 precision guards & state I/O
experiments/
  ├── e01_analytic_control.py      # Ground-truth weight recovery gate suite
  ├── e01h_cascade.py              # Fit_cascade 2D line search routines
  ├── e02_random_direction_null.py # Random direction null distribution sweep
  ├── e03_required_n.py            # Event-indexed sample complexity tables
  ├── e05_classical_baselines.py   # Closed-form STA/STC benchmark comparisons
  ├── kaggle_s3_positive_control.py# Gemma 3 27B Introspection GPU test suite
  ├── analyse_s3_full.py           # Paired Wilcoxon statistical tests
  └── rescore_s3.py                # Generated text decoding audit routines
results/                           # Execution logs & committed JSONL results
paper/                             # NeurIPS LaTeX template & script figure generation
```

---

### 6.2 `caliper/activations.py` (PyTorch Forward Hooks & State Extraction)
Registers forward hooks on target residual stream layers to capture activation tensors:
```python
import torch

def capture_residual_activations(model, tokenizer, prompts, layer_idx):
    captured_activations = []
    
    def hook_fn(module, input, output):
        # Extract post-LayerNorm residual stream output tensor
        tensor = output[0] if isinstance(output, tuple) else output
        captured_activations.append(tensor.detach().to(torch.float32).cpu())
    
    target_layer = model.model.layers[layer_idx]
    handle = target_layer.register_forward_hook(hook_fn)
    
    # Execute forward pass under torch.no_grad()
    with torch.no_grad():
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        model(**inputs)
        
    handle.remove()
    return torch.cat(captured_activations, dim=0)
```

---

### 6.3 `caliper/estimator.py` (Subspace Bottleneck Module & Cascade Optimization)
Defines rank-$K$ subspace bottleneck estimation modules in PyTorch:
```python
import torch
import torch.nn as nn

class BottleneckSubspaceEstimator(nn.Module):
    def __init__(self, d_in, rank=1, hidden_dim=64):
        super().__init__()
        self.V = nn.Parameter(torch.randn(d_in, rank)) # Subspace projection matrix
        self.nonlinearity = nn.Sequential(
            nn.Linear(rank, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, X):
        # X: (N_samples, d_in)
        projection = X @ self.V # (N_samples, rank)
        return self.nonlinearity(projection).squeeze(-1)
```

---

### 6.4 `caliper/batched.py` (Shared-Stimulus Grouped Projection Matrix Math)
Vectorizes bottleneck search across $N_{\text{neurons}}$ simultaneously over a shared stimulus matrix $X \in \mathbb{R}^{S \times D}$:
```python
import torch

def batched_subspace_projection(X, V_batch):
    # X: (S_samples, D_dim)
    # V_batch: (N_neurons, D_dim, K_rank)
    # Returns Z: (S_samples, N_neurons, K_rank)
    return torch.einsum('sd,ndk->snk', X, V_batch)
```
Replaces sequential loop calls with single parallel GPU tensor operations, accelerating processing by **4.5x to 20x**.

---

### 6.5 `caliper/runtime.py` (Float32 Precision Safeguards & Checkpointing)
- **Precision Guards:** Enforces float32 evaluation during Gemma layer activations to eliminate float16 overflow exceptions (peak activation magnitude $= 51,436$ vs. float16 max $= 65,504$).
- **State Persistence:** Writes trial outcomes to append-only `.jsonl` logs to enable immediate resumption following Kaggle session timeouts.

---

### 6.6 Hardware Setup & Kaggle Execution Pipeline
- Hardware: 2x Tesla T4 GPUs (16GB VRAM per GPU).
- Quantization: 4-bit NormalFloat (NF4) via `bitsandbytes`.
- Compute Precision: Float32 computation for activation manipulation stability.

---

# SECTION 7: COMPREHENSIVE LITERATURE SURVEY (40+ PAPERS IN 3 STRANDS)

### 7.1 Strand 1: Systems Neuroscience & Subspace Dimensionality Reduction
- **Sharpee, Rust & Bialek (2004):** Maximally Informative Dimensions (MID) for neural stimulus-response characterization.
- **Williamson, Sahani & Pillow (2015):** Equivalence proofs between information-theoretic MID and likelihood-based bottleneck estimators.
- **Willmore & Tolhurst (2001):** Single-cell selectivity vs. population sparsity dynamics.

### 7.2 Strand 2: Representation Probing, Steering Vectors & Persona Extraction
- **Distill Circuits (2020–2021):** Feature visualization and receptive field mapping in vision networks.
- **Anthropic Persona Vectors (2025):** Difference-of-means trait vector extraction validated via behavioral steering.
- **Kim & Paik (2026):** Geometry of latent sub-dimensional representations in LLMs.

### 7.3 Strand 3: AI Introspection, Self-Reports & Anomaly Detection
- **Macar et al. (2026):** Injected concept detection on Gemma 3 27B ($L=37, \alpha=4$).
- **Lindsey et al. (2025/2026):** Introspective awareness claims in Claude 4 / 4.1.
- **Lederman & Mahowald (2026):** Confabulation analysis demonstrating content-agnostic detection patterns.
- **Singh et al. (2026):** *Reality Check* analysis showing detection sensitivity reflects generic input/activation irregularity.
- **Godet (2025):** Logit shift observations under steering vectors on Mistral-22B.

---

### 7.4 Comparative Methodological Matrix

| Paper | Target Model(s) | Measurement Readout | Content-Free Control Vector? | Preamble-Matched Neutral Prompt? | Core Finding |
|---|---|---|---|---|---|
| **Lindsey (2025)** | Claude 4 / 4.1 | Generated Text | NO | NO | Reports introspective awareness |
| **Godet (2025)** | Mistral-22B | First-Token Logit | Informal (1 sentence) | NO | Demonstrates logit shift bias |
| **Lederman & Mahowald (2026)** | Qwen3 235B, Llama 405B | Generated Text + Logit Lens | NO (Inferred) | NO | Concludes content-agnostic anomaly detection |
| **Macar et al. (2026)** | Gemma 3 27B | Generated Text | NO | Partial ("Unprompted") | Attributes detection to post-training circuit |
| **Singh et al. (2026)** | Gemma 27B, Llama 70B | First Sampled Token | NO | NO | Attributes detection to generic irregularity |
| **CALIPER (This Work)** | **Gemma 3 27B** | **First-Token Logit AND Generated Text** | **YES (Pre-registered Random & Shuffle)** | **YES (Preamble-Matched Control)** | **Demonstrates perturbation alarm dominance; small concept residual at low $\alpha$; non-self-specific** |

---

# SECTION 8: PROJECT ROADMAP, WORK DIVISION & RISK MANAGEMENT

### 8.1 Semester 5 & 6 Milestone Timeline

```
  2026           2026           2026           2026           2027           2027
  SEPTEMBER      OCTOBER        NOVEMBER       DECEMBER       JANUARY        FEBRUARY
 ┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
 │ Sem 5 Rev 1  │ Sem 5 Rev 2  │ Sem 5 Rev 3  │ Sem 5 Rev 4  │ Paper 1      │ ICLR 2027    │
 │ Scope & Lit  │ System Design│ Unit Prototype│ Introspection│ arXiv        │ Workshop     │
 │ Presentation │ Requirements │ Demonstration│ Results      │ Preprint     │ Submission   │
 └──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
```

- **September 2026:** Department Review 1; $L_2$-normalized vector protocol execution.
- **October 2026:** Department Review 2; release Study 3 preprint on arXiv (cs.LG / cs.AI).
- **November 2026:** Department Review 3 & Review 4; finalize `caliper/batched.py` engine.
- **December 2026 – February 2027:** Execute Planted Persona depth sweep (Track 1); submit to ICLR 2027 Workshop.

---

### 8.2 Work Division Across 4 Independent Ownership Tracks
1. **Track Lead A (Estimation & Integration):** Bottleneck estimator, `fit_cascade` optimization, Study 3 pipeline integration.
2. **Track Lead B (Literature & Theory):** 60-paper literature matrix, neuroscience method mapping, related work writing.
3. **Track Lead C (Replication & Robustness):** Replicating Study 3 on Qwen 2.5 32B with trial-randomized prompts.
4. **Track Lead D (Calibration & Benchmarking):** Evaluating false discovery rates on InterpBench planted-latent networks.

---

### 8.3 Hardware & Compute Feasibility (CPU & Kaggle Free Tier)
- **Unit Level (Phase 0):** Evaluated locally on CPU (~1 minute per neuron fit).
- **Large Models (Study 3 & Planted Personas):** Evaluated on **Kaggle Free Tier** (2x Tesla T4 GPUs, 30 GPU-hours/week per account). Zero department budget required.

---

### 8.4 Technical Risk Matrix & Fallback Strategies

| Technical Risk | Impact | Likelihood | Mitigation Strategy |
|---|---|---|---|
| 1D Rank-1 Optimization Traps | High | High | `fit_cascade` 2D plane line search + Disagreement Flag warning light |
| Multi-Dimensional Degeneracy ($K \ge 2$) | Medium | High | Sequential 1D iterative extraction (project out & repeat) |
| Kaggle GPU Session Timeouts | Medium | Medium | Append-only `.jsonl` trial checkpointing |
| Float16 Activation Overflow | High | Medium | Enforce float32 evaluation during activation capture |
| Prior Work Scooping | Medium | Low | Instrument calibration framing is scoop-resistant; early arXiv release |

---

### 8.5 Publication Roadmap (arXiv, BlackboxNLP, ICLR Workshops)
- **Target 1:** arXiv Preprint (October 2026) — Title: *"What does a language model detect when it detects an injected thought?"*
- **Target 2:** ICLR 2027 Workshop Submission (February 2027).
- **Target 3:** Paper 1 (Phase 0 Calibration Methods) submitted to BlackboxNLP / NeurIPS ATTRIB.

---

# SECTION 9: FORMAL TECHNICAL GLOSSARY & PANEL DEFENSE GUIDE

### 9.1 40+ Formal Technical Terms Defined
1. **Activation:** The scalar or tensor output generated by a neural network layer or unit during forward pass execution.
2. **Activation Steering (Injection):** Adding a feature vector directly to residual stream activations to modify model outputs.
3. **Bottleneck Estimator:** An optimization model that projects high-dimensional input vectors into a low-dimensional subspace $K$ to predict unit activations.
4. **Calibration:** Evaluating diagnostic instruments against known reference states before deployment on unmapped representations.
5. **Cascade Optimization (`fit_cascade`):** Initializing search in a 2D subspace before isolating the optimal 1D vector to avoid local minima traps.
6. **Concept Vector:** A directional vector derived by subtracting baseline residual activations from concept-prompted activations.
7. **Convergent Validity:** The degree to which distinct measurement tools yield consistent readings on an identical target feature.
8. **Discriminant Validity:** The degree to which a measurement tool differentiates distinct target features without cross-contamination.
9. **Disagreement Flag:** A diagnostic metric comparing direct and cascade optimization fits to flag uncalibrated estimation failures.
10. **Effect Validation:** Validating a feature vector solely by demonstrating downstream behavioral intervention shifts (flawed due to non-unique vector alignment).
11. **First-Token Logit Readout:** Measuring normalized logit probabilities $P(\text{YES}) / [P(\text{YES}) + P(\text{NO})]$ at the initial decoding position.
12. **GELU:** Gaussian Error Linear Unit; a non-monotone activation function $f(x) = x \cdot \Phi(x)$.
13. **Ground Truth:** An authoritative reference state known independently of the diagnostic instrument being evaluated.
14. **Introspection:** A model reporting on its own internal computational or activation states via text generation.
15. **Maximally Informative Dimensions (MID):** An information-theoretic subspace reduction method developed in systems neuroscience.
16. **MLP Input Weight Column ($\mathbf{w}_j$):** The $j$-th column vector of an MLP layer's input weight matrix $W_{\text{in}}$.
17. **Multitrait-Multimethod (MTMM) Matrix:** A formal psychometric framework for evaluating construct validity across features and measurement tools.
18. **Null Distribution:** The empirical response distribution generated by an instrument when evaluating content-free random noise.
19. **Operating Point:** The minimum optimization budget (tokens $\times$ steps $\times$ restarts) required to guarantee worst-case recovery performance.
20. **Persona Vector:** An activation direction claimed to represent high-level behavioral or personality traits.
21. **Perturbation Alarm:** A computational shift where a model outputs detection reports in response to activation magnitude disturbance, irrespective of content.
22. **Planted Persona:** A synthetic direction vector injected into a model to establish mathematical ground truth for persona extraction algorithms.
23. **Preparation:** A biological or artificial system engineered to provide known reference states for instrument calibration.
24. **Pre-Registration:** Formally archiving experimental criteria, hypotheses, and analysis scripts prior to execution.
25. **Random Control Vector:** A Gaussian noise vector scaled to match the exact $L_2$ norm of a target concept vector.
26. **Residual Stream ($\mathbf{s}$):** The primary linear representation vector space passing through Transformer decoder layers.
27. **Shuffle Control Vector:** A target concept vector whose coordinate elements have been randomly permuted.
28. **Silent Estimation Failure:** An estimator converging to an incorrect direction while displaying high optimization stability and restart agreement.
29. **Spike-Triggered Average (STA):** A closed-form linear estimation technique derived from neural receptive field mapping.
30. **Subspace:** A linear vector space of dimension $K$ embedded within a higher-dimensional space $D$.
31. **Tuning Curve:** A functional characterization of unit output activation as a function of input stimulus parameters.
32. **Weight Vector ($\mathbf{w}$):** The input weight matrix column defining exact ground-truth selectivity for an MLP neuron in its own layer.
33. **Wilcoxon Signed-Rank Test:** A non-parametric paired statistical test used to evaluate paired experimental conditions.
34. **Wilson Score Interval:** A statistical confidence interval calculation for binomial pass rates.

---

### 9.2 Comprehensive Panel Q&A Defense Guide

**Q1: "What is the primary contribution of Project CALIPER in formal terms?"**  
*Answer:* CALIPER establishes a ground-truth calibration protocol for LLM interpretability tools by adapting measurement principles from systems neuroscience. It characterizes silent estimation failures, provides algorithmic remedies (`fit_cascade`), and evaluates frontier introspection claims using magnitude-matched content-free controls.

**Q2: "How is mathematical ground truth established inside a trained LLM?"**  
*Answer:* An MLP neuron's activation is $y = f(\mathbf{w}_j^{\top} \mathbf{s} + b_j)$. With respect to its layer's post-LayerNorm residual stream $\mathbf{s}$, the true input direction defining its 1D stimulus space is mathematically identical to its input weight column $\mathbf{w}_j$, which is extracted directly from model weight parameters.

**Q3: "Why do standard 1D rank-1 subspace search algorithms experience silent failures on ~23% of units?"**  
*Answer:* Direct 1D optimization landscapes contain local minima traps. However, 2D subspace landscapes are benign. CALIPER's `fit_cascade` algorithm initializes search in 2D space before optimizing a 1D direction, elevating recovery performance from 45.2% to 99.9%.

**Q4: "What did CALIPER's audit reveal regarding published LLM introspection claims?"**  
*Answer:* Evaluating Gemma 3 27B under magnitude-matched content-free control vectors (random Gaussian and shuffled vectors) revealed that detection reports are statistically indistinguishable from real concept vectors ($p = 0.33$). The model exhibits a generic perturbation alarm to activation magnitude rather than introspective access to concept content.

**Q5: "How is computational feasibility maintained for 27B parameter model evaluations?"**  
*Answer:* Models are executed in 4-bit NF4 quantization via `bitsandbytes` with float32 evaluation precision on Kaggle's free GPU tier (2x Tesla T4 GPUs), incurring zero hardware expenditure.
