# Capstone Review 1 Presentation Deck
## Title: Calibrating Interpretability Tools for Language Models (Project CALIPER)
**Subtitle:** *Does the Mind-Reading Kit Work? Building Ground-Truth Preparations for LLMs*

---

## Slide Outline Overview

1. **Slide 1: Title & Team** (Introduction, Team, Mentor)
2. **Slide 2: Introduction & Core Concept** (The Neuroscience Analogy)
3. **Slide 3: Problem Statement** (3 Tools, 3 Levels, Zero Ground-Truth Checks)
4. **Slide 4: Background Study & Literature Survey** (Three Literature Strands & Critical Gaps)
5. **Slide 5: Where Prior Work Collides vs. Our Difference** (Comparative Literature Matrix)
6. **Slide 6: Workflow & System Architecture** (3-Level Preparations: Unit, Trait, Self)
7. **Slide 7: Calibration Workflow & Requirements** (The 4 Calibration Rules & Disagreement Flag)
8. **Slide 8: Scope, Feasibility & Resource Planning** (Hardware, Datasets, Pre-registration & Work Division)
9. **Slide 9: Semester Roadmap & Key Deliverables** (Review 1 through Review 4)
10. **Slide 10: Expected Panel Q&A & Cheat Sheet** (Anticipated Questions & Short Answers)

---

# Detailed Slide Content & Speaker Scripts

---

### Slide 1: Title, Team & Project Overview

**Slide Header:** Project CALIPER: Calibrated Measurement Instruments for Language-Model Internals  
**Subtitle:** Adapting Systems Neuroscience Discipline to AI Interpretability  

**Slide Content:**
- **Team Name & Members:** [List Team Members] (Tracks: Estimator & Integration, Literature & Framing, Model Replication, Calibration)
- **Mentor:** [Mentor Name] (Selected 18 August 2026 for its neuroscience-inspired measurement framing)
- **Department Review:** Capstone Semester 5 — Review 1: Problem Identification, Literature Survey, & System Scope
- **Core Pitch:** *"Every reading tool in AI interpretability is a thermometer that nobody has ever tested in ice water. We build the ice water."*

> **🔊 Speaker Script (15–20 seconds):**  
> *"Good morning members of the panel. Our project is CALIPER. Interpretability tools claim to read what is happening inside a language model—which direction a neuron responds to, which vector encodes honesty or sycophancy, or whether a model can sense its own internal states. However, none of these tools has ever been checked against a case where the true answer was already known. We adapt the discipline of systems neuroscience to build known ground-truth 'preparations' for AI models and measure how reliably these reading tools work."*

---

### Slide 2: Introduction — The Neuroscience Analogy

**Slide Header:** Introduction: Why Systems Neuroscience?

**Slide Content:**
* **The "Thermometer in Ice Water" Analogy:**
  - Before a doctor trusts a clinical thermometer on a patient, it is calibrated in freezing (0°C) and boiling (100°C) water.
  - In AI interpretability, researchers use tools to read model activations, but skip calibration against known ground truth.
* **The Neuroscience Precedent ("Preparations"):**
  - Neuroscientists faced the exact same problem decades ago when reading signals from biological brains.
  - They solved it using **simple preparations** (squid giant axon, *Aplysia*, *C. elegans*) where physical structure was known, allowing them to calibrate instruments before applying them to complex brains.
* **Our Core Idea:**
  - Build **preparations for language models** across three levels: **Unit Level** (neurons), **Trait Level** (personas), and **Self Level** (thought detection reports).

> **🔊 Speaker Script (45 seconds):**  
> *"Why systems neuroscience? Neuroscience has been reading signals from complex units it didn't design for over a century. Early on, neuroscientists learned that every reading tool must be calibrated on a simple 'preparation'—like the giant squid axon or a sea slug—where the physical ground truth was known. Without that step, you can't tell if an unexpected reading is a discovery or an instrument failure. Today, AI interpretability is repeating early neuroscience's mistakes by using probes and steering vectors without calibration. We import neuroscience's measurement discipline into AI."*

---

### Slide 3: Problem Statement

**Slide Header:** Problem Statement: Reading Tools Without Calibration

**Slide Content:**

| Level | Current Published Tool | Published Claim | Ground-Truth Check |
|---|---|---|---|
| **1. Unit Level** | Bottleneck search / Probes | *"Neuron 2977 represents concept X"* | **NEVER** (Wiring unknown) |
| **2. Trait Level** | Persona / Steering vectors | *"This vector is the model's honesty trait"* | **NEVER** (Effect-only validation) |
| **3. Self Level** | Introspective prompts | *"Model reports detecting injected thoughts"* | **NEVER** (Controlled only vs. zero injection) |

* **The Core Problem:**
  - **Silent Failures:** When a tool returns a wrong answer, it looks 100% confident. Standard metrics (like held-out accuracy) fail to detect when the direction is wrong.
  - **Effect Validation Flaw:** Showing that steering along a vector changes behavior does *not* prove it is the right direction—steering along *any* correlated direction changes behavior!
  - **Perturbation Confound:** Prompts asking models to report on themselves confuse generic disturbance (a kick) with actual content detection.

> **🔊 Speaker Script (60 seconds):**  
> *"Here is our problem statement: Interpretability tools claim to read a model's mind at three distinct levels—single neurons, personality traits, and self-reports. But zero of these tools have been validated against known ground truth. At the unit level, probing algorithms fail silently 23% of the time with no warning. At the trait level, persona vectors are validated only by steering, which cannot separate the true direction from a correlated wrong one. At the self level, asking a model about itself causes it to confuse generic disturbance with content detection. We build the test bench that these tools have never had."*

---

### Slide 4: Background Study & Literature Survey

**Slide Header:** Background Study: Three Strands of Literature

**Slide Content:**
1. **Strand 1: Dimensionality Reduction & Receptive Field Estimation (Neuroscience)**
   - *Key Papers:* Sharpee et al. (2004), Williamson, Sahani & Pillow (2015).
   - *Takeaway:* Maximally Informative Dimensions (MID) and likelihood-based subspace estimators. Neuroscience suspects MID error on real neurons is large, but cannot verify it due to lack of ground truth.
2. **Strand 2: Representation Probing & Persona Extraction (AI Interpretability)**
   - *Key Papers:* Anthropic (2025/2026), Distill Circuits (2020), Kim & Paik (2026).
   - *Takeaway:* Difference-of-means and linear probes to extract concept/persona vectors. All existing validations rely purely on downstream steering effects.
3. **Strand 3: Introspective Awareness & Self-Reports (AI Safety)**
   - *Key Papers:* Macar et al. (2026), Lindsey et al. (2025/2026), Lederman & Mahowald (2026), Singh et al. (2026).
   - *Takeaway:* Claims that models can detect injected concept vectors. Critiques suggest generic irregularity, but lack direct norm-matched noise controls.

> **🔊 Speaker Script (45 seconds):**  
> *"Our literature survey covers over 40 papers across three distinct strands. First, classical systems neuroscience literature on subspace estimation, which gave us our core mathematical tools. Second, modern AI interpretability papers on probing and persona vectors, which establish current extraction techniques but lack ground-truth validation. Third, recent frontier papers on model introspection. By bridging these three strands, we identify a clear gap: no one has systematically measured instrument error rates against known planted ground truth."*

---

### Slide 5: Literature Matrix & Comparative positioning

**Slide Header:** Where Prior Work Collides vs. Our Contribution

**Slide Content:**

| Prior Work / Paper | What Prior Work Does | What Prior Work Lacks | How CALIPER Differs (Our Novelty) |
|---|---|---|---|
| **Distill Circuits / Cacioli (2026)** | Tuning curves on vision models & number representations | No mathematical ground truth for input directions | Uses MLP input weight columns as **free, exact ground truth** inside real networks |
| **Anthropic Persona Vectors (2025)** | Persona vector extraction via difference of means | Validated only by steering effect (effect-validation flaw) | **Planted Personas (Latest Pivot):** Plants synthetic vectors $\mathbf{v}$ and tests true recovery $|\hat{\mathbf{v}} \cdot \mathbf{v}|$ |
| **Macar et al. / Lindsey (2026)** | Injected thought detection claims on Gemma 3 27B | Controls against **no injection** ($\alpha=0$) only | Adds **norm-matched content-free controls** (random Gaussian & shuffled vectors) |

> **🔊 Speaker Script (45 seconds):**  
> *"This table summarizes how CALIPER differs from prior work. Where vision tuning-curve papers work without ground truth, we exploit the fact that an MLP neuron's weight column is exact ground truth for its own layer. Where Anthropic validates persona vectors by steering effects, we plant known synthetic vectors and measure exact recovery. And where recent introspection papers control only against zero injection, we introduce norm-matched content-free controls to separate content detection from generic disturbance."*

---

### Slide 6: Workflow & System Architecture

**Slide Header:** Workflow: The Three-Level Preparation Architecture

**Slide Content:**

```
                                SYSTEM ARCHITECTURE
                                         │
     ┌───────────────────────────────────┼───────────────────────────────────┐
     ▼                                   ▼                                   ▼
LEVEL 1: UNIT PREPARATION           LEVEL 2: TRAIT PREPARATION          LEVEL 3: SELF PREPARATION
• Stimulus: Corpus text             • Planted persona vector v          • Concept injection (Gemma 3 27B)
• Ground Truth: MLP weights w       • Extract persona vector v_hat      • Controls: Random & Shuffle noise
• Measure: |w_hat · w|              • Check: |v_hat · v|                • Framing: Introspective vs Neutral
• Result: 77/100 pass rate          • Status: Sem 6 execution           • Result: Perturbation Alarm found
```

* **Core Subsystems:**
  - `caliper/activations.py`: Forward-hook engine for stimulus/response extraction.
  - `caliper/estimator.py`: Rank-$K$ bottleneck search with `fit_cascade` optimization.
  - `caliper/batched.py`: Shared-stimulus batched projection matrix math (20× speedup).
  - `experiments/`: Dedicated execution scripts with JSON/JSONL logging and checkpointing.

> **🔊 Speaker Script (60 seconds):**  
> *"Our workflow operates across three preparation levels. Level 1 evaluates single neurons using laptop CPU compute, where neuron weight vectors provide instant ground truth. Level 2 plants synthetic trait vectors into instruct models to test persona extraction tools. Level 3 runs on Kaggle T4 GPUs, auditing model self-reports under injected concepts versus norm-matched noise. The software pipeline is built around modular PyTorch hooks, batched matrix operations, and resumable checkpointing."*

---

### Slide 7: Calibration Rules & Methodological Innovations

**Slide Header:** Methodology: The Four Calibration Discipline Rules

**Slide Content:**
1. **Rule 1: Mandatory Null Distribution ($R^2$ Baseline)**
   - Measured random-direction null on correlated text: max $R^2 = 0.044$. Establishes a strict false-alarm detection threshold.
2. **Rule 2: Disagreement Flag (Warning Light Without Ground Truth)**
   - When direct fit and `fit_cascade` search disagree, the reading is flagged as unreliable ($p = 7.7 \times 10^{-4}$). Enables auditing real models where ground truth is unknown!
3. **Rule 3: Required Sample Complexity ($N_{\text{eff}}$ Table)**
   - Sample complexity must be indexed by active firing events, not token count (GELU units sit below zero 90%–99% of positions). Requires ~200 events for $K=1$.
4. **Rule 4: Pre-registered Discipline**
   - All criteria, hypotheses, and confidence intervals are committed before running experiments.

> **🔊 Speaker Script (45 seconds):**  
> *"To ensure our calibration is rigorous, we established four strict methodological rules. First, a random-direction null establishing that random noise explains at most 4.4% of activation. Second, a disagreement flag—our warning light—which detects when an estimator is silently wrong without needing ground truth. Third, sample complexity indexed by firing events rather than token count. Fourth, pre-registering every hypothesis before running the code."*

---

### Slide 8: Scope, Feasibility & Resource Planning

**Slide Header:** Scope, Feasibility, & Risk Management

**Slide Content:**
* **Hardware & Compute Feasibility:**
  - **Level 1 (Unit Level):** Laptop CPU / RTX 4070 (~1 minute per neuron fit).
  - **Level 3 (Self Level):** 2× Tesla T4 GPUs on **Kaggle Free Tier** (30 hours/week quota, 4-bit NF4 weights, float32 compute). Zero compute cost!
* **Dataset & Software Feasibility:**
  - Public domain text (Project Gutenberg), open-weight models (GPT-2, Gemma 3 27B, Qwen 2.5 32B). No proprietary data.
* **Risk Mitigation & Fallbacks:**

| Technical Risk | Likelihood | Impact | Mitigation / Fallback Plan |
|---|---|---|---|
| Estimator fails silently | High | High | `fit_cascade` fix + Disagreement Flag warning light |
| Multi-direction ($K \ge 2$) search degenerates | High | Medium | Sequential 1D extraction (project out & repeat) |
| Free GPU memory limits | Medium | Low | 4-bit NF4 quantization with float32 compute stability |

> **🔊 Speaker Script (45 seconds):**  
> *"Regarding scope and feasibility: Level 1 runs locally on CPU, while Level 3 runs on Kaggle's free GPU tier using 4-bit model quantization. We rely entirely on open-source models and public-domain data, requiring zero department or cloud budget. On the technical side, we have identified key risks—such as silent estimation failures and multi-direction degeneration—and built concrete fallbacks for each."*

---

### Slide 9: Team Work Division & Semester Roadmap

**Slide Header:** Semester 5 Plan & Team Work Division

**Slide Content:**
* **Team Track Division (4 Separable Ownership Tracks):**
  - **Member A (Lead Integrator & Estimator):** Core bottleneck estimator, `fit_cascade` implementation, Study 3 execution.
  - **Member B (Literature & Framing):** 60-paper literature matrix, neuroscience method map, related work chapters.
  - **Member C (Model Replication & Robustness):** Replicating Study 3 on Qwen 2.5 32B with trial-randomized prompts.
  - **Member D (Calibration & Benchmarking):** Evaluating false discovery rates on InterpBench planted-latent networks.

* **Semester Milestone Roadmap:**
  - **Review 1 (Today):** Problem definition, neuroscience framing, literature survey, team track assignment.
  - **Review 2 (Late Sept):** 4 Calibration rules, system design, compute specifications.
  - **Review 3 (Late Oct):** Prototype demonstration on GPT-2 neurons (77/100 pass rate & warning light demo).
  - **Review 4 (Mid Nov):** Study 3 Introspective results on 27B model & arXiv preprint submission.

> **🔊 Speaker Script (45 seconds):**  
> *"Our team of four is divided into distinct, parallel tracks: core estimation, literature synthesis, second-model replication, and calibration benchmarking. Our semester plan maps cleanly to the department's four reviews. By Review 3, we will demonstrate the live unit-level prototype with its measured error rate. By Review 4, we will present the full 27B introspection audit and arXiv preprint."*

---

### Slide 10: Expected Panel Q&A Cheat Sheet

**Slide Header:** Panel Q&A Quick Reference

**Slide Content:**
* **Q1: "Isn't this just standard probing or steering?"**
  - *Answer:* Probing and steering are the *tools being tested*. We are not inventing a 101st probe; we are building the calibration test bench that existing probes have never had.
* **Q2: "Why import systems neuroscience into computer science?"**
  - *Answer:* Systems neuroscience is the only discipline with 100 years of experience reading activity from complex units it didn't design. It learned that every instrument requires a ground-truth preparation.
* **Q3: "What is your main deliverable?"**
  - *Answer:* A calibrated instrument with a measured error rate, a disagreement warning light for uncalibrated models, an open-source evaluation suite, and a published arXiv preprint.
* **Q4: "What have you actually accomplished so far?"**
  - *Answer:* Characterized unit estimation on 100 neurons (found 23% silent failure rate + built warning light), reproduced published 27B introspection claims on Kaggle, and completed the norm-matched noise control paper draft.

> **🔊 Speaker Script (30 seconds):**  
> *"Thank you for your time. We are ready to take your questions."*
