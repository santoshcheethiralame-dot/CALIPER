# Project CALIPER: Comprehensive Overview & Status Report

> **Simple Summary:**  
> When scientists try to read what is happening inside an Artificial Intelligence (AI) model's "brain", how do they know their tools are actually accurate? **CALIPER** is a project that adapts measurement principles from **systems neuroscience** to calibrate interpretability tools. It tests these tools where the exact answer is already known before using them on complex AI safety problems.

---

## 1. What is the Project About?

### The Core Problem in AI Interpretability
Imagine a doctor using an eye chart to test your vision. If the eye chart is printed blurry or upside down, the doctor might diagnose you with bad eyesight when the chart itself was broken. 

In AI research today, scientists try to look inside Large Language Models (LLMs) to extract "concept directions"—specific vectors in the AI's internal memory that represent ideas like "sycophancy", "honesty", or "apple". However, most researchers adopt these measurement tools without ever checking:
1. **Do these tools measure what we think they measure?**
2. **How often do they report finding something that isn't actually there?**
3. **If a tool gives a wrong answer, can we tell?**

### The Neuroscience Solution
Decades ago, **systems neuroscience** faced the exact same challenge when reading brain signals from biological neurons. Neuroscientists solved it by establishing a strict **discipline of measurement**:
- **Calibration against known ground truth** (testing the instrument where the exact physical signal is known).
- **Null models** (testing what random noise looks like to establish a baseline).
- **Response characterisation** (mapping the limits and sample demands of the tools).

**CALIPER** brings this exact neuroscience discipline into AI.

---

## 2. The Ground Truth Trick Inside Language Models

How do you get "ground truth" inside a deep AI network where everything seems like a black box?

In a Transformer model (like GPT-2 or Gemma), an MLP neuron's input weight column is **exact mathematical ground truth** for its own layer's inputs. 
- For that specific layer, the true direction that activates the neuron is known perfectly (correlation = 1.000000).
- This gives us a **free, exact calibration standard** inside real, working neural networks.

**The CALIPER Rule:** An instrument must pass the test on this free ground truth *first*. Only after it proves it can reliably measure known directions does it move to earlier layers where complex computation happens and the true answer is unknown.

---

## 3. Two Major Research Tracks

Project CALIPER is split into two major interconnected lines of work:

```
                          PROJECT CALIPER
                                 │
         ┌───────────────────────┴───────────────────────┐
         ▼                                               ▼
     TRACK 1                                         TRACK 2
 CALIPER Instrument Calibration                 Study 3 / Paper A
 (Phase 0 & Phase A)                            (AI Introspection Audit)
 • Calibrating subspace search                  • Testing "Thought Detection" claims
 • Fixing silent failure modes                   • Content vs. Perturbation controls
 • Required sample size tables                  • Reading out model states reliably
```

---

## 4. Key Discoveries & Results So Far

### Track 1: Instrument Calibration (Phase 0)

1. **Standard Search Tools Fail Silently on ~20% of Neurons:**
   - **The Problem:** When searching for 1-dimensional directions, standard tools got stuck on a **confidently wrong answer on ~20% of real neurons**.
   - **The Danger:** Optimization restarts agreed with each other (85%–99% confidence), giving **zero warning** that the answer was wrong!
   - **The CALIPER Fix (`fit_cascade`):** CALIPER discovered that fitting 2 dimensions first (where the landscape is friendly) and then searching inside that 2D space recovers the true 1D direction. Recovery rate jumped from **45.2% → 99.9%**!

2. **Common Evaluation Metrics Lie:**
   - Researchers usually judge fit quality using "held-out $R^2$" (how well the model predicts test data). CALIPER proved that a slightly wrong direction can absorb non-linearities and score a higher $R^2$ than the true direction. Testing held-out $R^2$ alone is **not sufficient**.

3. **Classical Shortcuts Fail on AI Activations:**
   - Simple math shortcuts (like Spike-Triggered Averages or decorrelated linear regression) fail completely—recovering only 1 out of 30 neurons (3%), compared to 26 out of 30 (87%) for CALIPER's fitted tool. Why? Because neural network activations are non-Gaussian and GELU activation functions cause signals to cancel out.

4. **False Alarm Threshold Established:**
   - On correlated text data, random directions explain at most **4.4%** ($R^2 = 0.044$) of neuron activation, whereas real directions explain 70%–98%. This establishes a clear false-alarm threshold for future claims.

5. **Multi-Direction Search Degenerates:**
   - Trying to fit multiple directions ($K \ge 2$) simultaneously caused the algorithm to recover 1 direction perfectly and completely miss the rest. Multi-direction search must be done iteratively (find 1 direction, project it out, then find the next).

---

### Track 2: AI Introspection & Thought Detection Audit (Study 3)

A headline 2026 paper (*Macar et al.*) claimed that Gemma 3 27B could **"detect when a thought/concept vector was injected into its activations"** with a 10.8% detection rate and 0% false positives, calling it evidence of AI introspective awareness.

CALIPER ran a pre-registered audit on Gemma 3 27B on Kaggle GPUs:

1. **The Effect Reproduces:**
   - CALIPER reproduced the published detection effect: 2/30 = 6.7% (Wilson 95% CI 1.8-21.3%) vs the published 10.8%, with 0/30 false positives.
     *(Corrected 7 Oct 2026 per notebook C40: the pre-registered scorer gives 2/30. The 10.0% figure was a hand-read 3/30 and is withdrawn.)*

2. **The "Bump / Perturbation Alarm" Discovery (Headline Result):**
   - **The Test:** CALIPER injected **content-free noise vectors** (random Gaussian vectors and shuffled vectors) of the exact same size.
   - **The Finding:** The model said *"YES, I detect an injected thought"* **just as often for random noise as for real concept vectors!**
   - **What it means:** The model isn't "reading the thought content" inside its mind—it is simply reacting to being bumped/disturbed!

3. **Self-Directed Prompts Inflate Panic:**
   - Asking the model self-directed questions ("Do you detect an injected thought?") acts like a suggestion that inflates the false alarm rate 8-fold (+18.4% vs +2.3%). A neutral prompt ("Is anything unusual present?") cleanly separates actual concepts from mere noise.

4. **The Reported Rate Depends On How You Read It:**
   - Changing the readout method (asking for generated text vs. checking the first-token probability) caused the reported detection rate to swing from **7% to 50%** on the exact same model and injection strength!

---

## 5. How Much is Done? (Project Status Breakdown)

| Project Phase / Milestone | Description | Status | Completion % |
|---|---|---|---|
| **Phase 0: Instrument Calibration** | E0.1–E0.5 synthetic controls, random null, recovery ceiling, classical baselines, cascade fix. | **DONE** | **95%** |
| **Phase 0: Batched Engine** | Re-architecting matrix operations across shared stimuli (4.5×–20× speedup). | In Queue | **30%** |
| **Study 3 / Paper A: GPU Runs** | Kaggle GPU runs on Gemma 3 27B (real, random, shuffle controls, prompt variants). | **DONE** | **100%** |
| **Study 3 / Paper A: Analysis** | Data analysis scripts, statistical tests (Wilcoxon), figures generation. | **DONE** | **100%** |
| **Study 3 / Paper A: Manuscript** | October arXiv preprint draft (`paper/main.tex` & `docs/paper-s3-draft.md`). | **DONE** | **90%** |
| **Phase A: Subspace Depth Sweep** | Measuring dimensionality degradation across network depths. | Next Up | **0%** |
| **Planted Personas Study** | Testing persona extraction tools using synthetic planted directions. | Planned | **0%** |

---

## 6. Immediate Next Steps & Publication Roadmap

1. **October 2026 arXiv Release (Paper A):**
   - Target: Release the Study 3 thought detection audit preprint titled *"What does a language model detect when it detects an injected thought?"*
2. **Batching Implementation:**
   - Complete `caliper/batched.py` to accelerate neuron sweeps by 20×.
3. **Paper 1 (Phase 0 Methods):**
   - Submit CALIPER's neuroscience calibration findings to an interpretability venue (BlackboxNLP / NeurIPS workshop).
4. **ICLR 2027 Workshop Submission:**
   - Submit the refined Study 3 paper to an ICLR 2027 workshop in early 2027.

---

## 7. Summary Table of Key Concepts

| Concept | What It Means (Simple Terms) |
|---|---|
| **Subspace Vector** | A direction in AI memory representing an idea, concept, or trait. |
| **Ground Truth** | Knowing the 100% true, unarguable answer in advance to test your tools. |
| **Silent Failure** | When a tool gives a completely wrong answer but claims high confidence. |
| **Fit Cascade** | CALIPER's fix: searching 2D space first to cleanly find the 1D direction. |
| **Perturbation Control** | Testing random noise of the same size to see if the AI is just reacting to a "kick". |
| **First-Token Readout** | Measuring the AI's instant logit probability before it generates text. |
