# Capstone Project Specification

## Measurement Methods from Systems Neuroscience for the Interpretation of Language Model Internals: Adaptation, Calibration and Validation

**Department of Computer Science and Engineering, PES University**

Santosh Cheethirala (PES1UG24CS127) · Nivas Reddy Dandu (PES1UG24AM075)
C. Sree Krishna Koushik (PES1UG24CS128) · Marreddy Rushi Eswar Reddy (PES1UG24AM158)

---

# 1. Problem Statement

> **Interpretability research reads the internals of neural language models without an agreed
> measurement standard. Methods are proposed, adopted, and used in deployment audits before
> anyone establishes what they measure, how reliably, or how often they report structure that
> is not there. Systems neuroscience faced the same problem across four decades of recording
> from biological neurons and answered it by building a discipline of measurement — response
> characterisation, population-level analysis, explicit null models, and calibration against
> known ground truth. Most of that discipline has never been carried across.**
>
> **This project adapts, implements and validates a set of established neuroscience measurement
> methods for language models, and — critically — calibrates each one against artificial
> networks whose internal structure is known by construction, so that every reported number
> carries a measured error rate rather than an assumed one.**

### 1.1 Why the statement is scoped this way

The statement is deliberately at the level of a **programme**, not a single hypothesis. Three
reasons:

1. **It is a build, not a claim.** Instruments are not scooped by a better abstract. A specific
   hypothesis can be published by a competitor in six weeks; a calibrated measurement toolkit
   cannot.
2. **Every branch produces a result.** A method that transfers cleanly is a contribution; a
   method that fails to transfer, diagnosed precisely, is also a contribution.
3. **It can be appended to.** Sub-questions attach underneath without renegotiating the frame.
   Section 3 lists the ones we commit to now; Section 12 lists ones we may append later.

### 1.2 Extended statement (for appendix or expanded submission)

Two specific gaps motivate the programme.

**At the level of the single unit**, interpretability's central difficulty is *polysemanticity*:
a unit responds to many apparently unrelated inputs, so no single description characterises it.
The field's principal response has been the sparse autoencoder — a dictionary learned across a
whole layer under a sparsity penalty. Recent evaluations have found trained SAEs scoring 0.90
against 0.87 for constrained random baselines on interpretability, 0.72 against 0.69 on sparse
probing, and *losing* to random baselines on causal editing (0.72 vs 0.73); recovering 9% of
true features on synthetic data while reporting 71% explained variance; and exhibiting feature
splitting and absorption severe enough to require a purpose-built detector. Sensory neuroscience
faced the same problem — a V1 complex cell also responds to many stimuli — and answered it
differently: **per unit, estimate the low-dimensional subspace of the input space on which the
unit's response actually depends.** That measurement has never been made for a language model.

**At the level of the population**, interpretability routinely reads the residual stream by
taking its dominant directions — principal components, top singular vectors, high-variance
features. Systems neuroscience asked the corresponding question directly and got a surprising
answer: when V1 drives V2, the communicated subspace is **distinct from V1's own dominant
fluctuations**. Whether the directions a transformer component actually reads are the dominant
directions of the stream it reads from is, as far as we can determine, untested.

---

# 2. Background

### 2.1 What neuroscience built that interpretability has not

| Neuroscience practice | Interpretability equivalent |
|---|---|
| Full response characterisation over a stimulus range | Max-activating exemplar (the peak, curve discarded) |
| Explicit null models — shuffled, circularly shifted, random-direction | Rarely reported |
| Bias correction on information estimates | Not standard |
| Cross-validated dimensionality selection | Hyperparameter chosen by convention |
| Calibration against a system with known answer | Almost never possible in biology; **possible here and rarely done** |

The last row is the leverage. A neuroscientist can never enumerate the true features a cortical
population encodes. We can — in compiled transformers, in semi-synthetic networks with known
circuits, and in networks we train ourselves with structure we plant. **Every instrument in this
project is scored on such a network before it is pointed at a real model.**

### 2.2 Method-transfer landscape (surveyed August 2026)

Verified by arXiv metadata queries plus targeted search.

**Occupied — excluded from scope:** double dissociation (10 of 12 hits formally apply it);
lesion–symptom mapping (arXiv:2608.12717, August 2026); psychophysics/Weber/signal-detection
(≈20 pre-registered papers, March–May 2026); adaptation and context effects (AMEL,
arXiv:2605.22714); sparse coding; RSA/MVPA; latent and rotational dynamics (arXiv:2603.13259).

**Open — in scope:** maximally informative dimensions and subspace estimation (zero applications
to any modern deep network); communication subspaces / reduced-rank regression (zero on
transformers); spike-triggered covariance (zero; retained as a *failing* baseline); divisive
normalization as empirical analysis; repetition suppression as an internal probe; ideal-observer
analysis; critical-period/deprivation designs.

---

# 3. Research Questions

- **RQ1 — Dimensionality.** How many dimensions of its input does a language-model unit's
  response actually depend on, and how is that number distributed across layers, models and
  training time?
- **RQ2 — Adjudication.** Does the per-unit subspace dimensionality agree with what sparse
  dictionaries claim about the same units, and when they disagree, which is right on a network
  where the answer is known?
- **RQ3 — Routing.** Are the directions a transformer component reads the dominant-variance
  directions of the stream it reads from, or a selected subspace, and is the distinction
  causally real?
- **RQ4 — Calibration.** What is the false-discovery rate of each instrument on networks with
  enumerated ground truth, and what sample size does each require for a stated precision?

---

# 4. The Idea, in Detail

### 4.1 Workstream A — unit-level subspace measurement

**The method.** Maximally Informative Dimensions (Sharpee, Rust & Bialek, *Neural Computation*
2004) estimates, per unit, the low-dimensional subspace of stimulus space on which the response
depends. It maximises the KL divergence between the distribution of stimulus projections
conditioned on a response event and the unconditioned distribution:

```
I(v) = D_KL[ P_v(x | event) ‖ P_v(x) ],     x = s·v
```

extended to K dimensions by maximising over the joint projection onto a K-dimensional subspace.
It assumes nothing about the stimulus distribution — which is why it was built, since its
predecessor (spike-triggered covariance) fails on correlated non-Gaussian stimuli. Text is the
most correlated, most non-Gaussian stimulus ensemble available.

**The critical technical finding from our research pass — we do not implement the 2004
algorithm.** Williamson, Sahani & Pillow (2015) proved that MID is exactly maximum likelihood
for a linear-nonlinear model:

```
Î_ss(K) = (1/n_sp)[ L(θ; D) − L(θ₀; D) ]     ⟹     K̂_MID = K̂_ML
```

For a **continuous** activation `a` and stimulus `s`, this makes the estimator a **rank-K
bottleneck regression**:

```
minimise over V ∈ ℝ^(D×K), f:     Σ_t ( a_t − f(Vᵀ s_t) )²
```

— a multi-index model, differentiable, trainable with Adam, no histograms and no simulated
annealing. This matters enormously for feasibility, and it must be stated honestly in the
paper: **the estimator is a bottleneck probe; the contribution is the surrounding measurement
discipline** — the saturation criterion for selecting K, the model-free information benchmark,
the random-direction null, bias correction, and the calibration layer. We frame the work as
*bringing dimensionality-selection rigour to unit analysis*, not as *implementing MID*.

**The one favourable inversion, and it is the strongest argument for the project.** MID is
capped at K = 2 in neuroscience (K = 3 is the published ceiling) because **spikes are
expensive**. The literature gives two accounts of the cost, and the difference matters:

- *Pessimistic (Williamson et al.).* A histogram with m bins along each of K axes needs `m^K`
  parameters — exponential, and the reason MID "has usually been limited to one or two filters."
- *Empirical (Rowekamp & Sharpee 2011).* Measured convergence curves **collapse when plotted
  against `K·D / N_spike`** — i.e. the requirement grows **linearly, not exponentially**, in K.
  Their explanation: although the number of bins grows exponentially, **most bins are empty**,
  and strong stimulus correlations hold the number of *occupied* bins to a polynomial in K.
  Reasonable three-dimensional reconstructions are obtained "as long as the number of spikes is
  greater than the number of parameters needed to define the dimensions."

Text is a strongly correlated ensemble, so the empirical account is the one likely to apply.
The operational anchors: **error ≈ D / (2·N_spike)**; ~50,000 events recovered one dimension at
projection 0.92 with D ≈ 900; three dimensions cost roughly 3× one dimension.

**At our working point — D ≈ 64 after reduction, K = 3 — this implies on the order of 10³–10⁴
response events per unit for good recovery.** For a language model that is minutes of forward
passes. **The binding constraint that has capped this method for twenty-two years simply does
not bind here**, and quantifying that is E0.3's job.

**The one fatal trap, identified in advance.** For *uncorrelated* stimuli the information along a
random direction has the analytic form `I_random ≈ K/D` — negligible. But Sharpee et al. showed
that with strongly *correlated* stimuli at D ≈ 10³, *even a projection onto a random vector has a
high probability of explaining ~60% of the total information per spike*. The residual stream is
far more anisotropic than natural images. **Running this at raw d = 768 will produce a confident,
wrong, publishable-looking answer, and the naive check "my direction explains most of the
information" passes trivially.** Two mandatory gates follow:

1. ~~**Whiten and PCA-reduce the stimulus space to D ≈ 50–100 before estimation.**~~
   **CORRECTED BY MEASUREMENT (E0.1, run 18 Aug 2026).** PCA truncation destroys the signal it
   was meant to preserve. Measured on GPT-2 layer 6: MLP input-weight directions retain only
   0.352 of their norm in the top-64 PCA subspace, against 0.267 for *random* directions and an
   isotropic expectation of 0.289 — i.e. read-directions sit barely above chance relative to the
   residual stream's principal components. Retaining 80% of the true direction requires D ≈ 512,
   which is not a reduction. Whitening without truncation was also tried and is numerically
   unsound at d = 768: the covariance is near-singular, the transform amplifies noise in
   near-zero-variance directions, and recovery falls to 0.02 where synthetic data gives 1.00.
   **The protocol is raw space**, pending a properly conditioned whitening.
   *Stated honestly:* **no MID paper does this.** The literature controls D by cropping and
   downsampling the stimulus (128×128 → 32×32 → a 16×16 patch, in the canonical V1 study), and
   Sharpee's group explicitly declines regularisation in the MID pipeline. Our reduction is a
   deliberate departure, justified by the correlation problem above, and its effect on the
   estimate is reported as a sensitivity analysis rather than buried.
2. **Build the random-direction null distribution first** — 10⁴–10⁵ random unit vectors, the
   histogram of their information — and require the fitted value to sit outside it. Compare
   against the analytic `K/D` to quantify how far from the uncorrelated ideal our substrate is.

**Two further requirements the literature is emphatic about.** *Joint, never sequential,
optimisation*: greedy dimension-by-dimension search is biased under correlated stimuli, because
a direction can carry large information merely by correlating with an already-found one. On a
two-dimensional model this cost subspace projection 0.875 → 0.60 and information explained
90% → 63%. *Bias correction*: the finite-sample bias on the information estimate is
approximately `(N_bins − 1) / N_spike`, and jackknife correction is what buys the `N^(−1/2)`
convergence rate — it is mandatory, not optional.

**A second trap.** An MLP neuron's pre-activation is exactly `w·x`, so its dependence on its own
immediate input is one-dimensional *by construction* and estimating it is vacuous. The stimulus
space must therefore be an **earlier** representation — token embeddings or an early-layer
residual stream — so that genuine nonlinear computation separates stimulus from response. This
is pre-registered, not chosen after seeing results.

### 4.2 Workstream B — population-level routing

**The method.** Reduced-rank regression from a source population to a target population
(Semedo, Zandvakili, Machens, Yu & Kohn, *Neuron* 2019):

```
min_B ‖Y − XB‖²_F   s.t.  rank(B) = m
B_RRR = B_OLS V Vᵀ,   V = top-m PCs of the fitted prediction X·B_OLS
```

Note this is **not** the truncated SVD of `B_OLS` — the eigenproblem lives in prediction space.
Ridge-RRR (Wu & Pillow, arXiv:2512.12467) substitutes the ridge estimator into the same three
steps and recovered additional communication dimensions on the original V1–V2 data.

**The transformer adaptation, with the two design decisions that make or break it:**

- **The skip connection would trivialise a naive fit.** `h^(L+1) = h^L + a^L + m^L`, so `B = I`
  already explains most of the variance. **The target must be the component's *write*** — a
  single attention head's output `a_{L,h}` or an MLP's output `m_L` — not the residual stream
  itself. This is the exact analogue of "V2 spiking activity": the new thing that population
  does.
- **LayerNorm is linear except for one scalar per token.** Mean subtraction is the linear
  projection `(I − 11ᵀ/d)`; γ and β are affine; only `1/σ(x)` is nonlinear. **Defining the
  source population as the *post-LayerNorm* activation** removes the approximation entirely,
  because that is literally what the block reads. We report the pre-LN version too; the gap
  between them is itself a measurement of how much LayerNorm's gain matters with depth.

**Attention heads are the better target than MLPs.** An MLP's output is a deterministic function
of the residual stream at the *same* position, so `R²` measures only nonlinearity. An attention
head's output depends on all positions ≤ t, so a per-position regression is genuinely
misspecified in the same way a recording from a subsampled source population is — which is the
condition Semedo's method was designed for.

**The deepest problem, stated plainly.** Semedo's epistemics rest on shared *trial-to-trial
noise*: he subtracts the PSTH so that what remains is covariability beyond common stimulus
drive. **A deterministic transformer has no trial-to-trial noise.** Every bit of the target's
activity is a deterministic function of the input, so "coupling beyond common drive" is not
well-posed. Two substitutions, and we need both:

1. **Condition** — the (stimulus, time-bin) unit maps onto **(token identity, position)**.
   Subtract the per-token-type mean from source and target and fit on the residuals. This asks:
   *given the same token in different contexts, do context-driven fluctuations in the source
   predict context-driven fluctuations in the target?* It also kills the "you are just measuring
   token identity" objection, which is real at early layers.
2. **Intervene** — and this converts the method's weakest point into our strongest. Wu & Pillow
   warn explicitly that RRR "relies on correlation, not causation… it is entirely possible that
   the estimate reflects shared common input from a third region, or signals propagating in the
   reverse direction." **That warning does not apply to us.** We know the causal graph exactly
   and can inject along candidate directions. The RRR fit is a cheap observational screen;
   **activation patching along predictive versus private directions is the actual experiment**,
   and it is the causal version of Semedo's Figure 6 that he could never run. It reuses the
   injection machinery the team already built and validated in APERTURE.

**Prior art we must beat, named.** *Talking Heads* (Merullo, Eickhoff & Pavlick, NeurIPS 2024,
arXiv:2406.09519) already established low-rank inter-layer communication channels in
transformers — from **weight SVD**, at rank 1–2, with causal validation. **Our delta must be
source-side geometry restricted to the data manifold**: which directions of *actually occurring*
residual-stream variation drive a component, and how they sit relative to that population's own
variance structure. Weight SVD cannot give this because it is distribution-free. If we cannot
state that delta in one sentence, we do not start.

### 4.3 Workstream C — calibration

Every instrument is scored before use, on three substrates of increasing realism:

1. **Tracr** — compiled transformers, exact weight-to-function map. Clean-case correctness only;
   no superposition, most activations zero.
2. **InterpBench** — 86 semi-synthetic transformers with known circuits, trained via Strict
   Interchange Intervention Training, so superposition is present.
3. **Planted-latent networks we train ourselves** — small transformers trained on synthetic data
   with a **known number of generative latents**, so the true subspace dimensionality per unit is
   known by construction. This is the direct descendant of the ground-truth-manufacturing move
   the team executed in APERTURE, and it is the only substrate that gives ground truth for
   *dimensionality* specifically.

---

# 5. Experiments

Twenty-six experiments across five phases. Each states purpose, the research question it serves,
what it depends on, protocol, the statistic and its threshold, the expected result, **what we do
if the opposite happens**, and cost in person-weeks.

**Every experiment is marked CORE or EXTENDED.** An external audit measured our earlier plan at
roughly twice the team's realistic capacity, so the core set is the smallest programme that
constitutes Paper 1; extended items run only if core lands early. Nothing in the core set is
optional and nothing in it depends on an extended item.

### 5.1 Dependency structure

```
E0.1 analytic control ──► E0.2 null ──► E0.3 scaling ──┬──► Phase A (E1.*)
        │                     │                        │
        │                     └────────────────────────┴──► Phase B (E2.*)
        │
        └──► E0.5 STC baseline

E3.2 planted-latent recovery ──► gates every K reported in Phase A
E3.1 FDR on InterpBench      ──► gates every "tuned"/"narrow" claim
E2.5 weight-vs-data rank     ──► gates all of Phase B (kill check, month 6)
```

Three hard gates. **If E0.1 fails, nothing else runs. If E0.2 finds no detection threshold, the
unit-level workstream stops. If E2.5 shows the measurement is recoverable from weights, Phase B
stops.** Each is cheap and each is scheduled early, which is the entire point.

---

## Phase 0 — Does the instrument work at all? (months 0–3)

### E0.1 · Analytic positive control — a unit whose answer is known exactly · CORE

**Purpose.** Before any synthetic test or any real claim, verify the estimator on a case where
the true answer is known analytically, in a real model, with no ambiguity.

**The observation this exploits.** An MLP neuron's pre-activation is exactly `w·x`, where `x` is
its own layer's post-LayerNorm residual stream and `w` is that neuron's input weight row.
Therefore, *with respect to that stimulus space*, its dependence is **exactly one-dimensional and
the true direction is exactly `w`**. This was first noted as a trap — it makes naive application
vacuous. It is better used as a gift: **free, exact ground truth in a real network.**

**Protocol.** Stimulus = own-layer post-LN residual. Response = the neuron's post-GELU
activation. Run the full estimator, K = 1…3, blind to `w`. Sweep N from 10² to 10⁶.
Repeat across ≥500 live neurons spanning all layers of GPT-2 small and Pythia-160m.

**Statistics.** (i) `|v₁ · ŵ|`, the alignment of the recovered first direction with the true
weight, bootstrapped over data splits. (ii) Held-out information gain from K = 2 over K = 1,
which must be within the noise band. (iii) The empirical error-versus-N curve, compared against
the theoretical `D/(2N)`.

**Threshold.** `|v₁·ŵ| > 0.95` and no significant K = 2 gain, for ≥95% of tested neurons, at some
achievable N. That N is then recorded as the estimator's calibrated sample requirement.

**Expect.** Clean recovery; an empirical error curve matching `D/(2N)` up to a constant.

**If the opposite** — the estimator does not return K = 1, or does not recover `w`: **the
estimator is broken and nothing downstream is admissible.** Debug: check the joint-optimisation
implementation, the bias correction, the restart count, and whether GELU's saturation is
destroying the signal at the chosen activation threshold. This is a debugging gate, not a
research finding, and it is scheduled first precisely so that a broken estimator is discovered in
week two rather than month eight.

**Cost.** 2 person-weeks. **Gates: everything.**

**RESULT (18 Aug 2026, GPT-2 layer 6, 20 neurons, 20k positions).** Median alignment with the
known direction **0.9905**, median k=1 R² **0.9985**, median k=2 gain **−0.0008**, median restart
stability 0.9841. **Fraction passing 0.75 — verdict FAIL** against the pre-registered ≥0.95.

Follow-up E0.1b re-fitted the five failures with a 20× larger optimisation budget (10 restarts,
3000 steps). **One of five recovered** (n1859: 0.879 → 0.999, k=2 gain +0.061 → −0.000). The
other four are unmoved, converge reproducibly (restart stability 0.91–0.97), and have k=1 R²
well below 1.0 (0.543, 0.822, 0.935, 0.974) *even though the ground truth is exactly
one-dimensional*. **The estimator confidently returns a wrong answer on roughly a fifth of
neurons, and it is not a local-minimum problem.**

Working hypothesis under test (E0.1c): the loss, not the subspace search. Squared error on a
heavy-tailed response is dominated by a handful of extreme events, so the effective sample size
is the event count rather than the position count — which is precisely why the source literature
counts spikes rather than stimulus presentations. Failing neurons are about twice as kurtotic as
passing ones (median 66.5 vs 31.6) and fire about half as often (2.12% vs 3.86%), though at
n = 20 neither correlation is significant (p = 0.12, p = 0.16) and there are counterexamples in
both directions.

**Consequence for the programme.** E0.3's sample-complexity table must be indexed by *events*,
not positions, and every downstream experiment needs a per-unit effective-N figure reported
alongside its estimate. The gate has done its job: this would have been invisible on data where
the answer was unknown.

### E0.2 · Random-direction null across stimulus dimensionality · CORE

**Purpose.** Establish what "found something" means before finding anything.

**Depends on.** E0.1 (supplies known-signal units to compare the null against).

**Protocol.** Sample 10⁴–10⁵ random unit vectors in the stimulus space; compute the information
statistic for each; build the null histogram. Repeat at D ∈ {768, 256, 100, 64, 32}, both raw
and whitened. Compare against the analytic uncorrelated expectation `I_random ≈ K/D`.

**Statistics.** The 99th percentile of the null at each D, versus the value achieved by E0.1's
analytically-known units.

**Threshold.** After whitening and reduction to D ≈ 64, the 99th percentile of the null must sit
**below** the value achieved on known-signal units, with clear separation. If it does not, no
detection threshold exists at that D.

**Expect.** At raw D = 768 the null is wide — reproducing Sharpee's warning that a random vector
can explain ~60% of the information under strong stimulus correlation. After whitening and
reduction the null collapses toward `K/D`.

**If the opposite** — (a) the null is already tight at D = 768: excellent, the PCA step becomes
optional and we report that residual streams are better behaved than natural images, which is
itself a publishable observation about representation geometry. (b) The null stays wide even at
D = 32: the unit-level workstream is unmeasurable at this stimulus definition. Pivot the stimulus
space to token embeddings; if that also fails, **stop the workstream and reallocate to Phase B.**

**RESULT (E0.5, 19 Aug 2026, n=30).** Bussgang's theorem guarantees that for `a = f(w·s)` with
*Gaussian* `s`, ridge regression recovers `w` up to scale for any `f`, with no optimisation.

| method | median | min | fraction > 0.95 |
|---|---|---|---|
| spike-triggered average | 0.287 | 0.003 | 0.00 |
| decorrelated STA (Bussgang) | 0.528 | 0.013 | 0.03 |
| STC top eigenvector | 0.322 | 0.002 | 0.00 |
| **fitted bottleneck** | **0.995** | 0.575 | **0.87** |

One of thirty neurons is recovered by any closed-form method, against twenty-six of thirty for
the fitted estimator. **The instrument is necessary, not ornamental** — the strongest single
result of the pilot. Cause: the residual stream is strongly non-Gaussian, and these units sit
mostly in GELU's non-monotone region, driving the Bussgang constant `E[f'(z)]` toward zero.

**Cost.** 1 person-week. **Kill gate.**

**RESULT (E0.3a, 19 Aug 2026).** Rather than trust a theoretical constant, the recovery ceiling
was measured directly: synthetic units with a known direction, built on the *real* residual
stream, swept across firing rates from 0.5% to 80%.

| N_eff/D | 0.96 | 1.57 | 2.59 | 4.49 | 7.45 | 11.5 | 18.2 | 24.8 |
|---|---|---|---|---|---|---|---|---|
| ceiling | 0.996 | 0.976 | 0.978 | 0.996 | 0.996 | 0.996 | 0.997 | 0.998 |

**The ceiling is ≈0.98–1.00 everywhere, even with fewer informative positions than stimulus
dimensions.** There is no information limit in this regime, which refutes the hypothesis that
E0.1's failures were units too sparse to be recoverable. Real neurons failing at 0.46 against a
ceiling of 0.98 have genuine headroom. **This curve is also the E0.3 deliverable in miniature
and should have been measured before E0.1's pass criterion was written, not after.**

### E0.3 · Which scaling law governs correlated text · CORE

**Purpose.** Two published accounts of MID's sample cost disagree, and which one applies decides
how ambitious K can be.

**Protocol.** Synthetic units with known K ∈ {1,2,3,4} embedded in real residual-stream
statistics. Sweep N from 10² to 10⁷, D ∈ {32, 64, 100, 256}. **Plot subspace recovery against
`K·D/N`.** Separately count *occupied* histogram bins as a function of K — the mechanism the
linear account proposes.

**Statistics.** Curve collapse under the `K·D/N` rescaling, tested by comparing residuals against
a per-K free-scaling fit.

**Threshold.** Curves collapse (linear account, Rowekamp–Sharpee) or separate by orders of
magnitude in K (exponential account, Williamson). Report which, with the occupied-bin evidence.

**Expect.** Collapse, because text is strongly correlated and most bins are empty — implying
K = 3 reachable at ~10³–10⁴ events per unit.

**If the opposite** — exponential scaling on text: cap the programme at K ≤ 2. The headline
becomes *"how often is a unit's dependence genuinely one- or two-dimensional?"*, which is still a
first measurement and still adjudicates against sparse dictionaries. Either way this is known in
month 2 and everything downstream is scoped against it.

**Cost.** 3 person-weeks. **Deliverable regardless:** a required-N table for (K, D, precision),
which nobody has for language models.

### E0.4 · Retrodiction on a published feature, re-aimed · EXTENDED

**Purpose.** Recover something already known before claiming something new — *and* test the
premise, not only the instrument.

**Protocol.** Two parts. (a) Sweep day-of-week and month with ≥80 carriers per level; fit
circular response functions; check we recover the circular structure Engels et al. (ICLR 2025)
found via an SAE-based search. (b) **The part the earlier plan missed:** count how many units in
our affordable models carry the target variables at all, under a pre-declared effect-size floor.
The published unit-level evidence for numeric structure is at 8B; our compute is at GPT-2/Pythia
scale, and that gap must be measured, not assumed.

**Threshold.** Circular fit preferred over flat and monotone by cross-validated likelihood on
held-out carriers, p < 0.01 against the permutation null, for ≥1 unit per model. Plus a reported
count of variable-carrying units per model with its confidence interval.

**If the opposite** — no circular structure recovered: report as a methodological note. *The
published feature is not visible to direct sweeps*, which is genuinely informative about what
dictionary-discovered features are. If part (b) finds too few units, escalate the substrate up
the Pythia ladder and **budget the escalation now** rather than discovering the need in month six.

**Cost.** 3 person-weeks.

### E0.5 · Spike-triggered covariance as the failing baseline · CORE

**Purpose.** Demonstrate the method's own necessity, cheaply.

**Protocol.** Classical STC on the same units and stimuli. The literature predicts failure under
correlated stimuli: the stimulus covariance acquires outstanding eigenvalues that swamp the
significance analysis.

**Threshold.** Compare recovered subspace against E0.1's known answer. STC either recovers `w` or
does not.

**If the opposite** — STC works fine on text: **that is the better outcome.** STC is far cheaper,
better understood and easier to defend. We adopt it, save three months, and report that the
correlated-stimulus objection does not bind in this substrate. We are not attached to the
complicated method.

**Cost.** 1 person-week.

---

## Phase A — Unit-level measurement (months 2–9)

### E1.1 · Stimulus depth sweep — turning a nuisance parameter into a result · CORE

**Purpose.** The measured K depends on where the stimulus is defined. Rather than fixing that
choice and defending it, we sweep it and report the curve.

**Protocol.** For a response unit at layer L, define the stimulus at layer L′ ∈ {embedding, 0, 1,
…, L−1} and measure K(L, L′). E0.1 fixes the boundary condition: **K(L, L−1) = 1 exactly.**

**The object this produces.** A curve of *dependence dimensionality against computational
distance* — how many input directions a unit's response depends on as a function of how much
computation lies between stimulus and response. This is the language-model analogue of receptive
fields growing along a sensory hierarchy, and no one has it.

**Statistics.** K(L, L′) with CIs, per layer, averaged over units; slope of K against (L − L′).

**Expect.** K = 1 at distance 1 by construction, rising with distance, plausibly saturating.

**If the opposite** — K stays flat with distance: units depend on low-dimensional projections of
the input regardless of how much computation intervenes. That is a strong and surprising
statement about compositional structure, and a better result than the expected one.

**Cost.** 4 person-weeks. Subsumes what was previously a pre-registration caveat.

### E1.2 · Per-unit subspace estimation · CORE

**Depends on.** E0.1, E0.2, E0.3.

**Protocol.** Fit `a ≈ f(Vᵀs)` with rank-K bottleneck, K = 1…4, `f` in a smooth
radial/cylindrical basis, **all K dimensions optimised jointly, never greedily** — greedy search
is biased under correlated stimuli, costing subspace projection 0.875 → 0.60 and information
explained 90% → 63% on a published two-dimensional benchmark. Adam; ≥20 random restarts with the
distribution of pairwise `|vᵢ·vⱼ|` reported. K selected by cross-validated held-out information;
secondary criterion, saturation against the model-free benchmark. Early stopping on the Sharpee
rule (evaluate on a held-out eighth periodically; stop if held-out information drops 25% below
its maximum). Bin sweep m ∈ {5,10,15,20,40} with per-bin occupancy disclosed — any bin under ~20
events is reported, never silently smoothed. Jackknife plus the `(N_bins − 1)/N_spike` bias term
throughout.

**Threshold.** Per-unit K with CI, FDR-controlled, for ≥10³ units across ≥2 model families.

**If K does not saturate for most units** — information keeps rising with K: this is the
signature of genuine high-dimensional polysemanticity, but it is **indistinguishable from
insufficient data without E0.3**, which is exactly why E0.3 runs first. Report as *"K exceeds the
resolvable ceiling for X% of units at sample size N"*, with the ceiling stated. That is an honest
and publishable measurement of how severe superposition is.

**If K = 1 for essentially everything** — units are effectively single-feature at this stimulus
definition. That contradicts the superposition picture and is a strong, surprising result.

**Cost.** 8 person-weeks.

### E1.3 · The dimensionality distribution · CORE

**Protocol.** Distribution of K by layer, by model scale across the Pythia ladder, and across the
Pythia checkpoint ladder from step 0 to final.

**Threshold.** Stable across seeds and across two disjoint corpora; survives the E0.2 null.

**If the distribution is flat and featureless** across layers, scales and training: report it.
*"Unit dimensionality is invariant to depth, scale and training"* would be surprising and would
constrain a good deal of informal argument about hierarchical abstraction.

**Cost.** 3 person-weeks.

### E1.4 · Causal validation of the recovered subspace · CORE

**Purpose.** This is what separates a measurement from a curve fit, and the earlier plan did not
have it for the unit-level workstream.

**Protocol.** Having recovered a K-dimensional subspace `V` for a unit, ablate the stimulus's
projection onto `V` and measure the unit's activation. Compare three ablations at matched
dimension: (a) the recovered subspace, (b) a random K-dimensional subspace, (c) the top-K
principal components of the stimulus.

**Statistics.** Fraction of activation variance removed under each; paired across units; effect
size with CI.

**Threshold.** Ablating the recovered subspace removes substantially more activation variance
than either control, with the gap surviving correction across units.

**Expect.** Recovered ≫ PC-matched > random.

**If the opposite** — ablating the recovered subspace does not collapse the activation: **the
subspace is a statistical artefact and the measurement does not mean what it appears to.** This
is the arXiv:2311.17030 lesson (*Is This the Subspace You Are Looking For?*) applied to our own
method, and reporting it honestly is more valuable than the positive result. It would also
redirect the project toward causal rather than observational subspace identification.

**Cost.** 3 person-weeks. Reuses APERTURE's intervention machinery.

### E1.5 · Semantic readout — what is the subspace? · EXTENDED

**Purpose.** Bridge from a number to an interpretation. Without this the result is a statistic
that interpretability researchers cannot act on.

**Protocol.** Project each basis direction of the recovered subspace through the unembedding
(logit lens); retrieve maximally activating text along each direction and along random directions
within the subspace; test whether the individual basis directions are separately interpretable or
only the subspace as a whole is.

**Threshold.** Blind human rating of coherence for basis directions versus random within-subspace
directions versus random outside-subspace directions.

**If individual directions are uninterpretable while the subspace is causally real (E1.4)** — a
genuinely important negative: it says the interpretable-direction assumption fails even when the
subspace identification is correct. That result bears directly on every method that reads meaning
off individual directions.

**Cost.** 4 person-weeks.

### E1.6 · The kurtosis heuristic versus direct measurement · CORE

**Purpose.** Test a heuristic in active use, against a principled reason to doubt it.

**Protocol.** *Universal Neurons* identifies monosemantic units by high pre-activation skew and
kurtosis. Willmore & Tolhurst (2001) showed lifetime kurtosis is uncorrelated with population
sparseness, with Gabor codes as the counterexample: high lifetime kurtosis, low population
sparseness. Compute all five measures — lifetime kurtosis, lifetime Treves–Rolls sparseness,
population kurtosis, population Treves–Rolls sparseness, activity sparseness — plus measured K,
and correlate them, per layer, across the scale ladder.

**Threshold.** Correlation coefficients with bootstrap CIs; a pre-registered decision rule for
what counts as "the heuristic works."

**If the opposite** — kurtosis predicts everything well: **the field's heuristic is vindicated**
and now rests on an empirical basis rather than an assumption. That is a clean, useful result and
we report it that way. We are not invested in the heuristic failing.

**Cost.** 2 person-weeks.

### E1.7 · Response characterisation and selectivity · CORE

**Protocol.** Parametric sweeps on **two variables only** — the scope cap is deliberate — with
≥80 carriers per level, cross-validated across **held-out carriers** rather than held-out levels,
because the carrier is the confound that matters. Shape families: flat, monotone, Gaussian, von
Mises, spline. Within-carrier permutation null, ≥1000 permutations, Benjamini–Hochberg at
q = 0.05. **Tuned fraction reported as excess over the FDR-controlled null, never as a raw
count.** Forward-model SNR calibration: simulate known families at measured noise levels, run the
identical pipeline, publish the family × family confusion matrix by SNR decile. **Do not invert
the confusion matrix** — report stratified by SNR instead.

**If shape classification is unidentifiable at our SNR** — report the resolution floor: the
minimum SNR at which each family is separable. That floor is a reusable methodological result and
a reviewer would demand it anyway.

**Cost.** 5 person-weeks.

### E1.8 · The absorption-notch prediction · CORE, pulled to month 3

**Purpose.** A falsifiable prediction handed to the sparse-dictionary community — and, on three
independent audit assessments, the most quotable item in the plan.

**Protocol.** Feature absorption makes a general feature stop firing exactly where a specialised
child feature takes over. **A response profile swept across a graded variable should therefore
show a notch at those values.** Prediction: SAE latents show notched profiles where specialised
latents exist; raw MLP neurons do not at the same values. Detection by fitting with and without a
notch term and comparing on held-out carriers.

**Threshold.** Notch detection rate in SAE latents versus matched raw neurons, with the
false-positive rate established on E3.2's planted networks.

**If the opposite** — no notches: absorption has no response-profile signature, which bounds what
sweep-based methods can detect. One paragraph and a figure. **If notches appear in raw neurons
too**, absorption is not dictionary-specific — a more interesting finding than the original
prediction.

**Cost.** 3 person-weeks. **Scheduled at month 3, not month 14.** Parked deliverables do not
happen.

### E1.9 · Cross-seed universality of K · EXTENDED

**Protocol.** Do corresponding units across differently-seeded models have the same K? Uses the
established cross-seed correspondence procedure (activation correlation over a large token
sample).

**If K varies wildly across seeds** — K is a property of the particular solution, not of the
computation, which bounds how much anyone should read into any single model's number. Important
and cheap to establish.

**Cost.** 2 person-weeks.

---

## Phase B — Population-level routing (months 6–12) · gated on E2.5

### E2.5 · Data-manifold rank versus weight-derived rank · CORE for Phase B, run FIRST

**Listed first because it is the kill check.** An attention head reads through `W_V` and writes
through `W_O`, both rank ≤ d_head = 64, so a rank-64 bottleneck exists *by construction* and a
reviewer will say any measured rank is recoverable from the weights.

**Protocol.** Compare effective rank measured on activations against the architectural ceiling
and against weight-SVD rank.

**Threshold.** Data-manifold rank strictly and substantially below weight rank.

**If they coincide** — the measurement is recoverable from weights, Phase B's contribution
collapses to a validation exercise, and **we stop Phase B and reallocate to Phase A.** Cheap to
check, decisive, scheduled at month 6.

**Cost.** 2 person-weeks.

### E2.1 · Communication-subspace sweep · CORE for Phase B

**Protocol.** Source `X` = post-`ln_1` residual stream at layer L — defining the source as the
post-LayerNorm activation makes the read exactly linear, since LayerNorm is linear apart from one
scalar per token. Target `Y` = **a component's write** (a single head's output, or an MLP's
output), never the residual stream itself — the skip connection would make `B = I` explain most
of the variance and the result meaningless. Attention heads are the better target than MLPs: an
MLP's output is a deterministic function of the same position's residual, so `R²` would measure
only nonlinearity, whereas a head's output depends on all positions ≤ t, which is genuinely the
subsampled-source condition the method was built for.

Sweep all 144 head channels in GPT-2 small; replicate on Pythia-410m. ~2M tokens, position 0
excluded (attention-sink norms would dominate least-squares), **sequence-level CV folds only**
(token-level leaks, because tokens within a sequence are dependent). Ridge-RRR.

**The conditioning step.** Subtract the per-token-type mean from source and target and fit on
residuals — the analogue of PSTH subtraction. This asks: *given the same token in different
contexts, do context-driven fluctuations in the source predict context-driven fluctuations in the
target?* It also answers the "you are just measuring token identity" objection, which is real at
early layers. Report with and without.

**Rank selection.** Semedo's "smallest rank within one SEM of peak" rule **will break here**: with
n = 10⁶ and no trial-to-trial noise, fold-to-fold variance is minuscule and the rule selects an
absurd rank. We substitute a variance-explained threshold — smallest r reaching 99% of the
full-rank ridge R² — and state the substitution explicitly.

**Report per channel.** `R²(r)/R²_full-ridge` (the Semedo normalisation, non-negotiable, and what
makes the skip-connection objection moot); effective rank; **α_in**, the input alignment index
(1 = aligned to top PCs, 0.5 = random, 0 = anti-aligned); communication fraction.

**Cost.** 5 person-weeks.

### E2.2 · Are the read directions the dominant directions? · CORE for Phase B

**The framing that survives either outcome.** We do not ask *"do communication subspaces exist in
transformers?"* We ask **"are the dominant variance directions of the residual stream the
directions the model actually reads?"** A large fraction of interpretability practice implicitly
assumes yes, so both answers are publishable.

**Expect, honestly.** α_in near 1 for most channels — components read the dominant modes. That is
what was found *within* a cortical area, and what a shared-bus architecture predicts.

**If that is the finding** — *"unlike cortex, transformer inter-layer communication is not
selective; the residual stream is a broadcast bus, not a set of private channels."* A clean
negative with real consequences for whether PCA of the residual stream is a meaningful tool, and
for whether steering directions should generalise across downstream consumers. It must be
airtight across ≥3 models and all channels, and we budget it as a short paper.

**If a low-α_in tail exists** — heads reading small-variance directions — those channels go to
E2.4 and this becomes the headline.

**Cost.** 2 person-weeks.

### E2.3 · Control battery · CORE for Phase B

Within-block control (predict another head at the same layer from the same source, matched
dimensionality — the workhorse control); shuffle null (permute token order between X and Y,
expect R² ≈ 0); identity/skip baseline (B = I); PCR baseline (regress on top-k PCs — the
dominant-dimensions comparison).

**If the within-block control shows the same rank and alignment as the cross-block case**, the
method discriminates nothing in this substrate and Phase B stops. A two-week finding, not a
twelve-month one.

**Cost.** 3 person-weeks.

### E2.4 · Causal test — predictive versus private directions · CORE for Phase B

**Purpose.** The experiment the source literature could never run. Its authors warn explicitly
that reduced-rank regression "relies on correlation, not causation" and that shared input from a
third region or reverse-direction flow cannot be excluded. **That warning does not apply to us:
we know the causal graph exactly and can intervene.**

**Protocol.** Construct `Q`, the **Σ-uncorrelated** complement of the predictive subspace (via
SVD of `B̃ᵀΣ`; plain orthogonality is *not* correct here and using it is a silent error). Inject
`δ·v` at matched `‖δ‖` for `v` in the predictive subspace versus `v` in the private subspace.
Measure change in the target component's write, in downstream logits, and in KL divergence.

**Expect.** Private-direction injections near-inert for that component while remaining potent for
others.

**If patching contradicts RRR** — the subspace is an artefact and we report it as one. Read
arXiv:2311.17030 before running; cite it either way.

**Cost.** 4 person-weeks.

### E2.6 · Does the read subspace move across data distributions? · EXTENDED

**Protocol.** Fit the read subspace on prose; test on code, and vice versa.

**Either outcome is informative.** Stable → fixed wiring, and the subspace is a property of the
model. Shifts → dynamic routing, i.e. what a component reads depends on what it is reading about,
which would be the positive-surprise version of this workstream.

**Cost.** 2 person-weeks.

---

## Phase C — Calibration (concurrent, months 1–12)

### E3.1 · False-discovery rate on enumerated ground truth · CORE

**Purpose.** A measured error rate on a substrate where the true feature set is enumerated is,
per the audit, the single rarest artefact in interpretability. It converts every other criticism
from fatal to answered.

**Protocol.** On InterpBench (86 semi-synthetic transformers with known circuits) and Tracr:
measure FDR for "unit is tuned", precision and recall of recovered subspaces against known
structure, and shape-classifier accuracy against known functional roles. Matched random-graph
baselines throughout.

**If the instruments score poorly** — that is the most important result in the project and it
publishes as such. An instrument with a *measured* high error rate is more useful than one with
an assumed low error rate, and it is a direct rebuke of current practice.

**Cost.** 5 person-weeks. **Runs before the atlas publishes, not alongside it.**

### E3.2 · Planted-latent networks — ground truth for dimensionality · CORE

**Purpose.** InterpBench gives ground truth for *circuits*, not for *dimensionality*. This
supplies the missing substrate.

**Protocol.** Train 2–4 layer transformers on synthetic corpora generated from a known number of
latent factors, one-token-per-value vocabulary, so true per-unit K is known by construction. Vary
planted K ∈ {1,…,6}; multiple seeds per condition; measure recovery.

**Threshold.** Estimated K within ±1 of planted K for ≥80% of units; bias and variance reported
as a function of planted K and N.

**If recovery fails here, no result on real models is admissible.** This is the strictest gate in
the project.

**Cost.** 5 person-weeks.

### E3.3 · Training-trajectory null · CORE

**Purpose.** The control whose absence the audit rated as fatal.

**Protocol.** Full pipeline at Pythia step 0, at 6–8 log-spaced checkpoints, and at final, for
two sizes. Only the weight file changes.

**Expect.** Structure at step 0 will **not** be zero. Number tuning curves have been found in
randomly initialised networks (*Science Advances*, 2021), and BPE merge structure exists before
training.

**If step-0 structure is substantial** — that is a *better* figure, not a worse one: it separates
architectural prior from learned structure, and the emergence curve across checkpoints becomes a
headline rather than a control. A monotone training-time trend is far more convincing than any
cross-sectional correlation.

**Cost.** 3 person-weeks.

### E3.4 · Adversarial self-check — can we make the pipeline hallucinate? · CORE

**Purpose.** Deliberately try to make our own instrument report structure that is not there.

**Protocol.** Feed the full pipeline: (a) shuffled stimulus–response pairings; (b) a unit whose
activation is replaced by noise with matched marginal statistics; (c) a unit that is a pure
linear function of a random direction; (d) a unit from an untrained model. Report the
false-positive rate for each, and the value of every reported statistic under each.

**Threshold.** False-positive rate below the nominal FDR for every adversarial condition.

**If any condition produces apparent structure** — we have found the failure mode ourselves,
before a reviewer does, and we publish the correction alongside the method. This is cheap
insurance against the single most damaging outcome available.

**Cost.** 2 person-weeks.

---

## Phase D — Application (months 12–18, second paper) · EXTENDED

### E4.1 · Planted-prior manipulation

**Purpose.** Convert a correlational claim into an intervention, using the validated instrument.

**Protocol.** Train small transformers on corpora whose value-frequency distribution we set:
uniform, Zipfian, bimodal, and — the key control — a comb with teeth at **non-round** values.
Then *move* the prior (make 37 the most frequent value) and test whether the measured allocation
follows. Multiple seeds per condition; report seed variance.

**If allocation follows the planted prior** — a causal result, with tokenisation, base-10
structure and embedding-quality confounds dead by construction.

**If it stays at round numbers regardless** — it was never resource allocation, and we have found
something important about tokenisation and base-10 structure instead, which is a paper in its own
right.

**There is no branch of this experiment that produces nothing** — which is the property we
previously claimed, falsely, for the whole programme.

**Cost.** 10 person-weeks.

---

### 5.2 Cost summary

| Phase | Core | Extended | Core cost |
|---|---|---|---|
| Phase 0 | E0.1, E0.2, E0.3, E0.5 | E0.4 | 7 pw |
| Phase A | E1.1, E1.2, E1.3, E1.4, E1.6, E1.7, E1.8 | E1.5, E1.9 | 28 pw |
| Phase B | E2.5, E2.1, E2.2, E2.3, E2.4 | E2.6 | 16 pw |
| Phase C | E3.1, E3.2, E3.3, E3.4 | — | 15 pw |
| Phase D | — | E4.1 | — |
| **Total core** | | | **66 pw** |

Against a realistic 50–70 productive person-weeks over the programme, **the core set fits only if
Phase B is treated as conditional.** Explicit rule: Phase B runs only if E2.5 passes *and* Phase A
is on schedule at month 6. Otherwise Phase B moves to Paper 2 and the core drops to 50 pw, which
fits with margin.

---

# 6. How We Build It

### 6.1 Architecture

```
data/          corpus loaders; Pythia exact-stream reader; carrier generators
extract/       streaming activation extraction (online moments, quantile sketches,
               reservoir sampling of high-activation contexts) — never materialise
               the full activation tensor
stim/          stimulus-space construction: whitening, PCA, effective-sample-size
               accounting, position-0 exclusion
estim/         subspace estimator (rank-K bottleneck, JAX/PyTorch); STC baseline;
               ridge-RRR; alignment indices
nulls/         random-direction null; within-carrier permutation; shuffle; step-0
stats/         FDR control, jackknife bias correction, bootstrap CIs, forward-model
               SNR calibration
calib/         Tracr / InterpBench / planted-latent harnesses
causal/        activation patching along predictive vs private subspaces (APERTURE
               injection machinery, reused)
report/        every figure regenerable from a single command
```

### 6.2 Stack and provenance

| Need | Choice | Note |
|---|---|---|
| Activation access | TransformerLens / nnsight | Standard |
| SAEs | SAELens + Gemma Scope + GPT-2 SAEs | Public |
| Precomputed activations | Neuronpedia API (Pile-Uncopyrighted, 36,864 × 128-token segments) | Removes pipeline work — **for activations only** |
| Prior counting (Phase D) | `EleutherAI/pile-{standard,deduped}-pythia-preshuffled` + `utils/batch_viewer.py` | **Never** `pile-uncopyrighted` for the prior — it is a reduced Pile |
| Subspace estimator | Port `pillowlab/LNPfitting` CBF/RBF negative-log-likelihood + gradient into JAX | MIT, self-contained. **No JAX/PyTorch MID exists — this is a genuine open-source contribution** |
| RRR | `bichanw/RRR` (Python) | Ridge-RRR + alignment indices |
| Calibration | InterpBench, Tracr | Public |

**Explicitly excluded:** `sharpee/mid` — non-redistributable Salk licence, Linux-only 2012 C,
headerless binary I/O that fails silently into plausible garbage. `emdodds/MID` — imports a
module that exists nowhere; broken as published.

### 6.3 Engineering discipline

- **Streaming statistics only.** Caching every MLP activation for GPT-2 small over 4.7M tokens is
  ~10¹¹ floats. Running moments, quantile sketches and reservoir samples instead. Budgeted as
  real work in Phase A.
- **Neuron-index integrity.** Off-by-one in layer/neuron indexing is the classic month-14 killer.
  Every index carries a `(model, revision, layer, component, index)` tuple; assertions on every
  join; a regression test that recovers a known neuron by identity.
- **Determinism.** Fixed seeds, pinned model revisions, pinned dataset revisions, hashed configs.
- **Pre-registration.** Every stated prediction, threshold and decision rule committed to the repo
  with a timestamp before the corresponding data is collected. Directly answers the
  garden-of-forking-paths objection, which the audit rated as a primary rejection risk.
- **Test suite.** Following APERTURE's precedent (98 tests): synthetic units with known K must be
  recovered; nulls must return null; every estimator has a fixture with a known answer.

### 6.4 Compute

Consumer GPU throughout; inference only, apart from Phase C/D small-model training (2–4 layers,
hours). RRR is a single 768×768 eigendecomposition. The full 144-channel sweep is hours. Free
cloud GPU suffices for the training arms.

---

# 7. Success Metrics

### Tier 0 — the instrument works (gate; by month 3)

| Metric | Threshold |
|---|---|
| Random-direction null documented at every D | Required |
| Fitted statistic outside the null | p < 0.01 for ≥1 unit |
| Restart reproducibility, pairwise \|v_i·v_j\| | Median > 0.9 across ≥20 restarts |
| Train/test information gap | Held-out information > 0 with CI excluding 0 |
| Bin-count sensitivity | Plateau, not monotone increase, over m ∈ {5…40} |
| Recovery on planted-latent networks | Estimated K within ±1 of planted K for ≥80% of units |
| Retrodiction (E0.1) | Circular fit preferred, p < 0.01 |

**If Tier 0 fails, the project pivots at month 3 rather than month 14.** That is the point of it.

### Tier 1 — minimum publishable (by month 11)

- Per-unit dimensionality for ≥10³ units, ≥2 model families, FDR-controlled.
- A measured false-discovery rate for every headline claim, from E3.1.
- Sample-complexity table (N for given K, D, ε) — reusable regardless of other outcomes.
- Step-0 baseline reported beside every quantity.
- Open-source JAX subspace-estimation library with tests.

### Tier 2 — strong paper

- A quantified agreement (or disagreement) rate between measured dimensionality and SAE claims,
  adjudicated on InterpBench.
- The kurtosis-heuristic verdict, with all five sparseness measures.
- The absorption-notch result, either direction.
- The α_in distribution across ≥144 channels and ≥2 models.

### Tier 3 — best case

- A causally confirmed private direction: large residual-stream variance, Σ-orthogonal to a
  component's predictive subspace, injection leaves that component inert while strongly moving
  another. **Selective routing in a transformer, demonstrated causally.**
- A dimensionality emergence curve across the Pythia checkpoint ladder.
- The planted-prior manipulation landing (E4.1).

### Process metrics (assessed throughout)

Pre-registrations filed before data collection · every figure regenerable by one command · test
count and pass rate · runs logged with config hashes · weekly written memo.

---

# 8. Risk Register

| Risk | Severity | Mitigation |
|---|---|---|
| Estimation at raw D = 768 returns confident garbage | **Highest** | E0.2 null + mandatory whitening/PCA. Scheduled first as a kill gate |
| K does not saturate; cannot distinguish polysemanticity from insufficient data | High | E0.3 calibration runs before E1.2; report the resolvable ceiling |
| Workstream B recoverable from weights (*Talking Heads* overlap) | High | E2.5 as an early kill check; lead with data-manifold restriction and α_in |
| Token autocorrelation inflates every significance test | High | One position per sequence, or explicit ESS discount; sequence-level CV only |
| Scope overrun (audit measured 2× capacity) | High | Two stimulus variables, not six. Workstream B gated on E2.5. Phase D is a second paper |
| Attention sinks dominate least-squares fits | Medium | Position 0 excluded; norm-based outlier filter |
| Subspace-patching illusion (arXiv:2311.17030) | Medium | Read before running E2.4; report the illusion check |
| No author has a top-venue publication record | Medium | Recruit a co-author who has been through the review process — flagged independently by two audit lenses as the largest single predictor |
| Competitor velocity in the adjacent lane | Medium | Instrument framing is scoop-resistant; date-stamped preprint on Tier-1 results |

---

# 9. Timeline and Team

| Months | Phase | Gate |
|---|---|---|
| 0–3 | Phase 0 + E3.2 planted-latent harness | **Tier 0 gate** — pass or pivot |
| 2–9 | Phase A (E1.1–E1.7); E1.7 pulled to month 3 | Dimensionality distribution exists |
| 1–12 | Phase C calibration, concurrent | FDR measured before atlas publishes |
| 6–12 | Phase B, gated on E2.5 | α_in distribution |
| 11–14 | **Paper 1** written and submitted | — |
| 12–18 | Phase D — planted-prior manipulation | **Paper 2** |
| 18–22 | Thesis assembly, artefact release | — |

**Work split.** (1) Extraction pipeline, streaming statistics, stimulus-space construction.
(2) Subspace estimator, JAX port, null models, restart/stability diagnostics. (3) Reduced-rank
regression, alignment indices, causal patching. (4) Calibration harnesses — Tracr, InterpBench,
planted-latent training — plus statistics, FDR, pre-registration and reproducibility.

Each member owns a component with a separable, defensible contribution.

---

# 10. Key References

**Unit-level.** Sharpee, Rust & Bialek, *Analyzing neural responses to natural signals:
maximally informative dimensions*, Neural Computation 16(2), 2004 (arXiv:physics/0212110) ·
Williamson, Sahani & Pillow, *The equivalence of information-theoretic and likelihood-based
methods for neural dimensionality reduction* (arXiv:1308.3542) · Paninski, *Convergence
properties of some spike-triggered analysis techniques*, 2003 · Fitzgerald et al., PLOS Comp
Biol 2011 · `pillowlab/LNPfitting` (MIT).

**Population-level.** Semedo, Zandvakili, Machens, Yu & Kohn, *Cortical areas interact through a
communication subspace*, Neuron 102(1), 2019 · Wu & Pillow, *Reduced rank regression for neural
communication* (arXiv:2512.12467) · `bichanw/RRR` · Merullo, Eickhoff & Pavlick, *Talking Heads*
(arXiv:2406.09519).

**Calibration.** Lindner et al., *Tracr* (arXiv:2301.05062) · *InterpBench* (arXiv:2407.14494) ·
Biderman et al., *Pythia* (arXiv:2304.01373).

**Motivation.** *Sanity Checks for Sparse Autoencoders* (arXiv:2602.14111) · *Automated
Interpretability Metrics Do Not Distinguish Trained and Random Transformers* (arXiv:2501.17727) ·
Chanin et al., *A is for Absorption* (arXiv:2409.14507, NeurIPS 2025) · Huang et al., *Rigorously
Assessing Natural Language Explanations of Neurons* (BlackboxNLP 2023) · Gurnee et al.,
*Universal Neurons in GPT2* (arXiv:2401.12181) · Willmore & Tolhurst, *Characterizing the
sparseness of neural codes*, Network 12(3), 2001 · Kim, Jang, Baek, Song & Paik, *Visual number
sense in untrained deep neural networks*, Science Advances, 2021.
