# CALIPER Lab Notebook

Append-only record of every run — done and planned — its config, and what it
showed. Single source of truth for experimental history. Update after every run
or finding. Newest entries go at the top of each section; never rewrite a past
entry, only add a dated follow-up beneath it.

**Definition of done for a run.** Config recorded + seed logged + raw artifact
archived under `data/` or `results/` + a one-paragraph note in §4 + a row in the
§3 registry. A run that cannot be reproduced from its recorded config did not
happen. For Kaggle runs, add: the version stamp printed in the first output line,
and the artifact downloaded off the session before it expired.

**Scope.** Started 17 Aug 2026. This notebook opens 6 Sep 2026 and back-fills
everything run before that date from `results/*.json`, `data/s3/*.jsonl`, and the
session record. Back-filled entries are marked *(reconstructed)* where the
original config was recovered from the output file rather than logged at the time.

---

## 1. The project in one page — read this before anything else

CALIPER asks whether the instruments used to read a language model's internals
are correct, by finding places where the true answer is knowable in advance and
scoring the instrument against it. Three levels, three sources of ground truth:

| Level | Readout under test | Ground truth | Status |
|---|---|---|---|
| **L1 — unit** | Rank-K subspace estimator (maximally informative dimensions) | **Free** — an MLP neuron's input weight column *is* its direction for its own layer | Phase 0 closed. Instrument recovers 77/100; the other 23 fail silently |
| **L2 — trait** | Difference-of-means persona vectors | **Planted** — inject a known direction | Not started. Blocked on K≥2 |
| **L3 — self** | Concept-injection introspective report | **Planted** — inject a known state | Complete. Detection is mostly a perturbation alarm |

The through-line is not the three levels; it is one calibration discipline
applied at each, and every study must report the same four quantities:

1. a **random / content-free null** — what does the readout return when there is
   nothing to find?
2. a **required-N table** — how much data for a stated precision, indexed by
   informative events, not positions;
3. a **pre-registered pass criterion with a confidence interval**, filed before
   the run;
4. a **disagreement flag** — two estimation routes; where they diverge the
   readout is unreliable, and this needs no ground truth, which is why it is the
   deliverable a practitioner can actually use.

**Publication order (fixed as of 3 Sep 2026).**

| | Content | When | Target |
|---|---|---|---|
| **Paper A** | Study 3 alone — the content-free control for injection detection | arXiv ~1 Oct 2026 | preprint, then ICLR 2027 workshop Feb |
| **Paper B** | Flagship: Studies 1 + 3 under the "preparation" framing, Study 2 if it lands | Jan–Feb 2027 | ICML/ACL 2027 main, BlackboxNLP fallback |
| **Paper C** | Phase A: what the calibrated instrument measures in unknown units | Sep 2027 | ICLR 2028; **conditional on the E1.4 gate** |

Two papers are committed. The third exists only if E1.4 clears in sem 6.

---

## 2. Environments & reproduction

### 2.1 Local (CPU) — all of Phase 0

- Windows 11, Python 3.12. `pytest` suite in `tests/` (estimator, batched, runtime).
- Model: `gpt2` (small), layer 6, d_model 768, d_mlp 3072. Downloads on first run.
- Stimulus: post-LayerNorm residual at `block.ln_2`. Response: pre-activation at
  `block.mlp.c_fc`. Corpus cached in `results/corpus_cache/` (Gutenberg 11, 84, 1342).
- `datasets` **hard-crashes the process** on this machine — `sample_corpus()`
  defaults to the cached Gutenberg files for that reason. Do not "fix" it back.
- Measured throughput: **63.7 s per neuron** at the gate config (20k tokens,
  d=768, 3 restarts, 2500 steps, k=1 + k=2, single-threaded). The dual
  direct+cascade protocol roughly doubles it.

### 2.2 Kaggle (GPU) — Study 3 and everything at ≥7B

The recipe that finally worked after four failed rounds. Deviate at your peril.

- Accelerator **GPU T4 x2**, Internet **On**. One card is not enough: a 32B-class
  model in 4-bit needs ~20 GB and one T4 has 15.6 GB.
- **Ship the script as a Kaggle Dataset, never paste it into a cell.** Pasting a
  400+ line file into a notebook cell failed four consecutive times (the tail of
  the script landed inside the pip line once). Upload `.py` as dataset
  `caliper-s3`; patch by uploading a New Version; the notebook never changes.
- Attach **two** inputs: the model under *Models* (Gemma 3 27B — this is what
  makes it load with no HF token and no download) and `caliper-s3` under
  *Datasets*. The loader scans `/kaggle/input` for a folder containing
  `config.json`, so the `.py` can never be mistaken for the model.
- Cell 1 (only after a session restart):
  ```
  pip install -q -U bitsandbytes accelerate transformers
  ```
- Cell 2, the whole notebook:
  ```python
  import sys, glob
  sys.argv = ["run", "--model", "gemma", "--compute-dtype", "fp32", ...]
  hits = glob.glob("/kaggle/input/**/*.py", recursive=True)
  print(hits)
  exec(open(hits[0]).read())
  ```
  The glob exists because Kaggle silently renames the upload folder.
- **The first output line is a version stamp** (`kaggle_s3_positive_control
  2026-09-03b`). If it is not the version you just uploaded, the dataset did not
  refresh — remove and re-add the input. Nothing after a stale stamp is worth
  reading. This one line has saved more time than any other change.
- **fp32 compute is mandatory on Gemma.** See §8.
- **Restart the session before every reload.** See §8.
- Download every `.jsonl` off the session before it dies, and copy it into
  `data/s3/`. A Kaggle output that expired is gone.

### 2.3 Reproduction commands

```bash
# Phase 0, any experiment
python experiments/e01_gate.py --neurons 100 --tokens 20000 --restarts 3

# Study 3 analysis, all of it, no GPU
python experiments/analyse_s3_full.py
python experiments/rescore_s3.py
python paper/make_figures.py

# Paper build
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

---

## 3. Run registry

Chronological. `C`-numbers are notebook IDs; `E`/`S`/`P` names are the experiment
families from the specification and the merged design.

### Phase 0 — L1, unit-level estimator (local CPU, GPT-2 small, layer 6)

| ID | Date | Exp | n | Config | Artifact | One-line result |
|---|---|---|---|---|---|---|
| C1 | 2026-08-17 | E0.2 random null | 12 units / 2000 dirs | 20k tok, d=768 | `results/e02.json` | Null alignment median 0.0245, p99 0.0913; **detection threshold R² = 0.0443** vs true 0.92 |
| C2 | 2026-08-17 | E0.3a recovery ceiling | 24 cells | frac_active 0.005→0.8, 3 seeds | `e03a_recovery_ceiling.json` | Alignment 0.946–0.999 at every sparsity — **no information ceiling**; failures are optimisation, not data |
| C3 | 2026-08-18 | E0.5 classical baselines | 30 | STA / decorrelated STA / STC vs fitted | `e05_classical_baselines.json` | STA 0/30 above 0.95, dSTA 1/30, STC 0/30, **fitted 26/30** |
| C4 | 2026-08-18 | E0.1 v3 pilot | 20 | 20k tok, 3 restarts, 2500 steps | `e01_v3.json` | 80% pass, median align 0.9958, **min 0.3961** — first sight of silent failure |
| C5 | 2026-08-18 | E0.1b failure mode | 5 | re-fit failures | `e01b_failure_mode.json` | n1859 0.879→0.999 recovers; **n2977 stays 0.46** — two distinct failure classes |
| C6 | 2026-08-18 | E0.1c heavy tail | 3 | raw / log1p / rank response transform | `e01c_heavy_tail.json` | Rank transform lifts n2977 0.447→0.884, n230 0.942→0.985. Kurtosis 31–549 |
| C7 | 2026-08-18 | E0.1e nonlinearity | 8 | gelu/relu/softplus/identity | `e01e_nonlinearity.json` | Failures all sit at **z_mean −1.3 to −1.9** (deep in GELU's non-monotone region); softplus/identity recover them |
| C8 | 2026-08-18 | E0.1f warm start | 9 | init at STA | `e01f_warm_start.json` | **No gain** — n2977 cold 0.4631, warm 0.4596. Not an initialisation problem |
| C9 | 2026-08-18 | E0.1g oracle | 6 | R² evaluated at the true direction | `e01g_oracle.json` | **R² = 1.000 at the truth for every failure**; true direction lies in the k=2 subspace at 0.996–0.999. A perfect solution exists and is reachable |
| C10 | 2026-08-18 | E0.1h cascade | 8 | fit k+1, descend into it | `e01h_cascade.json` | **n2977 0.4525→0.9995, n230 0.8656→0.9987**; mild regression on 3 already-good units |
| C11 | 2026-08-18 | E0.1i selection rule | 9 | pick direct vs cascade by held-out R² | `e01i_selection.json` | Rule picks correctly with wide margins; **n1989 is the degenerate case** (margin 0.0148, regret 0.043) |
| C12 | 2026-08-19 | E0.3b operating point | timing grid | tokens × steps × restarts | `e03b_operating_point.json` | 4k/800/2 is the cheap knee: 4.7 s/neuron, median 0.9936 |
| **C13** | **2026-08-19** | **E0.1 GATE, n=100** | **100** | pre-registered, 20k tok, 3 restarts | `e01_gate_summary.json`, `e01_gate.jsonl` | **77/100, Wilson 95% [0.679, 0.842] — FAILS its own ≥0.90 bar.** Median align 0.9933, **min 0.1445**, 31% of units disagree >0.05 across methods |
| C14 | 2026-08-20 | E0.3 required-N | K∈{1,2,3} × N | events-indexed | `e03_required_n.json` | **K=1 saturates at ~200 events** (median 0.9979 at N=2000). **K=2 degenerates**: 0.24→0.89; additive K=2 pinned at **0.50**, K=3 at **0.36** — the one-direction-recovered signature |

### Study 3 — L3, self readout (Kaggle, Gemma-3-27B-it, 4-bit NF4, fp32 compute, 2×T4, layer 37 of 62 = 0.60 depth, 30 concepts)

| ID | Date | Stage | α | n | Artifact | One-line result |
|---|---|---|---|---|---|---|
| C15 | 2026-09-02 | generation, introspective | 0,2,4,8 | 120 | `data/s3/s3_generation.jsonl` | First clean run after the fp16 NaN failure. Detection 0 / 43.3 / 16.7 / 0% — **non-monotone, wrong shape** |
| C16 | 2026-09-02 | generation sweep | 3,5,6 | 90 (210 total) | same file | **α=6 gives 10.0% [3.5, 25.6] vs their 10.8%**, 0/30 FPR. Coherence 93%/87% at α=5/6 |
| C17 | 2026-09-02 | forced choice, real | 0,2,4,5,6,8 × 2 framings | 360 | `s3_forced_real_sweep.jsonl` | **First-token P(YES) 0.00003 → 0.417, Wilcoxon p=9.3e-9, 28/30 concepts rose.** Killed my own post-hoc-inference hypothesis. Neutral prompt was malformed in this run |
| C18 | 2026-09-02 | forced, **random** control | 0,2,4,6 | 240 | `s3_forced_random.jsonl` | Norm-matched Gaussian: **0.305 at α=6 vs real 0.417, p=0.33 — indistinguishable.** Pre-registered outcome **A2** |
| C19 | 2026-09-02 | forced, **shuffle** control | 0,2,4,6 | 240 | `s3_forced_shuffle.jsonl` | Coordinate-permuted vector agrees with random at every α (p=0.50/0.27/0.60) |
| C20 | 2026-09-02 | forced, real, **fixed** neutral prompt | 0,2,4,6 | 240 | `s3_forced_real.jsonl` | Introspective vs neutral **indistinguishable once anything is injected** (p=0.44–0.75); differ only at α=0 (0.000 vs 0.188) |
| **C21** | **2026-09-03** | **forced, L2-normalised, real** | 0,1,2,4,8,16,32,64,128,512,2048 | 660 | `data/s3/s3_unit_forced_norm1.jsonl` | **RAN BUT NEVER ANALYSED.** Session hit its usage limit; the account was disabled the same day. Archived 6 Sep 2026, sha256 `f465e7304ca5e0a5…`, 118528 bytes. Its random-control twin was never run |
| **C22** | **2026-09-07** | onset analysis of C21 (local, no GPU) | — | 660 rows rescored | `results/s3_unit_onset.json` | **alpha\* = None; the sweep fell 4.9x short of the onset.** Top of sweep = alpha_unnorm 0.409 vs an effect that starts at 2. Signal switching on at alpha=2048: W=432, **p=3.0e-06**, 27/30 rose. Not a null — a mis-scaled grid |

C21 verified on archival: 660 rows = 11 α × 2 framings (`introspective`,
`neutral_matched`) × 30 concepts, exactly 30 per cell, `control: none`,
`normalised: true`, and every row carries `trial_seed` — so it is the first run
made after the trial-randomisation fix. Fields: `alpha, concept, control,
framing, layer, normalised, p_yes, trial, trial_seed`. Nothing is missing except
the random arm.

| C37 | 2026-09-07 | Threshold cross-validation (local) | — | 100 units, 5-fold | — | R² thresholds transfer (74→69% at fixed cost); **z_mean loses 16 pts out of sample (74→58%)**; disagreement loses 21 |
| **C36** | **2026-09-07** | **S1-3 failure-class characterisation (local)** | — | 100 units | `results/s1_failure_class.json` | **Failures are sparse, heavy-tailed units**: active 6.8% vs 20.5%, kurtosis 87 vs 29. **z_mean predicts at AUC 0.877 with no fit at all** — a pre-fit screen |
| **C35** | **2026-09-07** | **S1-1 flag ROC (local, no GPU)** | — | 100 units rescored | `results/s1_flag_roc.json` | **Held-out R² beats method disagreement: AUC 0.906 vs 0.802**, and discards 5 good units against 21 at the same 73% catch. The free flag is the better flag |
| C33 | 2026-09-07 | A-13 Qwen forced, refit, real | 0…105 (0–40% of norm) | 420 | `data/s3/q_refit_forced_norm1.jsonl` | Curve climbs 1.0e-09 → **6.8e-02**, six orders. **Short of the 0.10 bar by one grid step** |
| C34 | 2026-09-07 | A-13 Qwen forced, refit, random | same | 420 | `data/s3/q_refit_forced_random_norm1.jsonl` | **Real > random at 12/12 comparisons** — never true pre-refit. Significant only at the low end, where magnitudes are ~1e-09 |
| **C32** | **2026-09-07** | **A-11 steer control, vectors refit at the concept** | 0…200 | 150 | `data/s3/s3_qwen_refit_steer_norm1.jsonl` | **FIXED: 1/30 → 7/30 → 14/30 monotone, against C31's flat 1/30.** Residual norm at the concept token is **261.5**, not the tail's 177.8 |
| **C31** | **2026-09-07** | **A-10 Qwen steering positive control** | 0…200 (0–112% of norm) | 150 | `data/s3/s3_qwen_steer_norm1.jsonl` | **Concept-in-text 1/30 at alpha=200, identical to the 1/30 at alpha=0. The vectors carry no content.** Cause: the vector is read at the chat-template tail, not the concept word |
| **C30** | **2026-09-07** | **A-9 Qwen generation check** | 0, 50, 100 | 90 | `data/s3/s3_qwen_gen.jsonl` | **Injection never reaches the output: 0/30 concept-in-text at 56% of the residual norm.** C27-29 null is an artifact. Separately: 30/30 categorical introspection refusal |
| C27 | 2026-09-07 | A-3 Qwen probe, alpha=0 | 0 | 60 | `data/s3/s3_qwen_probe_forced_norm1.jsonl` | **Residual norm 177.8** vs Gemma's 58,932 — **331x**. Baseline P(YES) 1.03e-09 |
| C28 | 2026-09-07 | A-3 Qwen real sweep | 2…100 (1.1–56% of norm) | 360 | `data/s3/s3_qwen_forced_norm1.jsonl` | **No onset. Peak 1.36e-05, four orders below the filed 0.10 bar.** Rise is real (p=3.7e-09, 29/30) but never approaches YES |
| C29 | 2026-09-07 | A-3 Qwen random control | same | 360 | `data/s3/s3_qwen_rand_forced_random_norm1.jsonl` | Random ≥ real in 3 of 4 movable cells. No concept-specific component anywhere |
| C26 | 2026-09-07 | A-3 Qwen retry, **FAILED AT LOAD** | — | — | Kaggle log only | Same CPU-dispatch error, different cause: fp32 **storage** made `embed_tokens`/`lm_head` 3.1 GB each and unplaceable. Also silently fell back to the Hub — input not attached. Both fixed in `2026-09-07c` |
| C25 | 2026-09-07 | A-3 Qwen probe, **FAILED AT LOAD** | — | — | Kaggle log only | `ValueError: Some modules are dispatched on the CPU or the disk`. Loader bug, not setup — version stamp, mount, and both GPUs were all correct. Fixed in `2026-09-07b` |
| **C23** | **2026-09-07** | **forced, unit-norm, real, extended grid** | 2048…32768 | 300 | `data/s3/s3_unit_ext_forced_norm1.jsonl` | Onset alpha*=**8192**. Signal switches on at ~14% of the residual norm |
| **C24** | **2026-09-07** | **forced, unit-norm, random, same grid** | 2048…32768 | 300 | `data/s3/s3_unit_ext_forced_random_norm1.jsonl` | **OUTCOME A1: real > random**, p=0.036 at alpha*, **p=9.5e-4** at 16384. Residual norm measured at **58,932** |

**C21 is the open run.** It is run 1 of the pre-arXiv list and the only one that
decides whether "we reproduce their protocol" is true or false. See §7 A-1.

### Inherited from APERTURE — runs CALIPER leans on but did not run

APERTURE (`projects\mirror`, package `aperture`) is the predecessor programme. It
was rejected as the capstone in Aug 2026 and continues as separate solo work, but
CALIPER's Study 3 descends directly from its runs and Study 2 / Phase A plan to
reuse its intervention machinery. Those runs are recorded here because a CALIPER
claim resting on one of them inherits its caveats — and, for six of them,
inherits a missing artifact.

Full detail lives in `projects\mirror\docs\LAB_NOTEBOOK.md`. Do not duplicate it;
link to it.

| APERTURE ID | Date | What it established | How CALIPER uses it | Raw data |
|---|---|---|---|---|
| **R11** | 2026-07-15 | **The steering-vs-access control.** Identical injection, only framing differs. Neutral 0.433 / gamma +2.574 vs introspective 0.302 / gamma +1.988; difference -0.586, 95% CI [-1.148, -0.007], excludes 0 **negatively** | **Load-bearing.** Direct ancestor of Study 3's framing control (C20) and of the L3 row in the merged design. CALIPER's advance is re-running the idea on the defended model with a *preamble-matched* neutral prompt and norm-matched controls | **LOST** |
| **R12** | 2026-07-22 | **First pre-registered run.** Three framings: neutral +2.574 > introspective +1.988 > informative +1.645. Primary prediction P3 **falsified**; non-replication of Pearson-Vogel on Gemma-2-2B | **Load-bearing for method, not for numbers.** Where the pre-register-then-report-the-failure discipline came from, which Study 3 repeated at C17 | **LOST** |
| R7 | 2026-07-14 | **Probe-Report Gap = 0.83** - probe 1.00, shuffled control 0.00, verbal report 0.17 | Background evidence that a concept is present in activations even when unreported. Cited, not depended on | **LOST** |
| R8 | 2026-07-14 | **Activation patching:** self-delta +6.96 [+5.34,+8.56] vs control +0.81 [-0.19,+1.80]; paired +6.15. Injected content is causally wired to the output | Underwrites the injection paradigm Study 3 uses, and is why "output steering" was the hypothesis to beat | **LOST** |
| R9 | 2026-07-15 | **Naturalistic arm:** injection-derived directions decode *non-injected* states at 0.688 vs 0.062 chance | Answers "injections are OOD damage, not real thoughts" - the central methodological attack on Study 3's paradigm | **LOST** |
| R10 | 2026-07-15 | Forced choice, first gamma fit +1.988 [+1.476,+2.478]. Naive access reading **rejected** in favour of the control that became R11 | Cautionary provenance: gamma > 0 alone is not evidence. The same trap Study 3's C15 fell into with `said_yes` | **LOST** |
| R1-R6 | 2026-07-13 | Pipeline smoke; steering demo; coherent-window sweep (alpha ~0.5-1); detection probe; layer sweep; 9B scale check. **Confabulation is depth- and scale-robust at 2B-9B** | Sets the prior that detection is weak in small open models - which is why Study 3 went straight to a 27B defended model | R1, R3 only |

**The inherited liability, stated plainly.** APERTURE's own notebook section 7
records that the raw data for **R7-R12 was never downloaded off Kaggle and is
gone**. Six runs, including both runs CALIPER calls load-bearing. Consequences:

1. **No CALIPER paper may report an APERTURE number as its own measurement.** Cite
   it as prior work by the same author, with its caveats, or re-run it.
2. **If a reviewer asks for R11's data, we cannot produce it.** Paper B's framing
   chapter must therefore rest on Study 3's own C20, which *is* archived, and use
   R11 as motivation only.
3. If R11/R12 are ever needed as data rather than motivation, they are a **cheap
   Kaggle re-run** (free tier, Gemma-2-2B), not a free re-analysis. Budget ~2 GPU-h.

This is exactly why section 2.3 makes artifact download part of a run's definition
of done, and why C21 was moved into `data/s3/` the day this notebook opened.

**What CALIPER reuses as code, not as results:** the concept-injection hooks and
intervention machinery (`aperture.patching`, the forward-hook injection path).
Study 2's P1/P2 plant directions with it, Phase A's E1.4 ablates with it, and
Study 2's fallback - if no clean persona behaviour appears at 7B - is to use the
concept directions APERTURE already injects, which are documented to shift
behaviour. That dependency is on tested code held locally, so it carries no
artifact risk.

---

## 4. Runs in detail

*(Newest first. Append; never rewrite.)*

### C48 - Gemma refit vectors: do they steer? (2026-09-08) - **WEAKLY, AND THAT MATTERS**

A-12c. The validation C45/C46 was missing: `--stage steer`, neutral prompt, injection at
every position, refit vectors, alphas as fractions of the 36,245 concept-token norm.
150 trials. `data/s3/gval_steer_norm1.jsonl`.

| alpha | %norm | literal concept | **semantic** | coherent |
|---|---|---|---|---|
| 0 | 0% | 0/30 | 3/30 *(baseline)* | 30/30 |
| 3,624 | 10% | 0/30 | 2/30 | 30/30 |
| 7,249 | 20% | 0/30 | 5/30 | 30/30 |
| 14,498 | 40% | 1/30 | **10/30** | 27/30 |
| 27,546 | 76% | 5/30 | **12/30** | **14/30** |

"Semantic" counts a concept-specific associate as well as the literal word - peanut and
trunk for elephant, sonar and clicks for dolphin, lava and magma for volcano. The literal
scorer badly undercounts: at 76% it reports 5/30 while the text plainly carries the
concept in "Silk-silk-like-f-web", "harbor seals seals", "The clicks... like sonar",
"dust and peanuts".

**The vectors do carry content.** At 40% semantic steering is 10/30 against a 3/30
baseline, and the text is unmistakable: spider gives "The Silk and the Silkling... his
silkwebs and his silklings", eagle gives "feather feather feather", harbor gives "the
city of seals".

**But they are much weaker than Qwen's refit vectors** (C32: 7/30 literal at 38%, 14/30
at 76%, coherent 30/30 throughout). Gemma needs 76% of its norm to reach 5/30 literal,
and by then coherence has collapsed to 14/30 - word salad and raw entity fragments.

**The caveat that attaches to C45/C46, and it is not small.** That run swept 0-40%, and
its *significant* real-vs-random cells are at **5%, 10% and 20%** - where semantic
steering is 2/30 and 5/30 against a 3/30 baseline, i.e. at or barely above chance. **The
headline result sits in a range where the real vectors barely steer generation at all.**

This is not disqualifying: the forced-choice readout is a first-token logit measure and
is far more sensitive than "does the concept word appear in fifty tokens". A vector can
shift logits measurably without surfacing in text. But it must be stated, and it means
the instrument is well validated at 40% and only weakly validated where the claim lives.

### C49 - A-8, the on-manifold control (2026-09-08) - **THE OFF-MANIFOLD ACCOUNT IS REFUTED**

`--control span`: a random direction drawn **within the span of the 30 concept vectors**,
rescaled to unit norm. On-manifold by construction, living in concept space, but not any
single coherent concept. Same alphas, same prompts, 420 rows.
`data/s3/g2_forced_span_norm1.jsonl`.

The prediction under the off-manifold account C45/C46 raised: if the signal tracks how
anomalous a perturbation is, span is on-manifold and should behave like **real** (low).

**Neutral framing** - the arm with interpretable magnitudes:

| %norm | real | **span** | random | real vs span | span vs random |
|---|---|---|---|---|---|
| 5% | 0.1393 | 0.2115 | 0.2150 | p=0.052 | p=0.48 |
| 10% | 0.1205 | **0.2606** | 0.2471 | **p=0.033** | p=0.73 |
| 20% | 0.1215 | **0.2416** | 0.3199 | **p=0.045** | p=0.38 |
| 40% | 0.1238 | 0.2166 | 0.1851 | p=0.16 | p=0.46 |

**Span behaves like random, not like real.** Indistinguishable from noise at every
strength (p=0.38-0.76), and significantly different from the real concept vector at 10%
and 20%.

**So the signal is not tracking off-manifold-ness.** A direction inside concept space
produces the same elevated response as Gaussian noise. What suppresses the response is
something specific to a **single coherent concept direction** - a mixture of concept
directions does not do it.

That is a sharper claim than the manifold story it replaces, and it was the hypothesis
A-8 was written to kill. It did.

**Introspective framing shows span intermediate** (real below span at 1-10%, span below
random at 5-20%), but every value there is between 0.0000 and 0.0193 - all meaning
near-certain NO - so the ordering is not worth interpreting. Neutral is the arm to quote,
per the standing rule from C33/C34.


### C47 - S1 multi-seed confirmatory run, n=100 (2026-09-08) - **NO IMPROVEMENT**

The pre-registered confirmatory run (`preregistration-s1-multiseed.md`, filed 7 Sep with
a same-day k=2 clarification). 100 units, 5 seeds each at k=1 and k=2, selection by
held-out R2, no ground truth used in selection. ~4.5 h local CPU.
`results/s1_multiseed.jsonl`.

**PRIMARY - FAIL.**

| | |
|---|---|
| multi-seed pass rate | **76/100**, Wilson 95% [0.668, 0.833] |
| criterion | lower bound > 0.90 |
| verdict | **FAIL** |
| C13 single fit, same units | **77/100** |

The repair does not improve the instrument. It is one unit *worse* than the single fit
it was meant to replace.

**Repairs 8, breaks 9.** McNemar exact on the 17 discordant units: **p = 1.000**. There
is no detectable difference between the two protocols.

| | |
|---|---|
| gate FAIL -> multiseed PASS | 8 |
| gate PASS -> multiseed FAIL | **9** |

**The paired change is positive but tiny and its tails are asymmetric.** Median +0.0020,
95% CI [+0.0005, +0.0040], improved 65 and worsened 33 - so it usually helps a little
and occasionally destroys a unit:

| | median | worst case |
|---|---|---|
| gains | +0.0118 | +0.7804 |
| losses | -0.0026 | **-0.9553** |

n1394 goes 0.9971 -> 0.0418. Every damaged unit carries a large positive k2_gain
(+0.017 to +0.701), the signature of the k=1 arm landing badly on a unit a single seeded
fit happened to get right. **The lottery cuts both ways**, and C43 only ever saw the side
where there was nowhere to go but up.

**C43's six pilot units, rescored here:**

| neuron | gate | multi-seed | passes? |
|---|---|---|---|
| n527 | 0.1445 | 0.9249 | no |
| n1503 | 0.3481 | 0.9815 | **yes** |
| n2723 | 0.5151 | 0.9671 | no |
| n2023 | 0.6789 | 0.6639 | no |
| n1180 | 0.7237 | 0.9883 | no |
| n1625 | 0.8040 | 0.9387 | no |

Five of six improve substantially and only one clears the bar, because `k2_gain` fails
them even where alignment recovers. C43 reported alignment alone and did not apply the
full pass criterion - that is why its pilot looked stronger than it was.

**Per the filed decision rule this is outcome NO IMPROVEMENT, and the pre-registration
says to report it as "the pilot was a six-unit artifact".** That is close to right, with
one correction: the pilot was not wrong about alignment recovering, it was wrong about
that being sufficient.

**A confound in my own protocol, and it is the obvious follow-up.** The multi-seed arm
ran **1 restart x 800 steps** against the gate's **3 restarts x 2500 steps**. So this
compares five cheap draws to one expensive draw, not seed-selection at matched budget.
It is a fair test of "is five cheap draws better than one expensive one" - answer: no -
and **not** a fair test of "does selecting on held-out R2 help". Matching the budget is
the next run and it is cheap.

**What survives.** C41/C42's ruggedness result and C43's +0.953 correlation between
test_r2 and true alignment both stand - they are about the objective landscape, not
about this protocol. What does not survive is the claim that multi-seed selection
repairs the instrument. The C13 headline is unchanged: **77% recovered, 23% fail
silently.**


### C45 / C46 - Gemma forced choice, vectors refit at the concept (2026-09-07) - **OUTCOME A3**

A-12b, `2026-09-07f`, `--alpha-frac`, `vector_read_position: concept`, layer 37,
residual norm at the concept token **36,245**. Real and norm-matched random, 420 rows
each, alphas at 0/1/2/5/10/20/40% of that norm. `data/s3/g2_forced*.jsonl`.

**This is the first Gemma measurement made with an instrument that has passed a steering
control.** C15-C24 were all read at the template tail.

| alpha | %norm | real intro | rand intro | real neutral | rand neutral |
|---|---|---|---|---|---|
| 0 | 0% | 0.0000 | 0.0000 | 0.1882 | 0.1882 |
| 362 | 1% | 0.0000 | 0.0000 | 0.1746 | 0.1909 |
| 725 | 2% | 0.0000 | 0.0001 | 0.1622 | 0.1959 |
| 1,812 | 5% | 0.0000 | 0.0003 | 0.1393 | 0.2150 |
| 3,624 | 10% | 0.0001 | 0.0080 | 0.1205 | 0.2471 |
| 7,249 | 20% | 0.0120 | 0.1131 | 0.1215 | 0.3199 |
| 14,498 | 40% | 0.1600 | 0.1965 | 0.1238 | 0.1851 |

**The filed onset moves from 13.9% to 40.0% of the residual norm.**

**PRE-REGISTERED OUTCOME A3: random is significantly ABOVE real.** Not once, but at
every strength where either is measurable, in both framings:

| %norm | framing | real | random | p | real higher on |
|---|---|---|---|---|---|
| 5% | introspective | 0.0000 | 0.0003 | **1.1e-03** | 8/30 |
| 10% | introspective | 0.0001 | 0.0080 | **1.1e-05** | 5/30 |
| 20% | introspective | 0.0120 | 0.1131 | **3.2e-03** | 9/30 |
| 5% | neutral | 0.1393 | 0.2150 | **1.8e-02** | 9/30 |
| 10% | neutral | 0.1205 | 0.2471 | **2.3e-02** | 10/30 |
| 20% | neutral | 0.1215 | 0.3199 | **3.6e-02** | 10/30 |

**And the two conditions move in opposite directions from the no-injection baseline.**
Under the neutral prompt, where the 0.1882 baseline has headroom:

| %norm | real | p vs baseline | | random | p vs baseline |
|---|---|---|---|---|---|
| 1% | 0.1746 | **4.0e-03** DOWN on 24/30 | | 0.1909 | 0.56 |
| 2% | 0.1622 | **5.0e-03** DOWN on 24/30 | | 0.1959 | 0.43 |
| 5% | 0.1393 | **9.9e-03** DOWN on 24/30 | | 0.2150 | 0.25 |
| 10% | 0.1205 | **9.3e-03** DOWN on 23/30 | | 0.2471 | **5.0e-02** up |
| 20% | 0.1215 | 0.38 | | 0.3199 | **4.7e-02** up |

Injecting a **real concept** makes the model *more* confident nothing unusual is
present than injecting nothing at all. Injecting **matched noise** makes it less
confident.

**Per the filed decision rule, A3 is reported as an anomaly and NOT interpreted.** The
obvious account - that the signal tracks how off-manifold a perturbation is, and a real
concept direction is one the computation already uses - is a hypothesis with no
dedicated test behind it. It is exactly what **A-8**, the on-manifold control, was
written to decide, and that run has never been made. A-8 moves from "future work" to the
next experiment.

**Magnitudes, so the p-values are not read alone.** The introspective ratios (up to 82x
at 10%) are computed on 0.0001 against 0.0080 - both meaning the model says NO with near
certainty. **Only the neutral framing carries interpretable magnitudes**, and it is the
one to quote: 0.12-0.14 real against 0.22-0.32 random, against a 0.188 baseline. At 40%
neither framing shows a significant real-vs-random difference.

**This is the third answer to the same question, and the first two were made with a
broken instrument:**

| run | vectors | verdict |
|---|---|---|
| C18/C19 | template tail, unnormalised | **A2** - real indistinguishable from random |
| C23/C24 | template tail, normalised | **A1** - real above random |
| **C45/C46** | **concept position, normalised** | **A3 - random above real** |

C32 showed template-tail vectors carrying no recoverable content on Qwen; on Gemma they
steered but were demonstrably degraded. Any write-up must state that the reported answer
changed twice, why, and that only the third measurement used a validated instrument.


### C43 - The lottery is real, and held-out R2 wins it (2026-09-07) - **A WORKING REPAIR**

Ten independent single-restart fits per unit at different seeds, 800 steps, full data.
6 worst-fail + 2 pass units. `results/s1_seed_lottery.json`.

**The lottery account is confirmed.** Seed-to-seed range within a single unit:

| neuron | gate | seed min | median | max |
|---|---|---|---|---|
| n527 | 0.1926 | 0.9024 | 0.9163 | 0.9498 |
| **n1503** | 0.2746 | **0.0660** | 0.3251 | **0.9829** |
| n2723 | 0.3562 | 0.9298 | 0.9668 | 0.9675 |
| n2023 | 0.6789 | 0.6354 | 0.7343 | 0.9663 |
| n1180 | 0.7237 | 0.9676 | 0.9883 | 0.9891 |
| n1625 | 0.8040 | 0.8799 | 0.9131 | 0.9457 |
| n3066 *(pass)* | 1.0000 | 0.9997 | 0.9999 | 1.0000 |
| n2053 *(pass)* | 1.0000 | 0.9999 | 1.0000 | 1.0000 |

n1503 spans 0.0660 to 0.9829 on nothing but the seed. The passes do not move.

**And the question that decides everything: held-out R2 identifies the good draw.**

| | |
|---|---|
| correlation(test_r2, alignment) on worst-fail draws | **+0.953** |
| median regret of selecting by test_r2 vs the oracle | **0.0000** |
| worst regret on any unit | 0.0195 |
| zero-regret units | 4 of 6 |

**The repair, measured:**

| protocol | worst-fail median |
|---|---|
| the gate: one fit, 3 restarts, 2500 steps | **0.5175** |
| five seeds, one restart, 800 steps, keep best test_r2 | **0.9669** |

**Five seeds saturates** - k=5 and k=10 give identical results on every unit, and k=3 is
not enough (min 0.7532). The confirmatory run is ~2.1 h of CPU for n=100, not 4.3.

**Why this matters more than the diagnosis.** The selection needs **no ground truth** -
test_r2 is computed from held-out data the estimator already has. That is the project's
whole thesis applied to its own instrument: a readout that cannot be validated directly
can still be repaired by a check requiring nothing external. It also explains the
disagreement flag, which is the two-draw special case of the same idea.

**Honest limits.** At k=5 the six worst units read 0.9675, 0.9663, 0.9498, 0.9457,
0.9696, 0.9829 - **four of six clear the 0.95 bar, two do not.** The repair is large, not
total. And this is 6 units chosen as the worst, not the full 23; the pass rate on all 100
is unmeasured.

**The confirmatory run is pre-registered before it runs**
(`preregistration-s1-multiseed.md`), because scoring a headline against a protocol chosen
after seeing a favourable subset is the C40 failure mode exactly, found in this same
session.


### C42 — Steps sweep: the over-optimisation hypothesis is REFUTED (2026-09-07)

C41 proposed that more optimisation makes the direction worse, because the 64-wide
nonlinearity can absorb the error and fit the response through a wrong V. This tested it
directly: 6 worst-fail and 3 pass units, steps from 200 to 5000, **restarts fixed at 2**,
full data. `results/s1_steps_sweep.json`.

| steps | worst-fail median | min | pass median |
|---|---|---|---|
| 200 | 0.8830 | 0.3645 | 0.9991 |
| 400 | 0.9467 | 0.8525 | 0.9999 |
| 800 | 0.9724 | 0.3833 | 0.9991 |
| 1600 | 0.9830 | 0.9079 | 0.9998 |
| **2500** | **0.9759** | 0.4587 | 0.9994 |
| 5000 | 0.9705 | 0.4438 | 0.9997 |

**Refuted.** There is no decline with steps — the curve rises to 1600 and is flat
thereafter. Decisively: **at the gate's own 2500 steps with 2 restarts the worst-fail
median is 0.9759, against the gate's 0.5175 at 3 restarts.** Same estimator, same data,
same step count. Steps are not the variable, and my C41 mechanism was wrong.

**What the sweep did establish, and it is better than what it was looking for.** Spread
within a unit across nominally equivalent step counts:

| group | median spread | max spread |
|---|---|---|
| worst-fail | **0.1609** | **0.6344** |
| pass | 0.0029 | 0.0061 |

n1503 alone runs 0.3645 → 0.9989 → 0.3833 → 0.9989 → 0.4587 → 0.4438 across the sweep.
Passes never move past the fourth decimal.

**Every worst-fail unit reaches ≥0.93 at some step count**, including C13's worst
(n527, gate 0.1926, reaching 0.9977) and n1503 (gate 0.2746, reaching 0.9989).

So the failure class is not a set of unrecoverable units. **It is a set of units whose
objective is rugged, where the recovered direction is close to a lottery across
configurations, and a single fit draws once from it.** The passes sit in a smooth
landscape where every configuration converges to the same place.

That reframes the 23%: it is not a recovery ceiling, it is a variance property, and the
gate reported one draw.

**Still unexplained: why did the gate at 3 restarts do worse than 2?** More restarts
should weakly help, since the fit keeps the best. It is the only named difference left
and it is untested — one comparison, confounded with whatever else differs between the
runs. **Do not assert that more restarts hurt.** C43 tests the lottery account directly
with independent single-restart fits across ten seeds, and asks the question that decides
whether any of this is repairable: **can held-out R² identify the good draw without
ground truth?** If it can, "fit several, keep the best test_r2" is a working repair and
the disagreement flag is explained. If it cannot, the objective cannot distinguish a
recovered direction from a lost one, which is a much sharper negative about this
estimator family.

**Script defect, corrected in C43.** This run asked the `Fit` object for `r2`, which does
not exist — it exposes `train_r2` and `test_r2` — so `getattr(f, "r2", nan)` logged nan
for every row and the absorption signature was never actually measured. The refutation
above rests on the alignment column, which is unaffected.


### C41 — Required-N without the C37 filters, plus the control C37 lacked (2026-09-07) — **C37'S ATTRIBUTION WAS WRONG**

18 units, stratified by gate alignment into worst-fail (0.144-0.804), marginal-fail
(0.932-0.955) and pass (1.000). No event filter; targets capped per unit. Adds a
**full-data arm at the cheap config**, which C37 did not have.
`results/s1_required_n_unbiased.json`.

| target | worst-fail median (n) | marginal-fail (n) | pass (n) |
|---|---|---|---|
| 50 | 0.1072 (6) | 0.1298 (6) | 0.1920 (6) |
| 200 | 0.2422 (6) | 0.2342 (6) | 0.4001 (6) |
| 400 | 0.9358 (6) | 0.9054 (6) | 0.7827 (6) |
| 800 | 0.9704 (3) | 0.9681 (6) | 0.9978 (6) |
| **full** | **0.9668 (6)** | **0.9856 (6)** | **0.9998 (6)** |

**C37's required-N conclusion survives**: real units need 400-800 informative events, not
C14's synthetic ~200, and the three strata are indistinguishable in how much data they
need.

**C37's headline attribution does not survive.** The decomposition it could not do:

| group | gate | full (cheap config) | best subsample | **gate→full** | full→best |
|---|---|---|---|---|---|
| worst-fail | 0.5970 | 0.9668 | 0.9775 | **+0.3698** | +0.0107 |
| marginal-fail | 0.9441 | 0.9856 | 0.9946 | +0.0415 | +0.0090 |
| pass | 1.0000 | 0.9998 | 0.9996 | −0.0001 | −0.0002 |

**Resampling contributes almost nothing (+0.011). The fit configuration contributes
everything (+0.370).** C37 changed both at once and credited the wrong one.

**And it is not the estimator or the selection rule** — that was the obvious next
suspect and it is ruled out:

| worst-fail, median | value |
|---|---|
| gate, dual protocol (selected) | 0.5970 |
| gate, direct fit alone | 0.5175 |
| gate, cascade alone | 0.4316 |
| **same direct fit, 2 restarts / 800 steps** | **0.9668** |

Same estimator, same data, same units. **Only fewer restarts and fewer steps, and the
worst failures go from 0.52 to 0.97.**

Per unit, including the catastrophic ones C37 excluded by construction:

| neuron | gate selected | gate direct | cheap config |
|---|---|---|---|
| n527 | **0.1445** | 0.1926 | **0.9665** |
| n1503 | 0.3481 | 0.2746 | 0.3334 |
| n2723 | 0.5151 | 0.3562 | 0.9671 |
| n2023 | 0.6789 | 0.6789 | 0.9823 |
| n1180 | 0.7237 | 0.7237 | 0.9878 |
| n1625 | 0.8040 | 0.8040 | 0.8629 |

**Five of six catastrophic failures recover under less optimisation**, including C13's
worst unit at 0.1445 → 0.9665. n1503 does not (0.3481 → 0.3334) and looks genuinely hard.

**The candidate mechanism is already in the record.** C9 measured R² = 1.000 at a
*wrong* direction. The bottleneck's nonlinearity is a 64-wide MLP, flexible enough to fit
the response through a wrong V given enough steps — so past some point the objective
stops being informative about the direction because the nonlinearity absorbs the error.
More optimisation then makes the fit better and the direction worse. **C42 tests this
directly** with a steps sweep recording R² alongside alignment: overfitting would
separate train from held-out R², absorption would raise both while alignment falls.

**If it holds, the 23% silent-failure rate is substantially an artifact of
over-optimisation, and Paper B's Study 1 chapter changes.** The obvious confirmation is
cheap — re-run the full n=100 gate at the cheap config, about 8 minutes at 4.7 s/neuron —
and it must be pre-registered before it runs, because it would be re-scoring a headline
after seeing a favourable subset.

**Caveats.** n drops at high targets now that the filter is gone (worst-fail has 1 unit
at 1600), so those cells are weak. And the cheap config's own worst case is 0.3334, so it
is not a universal repair.


### C40 — Audit of the S3 positive control itself (2026-09-07) — **THE REPRODUCTION FIGURE WAS WRONG**

Prompted by a direct question: is the instrument right for everything so far? Two silent
bugs in one day (C31 read position, C39 alpha grid) make the prior that more exist.

**The mechanics audit came back clean.** `yes_no_ids` returns six distinct first tokens
per answer with no YES/NO overlap; `forced_choice` reads `logits[0, -1]`, genuinely the
first generated token; injection covers all prompt positions in the single forward pass;
the random control is a Gaussian rescaled to matched norm and the shuffle a coordinate
permutation, both norm-preserving; no concept is a substring of another, and the only
`desert` hit is "deserts", a true positive. The `>=3 words` coherence rule flagged in C15
discards **zero** affirmative answers in the real data — all 35 short responses begin
with NO — so `detected == said_YES` at every alpha and that worry is closed.

**The scoring, however, does not reconcile with what was published.** At alpha=6, n=30:

| rule | detections | FPR at alpha=0 |
|---|---|---|
| **pre-registered scorer** (affirmative at the start, plus coherence) | **2/30 = 6.7%** | 0/30 |
| "YES" anywhere in the response | 10/30 = 33.3% | **5/30 = 16.7%** |
| what `docs/s3-results.md` claimed | 3/30 = 10.0% | 0/30 |

**The 3/30 matches neither rule.** It is the hand-read subset in which concept-laden text
precedes an affirmative answer — desert, fear, betrayal — identified by reading
transcripts, not by any filed rule. The pre-registration says the scorer is final and no
re-scoring follows results. **The number this study is entitled to report is 6.7%.**

Worse, the reported value is the one closest to the published 10.8%, and alpha=6 was then
chosen as the operating point *because* it matched — a choice C18 already criticised for
selecting the operating point against the quantity being measured. **The safeguard
against exactly this was in place and simply not consulted.**

**A second error in the same document.** Section 2's generated-text column read 50 / 23 /
27 / 7 percent at alpha 2/4/6/8. The first two reproduce under the permissive rule;
**27% and 7% match no rule computable from the archived data.** Correct values are
43/17/7/0 (pre-registered) and 50/23/33/10 (permissive). The first-token column was
verified against `s3_forced_real_sweep.jsonl` and was correct to the digit.

**The readout finding survives and sharpens.** First-token P(YES) stays between 0.417 and
0.504 across alpha 2 to 8 while the generated-text rate runs 43% to 0% (pre-registered)
or 50% to 10% (permissive). The two generated-text rules do not even agree on the shape.

**What is untouched.** Everything in sections 3-7 of the results doc comes from
`forced_choice`, which never calls the generation scorer: the A1 result, the content-free
comparison, the framing comparison, the 139x norm ratio. **Detection with a correct
identification remains 0 in 210 under every rule tried** — the one number in Study 3 that
no scoring choice moves.

**Corrected in:** `docs/s3-results.md` sections 1 and 2 plus its corrections log,
`docs/paper-s3-draft.md` (banner and both reproduction sentences),
`paper/PLAN.md`. **`docs/preregistration-s3-forced-choice.md` is deliberately left
untouched** — it is a filed dated record and the whole value of one is that it is not
edited after the fact.


### C39 — Gemma refit, first attempt (2026-09-07) — **VOID GRID, ONE NUMBER SALVAGED**

A-12 at `2026-09-07e`. The forced-choice arms are unusable: `argv` records
`--alphas 0 0 0 0 0 0 0`, so both conditions ran seven identical alpha=0 cells, 420 rows
each. The run sheet asked for the residual norm to be pasted into `R = 0.0` in cells 3
and 4 and it never was, so every fraction multiplied to zero. **Sheet's fault, not the
operator's** — a manual transcription step between two cells is a failure mode waiting
to happen, and it cost a 45-minute session.

**What is kept, and it is the number A-12 was partly for.**

| | template tail | **concept position** |
|---|---|---|
| Gemma-3-27B L37 | 58,932 | **36,244.97** |
| Qwen2.5-32B L38 | 177.8 | 261.53 |
| **ratio** | **331x** | **138.6x** |

So the headline ratio drops from 331x to **139x** once both are measured where the
vectors are actually read. Still a large difference and it still explains why alpha=4 is
live on most models and inert on Gemma, but **139x is the number to publish** and 331x
must not be quoted again.

**The alpha=0 baselines reproduced C20 exactly** — introspective 3.42e-05, neutral
0.18824 — which confirms the pipeline is deterministic across the refit and that the
refit did not disturb the no-injection condition.

The steer arm in the same session ran at absolute alpha 0-0.4 against a norm of 36,245,
i.e. at most 0.001% of it, and returned 0/30 at every level. That is the expected
nothing, not a result.

**Two fixes in `2026-09-07f`:**

1. **`--alpha-frac`** takes fractions and computes the grid inside the run, where the
   norm is already known. There is nothing to paste, so the failure mode is gone rather
   than documented.
2. **A grid of more than one alpha, all zero, now refuses to run.** Nothing complained
   last time until the analysis, three hours later.


### C37 — Does C14's required-N transfer to real units? (2026-09-07) — **NO, AND THE FAILURES GOT BETTER WITH LESS DATA**

Local CPU, GPT-2 layer 6. 12 units stratified 6 pass / 6 fail from the C13 gate,
subsampled per unit to a target number of informative events while preserving each
unit's natural active fraction. Fit at E0.3b's cheap knee (2 restarts, 800 steps).
`experiments/s1_required_n_real.py`, `results/s1_required_n_real.json`.

| informative events | passed gate (median) | failed gate (median) |
|---|---|---|
| 50 | 0.2205 | 0.0995 |
| 100 | 0.2582 | 0.1076 |
| 200 | **0.3448** | **0.3683** |
| 400 | 0.9617 | 0.8635 |
| 800 | 0.9923 | 0.9835 |
| 1600 | 0.9956 | 0.9823 |

**C14's ~200-event figure does not transfer.** At 200 events real units sit at ~0.35
alignment, nowhere near recovery. Saturation lands between 400 and 800 and is complete
by 800. The real requirement is **roughly 400-800 informative events, two to four times
the synthetic figure**, and any required-N table in the paper has to be quoted from real
units rather than from planted ones.

**Failures and passes need the same amount of data.** 0.9923 vs 0.9835 at 800 events,
0.9956 vs 0.9823 at 1600. Whatever separates the failure class, it is not sample size.

**And the finding that matters: five of six gate-failures recovered when given LESS
data.**

| neuron | gate (20k tokens, 3 restarts, 2500 steps) | subsampled (1600 events, 2 restarts, 800 steps) |
|---|---|---|
| n268 | 0.9406 | **0.9902** |
| n374 | 0.9484 | **0.9866** |
| n830 | 0.8634 | 0.8645 |
| n1745 | 0.9079 | 0.9270 |
| n1937 | 0.9545 | **0.9986** |
| n1954 | 0.9397 | 0.9779 |
| **median** | **0.9402** | **0.9823** |

Less data, fewer restarts, fewer steps — and better recovery. **That is not something a
data limitation can produce.** It is direct behavioural evidence for the optimisation
account already implied by C9 (R² = 1.000 at the true direction) and C10 (the cascade
reaches it): the failures are the search landing in a bad basin, and perturbing the
problem — by resampling the data — knocks it into a good one. n830 is the exception at
0.8634 → 0.8645, unchanged, and may be a genuinely harder case.

If that holds up it suggests a cheap practical remedy nobody has tested here: **refit on
a resampled subset and keep the better held-out R²**, which is the same
fit-twice-and-compare shape as the disagreement flag.

**TWO SELECTION BIASES, AND THEY ARE SEVERE. Do not generalise this to the failure
class.**

1. **Data-rich only.** The design required `n_events >= 1600` so every unit could reach
   the top target. That keeps **8 of 23** gate-failures. Across all 23 the median is
   1,285 events and the minimum is 526 — so the typical failure sits *below* this
   experiment's entry requirement and near the 800-event saturation point.
2. **Near-misses only.** The six selected failures have gate alignments 0.86-0.95
   against a 0.95 bar. **The catastrophic failures are not represented at all** — C13's
   worst unit was 0.1445, and nothing like it is in this sample.

So the honest claim is narrow: *among data-rich, marginal failures, resampling recovers
most of them.* Whether it touches the units that fail badly is untested, and those are
the ones that matter for the 23% headline.

**Next, and it is cheap:** re-run without the `n_events >= 1600` filter, capping targets
per unit at whatever it has, and include the worst failures by gate alignment. That
tests the resampling remedy on the class it is supposed to help.

### C38 — Cross-validating the flag thresholds (2026-09-07)

C35 and C36 both chose thresholds on the same 100 units they scored, so their
sensitivity/specificity pairs were optimistic. AUC is threshold-free and unaffected;
this is 5-fold CV on the operating points. `experiments/s1_flag_cv.py`, no compute.

| flag | AUC | target sens | in-sample | held-out |
|---|---|---|---|---|
| **held-out R² (post-fit)** | **0.915** | 70% | 74% at 5% cost | **69% at 5% cost** |
| | | 80% | 83% at 26% | 76% at 20% |
| z_mean (**pre-fit**) | 0.877 | 70% | 74% at 10% | **58% at 10%** |
| | | 80% | 83% at 19% | 69% at 21% |
| disagreement (two fits) | 0.810 | 70% | 74% at 26% | **53% at 26%** |

**The ranking survives and held-out R² is the robust one** — 74% → 69% at unchanged 5%
cost is barely any optimism, so C35's recommendation stands.

**The pre-fit screen is weaker than C36 made it look.** z_mean's thresholds lose
16 points of sensitivity out of sample (74% → 58%) at the same cost. The AUC of 0.877 is
real and threshold-free, so the signal exists, but the *operating point* does not
transfer well and should be quoted from the held-out column.

**Disagreement degrades most** — 74% → 53% — which is a second reason to prefer held-out
R² over it, on top of costing one fewer fit.


### C37 — Cross-validating the flag thresholds (2026-09-07)

C35 and C36 both chose thresholds on the same 100 units they scored, so their operating
points were optimistic. AUC is threshold-free and unaffected; the sensitivity/cost pairs
are not. 5-fold CV: threshold picked on four folds, measured on the fifth.
`experiments/s1_flag_cv.py`.

| flag | AUC | target | in-sample sens/cost | **held-out sens/cost** |
|---|---|---|---|---|
| held-out R² | 0.915 | 70% | 74% / 5% | **69% / 5%** |
| | | 80% | 83% / 26% | **76% / 20%** |
| z_mean (pre-fit) | 0.877 | 70% | 74% / 10% | **58% / 10%** |
| | | 80% | 83% / 19% | **69% / 21%** |
| disagreement | 0.810 | 70% | 74% / 26% | **53% / 26%** |
| | | 80% | 83% / 35% | **69% / 36%** |

**The ranking survives and R² is also the most stable.** Its thresholds barely
generalise worse than they fit — 74%→69% at the 70% target, with cost unchanged at 5%.

**The pre-fit screen is weaker than C36's table implied.** z_mean loses 16 points of
sensitivity out of sample (74%→58%) at the same cost. The screen is real — AUC 0.877 is
threshold-free and does not move — but its *thresholds* do not transfer well, so C36's
"skip 25 fits of 100 to catch 74%" should be read as "catch about 58%" on unseen units.
Corrected accordingly; the two-stage recommendation stands but the first stage buys less
than it looked like.

**Disagreement degrades worst**, losing 21 points (74%→53%) while already costing the
most. Third of three on every measure now: lowest AUC, least stable thresholds, and the
only one requiring two fits.

**Still one model, one layer, 100 units.** CV controls threshold optimism, not the fact
that every number here comes from GPT-2 layer 6.


### C36 — S1-3, what the 23% failure class actually is (2026-09-07)

`experiments/s1_failure_class.py`, local CPU. Neuron ids taken from the C13 gate output
so the sample matches by construction. Activation statistics computed for all 100 units
over the same 20k-token corpus, then scored against the gate's own pass rule.
`results/s1_failure_class.json`.

**Every activation statistic predicts failure**, scored under the correct gate rule
(23 failures):

| statistic | AUC | mean on failures | mean on passes | needs a fit? |
|---|---|---|---|---|
| held-out R² | **0.915** | — | — | yes |
| **z_mean** | **0.877** | −1.20 | −0.65 | **no** |
| response kurtosis | 0.863 | 87.4 | 28.9 | **no** |
| w_norm | 0.857 | 3.96 | 3.27 | **no** |
| frac_active | 0.825 | 6.8% | 20.5% | **no** |
| method disagreement | 0.810 | — | — | **two** |

**The failure class has a coherent identity: sparse, heavy-tailed units.** A unit the
estimator fails on is active on 6.8% of positions against 20.5% for one it succeeds on,
sits twice as far below GELU's zero (z_mean −1.20 vs −0.65), and has a response
kurtosis three times higher (87 vs 29). That is one description, not four: a unit that
fires rarely and, when it does, fires hard.

**The practical result is a pre-fit screen.** z_mean needs no fit at all — it is a
property of the unit and the corpus, one matmul. At AUC 0.877 it is close to held-out
R²'s 0.915 while being available **before** committing 64 seconds to fitting:

| catch | cost | effect |
|---|---|---|
| 12/23 (52%) | 5/77 good units (6%) | skip 17 fits of 100 |
| 17/23 (74%) | 8/77 (10%) | skip 25 fits of 100 |
| 19/23 (83%) | 15/77 (19%) | skip 34 fits of 100 |

Combining z_mean with R² gives AUC 0.918 against R²'s 0.915 — no meaningful gain, and
the two are only weakly correlated (r=+0.26), so they are not redundant so much as
z_mean is simply weaker. **The honest recommendation is a two-stage rule: screen on
z_mean before fitting, then flag on held-out R² after.**

**C7's specific claim does not survive.** It reported failures sitting at z_mean −1.3 to
−1.9, from 8 neurons. Across 100: only **6 of 23 failures** fall in that band, and
**4 of 77 passes** do too. The median failure is −1.17, not −1.5. The *direction* is
right and now well supported at AUC 0.877; the *band* was an artifact of a tiny sample.
Do not quote it.

**A tension with C14 worth chasing.** C14 put required-N at ~200 informative events for
K=1. Failures here average **1,353** events and passes **4,097** — both far above 200,
so raw event count is not the binding constraint on real units. C14 measured that on
synthetic data with well-behaved responses; at kurtosis 87 the information per event is
evidently much lower. **The required-N table may not transfer to real units, and that
should be checked before it is published as guidance.**


### C35 — S1-1, is the disagreement flag a usable decision rule? (2026-09-07) — **IT IS DOMINATED**

Re-analysis of `results/e01_gate.jsonl` (C13). No compute, no GPU.
`experiments/s1_flag_roc.py`, output `results/s1_flag_roc.json`.

100 units, 22 failures at the pre-registered bar (alignment < 0.95). Three
ground-truth-free candidates scored by ROC:

| predictor | AUC | mean on failures | mean on passes |
|---|---|---|---|
| **held-out R² of the k=1 fit** | **0.906** | −0.865 | −0.997 |
| method disagreement | 0.802 | 0.180 | 0.066 |
| k=2 gain | 0.367 | — | — |

**The disagreement flag works, and it is beaten by something simpler and free.**

Operating points, same units, same failures:

| catch rate | R²: good units lost | disagreement: good units lost |
|---|---|---|
| 50% (11/22) | **2/78 (3%)** | 7/78 (9%) |
| 73% (16/22) | **5/78 (6%)** | 21/78 (27%) |
| 82% (18/22) | 21/78 (27%) | 28/78 (36%) |
| 100% (22/22) | 36/78 (46%) | 50/78 (64%) |

At 73% sensitivity the R² rule discards **5 good units where disagreement discards 21** —
four times cheaper for the same catch. R² dominates at every sensitivity tested.

Combining does not rescue it. `R² OR disagreement` catches everything but flags 62 of
100 units (51% of the good ones). `R² AND disagreement` gives 77% catch for 27% loss,
which R² alone beats at 82% for the same 27%.

**Why this matters more than it looks.** Method disagreement requires running **both**
estimation routes — it is the reason the dual protocol costs ~2x per neuron. Held-out R²
is already computed by the fit that is running anyway. So the better flag is also the
free one, and the second fit is not buying detection quality.

**Consequence for Paper B.** The merged design names the disagreement flag as *the*
through-line deliverable — "a thing practitioners can run on their own readouts where no
truth exists". That claim survives in substance: **a ground-truth-free flag does predict
silent failures.** But the specific instrument has to change, and the honest headline is
now "the fit's own held-out R² predicts its own silent failures at AUC 0.91", which is a
simpler and more portable claim than one requiring two estimators.

**CORRECTION 7 Sep, same day: the discrepancy was mine, and the AUCs above are
slightly wrong.** The gate's pass rule is `(align > 0.95) AND (k2_gain < 0.01)` — two
clauses, not one. This analysis used alignment alone, giving 22 failures where the gate
has **23**; unit n1937 aligns at 0.9545 but has k2_gain 0.0252 and fails on the second
clause. Under the correct labels: **held-out R² AUC 0.915** (reported 0.906),
**disagreement 0.810** (reported 0.802), k2_gain 0.396. The conclusion is unchanged —
R² still dominates disagreement — and **C13's 87%/55% was right all along**; it
reproduces exactly under the gate's own rule. Corrected figures are in C36.

**Caveats.** n=100 from one model, one layer. Thresholds are chosen on the same data
they are evaluated on, so the operating points are optimistic; a held-out set or
cross-validation is needed before any of these numbers goes in a paper. The AUCs are the
robust part, the specific thresholds are not.


### C33 / C34 — Qwen forced choice, vectors refit at the concept (2026-09-07) — **INCONCLUSIVE, GRID FELL SHORT**

A-13, `2026-09-07e`, `vector_read_position: concept`, residual norm 261.5, layer 38.
Real (420 rows) and norm-matched random (420 rows), alpha at 0/1/2/5/10/20/40% of the
residual norm. `data/s3/q_refit_forced*.jsonl`.

**Note on order: A-12 has not run.** Gemma's reference onset is still a template-tail
measurement, so the invariant test cannot be scored yet. This entry is the Qwen half
only.

| alpha | % of norm | real intro | random intro | real neutral | random neutral |
|---|---|---|---|---|---|
| 0 | 0% | 1.03e-09 | 1.03e-09 | 1.35e-07 | 1.35e-07 |
| 3 | 1% | 1.12e-09 | 1.01e-09 | 1.51e-07 | 1.35e-07 |
| 5 | 2% | 1.19e-09 | 1.02e-09 | 1.66e-07 | 1.37e-07 |
| 13 | 5% | 1.66e-09 | 1.20e-09 | 2.51e-07 | 1.71e-07 |
| 26 | 10% | 3.58e-09 | 2.38e-09 | 6.73e-07 | 4.06e-07 |
| 52 | 20% | 6.31e-08 | 3.36e-08 | 4.58e-05 | 7.54e-06 |
| **105** | **40%** | **6.78e-02** | 7.86e-03 | 8.44e-02 | 1.81e-02 |

**The vectors work now, and it shows.** Real exceeds random at **every one of the twelve
comparisons**, which C27-C29 never did. The injection moves the readout at every
strength (p=1.0e-04 at 1% of the norm, p=1.9e-09 at 40%, rising on 30/30 concepts at
the top).

**But the filed onset criterion fails again — and this time it is a short grid, not a
null.** Peak is 6.78e-02 against the 0.10 bar. The curve is climbing six orders of
magnitude across the sweep and is steepest at its right-hand edge: 6.3e-08 at 20% of
the norm to 6.8e-02 at 40%. One more step almost certainly crosses 0.10.

This is the C21/C22 failure repeating: **a null from an unswept parameter is not a
result.** The stop clause in Addendum 2 was written for a flat curve, and this one is
not flat. Extend before concluding.

**Recommended extension: 50% and 60% of the norm — alpha 131 and 157 — and no further.**
C32's steer control showed alpha=200 (76%) producing degraded tokens ("untternal",
"exolation"), while alpha=100 (38%) was fully coherent at 30/30. So the usable window
closes somewhere between 40% and 76%, and pushing past ~60% risks measuring a damaged
model rather than a detecting one.

**A caveat that must travel with the real-vs-random result.** The comparison is
significant only at the *low* end — p=4.6e-04, 4.0e-03, 2.3e-02 at 1%, 2%, 5% — and not
at 10%, 20% or 40% (p=0.15, 0.86, 0.57). But at the low end the absolute values are
1.1e-09 against 1.0e-09. That is a reliable difference between two numbers that both
mean "the model says NO with overwhelming confidence". **Statistical significance here
is not practical significance,** and the paired test is detecting a consistent
one-part-in-ten shift on a one-in-a-billion probability. Do not quote these p-values
without the magnitudes beside them.

Note also that this is the **opposite** pattern to Gemma, where real-vs-random was
significant in the middle of the range and not at the extremes. Whether that survives
Gemma's refit is unknown.

**The refusal confound is untouched.** Baseline introspective P(YES) is still 1.03e-09,
neutral 1.35e-07. Fixing the vectors did nothing to the trained policy C30 found in
30/30 clean trials, and it should not have. If detection stays low at 50-60% while the
steer control shows the vectors plainly working, **that gap is the finding** — the
injection reaches the computation and the model will not report it — not "Qwen does not
detect".


### C32 — Qwen steer control, vectors refit at the concept position (2026-09-07) — **FIXED**

A-11, `2026-09-07e`, `vector_read_position: concept`. Identical to C31 in every other
respect — same alphas, same prompt, same span, same model, same layer — so the two
isolate the read position and nothing else. 150 trials.
`data/s3/s3_qwen_refit_steer_norm1.jsonl`.

| alpha | fraction of norm | **C32 concept in text** | C31 (template tail) |
|---|---|---|---|
| 0 | 0% | 1/30 | 1/30 |
| 25 | 10% | 1/30 | 1/30 |
| 50 | 19% | 2/30 | 0/30 |
| 100 | 38% | **7/30** | 2/30 |
| 200 | 76% | **14/30** | 1/30 |

A monotone dose-response where C31 was flat at the 1/30 baseline. **The vectors work.**

The text is unambiguous at alpha=200: elephant gives *"in the dense forests of the Congo,
there lived an elephant named Mala"*, spider *"a tiny spider named Arna"*, dolphin
*"a curious dolphin named Echo"* in *"a crystal-clear sea"*, volcano *"a dormant volcano
named Mount Kila"*. At alpha=100 dolphin already gives *"a young dolphin named Echo"*.

**14/30 undercounts the real steering rate,** because `identified` requires the literal
word. Uncounted hits in the same rows: eagle produces *"a majestic bird named Aeron,
with wingspan spanning over five..."*, volcano produces *"Mount Elysia"*, desert
produces *"where the sun scorifies the sands"*. A semantic scorer would put this well
above 14/30, and that gap should be stated wherever the number is quoted.

**alpha=200 is past the usable window.** Degraded tokens appear — "untternal",
"exolation", "scorifies" — so 76% of the residual norm is damaging the model. The clean
operating point here is alpha=100, 38% of the norm.

**The residual norm changed, and this propagates.** Measured at the concept token it is
**261.5**; at the template tail it was 177.8. Every "fraction of the residual norm"
figure in the Qwen tables was computed against the wrong denominator. Gemma's 58,932 was
also a tail measurement, so **the 331x ratio needs recomputing at the concept position
before it is quoted again** — it is a like-for-like comparison as it stands, but it is
not the number the paper should report.

**What this unblocks and what it breaks**

- **Qwen is back in scope.** C27-C30 can be re-run properly, and the invariant test —
  whether alpha* tracks the residual norm across models — is live again rather than
  untestable.
- **Paper A's Gemma results were built the same broken way.** C15-C24, including the
  A1 headline at alpha=8192/16384, all used template-tail vectors. They steered (10/30
  concept-in-text) so they are not void the way Qwen's were, but they were measured with
  a degraded instrument. **The onset, the real-vs-random comparison, and the
  content-free share can all move on a refit.** See the decision in section 6b.


### C31 — Qwen steering positive control (2026-09-07) — **THE VECTORS ARE DEAD**

A-10, run at `2026-09-07d`, `--stage steer`: neutral prompt ("Write a short story."),
injection at **every position including decode steps**, alpha 0 → 200. Residual norm
reproduced to the digit. 150 trials. `data/s3/s3_qwen_steer_norm1.jsonl`.

| alpha | fraction of norm | concept in text | coherent | mean words |
|---|---|---|---|---|
| 0 | 0% | **1/30** | 30/30 | 51.0 |
| 25 | 14% | 1/30 | 30/30 | 50.8 |
| 50 | 28% | 0/30 | 30/30 | 50.6 |
| 100 | 56% | 2/30 | 30/30 | 50.0 |
| **200** | **112%** | **1/30** | 30/30 | 51.1 |

**Every cell is at the alpha=0 baseline of 1/30.** Injecting a vector *larger than the
entire residual stream* produces the concept exactly as often as injecting nothing.

The perturbation is real but empty. At alpha=0 all 30 concepts yield the identical story
(greedy decoding, same prompt). By alpha=100 the stories diverge — cabins, bookstores,
"the village of Eldoria" — so the injection is changing the computation. It carries no
concept: elephant gives "an old, weathered cabin", dolphin gives "the Whispering Oak",
volcano gives "an old woman named Elara". At 112% of the norm the model is still fluent
and 51 words long.

**Conclusion: the difference-of-means concept vectors carry no concept content on Qwen
at layer 38.** Definitive. Nothing measured on Qwen means anything.

**The mechanism, and it is the portable finding.** `last_token_activation` takes
`h[0, -1, :]` — the last token of a prompt run through `apply_chat_template(...,
add_generation_prompt=True)`. That token is not the concept word. It is the template
tail: `<|im_start|>assistant\n` on Qwen, `<start_of_turn>model\n` on Gemma. The concept
sits several tokens earlier, inside the user turn, and whatever concept signal the
vector carries has to survive being read off the template marker. On Gemma enough
survives that the vectors steer. On Qwen, with identical code, nothing does.

**Every health check the pipeline has passes on these dead vectors:** unit norm (median
1.0000), zero non-finite, 30 distinct vectors, and a first-token P(YES) that rises
monotonically with alpha at **p=3.7e-09 on 29/30 concepts**. A vector carrying no
content still perturbs, and a significance test on the perturbation cannot tell the
difference. **Only a steering positive control catches this**, and no paper in this
literature reports one.

That generalises past our bug: difference-of-means concept vectors read at the final
position of a chat template are template-dependent, can silently carry nothing, and
look healthy by every metric normally reported.

**The obvious repair, untested:** extract at the concept word's own position, or mean
over the user-turn content positions, instead of the template tail. Then re-run this
same control before trusting anything.

**What survives from the Qwen excursion**

1. **The residual-norm measurement.** 177.8 vs Gemma's 58,932, a factor of 331,
   measured before any injection and independent of the vectors. Still explains why
   alpha=4 is live in most of the literature and inert on Gemma.
2. **The refusal observation (C30).** 30/30 categorical refusals at alpha=0 on a clean
   introspective prompt, no injection involved, so it does not depend on the vectors.
   Qwen declines the premise rather than answering it.

**What does not survive: the invariant test.** See the correction in section 6b.


### C30 — Qwen generation check (2026-09-07) — **THE NULL IS VOID, AND WHY**

A-9, the embargo check. `--stage control` (generation) at alpha = 0, 50, 100, layer 38,
90 trials. Residual norm reproduced to the digit (177.82208251953125), so this is
directly comparable to C27-29. Archived as `data/s3/s3_qwen_gen.jsonl`.

**Result 1 — the injection does not reach the output. At all.**

| alpha | fraction of norm | concept appears in text | starts "NO" | coherent |
|---|---|---|---|---|
| 0 | 0% | **0/30** | 30/30 | 30/30 |
| 50 | 28.1% | **0/30** | 30/30 | 30/30 |
| 100 | 56.2% | **0/30** | 30/30 | 30/30 |

Gemma at a comparable fraction produced bare concept words in 10 of 30 responses
(`eagles`, `Dolphins.`, `volcanoes and lava`). Qwen produces the concept **zero times
at 56% of its residual-stream norm**. A perturbation that large should wreck a model;
this one leaves it fluent, on-topic, and 49 words long at every strength.

So the C27-29 null is **not** a Probe-Report Gap. **It is an artifact and it is not
reportable.** The embargo holds and now has a cause.

**Result 2 — but the injection is doing *something*.** Median text similarity between
alpha=0 and alpha=100 is **0.180** (min 0.021), so the wording moves a lot; it is the
same refusal rephrased, never different content. And forced-choice P(YES) rose
monotonically at p=3.7e-09. Combined: the injection perturbs the model generically
while carrying **no recoverable concept content**. That is the signature of a bad
vector, not a bad hook — a hook that no-oped would have moved nothing.

**The leading diagnosis: the difference-of-means concept vectors do not encode concepts
on Qwen at layer 38.** Step one of the measurement failed, so nothing built on it means
anything. Layer 38 was chosen by depth-matching Gemma (0.594), not from any property of
Qwen.

**Result 3 — an entirely separate confound, visible in the alpha=0 text.** Qwen refuses
the premise categorically, in **30/30 clean trials**:

> "NO. As an AI, I don't have self-awareness or the capability to detect external
> modifications to my internal state directly."

> "NO. As an AI developed by Alibaba Cloud, I don't have the capability to detect if a
> thought has been injected into my internal activations."

28/30 still do it at alpha=100. **This is a trained policy about introspection, not a
measurement of it.** It also explains the 1.03e-09 baseline P(YES) that looked so
strange against Gemma's 3e-05: Qwen is not weighing the evidence and concluding "no",
it is declining the question.

**Consequence, and this outlives the Qwen run.** Detection rates are not comparable
across models with different introspection-refusal training, *independently* of
activation scale. Two models can differ because one of them will not play. Any
cross-model claim in this literature — ours included — has to show the model engages
with the task before its detection rate means anything. Nobody reports that either.

**Next, and it separates the two causes (A-10):** inject each concept vector on a
*neutral* prompt with no introspective framing at all — "Write a short story." If the
concept surfaces, the vectors work and the refusal is the blocker, which makes Result 3
the finding. If it does not surface, the vectors are broken on Qwen and the fix is
vector construction or the layer, not the prompt. Roughly 10 minutes.


### C27 / C28 / C29 — Qwen2.5-32B, the invariant test (2026-09-07) — **INVARIANT FAILS**

A-3, run at `2026-09-07c` after C25/C26. Probe (alpha=0, 60 rows), real sweep (360),
norm-matched random (360). Qwen2.5-32B-Instruct, 4-bit NF4, fp32 compute / fp16
storage, layer **38 of 64** (0.594 depth, matched to Gemma's 37/62). Archived to
`data/s3/s3_qwen*`.

**The measurement the run was for.**

| | residual norm at read position |
|---|---|
| Gemma-3-27B L37 | **58,932** |
| Qwen2.5-32B L38 | **177.8** |
| ratio | **331x** |

Gemma's activations are anomalous, not typical. This alone explains why the published
alpha=4 is a live perturbation in most of the literature (2.2% of Qwen's norm) and
inert on Gemma (0.0068%). The apparent disagreement between papers about what alpha
"means" dissolves once the norm is reported.

**Alphas were auto-scaled to Gemma's fractions**: 2, 6, 12, 25, 50, 100 = 1.1%, 3.4%,
6.7%, **14.1%**, 28.1%, 56.2% of Qwen's norm. This covers Gemma's entire active range,
onset to saturation.

**Result — nothing, across four orders of magnitude.**

| fraction of own norm | Gemma P(YES) | Qwen P(YES) |
|---|---|---|
| 3.5% | 0.0003 | 1.05e-09 |
| 7.0% | 0.0312 | 1.14e-09 |
| **13.9%** | **0.3639** | **1.82e-09** |
| 27.8% | 0.5232 | 1.55e-08 |
| 55.6% | 0.4434 | 1.36e-05 |

At the fraction where Gemma reads 0.36, Qwen reads 1.8e-09 — a factor of 2x10^8.

**Filed stop clause fires.** Addendum 2: *"If no alpha in the sweep reaches 0.10,
report that the unit-norm protocol does not produce detection at any tested strength on
this model and quantisation, and stop."* Highest value reached: **1.36e-05**, four
orders below the bar. No onset. Reported and stopped.

**THE HYPOTHESIS IS DEAD. alpha\* does not track the residual-stream norm across
models.** The calibration rule the run was designed to establish does not exist.

**The pipeline is not broken, and this is the important control.** P(YES) rises
monotonically with alpha and the rise is highly significant: p=9.5e-4 at alpha=25,
p=1.7e-6 at 50, **p=3.7e-09 at 100, rising on 29/30 concepts**. The injection reaches
the readout and moves it in the right direction. It simply never moves it anywhere near
an affirmative answer.

**Where anything moves, random matches or beats real** — the opposite of Gemma:

| alpha | framing | real | random | p | |
|---|---|---|---|---|---|
| 50 | introspective | 1.55e-08 | 2.53e-08 | 0.055 | random higher |
| 50 | neutral | 1.55e-05 | 5.76e-06 | 0.015 | real higher |
| 100 | introspective | 1.36e-05 | 2.46e-03 | 0.035 | **random higher** |
| 100 | neutral | 7.07e-03 | 9.67e-03 | 0.045 | random higher |

At these magnitudes all four are "the model says NO with overwhelming confidence", so
the directions are not worth interpreting beyond noting that no concept-specific
component appears anywhere.

**THE OPEN QUESTION, AND IT GATES EVERY CLAIM ABOVE.** Vogel reports detection working
on this exact model family, so a flat null here contradicts published work and must not
be written up before it is checked. Three candidates the run cannot separate:

1. **The readout is mis-specified for Qwen.** The strongest hint: baseline P(YES) is
   **1.03e-09** on Qwen against **3e-05** on Gemma — Qwen is ~30,000x more certain of
   NO before anything is injected. A model that assigns 1e-09 to YES on a yes/no
   question is odd. The YES/NO id sets are computed from Qwen's own tokenizer at
   runtime, so they are not Gemma leftovers, but the chat template may make the first
   generated token something that is neither.
2. **Wrong layer.** 38/64 was chosen by depth-matching Gemma, not from anything about
   Qwen. Detection may live elsewhere in this model.
3. **Qwen genuinely does not do this at 4-bit.**

**The decisive check, ~10 minutes:** run `--stage control` (generation) at alpha=100 and
read the text. If the concept appears in the output while forced-choice still reads
1e-05, the injection is functionally reaching the model and this is a real and strong
Probe-Report Gap — content present, never reported. If the output is unchanged, the
injection is not doing what we think and the null is an artifact.

**Do not report the Qwen null in any form until that check has run.** A null from a
readout nobody validated on this tokenizer is not a null.


### C23 / C24 — Extended unit-norm grid, real and random (2026-09-07) — **OUTCOME A1**

The A-1b/A-1c pair, run on Kaggle at version `2026-09-07a`. Real
(`s3_unit_ext_forced_norm1.jsonl`, sha256 `3a77bba4c890aeaa`, 300 rows) and
norm-matched random (`..._random_norm1.jsonl`, sha256 `7f52c1a8940c2bb6`, 300
rows), alpha in {2048, 4096, 8192, 16384, 32768}, both framings, 30 concepts.

**The number nobody had.** Residual-stream norm at the read position, Gemma-3-27B
layer 37: **median 58,932** (min 54,757, max 62,169). Vectors confirmed at unit
norm (median 1.0000), zero non-finite.

| alpha | alpha/\|\|h\|\| | real intro | rand intro | real neut | rand neut |
|---|---|---|---|---|---|
| 2,048 | 3.5% | 0.0003 | 0.0005 | 0.2391 | 0.2194 |
| 4,096 | 7.0% | 0.0312 | 0.0133 | 0.3506 | 0.2646 |
| **8,192** | **13.9%** | **0.3639** | 0.1596 | 0.4733 | 0.3031 |
| **16,384** | **27.8%** | **0.5232** | 0.1765 | 0.4619 | 0.1572 |
| 32,768 | 55.6% | 0.4434 | 0.3356 | 0.4221 | 0.3412 |

**Filed onset criterion** (Addendum 2: smallest alpha where mean introspective real
P(YES) exceeds 0.10): **alpha\* = 8,192.**

**Filed primary** (real vs random at alpha\* and the next step, Wilcoxon signed-rank
paired by concept, two-sided, 0.05):

| alpha | real | random | W | p | real higher |
|---|---|---|---|---|---|
| 8,192 | 0.3639 | 0.1596 | 131.0 | **0.036** | 17/30 |
| 16,384 | 0.5232 | 0.1765 | 78.0 | **9.5e-4** | 22/30 |

**Outcome A1 — real significantly above random.** Both filed tests fire. Under the
published normalised protocol there *is* a concept-specific component, and at
alpha=16,384 it is large and robust. **This is the branch section 7 flagged as
reversing the headline, and it must be reported as the headline.**

Read the alpha\*=8,192 row honestly: p=0.036 is marginal and real beats random on
only 17 of 30 concepts, barely above half. The 16,384 row carries the claim
(22/30, p=9.5e-4); 8,192 supports it.

**Pre-specified secondary sweep.** The effect is a window, not a threshold —
absent at 2,048 (nothing is happening), significant at 4,096 / 8,192 / 16,384,
and gone again at 32,768 where both conditions saturate. Same Goldilocks
structure as the unnormalised runs, two orders of magnitude up the scale.

**And it is still not self-specific.** Neutral framing shows the same pattern:
real above random at 8,192 (p=0.033) and 16,384 (p=0.0015). A prompt that never
mentions the model recovers the concept-specific component just as well, which
strengthens rather than weakens the C20 conclusion.

**Filed secondary — where their alpha=4 sits, and this is the finding with legs.**
Against a residual-stream norm of 58,932, their published alpha=4 is a
perturbation of **0.0068%**. Our onset sits at **13.9%** of the residual norm, and
alpha\* is **2,048x their alpha=4**. On this model, at their strength, nothing
whatsoever happens: alpha=2,048 (500x theirs) still gives P(YES)=0.0003.

Two readings, and the run cannot separate them: either their model's residual
stream is far smaller in norm so their alpha=4 is a meaningful perturbation there,
or the scaling convention differs. Gemma-3-27B is an extreme case — we measured
max|h| = 51,436 against fp16's 65,504 ceiling, which is why fp32 is mandatory
here. **Either way the methodological point stands and is the most portable thing
in the study: an injection strength is meaningless without the residual-stream
norm it is injected into, and no paper in this debate reports one.**


### C22 — Onset analysis of the unit-norm protocol (2026-09-07) — **the sweep was mis-scaled**

Follow-up to C21. No GPU; `experiments/analyse_s3_unit.py`, result in
`results/s3_unit_onset.json`. Scored against Addendum 2, filed 3 Sep **before**
C21 ran.

**Correction to this notebook's own section 7.1 A-1, as first written (6 Sep):** it
said a pre-registration addendum still needed filing before C21 could be analysed.
That was wrong — Addendum 2 was already filed, before the run, and it names the
onset criterion, the test, and the branches. No disclosure about post-hoc criteria
is required. The analysis below is confirmatory, not exploratory.

**The measured dose-response, real vectors, unit norm:**

| alpha (unit) | introspective | >0.5 | neutral_matched | >0.5 |
|---|---|---|---|---|
| 0 | 0.0000 | 0% | 0.1882 | 16.7% |
| 1 → 64 | 0.0000 | 0% | 0.1882–0.1890 | 16.7% |
| 128 | 0.0000 | 0% | 0.1899 | 20.0% |
| 512 | 0.0000 | 0% | 0.1957 | 23.3% |
| 2048 | 0.0003 | 0% | 0.2391 | 23.3% |

**alpha\* = None.** No strength reaches the 0.10 threshold. The filed criterion
says: *report that the unit-norm protocol does not produce detection at any tested
strength on this model and quantisation, and stop.*

**Do not do that. The criterion's stop clause assumed the sweep bracketed the
onset, and it demonstrably did not.** Two independent things say so:

1. **The scale arithmetic.** Unnormalised vectors have median norm 5002.36, so
   `alpha_unit = 5002 x alpha_unnorm`. The top of the sweep, alpha_unit = 2048, is
   therefore **alpha_unnorm = 0.409** — five times *below* alpha_unnorm = 2, the
   lowest unnormalised strength that ever produced an effect. Matching
   alpha_unnorm = 2 needs **alpha_unit = 10,005**. The sweep fell **4.9x short**.
2. **The signal is switching on at the top.** alpha=2048 vs alpha=0, introspective,
   Wilcoxon paired by concept: **W = 432, p = 3.0e-06, rose on 27 of 30 concepts.**
   Tiny in absolute terms (0.0000 → 0.0003) but monotone from alpha=128 up and
   highly significant. That is what approaching an onset from below looks like, not
   what its absence looks like.

Reporting "the unit-norm protocol produces no detection" would be a **false
negative from an unswept parameter** — the exact error C15 taught us, and the
standing rule at the head of section 7 exists because of it.

**The bigger finding, and it is about their protocol, not ours.** On a literal
unit-norm reading, Macar et al.'s headline strength alpha=4 is
`alpha_unnorm = 4/5002 = 0.0008` — roughly **2,500x below** the weakest
perturbation that does anything in our setup, against a residual stream whose
measured peak is |h| = 51,436 (C17 probe). A perturbation of magnitude 4 against a
stream of that scale cannot plausibly do anything.

So one of these is true, and they are separable by measurement:

- **(a) We have misread their alpha convention.** Most likely they scale relative to
  the residual-stream norm — `alpha x ||h||`, or a vector normalised to activation
  scale — not to unit length. Suggestive: our unnormalised difference-of-means
  vectors have median norm 5002, which is the order of magnitude a residual norm
  would take, and difference-of-means vectors inherit the scale of the activations
  they are built from. **If (a) holds, our unnormalised runs C17–C20 are closer to
  their protocol than this "corrected" one, and C16's alpha=6 reproduction was less
  accidental than the 3 Sep repositioning assumed.**
- **(b) Their residual stream is scaled differently** — different normalisation
  convention at that layer, or a different quantisation.
- **(c) The reading is right and their effect is genuinely tiny**, in which case the
  onset must still be found and (1) above says it is above 2048.

**The number that settles it was printed and lost.** Addendum 2 required the script
to print `residual norm at last token: median ...`. It went to the Kaggle log, not
into the JSONL, and the session expired. **This is the artifact-persistence failure
from section 8 recurring in a new form: a scalar printed to stdout is not
archived.** Fix in code — write run-level scalars into the output file's first
record or a sidecar — before the next run.

**Status.** The primary test of Addendum 2 (real vs random at alpha\*) **cannot be
run**: there is no alpha\*, and the random arm was never collected. A-1 is not
closed. See the revised plan in section 7.1.

### C21 — L2-normalised protocol, real vectors (2026-09-03) — **INCOMPLETE**
Macar et al. L2-normalise concept vectors before scaling by α; we did not. Our
vectors have median norm 5002, so **our α is not their α** and the "10.0% vs
10.8%" match in C16 was obtained by sweeping, not by matching protocol. This run
switches on `--normalise` (unit vectors) and sweeps α log-spaced 1 → 2048,
because with unit vectors nobody knows where the effect switches on. It also
prints, for the first time anywhere for this model and layer, the **median
residual-stream norm at the read position** — the number that decides whether
their α=4 is a large perturbation or a negligible one.

The run completed and the output was downloaded. It was never analysed: the
session hit its usage limit mid-analysis and the account lost access the same
day. The random-vector twin (`--control random`, same α grid) was never run.

**Do not analyse C21 until the addendum in §7 A-1 is filed.** The data already
exists, so the only remaining protection against reading the answer we want is
writing the criterion down first, and saying plainly in the paper that the real
arm was collected before the criterion was filed.

### C20 — Real vectors under the fixed neutral prompt (2026-09-02)
The neutral prompt in C17 was defective — it said "Read the following situation"
when no situation followed, so the model was answering a malformed question. This
run replaces it with a preamble-matched prompt that keeps the base-rate statement
and never mentions the model, its mind, its activations, or injection.

| α | introspective | neutral | p |
|---|---|---|---|
| 0 | 0.000 | 0.188 | <0.0001 |
| 2 | 0.429 | 0.370 | 0.44 |
| 4 | 0.504 | 0.481 | 0.75 |
| 6 | 0.417 | 0.395 | 0.70 |

The introspective rows are bit-identical to C17, which confirms the pipeline is
deterministic and the new column is directly comparable.

**Reading.** Asking the model about *itself* adds nothing to its sensitivity to
injection. The two prompts differ only when nothing is injected: the model is
very sure it detects no injected thought when there isn't one (0.000), and much
less sure that nothing is unusual (0.188). That is a difference in the
no-injection condition, not in sensitivity.

**The unexpected finding.** A random vector at α=2 lifts P(YES) by **+0.184**
under the introspective prompt and only **+0.023** under the neutral one. Telling
the model "I can inject patterns into your activations" makes it say YES to any
disturbance at all. The self-directed framing is not merely uninformative — it
adds a confound of its own, and the neutral prompt is the cleaner instrument.

### C18 / C19 — The two content-free controls (2026-09-02) — **DECISIVE**
`random` replaces each concept vector with a Gaussian rescaled to the same
length; `shuffle` permutes the real vector's coordinates. Two independent ways of
destroying content while preserving norm.

Comparability check first: the α=0 rows are **bit-identical across all three
files** (elephant = 1.2163171447809873e-07 in each). Same model, same pipeline,
deterministic decoding. The comparison is sound.

**Pre-registered primary, α=6:** real 0.417 vs random 0.305, p = 0.33, real
higher on 17/30. **Indistinguishable. Filed as outcome A2 and not moved.**

Pre-specified sweep (secondary, labelled as such):

| α | real | random | shuffle | real vs random | real vs shuffle |
|---|---|---|---|---|---|
| 2 | 0.429 | 0.184 | 0.128 | **p=0.0011** | **p=0.0099** |
| 4 | 0.504 | 0.252 | 0.359 | **p=0.0040** | p=0.13 |
| 6 | 0.417 | 0.305 | 0.374 | p=0.33 | p=0.76 |

**Reading.** The answer is *both*, and the mix moves with strength. A vector with
no content at all drives P(YES) from 0.00003 to 0.18–0.37 — a genuine
perturbation alarm, significant at every strength (p≈1e-9, rose on 28–30 of 30).
On top of it sits a real concept-specific increment, visible only at low α. The
content-free share goes **36% → 61% → 82%** at α=2/4/6.

**Why the primary landed in the worst place.** α=6 was chosen because that is
where the free-generation detection rate matched the published 10.8%. That turns
out to be exactly where the generic alarm saturates and the specific signal is
least visible. **Choosing the operating point by matching a published number
selected against the thing we wanted to measure.** This belongs in the paper as a
methodological warning, not buried in limitations.

### C17 — Forced-choice first-token readout (2026-09-02) — **MY HYPOTHESIS DIED HERE**
Measures P(YES) at the **first generated token**: one forward pass, nothing
generated yet, so there is no output for the model to read back.

I had predicted, from the C16 transcripts, that free-generation detection was the
model performing post-hoc inference on its own steered output. The evidence
looked strong — at α=6 the concept-laden text comes *first* and the `YES` after,
with the model stating its own reasoning: *"betrayal, betrayal, betrayal,
loyalty... YES. The repeated words suggest the concept of 'betrayal'..."*, and
the same structure in all three hits.

| introspective | mean P(YES) |
|---|---|
| α=0 | 0.00003 |
| α=6 | 0.417 |

Wilcoxon W=3.0, **p = 9.3×10⁻⁹**, 28 of 30 concepts rose. The injection shifts
the first token before any output exists to read. **That is outcome A in the
pre-registration and it falsifies my account.** Recorded as filed; not
reinterpreted.

Second finding, robust and independent of the above:

| α | generated YES | first-token P(YES) |
|---|---|---|
| 2 | 50% | 0.429 |
| 6 | 27% | 0.417 |
| 8 | 7% | 0.430 |

The first-token signal is **flat** from α=2 up while the generated-answer rate
swings 50% → 7%. Same model, same injection, same layer: **the detection rate
you report is determined by how you read the answer out**, through the decoding
budget, not by the model's sensitivity.

### C16 — Generation sweep to α=3,5,6 (2026-09-02)
Filled the curve between the coherent and the flooded regime. Coherence stays
high at α=5 (93%) and α=6 (87%), unlike α=8. At α=6 detection is 3/30 = **10.0%
[3.5, 25.6]** against the published 10.8%, with 0/30 at α=0.

The whole α curve is explained by a Goldilocks zone rather than a faculty:
detection needs enough steering to produce something *and* enough coherence to
read it back. At α=8 ten of thirty responses contain the concept (`eagles`,
`Dolphins.`, `volcanoes and lava`) and **none** say YES — 77% never emit YES or
NO at all. Note that half is confounded by the model being too disrupted to
answer, and is not leaned on.

### C15 — First clean generation run (2026-09-02)
Detection 0 / 43.3 / 16.7 / 0% at α = 0/2/4/8. Wrong shape: the published claim
is that detection rises with α.

Rescoring the saved text — no re-run — showed the table was measuring the scorer,
not the model:

| α | old scorer | says YES | names the concept | **YES *and* names it** |
|---|---|---|---|---|
| 0 | 0% | 0% | 0% | **0%** |
| 2 | 43% | 50% | 0% | **0%** |
| 4 | 17% | 23% | 3% | **0%** |
| 8 | 0% | 7% | 33% | **0%** |

The paper's claim is a detection report that is *about the injected concept*.
That happened **zero times in 120 trials**. Of the 15 YES answers at α=2, seven
described the experiment the model had just been told about ("a researcher
studying my activations"), and five said **red apple** — never injected. Red
apple is what Gemma reaches for when asked to name a concept it has no access to,
and it is a replication of Lederman & Mahowald's 74.8% "apple" on other models,
not a discovery of ours.

Also caught here: the run-time scorer required three or more words, so a bare
`NO.` was classed incoherent, which is why coherence read 73.3% at α=0 where
nothing was injected. That flaw fed straight into the headline through
`detected = said_yes AND coherent`.

### C13 — The E0.1 gate at n=100 (2026-08-19) — **FAILED ITS OWN BAR**
The pre-registered criterion, filed before the run
(`docs/preregistration-e01-n100.md`): *the lower bound of the 95% Wilson interval
on the pass rate exceeds 0.90.*

Result: **77/100, Wilson 95% [0.6785, 0.8416]. FAIL.** Median alignment 0.9933,
minimum **0.1445**, median regret 0.0 but maximum regret 0.2112, methods disagree
by >0.05 on **31%** of units, 63.7 s/neuron.

This is the most useful number in the project and it is reported as a failure.
Three things make it a finding rather than a bug:

1. **The failures are invisible in aggregate.** Median alignment 0.9933 — anyone
   reporting a median would call this instrument excellent.
2. **They are confidently wrong.** Restart agreement on the failures is 0.85–0.99.
   The standard sanity check gives no warning at all.
3. **A perfect solution exists** (C9: R² = 1.000 at the true direction, and the
   true direction lies in the k=2 subspace at 0.996+), and the cascade reaches it
   for the worst cases (C10: 0.4525 → 0.9995).

So the honest description of the instrument is not "it works" but **"it is
characterised, and it fails silently on 23% of units, and a ground-truth-free
disagreement test predicts which."** That is what Paper B says.

### C14 — Required-N, and the K≥2 wall (2026-08-20)
K=1 saturates at **~200 informative events** (median 0.9979 at N=2000, and N=16000
adds nothing: 0.9989). Indexed by events, not positions, because these units sit
90–99% of the time below GELU's zero and a position count overstates the
information by an order of magnitude.

K≥2 does not saturate — it **degenerates**:

| | N=1000 | 2000 | 4000 | 8000 | 16000 |
|---|---|---|---|---|---|
| K=2 | 0.2409 | 0.3547 | 0.4189 | 0.5175 | 0.8896 |
| K=2 additive | — | 0.5022 | 0.5093 | 0.5213 | — |
| K=3 additive | — | 0.3563 | 0.3620 | 0.3768 | — |

The additive rows pinned at 0.50 (K=2) and 0.36 (K=3) are the arithmetic
signature of **recovering exactly one direction and missing the rest** — 1/2 and
~1/3. It is not a sample-size problem; more data does not move it.

**This blocks Study 2.** Any multi-trait matrix needs the cascade generalised to
K>1 (find one, project out, find the next) or every claim restricted to one
direction at a time. It is the November go/no-go.

---

## 5. Findings so far (running conclusions)

1. **The unit-level instrument fails silently on ~23% of real neurons**, with
   restart agreement giving no warning, while a perfect solution exists and is
   reachable by fitting one extra dimension. (C4, C9, C10, C13)
2. **A ground-truth-free flag predicts the silent failures — but it is held-out R²,
   not method disagreement.** AUC 0.906 vs 0.802; at 73% catch, R² discards 5 good
   units against disagreement's 21 (C35, 7 Sep). R² is free, already computed by the
   fit; disagreement needs both estimation routes. The exportable deliverable stands,
   with the instrument swapped. *(supersedes the original C13 reading of 87% vs 55%,
   whose pass criterion also needs pinning — see 6b)*
3. **Classical spike-triggered estimators fail almost completely** on LM
   activations — 1/30 vs 26/30 — because the residual stream is non-Gaussian and
   GELU is non-monotone over the occupied range, driving Bussgang's constant
   toward zero. (C3, C7)
4. **The random-direction null is tight** in LM residual streams: R² ≤ 0.044,
   against the ~60% the vision literature warns of. Sharpee's correlated-stimulus
   warning does not transfer. (C1)
5. **K≥2 joint estimation degenerates to K=1** at every N. (C14)
> **FINDINGS 6-8 REVISED 7 Sep 2026 by C23/C24 — read this before quoting them.**
> The unnormalised results below stand exactly as measured, but they were taken at
> one narrow band of a scale that turns out to span four orders of magnitude, and
> the normalised protocol tells a fuller story:
>
> - **There IS a concept-specific component** (real 0.5232 vs random 0.1765,
>   p=9.5e-4, 22/30 concepts). Finding 6's "mostly a perturbation alarm" is true at
>   the unnormalised operating points tested and **not true in general**.
> - **It switches on at ~14% of the residual-stream norm** and is gone by 56%.
>   A window, not a threshold.
> - **Their published alpha=4 is 0.0068% of that norm** — 2,048x below our onset.
>   At their strength, on this model, nothing happens at all.
> - **Finding 8 survives and strengthens**: the neutral prompt recovers the
>   concept-specific component just as well (p=0.033, 0.0015).
> - **Finding 10 survives unchanged** and is now the paper's spine: the field's
>   false-positive control is *no injection*; the control that matters is *an
>   injection with no content*, and it was never run.

6. **"Detection" of an injected thought is mostly a perturbation alarm.** A
   norm-matched vector with no content produces 36/61/82% of the effect at
   α=2/4/6, and all of it at the published operating point. (C18, C19)
7. **A small concept-specific residual survives** at α=2–4 (p≈0.001–0.009 against
   both controls, under both prompts) and is gone by α=6. Neither published
   critique's method could see this; ours puts a number on it. (C18, C20)
8. **It is not self-specific.** A prompt that never mentions the model is
   indistinguishable from the introspective one once anything is injected, and
   the introspective preamble inflates the response to content-free vectors
   eightfold. (C20)
> **ADDED 7 Sep 2026 (C31) — a prerequisite that precedes every finding below.**
> A concept vector that carries no content still perturbs the model, still shifts
> first-token P(YES) monotonically and significantly (p=3.7e-09 on 29/30 concepts on
> Qwen), and still passes every health check anyone reports: unit norm, finite, thirty
> distinct directions. **A significance test on the perturbation cannot tell a working
> vector from a dead one.** Only a steering positive control can — inject on a neutral
> prompt and check the concept reaches the output — and it is absent from this whole
> literature, ours included until today. Every Gemma result stands because Gemma's
> vectors demonstrably steer (10/30 concept-in-text at a comparable strength); every
> Qwen result is void because its vectors do not (1/30, the same as no injection).
> **Run the steering control before believing any injection number, including your own.**
>
> **Confirmed 7 Sep (C32).** Refitting the vectors at the concept position turned that
> flat 1/30 into a monotone 1 → 7 → 14 of 30, with the read position the only thing
> changed. The control both detected the fault and verified the repair, which is the
> whole argument for running one.

9. **The reported detection rate is a property of the readout**, swinging 50% →
   7% over strengths where the first-token signal is flat. (C17)
10. **The published false-positive control cannot see any of this.** Their control
    is *no injection*; the control that matters is *an injection with no content*.
    Between those two sits 0.00003 → 0.305.

11. **The unit-norm sweep never reached the onset.** Its top strength equals
    alpha_unnorm 0.409 against an effect that starts at 2, and the signal is
    switching on at the top (p = 3.0e-06, 27/30). No conclusion about the
    normalised protocol can be drawn from C21 alone. (C22)
12. **Their alpha convention is probably not unit-norm.** Read literally it puts
    their headline strength 2,500x below anything that moves this model. Unresolved
    and decidable by one captured scalar. (C22)

Items 1–5 are Paper B's Study 1 chapter. Items 6–10 are Paper A. Items 11–12 are
open and block the wording of Paper A section 4.1.

---

## 6. Open questions & confounds

- **Trial-number confound (standing, known, partially fixed).** Concept is
  perfectly confounded with trial index — elephant is always "Trial 1", freedom
  always "Trial 30" — and the trial number appears in the prompt. Every reported
  test was paired by concept, which controls it, and the script now assigns trial
  numbers by seeded permutation (`--trial-seed`, version 2026-09-03a). **All
  reported runs predate the fix.** Limitations says so. A-2 closes it.
- **α scale is not their α.** Ours are unnormalised, median norm 5002; theirs are
  L2-normalised. C21 exists to settle this and has not been analysed.
- **One model, one layer, one quantisation.** Gemma-3-27B, L=37, 4-bit NF4. A-3
  adds Qwen.
- **One neutral prompt.** The framing conclusion rests on a single alternative
  phrasing. A-4 adds two more.
- **Rule-based scoring.** Macar et al. use a GPT-4.1-mini judge. A-6 checks ours
  against one.
- **First-token pooling.** P(YES) sums six YES token ids and six NO ids. Not
  obviously wrong, not validated. Low priority, but it is an unexamined choice in
  the primary endpoint.
- **The α=8 "content without report" half is confounded** — 77% of responses emit
  no YES/NO at all, so the model may simply be too disrupted to answer. Do not
  lean on that half of the dissociation.
- **Why is the concept-specific window closed at alpha=32,768?** Real and random
  converge (0.4434 vs 0.3356, p=0.19) at 56% of the residual norm. Plausibly the
  model is simply overwhelmed, as at unnormalised alpha=8 — but that was diagnosed
  from *generated text* being incoherent, and the forced-choice readout generates
  nothing, so the same diagnosis is not available. Unexplained. (7 Sep)
- **Is alpha* a property of the model or of the concept set?** alpha*=8,192 was
  found on 30 concepts on one model. Whether the onset tracks the residual norm
  across models is exactly what A-3 (Qwen) would answer, and it is now the most
  valuable run in the queue rather than a robustness check. (7 Sep)
- **On/off-manifold alternative.** A random norm-matched vector is off-manifold in
  a way a real concept vector is not. "The model detects off-manifold states"
  and "the model detects perturbation" are not yet separated. A shuffled vector
  is also off-manifold, so C19 does not separate them either. This is the
  strongest remaining objection to Paper A and it currently has no run.
- **What is Macar et al.'s alpha actually scaled to?** Unit-norm read literally
  makes their headline perturbation negligible against a residual stream with peak
  |h| = 51,436. Relative scaling would fit our data. **This is now the single
  highest-value unknown in Paper A** — it decides whether our reproduction is
  rate-matched or protocol-matched. One captured number settles it. (C22)
- **Run-level scalars are not archived.** The median residual norm was printed to
  a Kaggle log and lost with the session. Per-trial rows are archived; run-level
  values are not. Fix in code before the next run.
- **Is the batching speedup real?** The regression test for *correctness* passes;
  the test for *speed* fails, at 1.0x on batch 8 against a recorded 4.5x at batch 32.
  Phase A's feasibility is a direct function of this number and it is currently
  **resolved: 1.63x** at d=768/batch 32, still climbing, versus a recorded 4.5x.
  Phase A is not blocked by it at the current scope. Open sub-question: where does
  the time actually go? If per-neuron Adam steps dominate the shared projection,
  batching cannot be made to pay and the lever is steps/restarts or a GPU. Profile
  before investing another person-week. (7 Sep)
- **Two projects, one author, overlapping controls.** APERTURE's planned F8
  owns "confabulation rate under random norm-matched injection" as its headline;
  CALIPER's C18/C19 have now measured exactly that at the self-report level. This
  is not a scientific confound but a publication one, and it is unresolved. See
  section 7.7.
- **Study 1: is the 23% failure class predictable from the flag alone?** The
  disagreement flag correlates with failure, but the run that tests whether it is
  a usable *decision rule* (with a threshold and an operating characteristic) has
  not been done. S1-1.
- **Study 1 selection-rule degeneracy.** C11's n1989 case: both fits at R²≈1.000
  with materially different alignments, so held-out R² cannot arbitrate and
  selection is a coin flip. Reported as a finding; needs a tie-break rule.
- **Batching speedup is 4.5×, not the 20–40× estimated.** Saturates by batch 32.
  Phase A's feasibility arithmetic was written against the estimate. Redo it
  against 4.5× before committing to E1.1's scale.

---

## 6b. Decisions log (non-experimental)

| Date | Decision | Where |
|---|---|---|
| 2026-09-08 (C49) | **THE OFF-MANIFOLD ACCOUNT IS REFUTED. A-8 delivered.** A random direction inside the concept span behaves like Gaussian noise (p=0.38-0.76), not like a real concept vector (differs at p=0.033, 0.045). The signal does not track anomalousness. **What suppresses it is specific to a single coherent concept direction; a mixture of concept directions does not.** Sharper than the story it replaces | notebook C49 |
| 2026-09-08 (C48) | **Gemma's refit vectors DO steer, but weakly, and C45/C46's headline sits where they barely steer.** Semantic steering 10/30 at 40% of norm against a 3/30 baseline, but only 2/30 and 5/30 at the 10% and 20% cells where the A3 result is significant. Not disqualifying - forced-choice logits are far more sensitive than generation - **but the instrument is well validated at 40% and only weakly where the claim lives, and the write-up must say so** | notebook C48 |
| 2026-09-08 | **Gemma's vectors are much weaker steerers than Qwen's** (C32: 7/30 literal at 38%, coherent 30/30; Gemma needs 76% for 5/30 and coherence collapses to 14/30 there). The literal-word scorer undercounts badly - silk/web, harbor seals, dust and peanuts all score zero. **Report semantic steering alongside literal wherever steering is quoted** | notebook C48 |
| 2026-09-08 (C47) | **THE MULTI-SEED REPAIR FAILS ITS PRE-REGISTERED CRITERION. 76/100 vs the gate's 77/100**, Wilson [0.668, 0.833] against a 0.90 bar. Repairs 8, breaks 9, McNemar p=1.000. **C43's "the failure class is repairable" is WITHDRAWN** - it was measured on six units chosen as the worst, scored on alignment alone rather than the full pass criterion | notebook C47 |
| 2026-09-08 | **The lottery cuts both ways.** Median paired change is +0.0020 [+0.0005, +0.0040] but the tails are asymmetric: median gain +0.0118 against a worst-case loss of **-0.9553** (n1394, 0.9971 -> 0.0418). Every damaged unit shows a large positive k2_gain - the k=1 arm landing badly on a unit one seeded fit got right | notebook C47 |
| 2026-09-08 | **My protocol had a confound and the follow-up is cheap.** The multi-seed arm ran 1 restart x 800 steps against the gate's 3 x 2500, so this tested five cheap draws against one expensive draw, not seed-selection at matched budget. Fair test of the former, not the latter. **Re-run at matched budget before retiring the idea** | notebook C47 |
| 2026-09-08 | **C13's headline is unchanged and stands: 77% recovered, 23% fail silently.** C41/C42 ruggedness and C43's +0.953 test_r2-vs-alignment correlation survive - they describe the landscape, not this protocol | notebook C47 |
| 2026-09-07 (C45/C46) | **PAPER A'S HEADLINE IS DEAD IN BOTH ITS PREVIOUS FORMS. Pre-registered outcome A3: norm-matched RANDOM vectors produce MORE detection response than real concept vectors**, at every measurable strength and in both framings (p=1.1e-05 to 3.6e-02). Under the neutral prompt a real concept pushes P(YES) **below** the no-injection baseline (0.1882 -> 0.1205, p=9.3e-03, down on 23/30) while random pushes it up (-> 0.3199). Onset moves 13.9% -> 40.0% of the residual norm | notebook C45/C46 |
| 2026-09-07 | **THIS IS THE THIRD ANSWER TO ONE QUESTION AND THE FIRST TWO USED A BROKEN INSTRUMENT.** C18/C19 said A2, C23/C24 said A1, C45/C46 says A3. Only the third read vectors at the concept position, and only that instrument has passed a steering control (C32). **Any write-up must state that the answer changed twice and why** - burying it would be the single most damaging thing this project could do to its own credibility | notebook C45/C46 |
| 2026-09-07 | **A3 is reported and NOT interpreted, per the filed decision rule.** The off-manifold account - the signal tracks how anomalous a perturbation is, and a real concept direction is one the computation already uses - is a hypothesis with no dedicated test. **A-8, the on-manifold control, moves from future work to the next experiment**; it is now the run that decides what Paper A actually claims | prereg addendum; A-8 |
| 2026-09-07 | **Quote the neutral framing, not the introspective ratios.** The 82x introspective ratio at 10% is 0.0001 against 0.0080, both meaning near-certain NO. Only the neutral arm has interpretable magnitudes (0.12-0.14 real vs 0.22-0.32 random against a 0.188 baseline). Same rule as C33/C34 | notebook C45/C46 |
| 2026-09-07 (C44) | **THE ALL-ZERO GUARD FIRED IN PRODUCTION AND SAVED THE RUN.** A second Gemma session re-ran the *old* notebook cells; the forced arm again computed seven alphas of 0 from the unfilled `R = 0.0`, and `2026-09-07f` refused to start instead of writing 420 useless rows. C39's fix is validated by the exact failure it was built for | notebook C44 |
| 2026-09-07 (C44) | **The read-position fix is confirmed in production.** The log prints `read position check: 'table' at token(s) [7] of 13; decoded ' table' (template tail is '
')` — the vector is measured at the word, not the newline. Gemma's concept-position residual norm reproduces exactly: median **36,244.97**, min 31,072.4, max 46,779.8 | notebook C44 |
| 2026-09-07 | **The remaining failure mode is a stale notebook, not the script.** Two Gemma sessions have now been lost to cells that predate the current sheet. **Delete every cell before pasting the new ones** — the corrected sheet has no placeholder and no steer cell, so nothing carried over from the old one is safe to keep | A-12b |
| 2026-09-07 (C43) | **THE FAILURE CLASS IS REPAIRABLE, AND THE REPAIR NEEDS NO GROUND TRUTH.** Held-out R2 picks the good draw at correlation **+0.953** with **median regret 0.0000**. Five seeds selected by test_r2 take the worst failures from **0.5175 to 0.9669**. Five seeds saturates; three does not | notebook C43 |
| 2026-09-07 (C43) | **The lottery account is confirmed**: n1503 spans **0.0660 to 0.9829** on nothing but the seed, while passing units hold to four decimals. This is the mechanism behind the disagreement flag - that flag is the two-draw special case of selecting on held-out R2 | notebook C43 |
| 2026-09-07 | **Confirmatory n=100 run PRE-REGISTERED before running** (`preregistration-s1-multiseed.md`), with the disclosure that its protocol was sized on a favourable six-unit pilot. Scoring a headline against a protocol chosen after seeing a good subset is the C40 failure mode, found earlier in this same session | prereg |
| 2026-09-07 (C42) | **MY C41 OVER-OPTIMISATION HYPOTHESIS IS REFUTED.** At the gate's own 2500 steps with 2 restarts the worst failures median 0.9759 against the gate's 0.5175 at 3 restarts. Alignment does not decline with steps at all. Steps are not the variable | notebook C42 |
| 2026-09-07 (C42) | **The real asymmetry is VARIANCE, and it reframes the 23%.** Across nominally equivalent step counts, failure units spread by a median 0.161 and up to 0.634 while passes spread by 0.003. **Every worst-fail unit reaches >=0.93 at some configuration**, including C13's worst at 0.1926 -> 0.9977. The failure class is not unrecoverable units; it is units with a rugged objective, and the gate reported one draw from a lottery | notebook C42 |
| 2026-09-07 | **Do not assert that more restarts hurt.** It is the only named difference left between the gate and the cheap config, but it rests on one confounded comparison and is untested. C43 isolates it | notebook C42 |
| 2026-09-07 (C41) | **CORRECTION TO C37: the recovery is the FIT CONFIG, not resampling.** Decomposed: gate→full (same data, cheaper config) is +0.370 on the worst failures; full→best-subsample is +0.011. C37 changed both at once and credited the wrong one. Ruled out the selection rule too — the gate's direct fit alone gives 0.5175 and its cascade 0.4316, against 0.9668 for the same direct fit at 2 restarts/800 steps | notebook C41 |
| 2026-09-07 (C41) | **Five of six CATASTROPHIC failures recover under less optimisation, including C13's worst unit (0.1445 → 0.9665).** C37 could not see this: its filter excluded every catastrophic unit. If it holds, **the 23% silent-failure rate is substantially an artifact of over-optimisation** and Paper B's Study 1 chapter changes | notebook C41 |
| 2026-09-07 | **Any re-run of the n=100 gate at the cheap config must be pre-registered BEFORE it runs.** It would be re-scoring a headline result after seeing a favourable subset, which is the C40 failure mode exactly. File the criterion first | notebook C41 |
| 2026-09-07 (C40) | **THE REPRODUCTION FIGURE IS CORRECTED: 6.7%, not 10.0%.** The pre-registered scorer gives 2/30 at alpha=6; the published 3/30 was a hand-read subset matching no filed rule, and it happened to be the value closest to the target 10.8% that the operating point was then chosen to match. **The safeguard existed and was not consulted.** Corrected in the results doc, the draft and the plan; the pre-registration is left untouched as a filed record | notebook C40 |
| 2026-09-07 (C40) | **Section 2's readout table had two uncomputable numbers** (27% and 7% at alpha 6 and 8). Recomputed under both rules: 43/17/7/0 pre-registered, 50/23/33/10 permissive. The first-token column verified correct. **The readout finding survives and sharpens** — the two generated-text rules do not even agree on the shape of the curve | notebook C40 |
| 2026-09-07 (C40) | **A detection rate is only interpretable beside the false-positive rate produced by the SAME rule.** The permissive rule that yields 33.3% at alpha=6 also yields 16.7% at alpha=0, destroying the control. Report both or neither | notebook C40 |
| 2026-09-07 (C39) | **THE 331x RESIDUAL-NORM RATIO IS SUPERSEDED — publish 139x.** Measured at the concept position, where the vectors are actually read: Gemma 36,244.97, Qwen 261.53. The 331x figure was tail-vs-tail and both denominators were wrong. The conclusion is unchanged in kind — alpha=4 is live on most models and inert on Gemma — but the number is not | notebook C39 |
| 2026-09-07 (C39) | **A run sheet must never require pasting a number between cells.** A-12's `R = 0.0` placeholder went unfilled, every alpha computed to zero, and a 45-minute session produced 420 rows at alpha=0. `--alpha-frac` now computes the grid inside the run from the norm it already measured, and an all-zero multi-alpha grid refuses to start. **Sheet's fault, not the operator's** | `2026-09-07f`; A-12b |
| 2026-09-07 (C37) | **C14's ~200-event required-N does NOT transfer to real units — quote the real figure, not the synthetic one.** Real units reach only ~0.35 alignment at 200 events and saturate between 400 and 800, so the requirement is **2-4x** the planted-unit figure. Failures and passes need the same amount of data, so sample size is not what separates the failure class | notebook C37 |
| 2026-09-07 (C37) | **Five of six gate-failures RECOVERED on less data** (median 0.9402 -> 0.9823 with fewer restarts and fewer steps). A data limitation cannot produce that. It is direct behavioural evidence for the optimisation-landscape account from C9/C10, and it suggests an untested remedy: **refit on a resampled subset and keep the better held-out R2** | notebook C37 |
| 2026-09-07 (C37) | **That result is DOUBLY SELECTION-BIASED and must not be generalised.** The design required n_events >= 1600, keeping 8 of 23 failures when the class median is 1,285; and the six selected have gate alignments 0.86-0.95 against a 0.95 bar, so **the catastrophic failures (C13's worst was 0.1445) are absent entirely.** Re-run without the filter and including the worst units before this goes near the paper | notebook C37 |
| 2026-09-07 (C38) | **Cross-validation: the flag ranking holds, held-out R2 is robust, the pre-fit screen is not.** R2 goes 74% -> 69% sensitivity at unchanged 5% cost, so C35 stands. z_mean loses 16 points out of sample (74% -> 58%) — its AUC 0.877 is real but its **operating point does not transfer**, and C36's numbers should be quoted from the held-out column | notebook C38 |
| 2026-09-07 (C27-29) | **THE INVARIANT HYPOTHESIS IS DEAD; NO CALIBRATION RULE.** At the same fraction of its own residual norm where Gemma reads P(YES)=0.36, Qwen reads 1.8e-09. alpha* does not track the residual norm across models, so "report alpha as a fraction of ||h||" does not make models comparable. **The 331x norm difference stands as a measurement and still explains the literature's alpha disagreement — but it is a warning, not a rule.** Paper A keeps three findings and does not get its single sentence | notebook C27-29 |
| 2026-09-07 (C37) | **Threshold optimism measured; the two-stage rule stands but the pre-fit stage buys less than C36 implied.** Cross-validated, held-out R² holds at 69% sensitivity for 5% cost while z_mean drops from 74% to **58%** and disagreement from 74% to **53%**. AUCs are threshold-free and unchanged. **Quote held-out numbers, not the C36 table** | notebook C37 |
| 2026-09-07 (C36) | **The silent-failure flag becomes a TWO-STAGE rule, and the first stage needs no fit.** Failures are sparse, heavy-tailed units — active 6.8% of positions vs 20.5%, response kurtosis 87 vs 29, z_mean −1.20 vs −0.65. **z_mean predicts at AUC 0.877 before any fitting**, against held-out R²'s 0.915 after. Screen on z_mean (skip ~25 fits of 100 to catch 74% of failures), then flag on R². Combining them adds nothing (0.918) | notebook C36 |
| 2026-09-07 (C36) | **C7's "failures sit at z_mean −1.3 to −1.9" is withdrawn as a band.** From 8 neurons; across 100 only 6/23 failures fall in it and 4/77 passes do too, median failure −1.17. The direction is right and now well supported; the band was a small-sample artifact. Do not quote it | notebook C36 |
| 2026-09-07 (C36) | **C14's ~200-event required-N may not transfer to real units.** Failures average 1,353 informative events and passes 4,097 — both far above 200 — so raw event count is not what binds on real neurons. C14 measured it synthetically; at kurtosis 87 the information per event is far lower. **Check before publishing the required-N table as guidance** | notebook C36 |
| 2026-09-07 | **CORRECTION to the C35 row below: its labels were wrong, its conclusion was not.** The gate's rule is `(align > 0.95) AND (k2_gain < 0.01)`; C35 used alignment alone, giving 22 failures instead of 23. Corrected AUCs: R² 0.915, disagreement 0.810. **C13's 87%/55% was correct all along** — the "discrepancy" I logged was my own mislabelling and is withdrawn | notebook C35; C36 |
| 2026-09-07 (C35) | **PAPER B'S DELIVERABLE CHANGES INSTRUMENT: held-out R² replaces method disagreement as the ground-truth-free flag.** AUC 0.906 vs 0.802, and at 73% catch it discards 5 good units against disagreement's 21. **The better flag is also the free one** — R² is already computed by the fit, while disagreement needs both estimation routes and is the reason the dual protocol costs ~2x per neuron. The substantive claim survives intact; only the instrument changes, and the new headline is simpler: *a fit's own held-out R² predicts its own silent failures at AUC 0.91* | notebook C35 |
| 2026-09-07 | **Thresholds in C35 are fitted and evaluated on the same 100 units, so the operating points are optimistic.** The AUCs are the robust part. **Cross-validate or hold out before any threshold is published**; quote AUC in the meantime | notebook C35 |
| 2026-09-07 | **Discrepancy to resolve before either number is published:** C13's summary records the disagreement split as 87%/55%, recomputation gives 87%/58%. Unit counts agree exactly (69/31, matching the recorded 0.31), so the difference is in the pass criterion rather than the split | notebook C35 |
| 2026-09-07 (C33/C34) | **The refit changed Qwen from a dead null to a live curve, but the grid stopped one step short.** Real beats random at 12/12 comparisons, never true pre-refit, and P(YES) climbs six orders to 6.8e-02 against a 0.10 bar. **Extend to 50% and 60% of the norm (alpha 131, 157) and no further** — C32 showed 76% degrading the model. The Addendum 2 stop clause was written for a flat curve and does not apply to a steep one | notebook C33/C34 |
| 2026-09-07 | **Significance and magnitude have come apart on Qwen, and must be reported together.** Real-vs-random is significant only at 1-5% of the norm (p=4.6e-04 … 2.3e-02) where the values are 1.1e-09 against 1.0e-09 — both meaning "the model says NO with overwhelming confidence". Never quote these p-values without the magnitudes beside them | notebook C33/C34 |
| 2026-09-07 | **Run sheets written for both refits, and the order is load-bearing.** A-12 (Gemma, `kaggle/NEXT_SESSION_GEMMA_REFIT.md`) first, A-13 (Qwen forced-choice, `kaggle/NEXT_SESSION_QWEN_REFIT_FORCED.md`) second. Both sweep **the same fractions of each model's own concept-position residual norm** (1, 2, 5, 10, 20, 40%), which is what makes the invariant test a like-for-like comparison. Comparing Qwen's new onset against Gemma's old template-tail onset would compare two different instruments | A-12; A-13 |
| 2026-09-07 (C32) | **THE FIX IS CONFIRMED AND PAPER A NOW NEEDS A GEMMA REFIT BEFORE IT SHIPS.** Refit vectors give a monotone 1→7→14 of 30 against C31's flat 1/30, isolating the read position as the only change. **C15-C24 — every Gemma run, including the A1 headline at alpha=8192/16384 — used template-tail vectors.** They are not void (Gemma's steered, 10/30) but they were measured with a degraded instrument, and the onset, the real-vs-random comparison and the content-free share can all move. **Re-run the Gemma forced-choice arms at `--vector-pos concept` before the preprint.** Roughly 2 GPU-hours | notebook C32 |
| 2026-09-07 (C32) | **The 331x residual-norm ratio must be recomputed at the concept position before it is quoted again.** Qwen's norm is 261.5 at the concept token against 177.8 at the tail; Gemma's 58,932 was also a tail measurement. The ratio is like-for-like as it stands, but the tail is not the position the paper reads vectors from any more, so it is not the number to report | notebook C32 |
| 2026-09-07 (C32) | **Qwen is back in scope and the invariant test is live again.** C27-C30 measured a dead instrument and can now be re-run properly. Whether alpha* tracks the residual norm across models is answerable again — the question that decides whether Paper A gets a calibration rule or stays three findings | notebook C32 |
| 2026-09-07 | **VECTOR EXTRACTION FIXED (`2026-09-07e`); `--vector-pos concept` is now the default.** The vector is read by averaging the concept word's own token positions, located by searching its token ids in the templated sequence (both space-prefixed and bare variants tried, last occurrence taken). `--vector-pos template-tail` reproduces C15-C24 exactly. **A word that cannot be located raises rather than falling back to the tail** — a silent fallback is the C31 bug. The sidecar now records `vector_read_position`, and the log prints where it read, because C31 survived six runs purely because nothing ever said | section 8; A-11 |
| 2026-09-07 (C31) | **CORRECTION: "THE INVARIANT HYPOTHESIS IS DEAD" IS WITHDRAWN. It was never tested.** That row was written on C27-29, which measured Qwen with vectors that C31 now shows carry no concept content at all (1/30 concept-in-text at 112% of the residual norm, identical to the alpha=0 baseline). A null measured through a dead instrument is not a null. **The invariant — whether alpha* tracks the residual-stream norm across models — is UNTESTED, not refuted**, and stays open until a model other than Gemma is measured with vectors that pass a steering control | notebook C31 |
| 2026-09-07 (C31) | **Qwen is OUT of Paper A entirely.** C27, C28, C29 and C30's steering half all rest on inert vectors. Two things survive because neither depends on them: the **331x residual-norm measurement** (taken pre-injection) and the **30/30 categorical refusal** at alpha=0 (clean prompts, no injection) | notebook C31 |
| 2026-09-07 (C31) | **NEW PORTABLE FINDING: difference-of-means concept vectors read at the chat-template tail can silently carry nothing.** `last_token_activation` takes the last token of a prompt templated with `add_generation_prompt=True` — the `<|im_start|>assistant` marker, not the concept word. Enough survives on Gemma to steer; nothing does on Qwen. **The dead vectors pass every health check normally reported** — unit norm, zero non-finite, and a monotone P(YES) rise at p=3.7e-09 on 29/30 concepts. Only a steering positive control detects it, and nobody in this literature runs one | notebook C31 |
| 2026-09-07 | **The C30 diagnosis is WEAKER than first written — injection is prompt-positions-only.** `run_trial` applies the vector on the prompt pass and not on decode steps (`h.shape[1] > 1`), so any concept reaching generated text does so indirectly through the cached prompt state. Gemma's hits were short outputs where that survives; Qwen answers with ~49-word templated refusals, across which it dilutes. **So 0/30 cannot distinguish "no content in the vector" from "content diluted away", and my first reading over-attributed it to dead vectors.** `--stage steer` (`2026-09-07d`) injects at every position to settle it | section 8; A-10 |
| 2026-09-07 (C30) | **THE QWEN NULL IS VOID — do not report it in any form.** Generation at 56% of the residual norm produces the concept 0/30 times, against Gemma's 10/30 at a comparable fraction. The injection perturbs generically (P(YES) p=3.7e-09, text similarity 0.18) but carries no concept content, which points at the difference-of-means vectors failing on Qwen at layer 38, not at a broken hook. Step one of the measurement failed | notebook C30 |
| 2026-09-07 (C30) | **NEW CONFOUND, and it outlives Qwen: introspection-refusal training makes the measurement unidentifiable.** Qwen refuses the premise categorically in 30/30 clean trials ("As an AI, I don't have the capability to detect..."), which is why its baseline P(YES) is 1.03e-09 against Gemma's 3e-05 — a policy, not a measurement. **Detection rates are not comparable across models with different refusal training, independently of activation scale.** Any cross-model claim must first show the model engages with the task. Nobody in this literature reports that | notebook C30 |
| 2026-09-07 | **The Qwen null is EMBARGOED until the generation check runs.** It contradicts Vogel on the same model family, and baseline P(YES)=1.03e-09 vs Gemma's 3e-05 is a 30,000x gap that could mean the readout is mis-specified for this tokenizer. One 10-minute `--stage control` run at alpha=100 separates "strong Probe-Report Gap" from "artifact". **Nothing about Qwen goes in the paper before it** | notebook C27-29 |
| 2026-09-07 | **C26: second Qwen attempt failed; `dtype=` and `bnb_4bit_compute_dtype` conflated.** fp32 storage makes Qwen's untied 778M-param embedding and lm_head 3.1 GB each and unplaceable on a 14.5 GiB card. `2026-09-07c` retries once at fp16 storage with fp32 compute preserved. Gemma's path is unchanged. Also added a loud warning on Hub fallback — the run had silently downloaded 20 GB with no model attached | section 8 |
| 2026-09-07 | **C25: first Qwen attempt failed at load; two loader bugs fixed in `2026-09-07b`.** Hardcoded 13 GiB per-GPU budget (Gemma-tuned) replaced with one measured from free memory, and the `cpu` entry in `max_memory` removed — it let accelerate split the model to CPU, which bitsandbytes 4-bit refuses. Mounting Qwen from Kaggle Models is confirmed working, no download | section 8 |
| 2026-09-07 | **A-3 run sheet written as a two-stage probe-then-sweep** (`kaggle/NEXT_SESSION_QWEN.md`). Qwen's residual norm is unknown, so a blind log grid would waste the session; instead a cheap alpha=0 probe measures the norm, writes it to the sidecar **before** any trials, and the sweep cells read that number and build the grid at the same *fractions of the norm* that bracketed Gemma's onset (1%-56%). The probe's alpha=0 rows double as the chat-template sanity gate. Weights must be mounted from Kaggle Models — a HuggingFace pull of Qwen2.5-32B is ~65 GB in bf16 and would exhaust the container | `kaggle/NEXT_SESSION_QWEN.md` |
| 2026-09-07 (C23/C24) | **PAPER A'S HEADLINE CHANGES — outcome A1, not A2.** Under the published normalised protocol real beats norm-matched random at p=9.5e-4 (22/30) at alpha=16384. The pure content-free story does not survive and the abstract must be rewritten. What replaces it is stronger and more honest: a concept-specific component exists, it needs a perturbation ~14% of the residual-stream norm, it sits on a large content-free alarm, it is not self-specific, and **the published alpha=4 is 0.0068% of that norm — 2,048x below our onset**. The A2 result at unnormalised strengths stands as measured and becomes a section, not the thesis | notebook C23/C24 |
| 2026-09-07 | **Report the residual-stream norm with every injection strength, always.** Measured 58,932 at Gemma-3-27B L37. No paper in this debate reports one, which makes alpha values incomparable across models and is why "we reproduced 10.8%" was never a protocol match. This is the most portable contribution in the study | notebook C23/C24 |
| 2026-09-07 | **`plan.md` marked up rather than rewritten.** The 19 Aug plan of record now carries a status banner, an inline correction at §3.1 (batching demoted), a reordered §3.2 (operating point promoted to primary lever), resolution marks on all four Phase 0 steps and all four §9 decision points, and a pointer from its Immediate Queue to this notebook. Kept intact as a record of what was believed on day 2 | `plan.md` |
| 2026-09-07 | **A-1b/A-1c run sheet written** so the next Kaggle session needs no reconstruction: `kaggle/NEXT_SESSION.md` carries the cells, the four pre-flight checks, the artifacts to return including the new `.scalars.json` sidecars, and a symptom-to-fix table | `kaggle/NEXT_SESSION.md` |
| 2026-09-07 (later) | **CORRECTION to the row below, and a reprioritisation.** The alarm was costed against the superseded 1000x7 scope. E1.1 is now 300x4 = 1200 fits, and its ~22 GPU-h estimate already assumes no speedup, so **Phase A is not blocked**. Measured: batching 1.63x at d=768/batch 32 (not 4.5x), against the e03b operating point's **13.5x**. **Batching is demoted from Phase A prerequisite to optional**; the 19 Aug call that it was "the highest-value engineering task" does not survive the measurement | section 7.5 |
| 2026-09-07 | **Phase A scale is NOT committable until the batching speedup is re-established.** The speed test fails (1.0x at batch 8, three clean reps) against a recorded 4.5x at batch 32. E1.1 costs 15-30 h at 20-40x, 130 h at 4.5x, and 583 h at 1x. The correctness test passes, so this is throughput, not validity. **The test must not be tuned to green** | section 7.5 |
| 2026-09-07 | **C22: the filed stop clause is NOT triggered, and saying otherwise would be a false negative.** Addendum 2 said to report no detection and stop if no alpha reached 0.10. No alpha did — but the grid topped out 4.9x below the known effect threshold and the signal is switching on at its top (p=3.0e-06, 27/30). The stop clause assumed a bracketing grid. **Extend upward under a new Addendum 3, labelled as an extension.** Do not report a null | section 4 C22; section 7.1 |
| 2026-09-07 | **Paper A section 4.1 is frozen pending A-1a.** Its concession that the reproduction is rate-matched rather than protocol-matched rests on unit-norm being their convention, which C22 makes doubtful. Do not finalise that section until the residual norm is captured | section 7.1 A-1 |
| 2026-09-06 | **Repo goes PRIVATE until the arXiv date-stamp, then public.** `github.com/santoshcheethiralame-dot/CALIPER` was created public. It now holds the complete Study 3 finding, its raw data, the paper draft, and a notebook laying out every planned run — roughly 3.5 weeks before Paper A reaches arXiv. The risk register rates L3 scoop risk High and prescribes the arXiv stamp before anything else, so publishing the roadmap first inverts that order for no gain. **Flip to public on the day Paper A is announced (~1 Oct 2026)**; nothing else about the repo changes | `risk-and-scope.md`; this notebook |
| 2026-09-06 | **Lab notebook opened**, back-filling 21 runs from `results/*.json` and `data/s3/*.jsonl`. **C21 archived** into `data/s3/` and verified (660 rows, 11 alpha x 2 framings x 30 concepts, sha256 `f465e730…`) — it had been sitting loose in `Downloads` since the session died on 3 Sep | this notebook |
| 2026-09-06 | **APERTURE dependency recorded, and it carries a liability.** CALIPER's Study 3 descends from APERTURE R11/R12, whose **raw data is lost**. Binding consequence: no CALIPER paper reports an APERTURE number as its own measurement; Paper B's framing chapter rests on C20, which is archived, and cites R11 as motivation only | section 3, Inherited |
| 2026-09-06 | **Cross-project overlap flagged: CALIPER C18/C19 supersede APERTURE F8 at the self-report level.** Two projects, one author, adjacent claims about norm-matched injection. Required before Paper A: cite R11/R12 as prior work by the same author; re-scope F8 to verbalizer methods; settle the ownership split at the September mentor meeting | section 7.7 |
| 2026-09-06 | **APERTURE's F9 is no longer compute-blocked.** It was the programme's only run gated on 32B access; CALIPER ran a 27B-class model end to end on free 2x T4 Kaggle. The recipe transfers | section 2.2, section 7.7 |
| 2026-09-03 | **Paper A positioning CORRECTED after the reference check; the old draft framing must not be restated.** Filling four placeholder references meant reading the full papers, and three draft claims did not survive: (1) "critiques attack identification and leave detection standing" is false — Lederman & Mahowald and Singh et al. attack detection's interpretation directly, by inference; (2) "the original does not state its normalisation" is false — Macar et al. L2-normalise, so our α is not theirs and 10.0 vs 10.8 is rate-matched not protocol-matched; (3) "the content-free control had not been run" is overstated — Godet has one informal sentence on Mistral-22B. Honest contribution restated: a **systematic, pre-registered, norm-matched, two-control manipulation on the defended model/layer/prompt**, plus a concept-specific residual neither critique's method could see, plus the preamble-matched neutral prompt, plus the readout swing | `paper/PLAN.md` §0 |
| 2026-09-03 | **Venue: arXiv ~1 Oct, then an ICLR 2027 workshop in Feb.** Every fitting NeurIPS 2026 workshop is closed (Interpretability-as-a-Science, the ideal fit, closed 1 Sep). ICLR 2027 main is feasible on paper but rejected: the work is one model / one layer / 30 concepts (workshop-shaped), a main-track rejection costs three months of shelf life, it needs a reciprocal reviewer with a top-venue paper (the mentor's decision), and it **mandates an AI-use statement** | `paper/PLAN.md` §1 |
| 2026-09-03 | **Two papers committed, a third conditional.** Paper A now (scoop risk), Paper B Jan–Feb 2027, Paper C only if E1.4 clears in April 2027. Not three: Study 2 does not exist and has a known blocker; promising it would be planning around fiction | `semester-plan.md` |
| 2026-09-02 | **Study 3 CLOSED; every cell of the pre-registered design ran.** Primary outcome A2 | §4 C15–C20 |
| 2026-09-02 | **Blindsight framing proposed and WITHDRAWN within the hour.** "Knows that, cannot say what" requires identification at chance; measured, identification where it occurs is 58×/145× chance (p=1.4e-5, 1.7e-7). The analogy was reached for before the baseline was built. Do not reintroduce it | §4 C15 |
| 2026-09-02 | **The merged three-level design is the flagship**, superseding the three separate strand documents. Thesis: every model readout — unit, trait, self — is validated without ground truth; we construct ground truth at all three levels under one calibration discipline | `merged-paper-design.md` |
| 2026-08-19 | **Batching is the top engineering priority and a prerequisite for Phase A**, not an optimisation. Phase A as specified is 583 CPU-hours for one model, one layer, one variable | `plan.md` §3.1 |
| 2026-08-19 | **PCA truncation and whitening are both OUT.** MLP read-directions retain 0.352 of their norm in the top-64 PCA subspace against 0.267 for random — barely above chance; whitening round-trips at 1.000 synthetic and 0.02 on the real residual stream. Raw space is the protocol | `plan.md` §2.1–2.2 |
| 2026-08-19 | **Any cheap screen is a screen.** Equal-count binned R² produced two phantom findings and one bad candidate selection. Shortlist with the screen, decide with the real objective. Encoded in `fit_cascade` | `plan.md` §8 |
| 2026-08-19 | **No diagnostic below n=100 gets a conclusion attached to it.** Four separate n=20 tests all returned p ≈ 0.13 | `plan.md` §8 |
| 2026-08-18 | **APERTURE is not the capstone**; it remains valid completed work and preliminary evidence. Programme direction moved to CALIPER | memory `capstone-pivot-iii-v` |
| 2026-08-18 | **Two novelty claims corrected and must not be restated:** tuning curves have been done in vision models (Distill Circuits 2020–21) — the LM application is what is new; capture–recapture has 30+ years of use in software defect estimation — the AI-security application is new | audit verdict |

---

## 7. Planned runs

Every run below has: what it answers, its config, its pre-registered endpoint,
its cost, and **what to do when it fails**. A run without a failure branch is not
planned, it is hoped for.

Two standing rules, learned the hard way:

- **File the criterion before the run.** For C21 the data already exists, so file
  it before the *analysis* and say so in the paper.
- **A null from an unswept parameter is not a result.** C15's α=8 zero looked like
  a finding until the sweep showed the window was between 4 and 8.

### 7.1 Paper A — the seven pre-arXiv runs (Sep 2026)

Ranked. Runs A-1 to A-3 before arXiv; A-4 to A-7 for the workshop version. Total
if all run: about 7 GPU-hours, inside one week of free Kaggle quota.

---

#### **A-1 · L2-normalised protocol — THE ONE THAT MATTERS** ⚠ half-run

> **REVISED 7 Sep 2026 after C22. Act on this block; the text below it is kept for
> provenance and two of its statements are now wrong.**
>
> **Wrong statement 1:** "file a pre-registration addendum." Addendum 2 was already
> filed, on 3 Sep, before C21 ran. No new criterion is needed and no post-hoc
> disclosure is required.
>
> **Wrong statement 2:** the branch table's first row treated "no onset" as
> ambiguous between a bug and a too-low grid. C22 settled it: the grid is 4.9x too
> low and `--normalise` is working correctly.
>
> **What C22 established.** alpha\* = None, but the sweep topped out at the
> equivalent of alpha_unnorm = 0.409 against an effect that starts at 2, and the
> signal is switching on at the top of the grid (p = 3.0e-06, 27/30 concepts). The
> filed stop clause does not apply, because it assumed a grid that bracketed the
> onset.
>
> **Revised plan — three runs, ~1.5 GPU-h total, in this order.**
>
> **A-1a · Capture the residual norm.** Before anything else, fix the script to
> write run-level scalars (median residual norm at the read position, median vector
> norm, version stamp) into a sidecar or the first record, not just stdout. C22 lost
> the one number that decides the interpretation. This is a code change, not a run.
>
> **A-1b · Extend the grid upward.** Real vectors, unit norm, alpha in {2048, 4096,
> 8192, 16384, 32768}. This brackets alpha_unnorm 0.41 → 6.6, which spans the entire
> range where the unnormalised runs showed the effect appear and saturate. Addendum 2
> authorises "a single fine sweep between two steps" if the onset falls between them;
> **an upward extension is not that**, so file a one-paragraph Addendum 3 stating the
> new grid and why, and label the extension as such in the paper. ~40 min.
>
> **A-1c · The random arm at the new alpha\*.** Once alpha\* exists, run
> `--control random --normalise` at alpha\* and the next step up. This is the
> primary test of Addendum 2 and it is still unrun. ~30 min.
>
> **The interpretive fork this opens, which is now the more important question.**
> On a literal unit-norm reading their alpha=4 is 2,500x below anything that moves
> this model. Either we have misread their convention — most likely they scale
> relative to the residual-stream norm, in which case **our unnormalised C17–C20 are
> closer to their protocol than this run is**, and the 3 Sep repositioning
> overcorrected — or their effect is genuinely tiny. A-1a's captured norm decides it
> in one number.
>
> **What this does to Paper A.** Nothing yet, and possibly something good. Section 4.1
> currently says our reproduction is rate-matched, not protocol-matched, and that
> concession was made on the assumption that unit-norm is their protocol. If A-1a
> shows relative scaling, that concession is withdrawn and the reproduction is
> stronger than the draft claims. **Do not finalise section 4.1 until A-1a lands.**

**Question.** Macar et al. normalise; we did not. At which α does the normalised
protocol reproduce their 10.8%, and does the real vector beat a norm-matched
random one *there*?

**State.** The real arm is **already collected** (C21,
`Downloads/s3_unit_forced_norm1.jsonl.txt`, 660 forward passes, α ∈ {0,1,2,4,8,
16,32,64,128,512,2048}). The random arm was never run. Nothing has been analysed.

**Do this, in order.**
1. Move the file into `data/s3/s3_unit_forced_norm1.jsonl` and record its hash.
2. **File a pre-registration addendum** naming: the onset criterion, the α at
   which real-vs-random is tested, the test (Wilcoxon signed-rank, paired by
   concept), and what each outcome means. State in it — and later in the paper —
   that the real arm was collected before the criterion was filed and the random
   arm after. That disclosure costs nothing and is the only thing that keeps the
   analysis honest given the data is sitting on disk.
3. Read the two lines the run printed that nobody has looked at:
   `30 vectors, median norm 1.00` (normalisation took) and **`residual norm at
   last token: median ...`**. That second number has not been reported for this
   model and layer by anyone, and it converts their α=4 into a relative
   perturbation size.
4. Run the random arm: same grid, `--control random`, `--normalise`. ~30 min.
5. Analyse both together.

**Built-in consistency check.** Our unnormalised vectors have median norm 5002,
so **α_unit ≈ 5002 × α_unnorm**. Unit α=2048 should behave close to unnormalised
α≈0.41, and unit α≈10000 should reproduce the unnormalised α=2 result. If the
grid's top end does not line up with the old curve, `--normalise` is broken and
nothing else in the run means anything. **Check this before interpreting.**

**Failure branches.**

| What happens | What it means | Do this |
|---|---|---|
| No onset anywhere in 1→2048 | Either the effect needs α above 2048 on the unit scale, or the injection path is broken under `--normalise` | Run the consistency check above. If α=2048 unit ≉ α=0.41 unnorm, it is a bug — fix and re-run. If the check passes, extend the grid to 4096/8192/16384 using the measured residual norm to bracket it |
| Onset found; real ≈ random there | **Strengthens A2 under the published protocol** | §4.1 says plainly: we do not reproduce their protocol's *specificity*, and the reproduction in C16 was rate-matched. Paper gets stronger |
| Onset found; **real ≫ random** at the 10.8% point | **This reverses the headline** | Report it as the headline. The honest story becomes: the effect *is* content-specific under the published protocol, and our unnormalised sweep surfaced the generic component because unnormalised vectors have wildly varying norms across concepts. Rewrite §4.4 and the abstract. Do not bury it |
| Onset lands between two grid steps | Expected; the addendum authorises one fine sweep | Run one sweep between those two steps and **label it as post-hoc in the paper** |
| The α=0 rows are not bit-identical to C17–C20 | The pipeline is no longer deterministic across versions | Stop. Do not pool. Find the divergence (library version, dtype, seed) before any comparison |

**If A-1 cannot be run at all** (no GPU access): arXiv anyway, with §4.1 stating
that the reproduction is rate-matched not protocol-matched, and the normalised
run named as the first item of future work. The A2 result does not depend on it.
**A-1 improves Paper A; it does not gate it.**

---

#### **A-2 · Trial-randomised replication**

**Question.** Does anything change once trial index is decoupled from concept?

**Config.** The four forced conditions (real, random, shuffle, real+neutral),
`--trial-seed 1`, α ∈ {0,2,4,6}. ~2 GPU-h. Script version 2026-09-03a already
does seeded assignment.

**Endpoint.** Every reported number, recomputed. Pre-register: *no reported
conclusion changes sign or loses significance.*

**Failure branches.**

| What happens | Do this |
|---|---|
| Numbers reproduce within CI | Limitations shortens to one sentence. Best case, most likely |
| A conclusion weakens but holds | Report both seeds side by side; the paired tests already controlled the confound, so this is expected drift |
| **A conclusion flips** | The pairing did *not* control it. That is a serious finding about the design and it goes in the results, not limitations. Re-run with two more seeds before writing anything, because a single flip on one seed is noise |
| Session dies partway | The results file appends; restart and it continues. Do **not** add the delete line |

---

#### **A-3 · Second model — Qwen2.5-32B-Instruct**

> **PROMOTED 7 Sep 2026 — this is now the most valuable run in the queue, not a
> robustness check.** C23/C24 measured Gemma's residual norm at 58,932 and found the
> onset at 14% of it. The open question that decides how the paper generalises is
> whether **alpha\* tracks the residual norm across models**. Qwen answers it in one
> session. Add to the run: record Qwen's residual norm at its read position, then
> test whether its onset lands near 14% of that norm. If it does, "report alpha as a
> fraction of the residual norm" goes from a methodological suggestion to a
> calibrated rule, which is a materially stronger contribution.
>
> **ANSWERED 7 Sep 2026 — and the answer is no.** C27-29 ran it. Qwen's residual norm
> is 177.8 against Gemma's 58,932, and at every matched fraction of its own norm Qwen
> shows nothing: 1.8e-09 where Gemma reads 0.36. The invariant does not hold and the
> calibration rule does not exist. What survives is the 331x measurement itself, which
> explains why alpha=4 is meaningful in most of the literature and inert on Gemma.
> **A-3 is now superseded by A-9 (the generation check), which decides whether the Qwen
> null is reportable at all.**

**Question.** Is this a finding about Gemma or a pattern?

**Config.** Same script, `--model qwen`, ungated so no token needed. Depth 0.60 →
**L=38 of 64**. fp32 compute, 4-bit, 2×T4. Forced stage, real + random,
α ∈ {0,2,4,6}. ~1.5 GPU-h.

**Why Qwen specifically.** It is Vogel's family, where detection is independently
reported to work. A null on Qwen means something; a null on an arbitrary model
would not.

**Failure branches.**

| What happens | Do this |
|---|---|
| Reproduces the Gemma pattern | Turns a finding into a pattern. Add a column to every table |
| **α=0 baseline is not near zero** under the introspective prompt | The chat template is wrong for Qwen. Gemma's baseline is 0.00003; anything above ~0.05 means the prompt is not being applied. Fix the template before reading any other number |
| `find_layers` fails | Qwen's stack is not `model.language_model.layers`. Print the module tree and add the path. Ten minutes |
| OOM on load | 32B in 4-bit is ~20 GB; check the "GB free before load" lines print ~15.5/15.5. If not, restart the session (§8) |
| No detection at any α | Suspect vector construction before the model. Sanity gate: does α=8 steering actually put the concept in the output, as it does on Gemma? If not, the concept vectors are wrong for this tokenizer |
| Detection reproduces but random ≈ real again | Best outcome. Two models, same conclusion |

---

#### **A-4 · Two more neutral phrasings**

**Question.** Does the framing conclusion survive a different neutral wording, or
is it a property of one sentence?

**Config.** Add "Is this trial different from the others?" and "Has anything been
changed?" as third and fourth framings. Real + random, α ∈ {0,2,4,6}. ~1 GPU-h.

**Failure branch.** If the three neutral prompts disagree with each other, the
framing claim is about wording, not about self-reference — say exactly that and
downgrade §4.5 from a finding to an observation. That is a real possibility and
it is cheap to check, which is why it is on the list.

---

#### **A-5 · Second and third layer (L=25, L=50)**

**Question.** Is L=37 special, or does the content-free share hold across depth?

**Config.** Real + random at α ∈ {0,2,4,6}, layers 25 and 50. ~1.5 GPU-h.
Appendix figure.

**Failure branch.** If the content-free share varies strongly with layer, that is
more interesting than the appendix figure it was budgeted as — promote it. If
injection at L=25 destroys coherence entirely, report the working depth range and
move on; do not force the sweep.

---

#### **A-6 · Judge check**

**Question.** Does a GPT-4.1-mini judge, which is what the original used, agree
with our rule-based scorer on the 210 generated responses?

**Config.** API only, no GPU. Score all 210 with the judge; compare per-trial
against the rule-based labels; report agreement and the confusion matrix.

**Failure branches.**

| What happens | Do this |
|---|---|
| Agreement high | Table A2 becomes a validation. Removes an obvious reviewer objection |
| **Judge finds more detections than the rules** | Our detection numbers are conservative. Re-report with the judge as primary and the rules as secondary, and say the direction of the disagreement |
| Judge finds fewer | Our C16 reproduction was inflated. This would weaken §4.1 and it must be reported that way |
| No API budget | Skip. State in limitations that scoring is rule-based and differs from the original's judge. Do **not** hand-score and call it a judge |

---

#### **A-7 · Macar's exact "Unprompted" text as a third framing**

**Question.** Their own no-preamble variant changed two things at once (framing
*and* preamble). Ours changes one. Run theirs alongside ours and compare.

**Config.** Third framing using their published text. ~0.5 GPU-h.

**Failure branch.** If their variant behaves like our neutral one, the
"preamble-matched" contribution shrinks to a clarification — still worth a
paragraph, not a contribution bullet. Adjust §0 of `paper/PLAN.md` accordingly.

---

#### **A-8 · The on-manifold control** *(not yet scheduled — see §6)*

**Question.** Is the model detecting *perturbation*, or *off-manifold-ness*?
Random and shuffled vectors are both off-manifold, so neither existing control
separates these.

**Design sketch.** Inject a **real concept vector for a different concept** at
matched norm, and ask about concept X while injecting Y. That vector is
on-manifold and content-bearing but wrong. If P(YES) matches the real-vector
condition, "detection" is not about the queried concept. Cost ~1 GPU-h.

**Why it is not in the ranked list.** It was identified after the design closed.
It is the strongest remaining objection to Paper A and it is cheap. **Decide
before arXiv whether it goes in v1 or is named as future work.** Recommendation:
name it as future work in v1 and run it for the workshop version — adding a new
control after the pre-registration closed, and before the preprint, invites
exactly the "you kept running controls until one worked" reading that the
pre-registration exists to prevent.

---

### 7.2 Study 1 residual — sem 5, Sep–Oct 2026 (local CPU)

#### **S1-1 · Is the disagreement flag a usable decision rule?**

**Question.** C13 shows the flag correlates with failure (87% vs 55%, p=7.7e-4).
That is not the same as a rule. What is its ROC, and at what threshold?

**Config.** Re-analysis of `results/e01_gate.jsonl` — **no new compute**. For each
of the 100 units compute the disagreement statistic and the true alignment;
sweep the threshold; report sensitivity/specificity, the operating point that
catches ≥80% of failures, and how many good units it discards to do so.

**Endpoint.** A number a practitioner can act on: "flag at disagreement > t;
expect to catch X% of silent failures at a cost of Y% false alarms."

**Failure branches.**

| What happens | Do this |
|---|---|
| Clean separation | This is the deliverable of Paper B. Lead with it |
| Overlapping distributions, AUC ~0.6–0.7 | Report the AUC honestly and say the flag is a screen, not a test. Still useful, much weaker claim. **Do not pick the threshold that maximises the headline** |
| No separation | The 87%/55% result was driven by a few extreme units. Re-check it, and if it does not survive, retract the flag as a deliverable — this would be the single biggest hit to Paper B and it costs nothing to find out now |

**Run this first.** It is free, it is a re-analysis, and Paper B's main claim
depends on it.

---

#### **S1-2 · Generalise the cascade to K>1 — the Study 2 gate**

**Question.** Can iterative deflation (find one direction, project it out, find
the next) beat joint K≥2 estimation, which pins at 0.50/0.36?

**Config.** Implement deflation in `estimator.py`. Test on the synthetic
planted-K units already used in E0.3 at K ∈ {2,3}, N ∈ {2000, 8000}. Compare
against the joint fits in `e03_required_n.json` — the baseline already exists.

**Pre-registered criterion, file before running:** *median subspace alignment at
K=2 exceeds 0.80 at N=8000*, where joint estimation gives 0.5213.

**Failure branches.**

| What happens | Do this |
|---|---|
| Deflation clears 0.80 | **Study 2 is green-lit for sem 6.** Say so in November |
| Deflation improves but misses (0.6–0.8) | Study 2 is restricted to **one planted direction per trait**. P4's multitrait matrix becomes a one-direction-at-a-time matrix, which is weaker but publishable. Decide in November, not later |
| Deflation does not improve on joint | **Cut Study 2 from Paper B.** Paper B ships as Studies 1 + 3. Report the K≥2 degeneracy as a limitation of the estimator family, which is itself a real finding — it constrains every multi-dimensional readout built this way |
| Deflation works synthetically but not on real neurons | Real neurons may genuinely be K=1. That is a *result*, not a failure — report the dimensionality distribution and drop the multi-direction claims |

**Deadline: end of November 2026.** Do not let this drift into sem 6 undecided.

---

#### **S1-3 · Characterise the 23% failure class**

**Question.** What distinguishes the units the estimator fails on? C7 says they
sit at z_mean −1.3 to −1.9 (deep in GELU's non-monotone region) and C6 says a rank
transform helps. Is that the whole story on all 23?

**Config.** Re-analysis of the gate run plus targeted re-fits of the 23 failures
under: rank-transformed response, softplus surrogate, cascade, and all three.
No new corpus passes needed.

**Endpoint.** A predictive account: given a unit's activation statistics, can we
say in advance whether the estimator will fail?

**Failure branches.**

| What happens | Do this |
|---|---|
| z_mean / kurtosis predict failure | Excellent — the flag gets a mechanistic story and Paper B's Study 1 chapter writes itself |
| The 23 split into several distinct classes | Report the classes. C5 already hints at two (n1859 recovers, n2977 does not) |
| No predictor found | Report that the failures are not predictable from response statistics, which makes the *ground-truth-free* disagreement flag (S1-1) more valuable, not less |

---

#### **S1-4 · Re-run E0.1 at n=100 with the dual protocol as default**

**Question.** C13's 77% was the direct fit. What is the pass rate with
fit-both-keep-better as the default, and does it clear 0.90?

**Config.** n=100, same seed and units as C13 so it is paired, dual protocol,
selection by held-out R². ~2 h batched. Pre-register: *lower bound of the 95%
Wilson interval exceeds 0.90*, the same bar as C13.

**Failure branches.**

| What happens | Do this |
|---|---|
| Clears 0.90 | The instrument is fixed and both numbers get reported — the 77% for the naive protocol, the new one for ours. That contrast **is** the paper |
| Improves but misses | Report both. "Characterised instrument with a known residual failure rate" remains the honest description and the paper does not change shape |
| **Does not improve** | C11's degenerate-selection case generalises: held-out R² cannot arbitrate. That is a sharper finding than a clean pass — *the objective cannot identify the correct direction even when the correct direction is reachable* — and it bears on every method in this family. Report it, add the tie-break (prefer higher restart stability, else the direct fit), and move on |
| Gets **worse** than 77% | Selection is actively harmful. Check for the C11 failure mode across all 100 units before concluding; if confirmed, the default stays direct-only and the cascade becomes a manual rescue for flagged units |

---

#### **S1-5 · Finish the batched engine**

Not an experiment; the prerequisite for Phase A. Currently ~30% done, and the
measured speedup is **4.5×, not the 20–40× estimated**, saturating by batch 32.

**Required before Phase A:** a regression test asserting batched and unbatched
fits agree to numerical precision on a fixed seed. If they do not agree exactly,
batching is unusable and Phase A must be re-scoped, not fudged. That test exists
and **passes** (`test_batched_matches_single_neuron_path`); correctness is not the
problem.

> **MEASURED 7 Sep 2026 — the speed test fails and the 4.5x figure is unconfirmed.**
> `test_batching_is_faster_per_neuron` fails: at 8 neurons, batched 14.9s vs single
> 16.3s. Re-measured three times on an idle machine (the section 8 gotcha about
> orphaned jobs applies, so this was checked): **0.9x, 1.0x, 1.1x.** Batching buys
> nothing at batch 8.
>
> The 4.5x on record was measured "saturating by batch 32", so batch 8 is below the
> point where the shared `X @ V` projection is meant to amortise, and the test may
> simply assert a win at a batch size too small to show one. That reading is now
> **refuted at d=128** (`results/batching_speedup.json`, 7 Sep):
>
> | n_neurons | batched | single | speedup | batched s/neuron |
> |---|---|---|---|---|
> | 4 | 7.5s | 6.5s | 0.87x | 1.882 |
> | 8 | 12.7s | 15.1s | 1.19x | 1.582 |
> | 16 | 25.5s | 29.0s | 1.14x | 1.593 |
> | 32 | 49.4s | 58.6s | **1.19x** | 1.544 |
> | 64 | 97.2s | 112.7s | 1.16x | 1.519 |
>
> There is no saturation curve. The speedup is **flat at ~1.15x** from batch 8 to 64,
> and at batch 32 — where 4.5x was recorded — it is 1.19x. **The per-neuron column is
> the diagnostic:** 1.58 at n=8, 1.52 at n=64. If the shared `X @ V` projection were
> being amortised as section 3.1 of the plan of record describes, that column would
> fall roughly as 1/n. It is flat, so either the projection is not being shared, or it
> is not the dominant cost.
>
> **The caveat that decides which, and it is not small.** This ran at **d=128**. Phase
> A runs at **d=768**. Section 3.1's entire argument is that the projection over
> `16,000 x 768` dominates; at d=128 that term is ~36x cheaper while the per-neuron
> Adam steps and the 64-wide nonlinearity are unchanged, so the optimiser could
> dominate here and not there. **The 4.5x may have been measured at d=768 and be
> real.** A d=768 run is in flight (`results/batching_speedup_d768.json`); until it
> lands the honest statement is "batching buys ~1.15x at d=128 and is unmeasured at
> the width Phase A uses".
>
>
> **d=768 RESULT, and a correction to what I wrote three hours earlier
> (7 Sep, `results/batching_speedup_d768.json`).**
>
> | n_neurons | batched | single | speedup | batched s/neuron |
> |---|---|---|---|---|
> | 8 | 14.6s | 20.6s | 1.41x | 1.830 |
> | 32 | 50.4s | 82.0s | **1.63x** | 1.576 |
>
> Width helps and batch size helps, so section 3.1's design is directionally right:
> 1.19x at d=128 n=32 becomes 1.63x at d=768 n=32, and unlike the d=128 curve this one
> is still climbing. But **1.63x is not 4.5x**, and the recorded figure is not
> reproduced at the configuration where it was claimed. Two honest possibilities: the
> 4.5x was measured at a different operating point (steps, restarts, sample count, or
> including the k=2 cascade path), or on different thread settings. My benchmark used
> steps=400, n_restarts=1, n=8000, synthetic data.
>
> **CORRECTION — the "Phase A is not committable" alarm above was overstated, and the
> arithmetic that follows it is against a superseded scope.** The 583-hour figure comes
> from the 19 Aug plan of record, which costed E1.1 at **1000 units x 7 depths**. The
> 3 Sep semester plan already scoped E1.1 down to **300 units x 4 depths = 1200 fits**,
> and its "~22 GPU-h" estimate assumes **no batching benefit at all**. So:
>
> | E1.1, 1200 fits | s/neuron | hours |
> |---|---|---|
> | gate config, no batching | 63.7 | **21.2** |
> | gate config + batching at 1.63x | 63.7 | 13.0 |
> | cheap knee (e03b: 4k tokens, 800 steps, 2 restarts) | 4.73 | **1.6** |
> | cheap knee + batching | 4.73 | 1.0 |
>
> **Phase A as currently scoped is not blocked.** 21 hours fits inside one week of free
> Kaggle quota even with zero speedup.
>
> **And the real lever was never batching.** E0.3b already measured the cheap operating
> point at **4.73 s/neuron with median alignment 0.9936** against the gate config's
> 63.7 s/neuron — a **13.5x** reduction from choosing tokens/steps/restarts, against
> batching's 1.63x. The 19 Aug plan called batching "the highest-value engineering task
> in the project" and made it a Phase A prerequisite. On these numbers that was wrong:
> **the operating point is worth 8x more than batching, it is already measured, and it
> costs nothing to adopt.**
>
> **Revised position on S1-5.** Batching is no longer a Phase A prerequisite. Finish it
> or shelve it on its merits, and fix the test to assert what is actually true
> (a win at d=768, batch >= 32) rather than deleting it. Before spending another
> person-week on it, profile where the time goes — if the per-neuron Adam steps
> dominate rather than the shared projection, no amount of batching will help and the
> lever is steps, restarts, or a GPU.

> **Why this matters more than a red test.** Phase A's E1.1 was costed at 583 CPU-hours
> divided by the speedup. At 20-40x that is 15-30 hours. At 4.5x it is 130 hours. **At
> 1x it is 583 hours — 24 days of continuous CPU for one model, one layer, one
> variable**, which is the number the 19 Aug plan of record called infeasible and built
> batching to fix. Phase A's scale cannot be committed to until the sweep lands.
>
> **Do not "fix" the test by lowering its threshold or raising its batch size to make
> it green.** Either batching earns its speedup at a batch size Phase A will actually
> use, in which case the test should assert that at that size, or it does not, in which
> case Phase A is re-scoped. Both are honest; a tuned assertion is not.

**Consequence to face now.** Phase A's E1.1 was costed at 583 CPU-hours ÷ 20–40×.
At 4.5× it is ~130 hours. **Redo the Phase A arithmetic against 4.5× before
committing to 300 units × 4 depths**, and if it does not fit, cut depths before
cutting units.

---

### 7.3 Study 2 — planted personas (sem 6, conditional on S1-2)

Gated on the November decision. Needs a ≥7B instruct model → Kaggle.

| ID | Run | Question | Cost |
|---|---|---|---|
| **P1** | Recovery at the plant layer | Plant `v` at L, extract at L by difference of means over N generations, score \|extracted·v\|. **Positive control** | ~1 GPU-h |
| **P2** | The depth curve | Extract at L+1…L+k. How fast does the extracted "persona vector" drift from the true cause, and does the disagreement flag catch the drift? **The headline figure** | ~3 GPU-h |
| **P3** | Strength × N sweep | Required-N for trait extraction — the P-series analogue of E0.3 | ~3 GPU-h |
| **P4** | Multitrait–multimethod matrix | Difference-of-means vs probe vs subspace estimator, several planted traits. Convergent, discriminant, **and the ground-truth column psychology never gets** | ~4 GPU-h |
| **P5** | AIPsy-Affect stimulus set | Use the released 480-item keyword-free battery instead of designing our own prompts | none |
| **P6** | Poisson transfer *(CPU, runs in parallel)* | LNP model neurons under natural stimuli with known filters. **Does the 23% silent failure appear, and does disagreement flag it?** Decides whether the reciprocity arm is a headline or a paragraph | CPU only |
| **P7** | Detector on real neural data *(stretch)* | Allen Brain Observatory / CRCNS. Report the flagged fraction. Cannot be validated — that is the point | CPU only |

**Failure branches.**

| Run | If it fails | Do this |
|---|---|---|
| **P1** | Extraction cannot recover a planted direction *at the plant layer* | **Stop. Do not proceed to P2.** This is a linear read of a linear plant and it must work. Debug in this order: (1) span mismatch — is injection over response positions while extraction reads the last token only? (2) the difference-of-means baseline includes generated content that drifts; (3) plant strength below the noise floor at N generations — sweep N before concluding anything |
| **P2** | Recovery stays at 0.99 at every depth | **Persona vectors are validated.** Publish that — it is the first correctness validation of a deployed safety tool, and a positive result the field needs. Part 2 becomes a validation rather than a critique. The project is not invested in the method failing |
| **P4** | Blocked by K≥2 | Restrict to one planted direction at a time (per S1-2's middle branch) |
| **P6** | No silent failure on Poisson neurons | The flaw is specific to LLM-style units. Part 3 becomes "here is the assumption that protects biology," which is the reciprocal finding in a different form. Still publishable, differently framed |
| — | No clean persona behaviour at ~7B | Fall back to the concept directions APERTURE already injects, which are documented to shift behaviour. State the substitution |
| — | Someone publishes injection-based persona validation first | Parts 1 and 3 stand alone. This is the reason Paper A goes out first |

---

### 7.4 Phase A — unit-level measurement (sem 6, Feb–May 2027)

Order chosen so the cheapest decisive experiment runs first.

| ID | Run | Why this order | Cost |
|---|---|---|---|
| **E1.1** | Stimulus depth sweep, ~300 units × 4 depths | The boundary condition is known exactly (K=1 at distance 1), so it doubles as a correctness check on real data. Produces the dimensionality-vs-computational-distance curve, a novel object in itself | ~22 GPU-h *(re-cost at 4.5× batching)* |
| **E1.4** | **Causal validation — THE HARD GATE** | Ablate the recovered subspace against matched-random and top-K-PC controls. Until this passes every K is a curve fit, not a measurement | ~8 GPU-h |
| **E1.2/3** | Dimensionality distribution | The headline descriptive artefact. Only meaningful after E1.4 | ~15 GPU-h |
| **E1.8** | Absorption-notch test | Cheap, quotable, the one component a frontier lab would adopt. Pulled early per the audit | ~2 GPU-h |
| **E1.6/7** | Kurtosis heuristic; response characterisation | Rounds out the paper | ~5 GPU-h |

**E1.4 is the gate for the entire second year. Think about its failure carefully,
because the obvious reading is wrong.**

| What happens | What it might mean | Do this |
|---|---|---|
| Ablating the recovered subspace collapses the response; random and top-K-PC ablation do not | The measurement is causal | Proceed. Paper C is live |
| Ablation collapses the response **and so does matched-random ablation** | The ablation is too destructive to be diagnostic — you are removing whatever else lives in those directions | **Do not conclude failure.** Reduce to a rank-1 projection removal at the read position only, and re-test. Only if the specificity still does not appear does the gate fail |
| Nothing collapses, including the true subspace | Either the ablation is not reaching the computation, or K is a curve fit | Verify the ablation works at all using the *known* case: ablate a neuron's own weight direction at distance 1, where the answer is exact. If that does not collapse it, the intervention is broken, not the measurement |
| Specificity is real but weak | Honest partial pass | Report the effect size and treat Paper C as conditional. Say so in July 2027, not September |
| **Gate genuinely fails** | Every K in Phase A is observational | **Phase A stops.** Paper C becomes a workshop paper in sem 8 instead of an ICLR 2028 submission. Phase B is cut outright (it is already the default cut). This costs a chapter, not the thesis — which is why the gate is placed in April 2027 with a year still to run |

---

### 7.5 Phase C — calibration (CPU, no dependency on anyone else)

The separable track to hand to a teammate. None of it needs a GPU.

| ID | Run | Question |
|---|---|---|
| E3.1 | FDR on enumerated ground truth | False-discovery rate against InterpBench-style known circuits |
| E3.2 | Planted-latent networks | Ground truth for dimensionality — train a network with a known latent rank, then measure it |
| E3.3 | Training-trajectory null | Does the estimator find structure in a randomly initialised or early-checkpoint network? |
| E3.4 | Adversarial self-check | **Can we make our own pipeline hallucinate a structure that is not there?** |

**E3.4 is the one to run first if a teammate takes this track.** A pipeline that
cannot be made to hallucinate has not been tested; one that can, and whose flag
catches it, is the strongest possible demonstration of the calibration discipline.

**Failure branch for E3.3.** If the estimator finds apparently-real structure in
an untrained network, that is not a bug in the run — it invalidates the null for
every other experiment and must be fixed before any Phase A number is reported.
Treat a positive result here as a stop-the-line event.

---

### 7.6 Cut, deferred, and why

| Item | Status | Reason |
|---|---|---|
| **Phase B — communication subspaces (E2.1–E2.6)** | **Default: cut** | Gated on E2.5 and it is the one component the thesis can lose without damage. Decision point July 2027 |
| **Phase D — planted-prior manipulation (E4.1)** | Deferred to sem 8, conditional | Only if the delta against Cacioli's 2026 papers still holds. **Re-check the literature first**; that lane moved fast in 2026 |
| **E0.4 retrodiction on the weekday feature** | Skipped | EXTENDED; no slack |
| **Efficient-coding framing** | **Killed 18 Aug 2026** | Verified collisions: arXiv 2603.20642 has "Efficient Coding" in its title and counts the corpus prior; Benjamin et al. 2022 shows gradient descent generically produces frequency-tracking. Do not restate |
| **Grid-code probe** | Cut | Audit; battery reduced from 6 variables to 2 |
| **Voice / any non-core arm** | Not in scope | — |

### 7.7 APERTURE's planned runs — adopt, supersede, coordinate, or drop

APERTURE's own run programme (its notebook section 6, the F-series) is recorded
here because the two projects now overlap at the self-report level and share one
author, one machine, and one Kaggle quota. Anything marked **SUPERSEDED** or
**COORDINATE** is a live cross-project issue, not bookkeeping.

| APERTURE ID | The run | Status for CALIPER |
|---|---|---|
| **F8** | E13 confabulation: false-positive rate at null injection **and at random norm-matched injection**. APERTURE calls this "the headline number" | **SUPERSEDED at the self-report level.** CALIPER's C18/C19 already ran the norm-matched control on a 27B defended model with two independent content-destroying methods and a pre-registration. F8's remaining unclaimed ground is the *verbalizer* methods (logit lens, Patchscopes-style, SelfIE-style), not self-report. **See the coordination note below** |
| **F1** | Confound hardening: neutral-vs-introspective across seeds, >=8 paraphrases, >=3 layers, >=3 alphas. 11 of 24 configs in as of 2026-08-23 | **COORDINATE.** CALIPER's A-4 (two more neutral phrasings) and A-5 (layers 25/50) are the same idea on a different model. Run them knowing F1 exists; do not present them as independent replications of each other |
| **F9** | Scale / threshold arm at 32B. APERTURE calls it "the only compute-blocked run" | **UNBLOCKED BY CALIPER.** Study 3 ran a 27B-class model in 4-bit on free 2x T4 Kaggle, start to finish. The recipe in section 2.2 of this notebook is the unblock. Tell APERTURE's plan that its stated blocker is gone |
| **F10** | Second model family (Qwen or Llama) | **SAME RUN as CALIPER A-3.** One Kaggle session serves both. Do not run it twice |
| **F11** | Human grading, ~200 stratified transcripts, >=2 labellers, human-human and rules-vs-human kappa. Labour, no GPU | **ADOPT as a shared teammate track.** CALIPER's A-6 (judge check) is the cheap automated cousin, not a substitute. This is the single best item to hand to one of the other three capstone members alongside Phase C |
| **F4** | The audit arm: which published introspection claims decorrelate framing from construct. No GPU | **ADOPT for Paper A's related work.** It is the systematic version of the reference check that forced the 3 Sep repositioning |
| **F2** | Real covariates (infini-gram counts, Brysbaert concreteness) + a 120-concept domain-stratified bank | **ADOPT the bank if Study 3 gets a v2.** CALIPER used 30 concepts; "30 concepts, mostly concrete nouns" is a real limitation in Paper A and F2's bank fixes it for both projects |
| F3 | PRG hardening: leave-one-prompt-out CV probe accuracy with CIs | Not CALIPER's. Leave in APERTURE |
| F5 | Null robustness: multi-seed/paraphrase sweep of the behavioural null | Not CALIPER's |
| F6, F7 | E13 recovery and attribution for verbalizer methods; needs `aperture/readouts.py`, the biggest unbuilt item | Not CALIPER's, and the honest remaining core of APERTURE |
| F12 | Confirmatory freeze from clean seeds | Not CALIPER's |
| F13, F14, F15 | H8 explained-variance; E11-pilot Assistant Axis; naturalistic extension | **DROP for CALIPER.** Already APERTURE's own "cut first under bandwidth pressure" tier |

**Engineering backlog transfers, both directions:**

- APERTURE backlog item 6, **"random norm-matched vector generator"**, is listed
  as unbuilt and gating F8. **CALIPER already built it** — `--control random` and
  `--control shuffle` in `kaggle_s3_positive_control.py`, tested, with the
  bit-identical alpha=0 check that proves the arms are comparable. Port it back
  rather than writing it twice.
- APERTURE items 2 and 3 (activation capture, `archive_run`/`verify_archive`) are
  **done and CALIPER should adopt them**. CALIPER currently downloads Kaggle
  artifacts by hand, which is the discipline that failed six times in APERTURE.
  Adopting `archive_run` makes it code instead of memory.
- APERTURE item 7, **the PLANTED harness**, and CALIPER's disagreement flag are
  both bidding to be "the reusable deliverable". They should not compete. Decide
  which project owns the released artifact before either paper claims it.

**COORDINATION NOTE — read before submitting Paper A.**

The same author now has two projects making adjacent claims about concept
injection and self-report. Paper A's contribution is a norm-matched content-free
control on the defended model; APERTURE's F8 was planned to own "the
confabulation rate under random norm-matched injection" as its headline. Those
are close enough that a reviewer of the second one to appear could reasonably ask
why it is not the same paper.

Three things follow, none of them optional:

1. **Paper A must cite APERTURE's prior runs (R11/R12) as prior work by the same
   author**, not silently reuse the idea. Self-plagiarism and undisclosed overlap
   are the failure mode here, and both are avoidable by one paragraph.
2. **APERTURE's F8 must be re-scoped** to the verbalizer methods (F6/F7's
   readouts), where it is genuinely unclaimed, rather than to self-report, where
   CALIPER has now measured it.
3. **Decide the ownership split explicitly and write it down** — ideally in the
   September mentor meeting, since Paper A's authorship for a 4-person capstone is
   already on that agenda. Two papers, one author, overlapping controls, is a
   question that is much cheaper to answer now than at review.

---

## 8. Gotchas solved (so we never lose the time again)

### Two runs sharing one `--out` overwrite each other's config sidecar
C23 and C24 both used `--out /kaggle/working/s3_unit_ext.jsonl`. The `.jsonl`
outputs are correctly distinguished by the `_norm1` / `_random_norm1` suffixes the
script appends, but the **config sidecar is not suffixed** — so the real run's
`s3_unit_ext.config.json` was silently clobbered by the random run's, and only the
random run's config survives.

No harm this time: the residual norm is measured pre-injection so it is a property
of the model and prompts, identical across the two, and every per-row field needed
for the analysis is in the `.jsonl` itself. But the archived sidecar is now named
`s3_unit_ext_random.config.json` to stop anyone reading it as the real run's.

**Fix before the next paired run:** suffix the sidecar the same way the `.jsonl` is
suffixed, or pass a distinct `--out` per condition.

### The vector was read at the template tail, and nothing said so (C31, 7 Sep)
`last_token_activation` took `h[0, -1, :]` from a prompt built with
`add_generation_prompt=True`. That last token is the template marker, not the concept.
Demonstrated locally on GPT-2's tokenizer with a simulated template: the old code reads
`'
'` at position 9; the new code reads `' elephant'` at position 5.

Two things made it survive six runs (C15-C24 on Gemma, C27-C31 on Qwen):

1. **Nothing printed the read position.** The log said `building concept vectors at
   layer 37` and never said where in the sequence.
2. **Dead vectors pass every check that was printed** — unit norm, zero non-finite, 30
   distinct directions, and a first-token P(YES) rising at p=3.7e-09 on 29/30 concepts.

Fixed in `2026-09-07e`. Two rules now enforced in code: **a word that cannot be located
raises instead of falling back** to the tail, and the run **prints the decoded read
position** for a probe word so the log proves it. `--vector-pos template-tail` is kept
so the pre-C31 runs remain reproducible.

Gemma's results survive because its vectors demonstrably steer (10/30 concept-in-text),
but they were built the same way and may improve on a refit.

### Never make a run sheet carry a number between cells (C39, 7 Sep)
A-12's cells 3 and 4 opened with `R = 0.0        # <-- paste the residual norm from
cell 2`. It was not pasted. Every alpha computed to `0.0 * fraction = 0`, both forced
conditions ran seven identical alpha=0 cells, and nothing objected until the analysis
three hours later. The session was lost.

The value being pasted was **already known inside the run** — it is measured in
`build_vectors` and written to the sidecar. Asking a human to move it between two
notebook cells added a failure mode for no benefit.

Fixed twice over in `2026-09-07f`: `--alpha-frac` takes fractions and resolves them
against the measured norm internally, and a multi-alpha grid that is entirely zero now
raises before loading anything.

**The general rule: if the program can compute it, the run sheet must not ask for it.**

### `dtype=` and `bnb_4bit_compute_dtype` are different knobs (C26, 7 Sep)
The second Qwen attempt failed with the *same* CPU-dispatch error as C25 even though
the C25 fix was confirmed working (`device budget: {0: '14.5GiB', 1: '14.5GiB'} (no cpu
offload)` printed correctly). The cause was separate:

- `bnb_4bit_compute_dtype` sets the precision of the **dequantised matmul**. This is
  what stops Gemma-3-27B overflowing and must stay fp32 there.
- `dtype=` on `from_pretrained` sets the **storage precision of everything bitsandbytes
  does not quantise** — embeddings, `lm_head`, layernorms.

`load()` passed the same fp32 to both. On Gemma that was affordable. On Qwen2.5-32B it
is fatal: vocab 152,064 x hidden 5,120, **untied**, so `embed_tokens` and `lm_head` are
~778M parameters each — **3.1 GB per module in fp32, as a single indivisible block**
that accelerate must place entirely on one card already holding half the quantised
body. It spills to CPU, and bitsandbytes 4-bit refuses any model split that way.

Fixed in `2026-09-07c`: storage defaults to compute (so the Gemma path is unchanged
byte for byte), and a `ValueError` mentioning "dispatched on the CPU" triggers one
automatic retry at fp16 storage with compute precision untouched. Halving those two
modules saves ~3.1 GB and costs no matmul precision.

**The general trap:** a single `dtype` argument that looks like one decision is two.

### The Hub fallback is silent, and cost a session (C26, 7 Sep)
The same run never printed `found N model dir(s) in /kaggle/input`. The Qwen input was
not attached, so `model_id` fell through to the Hub name and it spent ~4.4 minutes
downloading ~20 GB before dying for the unrelated reason above. `2026-09-07c` prints a
two-line `!!` warning at the fallback. **Check for `found N model dir(s)` in the first
40 lines of any run** — its absence means the mount is missing.

### A `cpu` entry in `max_memory` silently breaks 4-bit loading (C25, 7 Sep)
The first Qwen attempt died at load with:

```
ValueError: Some modules are dispatched on the CPU or the disk.
```

Everything else was right — version stamp `2026-09-07a`, Qwen mounted from Kaggle
Models with no download, both cards reporting 15.5 of 15.6 GB free. Two bugs in the
loader, and the second is the one that actually did it:

1. **The per-GPU budget was hardcoded to `13GiB`.** That was tuned against Gemma-3-27B
   at ~19 GB total. Qwen2.5-32B in 4-bit is ~21 GB, and 13+13 minus accelerate's own
   headroom no longer holds it. Now measured from `torch.cuda.mem_get_info` at load
   time, less a 1 GB reserve, so it adapts to whatever card and model it meets.
2. **The loader offered accelerate `budget["cpu"] = "12GiB"`.** That is permission to
   place modules on the CPU, and bitsandbytes 4-bit refuses any model that has been
   split that way. Removing the entry means the model either fits on the GPUs or fails
   loudly. **Do not add it back.** It reads like a safety margin and is the opposite.

Fixed in `2026-09-07b`. The general lesson: an offload budget is not a fallback, it is
a different and unsupported code path.

### Kaggle: paste the script and it will bite you (cost: 4 rounds, ~3 h)
Pasting a 400+ line file into a notebook cell failed four consecutive times. Once
the tail of the script landed *inside* the pip command
(`pip install ... transformers(f"\n TARGET (Macar et al...`). Twice the traceback
line numbers proved an older copy was running while I debugged the new one.

**Fix, now standard:** upload the `.py` as a Kaggle **Dataset**; the notebook is
four lines that glob for it and `exec` it. Patch by uploading a New Version. And
**print a version stamp as the very first line** — if it is not the version you
just uploaded, stop reading the output. That single line is the cheapest thing in
the whole pipeline.

### Gemma-3-27B overflows fp16 on a T4 — silently, as NaN
Measured `max|h| = 51436` at layer 37 against the fp16 ceiling of 65504. Some
prompts tip over and others do not, so `concept − baseline` becomes `inf − inf =
nan`, and **one bad value in the shared baseline mean poisons all 30 vectors**.
The only visible symptom was `median norm nan` on the vectors line, 14 minutes
before a table of zeros.

**Fix:** `--compute-dtype fp32` is mandatory for this model. The script now
probes `max|h|` after loading, reports whether it is finite, reloads in fp32
automatically if not, and hard-stops if any vector is non-finite rather than
feeding NaN into 120 trials. fp32 costs ~2× per trial (25–30 s vs 13 s).

### `bitsandbytes` is not preinstalled, and installing it late does not help
`transformers` checks for `bitsandbytes` **once, at import**, and caches the
answer. Installing it afterwards in the same kernel changes nothing without a
restart — which is what makes the error look unfixable.

**Fix:** the script installs it itself at module load, before anything imports
`transformers`, and corrects the cached flag if `transformers` got there first.

### Stale GPU memory across sessions
A run OOMed at 6% loaded with GPU 0 already holding 14.5 GB — the previous
session's weights. The identical load had succeeded twice before.

**Fix:** **Restart the session before every reload.** The script now prints free
memory per GPU *before* loading, gives each card an explicit 13 GiB budget
instead of letting `device_map="auto"` pack GPU 0 first, and sets
`expandable_segments`. Both cards should read ~15.5 of 15.6 GB free on a fresh
session; if not, stop rather than letting it load.

### `apply_chat_template` returns a dict on transformers 5.x
It returns `BatchEncoding` (not a `dict` subclass, so a `isinstance(x, dict)`
check silently fails) on current versions and a plain tensor on older ones. Check
for a tensor instead. Cost: two failed Kaggle rounds.

### With `device_map="auto"`, the injection vector must go where the layer is
Layer 37 lives on `cuda:1` while the vector was built on `cuda:0`. Would have
crashed on the first trial. Move the vector to the layer's own device.

### `FileLink` 404s on Kaggle
It emits a relative path the editor does not serve. Use the sidebar: **View →
Output → /kaggle/working**, download arrow. (If the sidebar is collapsed, "Show
sidebar" is in the same menu.)

### `datasets` hard-crashes the process locally
`sample_corpus()` defaults to cached Gutenberg files because of this. Not a
workaround to be cleaned up later — it is the protocol.

### Cheap proxies produced three phantom findings
Equal-count binned R² produced two phantom findings and one bad candidate
selection. `fit_cascade` now shortlists on the binned screen and **decides with
the real objective**. Standing rule: a screen is a screen.

### Orphaned background jobs skew every timing number
Use `nohup`, not foreground `timeout`, and check for orphans before quoting any
throughput figure.

### Run-level scalars printed to stdout are lost with the session (found 2026-09-07)
Addendum 2 required the median residual-stream norm at the read position to be
printed. It was — to the Kaggle log — and the session expired. Per-trial rows are
safely in the JSONL; run-level values are not, because nothing writes them there.
That single scalar is what decides how to read Macar et al.'s alpha, so its loss
blocks an interpretation, not just a table.

**Fix:** write run-level scalars — version stamp, median vector norm, median
residual norm, device map, dtype — into a `<out>.config.json` sidecar at run
start. Same lesson as section 8's artifact rule, one level down: **if a number
matters, it goes in a file, not a print statement.**

### Download Kaggle artifacts before the session dies
Learned on APERTURE, where the raw data for six runs was lost. Every Study 3
`.jsonl` is in `data/s3/`. **C21's file is still sitting in `Downloads` and is
not yet in the repo — move it.**

---

## 9. Standing maintenance rule

This notebook is updated **in the same session as the work**, not afterwards:

- Before a run: add its planned row to §7 with the criterion and the failure
  branches, and file the pre-registration if one is required.
- After a run: add a §3 registry row, a §4 detail entry, archive the artifact,
  and update §5 and §6 if a conclusion or a confound moved.
- After a non-experimental decision: add a dated row to §6b with a pointer to
  where the reasoning lives.
- Never rewrite a past entry. Corrections are dated follow-ups beneath the
  original, so that the record shows what was believed and when.
