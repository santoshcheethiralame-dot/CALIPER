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
- Measured throughput: **63.7 s per neuron** at the gate config. **Corrected
  8 Sep 2026:** that config is **8,000 tokens, 1600 steps, 2 restarts, batch 32**
  (`results/e01_gate.log`: `stimulus (8000, 768)`), i.e. the script defaults - not
  the 20k/2500/3 previously written here. That figure belongs to the separate
  throughput measurement in `plan.md` section 1 and was conflated with the gate.

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

### Runs C38-C58 and the B-series — registry rows added 5 Oct 2026 *(reconstructed)*

The registry stopped at C37 on 7 Sep. Everything after that was written up in §4, §6b or
§7.8 but never got its row here, so the table below back-fills it from the archived
artifacts. Pass counts were recomputed from the jsonl on 5 Oct, under both rules in use:
**align** = `align_selected > 0.95` (the rule the AUC failure labels use) and **gate** =
align plus `k2_gain < 0.01` (the rule `e01_gate.py` reports). The two differ by 0-4 units
per arm, which is where the 247-vs-248 and 283-vs-287 discrepancies came from.

**Two ID problems, recorded rather than renumbered.** There are two §4 headings for
"Cross-validating the flag thresholds", one labelled C37 and one C38. The C37 *registry*
row below is the threshold CV. C56, C57 and C58 were used as IDs in code and in
`PIVOTS.md` but never appeared in this notebook at all.

| ID | Date | Exp | n | Config | Artifact | One-line result |
|---|---|---|---|---|---|---|
| C38 | 2026-09-07 | threshold CV (duplicate of C37's heading) | 100 | 5-fold | — | Held-out R2 transfers (74 -> 69% at 5% cost); z_mean 74 -> 58%; disagreement 74 -> 53% |
| C39 | 2026-09-07 | Gemma refit, forced, first attempt | 420 | all seven alphas transcribed as 0 | `data/s3/g_refit_*.jsonl` | **VOID GRID.** One number kept: concept-position residual norm 36,245 |
| C40 | 2026-09-07 | audit of the S3 positive control | 30 | rescoring of `s3_generation.jsonl` | — | **The reproduction figure was wrong.** Pre-registered scorer gives **2/30 = 6.7%**, not 10.0%; "27% / 7%" matched no rule |
| C41 | 2026-09-07 | required-N without C37's filters | 18 | stratified worst / marginal / pass | `results/s1_required_n_unbiased.json` | C37's attribution was wrong; failures are not a data shortage |
| C42 | 2026-09-07 | steps sweep | 9 | 200-5000 steps, 2 restarts | `results/s1_steps_sweep.json` | Over-optimisation hypothesis refuted |
| C43 | 2026-09-07 | seed lottery | 9 | many seeds per unit | `results/s1_seed_lottery.json` | n1503 spans 0.066 to 0.983 on the seed alone; corr(test R2, alignment) +0.953, so picking by held-out R2 wins the lottery |
| C44 | 2026-09-07 | Gemma session re-run (guard event) | — | — | Kaggle log | All-zero guard fired and saved the run; read-position fix confirmed in production |
| C45/C46 | 2026-09-07 | Gemma forced, vectors refit at concept, real + random | 420 + 420 | alpha-frac 0-0.40 | `data/s3/g2_forced_norm1.jsonl`, `g2_forced_random_norm1.jsonl` | **OUTCOME A3: random significantly ABOVE real**, reported as an anomaly per the filed rule |
| C47 | 2026-09-08 | S1 multi-seed confirmatory | 100 | | `results/s1_multiseed.jsonl` | 76/100 [0.668, 0.833]: no improvement, FAIL |
| C48 | 2026-09-08 | Gemma refit vectors, steering control | 150 | alpha-frac 0-0.76 | `data/s3/gval_steer_norm1.jsonl` | Semantic steering 10/30 at 40% of norm: weak but real |
| C49 | 2026-09-08 | A-8 on-manifold (span) control | 420 | same grid as C45 | `data/s3/g2_forced_span_norm1.jsonl` | Off-manifold account refuted |
| C50-C52 | 2026-09-08 | A-14 validated-steering window, real / random / span | 300 each | alpha-frac 0.30-0.60 | `data/s3/gw_forced_*.jsonl` | OUTCOME B2: no significant difference |
| C53 | 2026-09-08 | E0.1 gate, Pythia-160m | 100 | gate config | `results/e01_gate_pythia.jsonl` | 94 align / **93 gate** of 100; FAIL its bar at a different rate |
| C54 | 2026-09-08 | Study 2 P1, planted random directions | 24 | 8 plants x 0.10/0.20/0.40 | `data/s2/p1_plant.jsonl` | **VOID.** Median recovery 0.0069 |
| C55 | 2026-09-08 | Study 2 P2, same plants, depth | 40 | 0.40, extract L37-L55 | `data/s2/p2_plant.jsonl` | **VOID.** Median recovery 0.0073 at every depth |
| **C56** | 2026-09-08 | **Study 2 P1b, planted real concept vectors** | 24 | 8 plants x 0.20/0.40/0.60, primary 0.40 | `data/s2/p1b_plant.jsonl` | **FAIL as filed:** 3/8 plants beat null A at 0.40 (needed 6/8), median recovery **0.1557** vs null A **0.1870** (needed > 0.30). Manipulation check passed. Detail in §4 |
| C57 | 2026-09-08 | P1b residual analysis | — | offline | **no artifact** | **NEVER RUN.** Needs the extracted vectors as `.npz`; no `.npz` exists anywhere in the repo |
| **C58** | 2026-09-08 | concept-bank Gram matrix | 30 vectors | alpha 0 | `data/s2/gram.config.json` | Median off-diagonal \|cos\| **0.4216**, p90 0.733, max 0.852 |
| B-0 | 2026-09-09 | device equivalence, n=16 | 16 | CPU vs CUDA | `data/b11/device_equivalence.json` | 5/16 flip pass/fail; max \|d\| 0.66 |
| B-0b | 2026-09-10 | device equivalence, n=50 | 50 | | `data/b11/device_equivalence_n50.json` | 4/50 flip; batch composition changes the answer |
| B-1 | 2026-09-09 | restart agreement, 5 restarts | 100 | GPT-2 L6 | `results/b1_stability_gpt2.jsonl` | 91/100; AUC R2 0.913 vs restart 0.845, DeLong p=0.11: underpowered as filed |
| **B-1b** | 2026-09-09 | **primary endpoint** | 300 | GPT-2 L6, 2 restarts | `results/b1b_primary_gpt2.jsonl` | 248 align / 247 gate; 52 failures; **AUC R2 0.942 vs restart 0.778, DeLong p=1.8e-07** (recomputed 5 Oct) |
| B-2b | 2026-09-10 | primary endpoint, Pythia-160m | 300 | L6, 2 restarts | `results/b1b_primary_pythia.jsonl` | **287 align / 283 gate**; 13 failures; AUC 0.993 vs 0.899, p=0.11 |
| B-2s | 2026-09-10 | Pythia n=100, 5 restarts | 3 | | `results/b1_stability_pythia.jsonl` (untracked) | **ABANDONED** after two empty exits. 3 orphan rows on disk |
| B-11 | 2026-09-09 | Pythia scale ladder | 50 x 4 | Kaggle T4 | `data/b11/b11_pythia-*.jsonl` | 48/48/48/31 of 50 (70m / 160m / 410m / 1.4b) |
| B-11s | 2026-09-10 | steps check, 3200 steps | 50 x 2 | Kaggle T4 | `data/b11/*_s3200.jsonl` | 1.4b 31 -> 36; scale claim withdrawn as under-fitting |
| B-7 | 2026-10-04 | layer sweep, GPT-2 | 50 x 3 | L2 / L6 / L10, 5 restarts, pool 300 | `results/b7_layer{02,06,10}_gpt2.jsonl` | 25/43/45 align (24/43/45 gate). **Legs are independent samples, not paired** (see §4, 5 Oct audit) |
| B-8 | 2026-10-04 | GPT-Neo-125m | 100 | L10 | `results/b8_gptneo125m.jsonl` | 59/100; **AUC R2 0.715 vs restart 0.582**, p=0.0086 |
| B-10 | 2026-10-04 | required-N per signal | 50 x 4 | 2k/4k/8k/16k nested prefixes, 2 restarts | `results/b10_required_n.jsonl` | Pass 0.16/0.56/0.66/0.80; R2 ahead of restart at every N. Script `experiments/b10_required_n.py` was **untracked** at commit time |
| B-12 | 2026-10-04/05 | batch invariance, per-neuron seed on | 50 x 3 (+ control running) | batch 32 / 8 / 1 | `results/b12_fix_b{032,008,001}.jsonl` | 40/36/37 pass. **10 and 9 of 50 flip vs batch 32 with seeding fixed.** Control arm in flight |
| **B-11c** | 2026-10-06 | **Pythia-1.4B L12, 3,200 steps, fixed estimator (Kaggle T4)** | 50 | `--independent-units`, saves directions | `data/b11/b11c_pythia-14b_s3200_indep.jsonl`, `_dirs/` | 31/50 pass (B-11s coupled: 36); restart 0.847 vs R2 0.973, diff -0.126, DeLong p=0.032 (permutation p=0.074); **all 19 failures under-fitted**; route agreement AUC 0.910 |
| **B-14** | 2026-10-05/06 | **primary endpoint re-run, fixed estimator** | 300 | GPT-2 L6, 2 restarts, `--independent-units`, same units as B-1b | `results/b14_primary_gpt2_indep.jsonl`, `results/b14_analysis.json` | 254 align / 252 gate; 46 failures (16 wrong basin, 30 under-fitted); **restart 0.792 vs R2 0.919, diff -0.127, DeLong p=1.8e-04, bootstrap [-0.193, -0.063], permutation p=0.001: HEADLINE STANDS**. By failure class: under-fitted R2 0.993 vs restart 0.796; **wrong basin R2 0.779 vs restart 0.782 (tied)** |

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

> **Follow-up, 6 Oct 2026: APERTURE's full notebook re-read after the merge (PIVOTS P11).**
> This table was written on 6 Sep from memory of APERTURE's state. Corrections and additions
> follow; the rows above are left as they were.
>
> **Every APERTURE run, now prefixed A- for CALIPER's registry.** "Data" means raw data on
> disk under `projects/mirror/runs/`, checked today.
>
> | ID | Date | Model, setup | Result as logged | Data |
> |---|---|---|---|---|
> | A-R1 | 07-13 | pythia-70m CPU, L3, alpha 0/4/8 | Pipeline smoke; KL exactly 0 at alpha 0 | **on disk** (`dev.jsonl`) |
> | A-R2 | 07-13 | Gemma-2-2B-it, L13, alpha 0/4/8, 8 concepts | Every vector steers to its concept (incl. multilingual: fear -> miedo/peur); alpha 4/8 derails (KL 24-29) | lost |
> | A-R3 / G1 | 07-13 | Gemma-2-2B, L13, alpha 0-3, 4 concepts | Coherent window alpha ~0.5-1 (KL 0.01-0.25). **Identification appears almost only once the model derails**: 2 of 24 cells show the concept in the coherent band (KL < 0.5); every other "exact" sits at KL >= 0.5, mostly > 5. Disclaimer ("As an AI...") in 8/8 responses at alpha <= 0.5, 0/16 at alpha >= 1 | **on disk** (`gemma_sweep.graded.jsonl`) |
> | A-R4 | 07-13 | Gemma-2-2B, L13, detection prompt | 0/4 false alarms at alpha 0; coherent volcano/telescope injections answered NO; the only YES is joy (affect confound) | lost |
> | A-R5 | 07-13 | Gemma-2-2B, L5-21, alpha 1/2 | Depth-robust NO. Early layers barely perturb (KL ~0). L21 volcano at KL 0.01 answers "NO ... caldera": the concept leaks into the reply that denies it | lost |
> | A-R6 | 07-13 | Gemma-2-9B 8-bit, L21 | Same at 9B: NO on 5/6 coherent cells; joy affect confound again | lost |
> | A-R7 | 07-14 | Gemma-2-2B 8-bit, inject L13, probe L20 | Probe-Report Gap 0.83: probe 1.00, shuffled control 0.00, report 0.17 (10-sample test set) | lost |
> | A-R8 | 07-14 | same, patch L20 | Patching self +6.96 nats [5.34, 8.56], control +0.81 [-0.19, 1.80], paired +6.15 [4.50, 7.89] | lost |
> | A-R9 | 07-15 | Gemma-2-2B, no injection, 16 passages | Injection-derived directions decode natural states 11/16 = 0.688 [0.438, 0.875] vs 0.062 chance, **only after mean-centring** (raw dot product: 14/16 predicted "dolphin", exactly chance) | lost |
> | A-R10 | 07-15 | Gemma-2-2B, forced choice, 16 x 6 orders | Hit 0.302 vs 0.062 chance; gamma +1.988 [1.476, 2.478]; beta log-frequency negative (do not report) | lost |
> | A-R11 | 07-15 | + neutral framing | Neutral hit 0.433, gamma +2.574 [2.163, 2.999] > introspective +1.988. Difference -0.586 [-1.148, -0.007] | lost; **reference cell reproduced by A-F1 c00** |
> | A-R12 | 07-22 | + informative framing, pre-registered | gamma neutral +2.574 > introspective +1.988 > informative +1.645 [1.117, 2.152]; informative 15/96 unparseable. Primary P3 falsified; Pearson-Vogel not replicated at 2B | lost |
> | **A-F1** | 08-18 to 08-23 | Gemma-2-2B 8-bit, 12 OFAT configs x 2 framings, frozen prereg | 11/24 files in, hash-verified. **P4 holds:** R11's gammas fall inside c00's CIs ([2.19, 3.07], [1.31, 2.25]). c04 introspective is 41.7% unparseable (20/40 "thought", 13 empty). Hit rates differ from R11 by 3 and 2 of 96 under greedy decoding (8-bit kernel drift across library versions) | **on disk** (`runs/F1/`); the remaining 13 files are S-3 |
>
> **Corrections to the table above.** R11 is no longer only "load-bearing and lost". F1's
> reference cell reproduces it inside its CIs and is archived, so Paper 2 cites A-F1 c00 as
> the measurement and R11 as its origin. The R7-R12 numbers in the table above were checked
> against APERTURE's notebook today and match.
>
> **Interpretations that bear on CALIPER** (detail in `docs/APERTURE_INHERITANCE.md` §5).
> 1. **The framing control now spans two scales and two readouts.** At 2B with forced-choice
>    identification, neutral framing beats introspective (A-R11, A-F1 c00). At 27B with
>    first-token detection, neutral and introspective are indistinguishable once anything is
>    injected (C20). Neither scale shows a positive introspective-framing effect. This is
>    Paper 2's framing result.
> 2. **"Identification" is mostly what a derailing model emits** (A-G1), the same pattern as
>    our generated-text readout swing (C17) and the concept window closing by 56% of the norm
>    (C23/C24). Per-trial KL is the variable that separates the two, so S-2 records it and
>    reports identification by KL band.
> 3. **Prompt vocabulary leaks into the answer.** The informative framing lost 15/96 answers,
>    and 20/40 of F1 c04's unparseable answers are the word "thought". A framing control must
>    count off-list answers, not only hits.
> 4. **The perturbation removes the disclaimer** (A-G1, 8/8 -> 0/16; Qwen in C30 refused
>    30/30 at baseline). It is untested whether a content-free vector does the same. S-2 tests
>    it.
> 5. **Same seeds and greedy decoding, but a different library version: 5 of 192 answers
>    changed** (A-F1 c00 vs A-R11). This is a fourth reproducibility source for Paper 1's error
>    budget, beside restart count, batch and device.
> 6. **Centring is not optional** (A-R9). It is the same failure as P1b's anisotropic bank.

---

## 4. Runs in detail

### S-4 - Study 3 reanalysis with equivalence tests (2026-10-07, offline) - **MOST "INDISTINGUISHABLE" CLAIMS ARE INCONCLUSIVE**

Prereg `docs/preregistration-s4-reanalysis.md` (paired TOST, margin +/-0.10 P(YES), filed
before the script ran). Script `experiments/analyse_s3_reanalysis.py`, output
`results/s3_reanalysis.json`. No new data.

| comparison (introspective, paired over 30 concepts) | mean diff | 90% CI | verdict |
|---|---|---|---|
| C18 real vs random, alpha 6 | +0.112 | [-0.052, +0.275] | **inconclusive** |
| C19 shuffle vs random, alpha 2 / 4 / 6 | -0.056 / +0.108 / +0.069 | all straddle +/-0.10 | inconclusive |
| C20 introspective vs neutral, real, alpha 2 / 4 | +0.059 / +0.023 | [-0.003, +0.121] / [-0.057, +0.103] | inconclusive (alpha 4: p_tost 0.057) |
| **C20 introspective vs neutral, real, alpha 6** | +0.022 | [-0.023, +0.067] | **equivalent** |
| C24 real vs random, alpha 32768 | +0.108 | [-0.037, +0.253] | inconclusive |
| C50-52 real vs random, frac 0.30 | **-0.146** | [-0.265, -0.027] | **different (random above real)** |
| C50-52 real vs span, frac 0.30 | +0.019 | [-0.051, +0.089] | equivalent |
| C50-52 at frac 0.40-0.60, both controls | -0.04 to +0.04 | all straddle +/-0.10 | inconclusive |

**Exact interval for the positive control:** 2/30 = 6.7%, Clopper-Pearson 95% [0.8%, 22.1%].
It does not contradict Macar et al.'s 10.8%; it also does not reproduce it in any useful sense.

**What changes in the record.**
- §5 findings 6 and 8 and Paper 2 can no longer say real and random vectors are
  "indistinguishable" at alpha 6, or at 32768. The data cannot exclude a 0.10 difference either
  way.
- What survives as equivalence: the neutral and introspective prompts at alpha 6 (C20), and real
  vs on-manifold span at 30% of norm.
- What survives as a difference: random *above* real at 30% of the residual norm on validated
  vectors, the A3 direction.
- The "perturbation alarm" reading rests on the controls producing large effects of their own
  (dose-response, in the JSON). It does not rest on equivalence with real vectors, and Paper 2
  should argue it that way.

Also found: `data/s3/g_refit_steer_placeholder.jsonl` is byte-identical to
`g_refit_steer_norm1.jsonl` (C39, void grid). It is a duplicate, kept for history and ignored by
every analysis.

### B-14 - the primary endpoint on the fixed estimator (2026-10-06 08:23) - **HEADLINE STANDS; THE ADVANTAGE IS ENTIRELY UNDER-FITTING**

Prereg `docs/preregistration-b14-primary-rerun.md` with Addenda 1 and 2, all filed before the
outputs were viewed. Analysis by `experiments/analyse_b14.py`, developed on B-1b and run once
on B-14; output in `results/b14_analysis.json`. GPT-2 small L6, the same 300 units, tokens,
restarts and steps as B-1b, with `--independent-units`. Local CPU, 227 s/unit, started 13:26
on 5 Oct after the follow-on task died twice. Rows were written by the orphaned Python
process after its launcher shut down. No directions were saved: the run started before
direction saving existed.

**Primary (pre-registered decision table).**

| | B-1b (coupled) | **B-14 (fixed)** |
|---|---|---|
| pass (alignment) | 248/300 | **254/300 = 0.847** [0.80, 0.88] |
| failures: wrong basin / under-fitted | 12 / 40 | **16 / 30** |
| AUC held-out R2 | 0.942 | **0.919** |
| AUC restart agreement | 0.778 | **0.792** |
| restart minus R2 (DeLong) | -0.164, p=1.8e-07 | **-0.127, p=1.8e-04** |
| paired bootstrap 95% | [-0.224, -0.106] | **[-0.193, -0.063]** |
| label-permutation control | — | **p = 0.001** (null mean -0.0001, sd 0.038) |

**Verdict per the frozen table: diff <= -0.10 and p < 0.05, so the headline stands on the fixed
estimator.** B-14 replaces B-1b in the paper, and B-1b moves to the appendix as the run that
found the coupling.

**Secondary (pre-registered).**
1. **Verdict agreement with B-1b:** 256/300. 25 units pass only on the fixed estimator and 19
   only on the coupled one, McNemar p = 0.45. On GPT-2 L6 the coupling did not shift the pass
   rate systematically, unlike B-12 and B-11c.
2. **Failure classes:** 16 wrong basin (held-out R2 > 0.99), 30 under-fitted.
3. **PR-AUC** (prevalence 0.153): R2 0.765, restart 0.322.
4. **Other signals:** the "disagreement" field (ground-truth-dependent, withdrawn) scores 0.757
   and r2_spread 0.691. Both are below R2 at BH q ~ 0.
5. **Threshold sweep:** R2 is above restart at every bar from 0.90 to 0.98 (R2 0.91-0.98,
   restart 0.77-0.79).
6. **Nuisance baselines and incremental value:** baselines alone (active fraction, z_mean,
   kurtosis, GELU coefficient) give CV AUC **0.825**; adding R2 gives **0.892**, adding restart
   agreement **0.831**. Restart agreement adds nothing beyond unit difficulty, and R2 does,
   replicating B-1b.
7. **Calibration:** Platt-scaled R2 has Brier score 0.080.
8. **Controls:** permuted labels p = 0.001; the ground-truth signal scores AUC 1.0.
9. **LayerNorm-null ceiling:** no unit limited below the bar (median ceiling 0.99976).

**Addendum 2: by failure class (the result that reshapes the paper).**

| class (R2 cut) | n | AUC held-out R2 [95%] | AUC restart agreement [95%] |
|---|---|---|---|
| under-fitted (0.99) | 30 | **0.993** [0.985, 0.999] | 0.796 [0.729, 0.858] |
| wrong basin (0.99) | 16 | **0.779** [0.700, 0.849] | **0.782** [0.664, 0.872] |
| wrong basin (0.995) | 14 | 0.762 [0.676, 0.832] | 0.768 [0.640, 0.867] |

The split is identical at cutoffs 0.98 and 0.99 and moves little at 0.995. **Held-out R2's
whole advantage is the under-fitted class. On fits that converged to a wrong direction, the
two checks are tied at about 0.78.** That is better than chance, far from reliable, and
replicates the exploratory B-1b split (0.776 vs 0.729). The R2 number on that class is
attenuated by construction; restart agreement's is not, and it is no better.

**Consequences.**
- **The claim becomes:** held-out R2 is the better check overall because it catches
  under-fitting, which is about two-thirds of failures. Neither check reliably catches the
  converged-wrong third, about 5% of units. A practitioner who relies on restart agreement is
  worse off on the first and no better off on the second.
- **Re-run scope:** B-14's gap lies inside B-1b's bootstrap interval and 85% of verdicts
  agree. So the reduced path applies: re-run Pythia-160m (B-2c) and GPT-Neo (B-8b) on the
  fixed estimator, plus B-15. The B-7 depth legs stay as coupled-estimator replications in an
  appendix.
- **Re-pooled** with B-14 replacing B-1b (B-11c already replaced): random effects
  **-0.125 [-0.165, -0.084]**, I2 = 0, arm bootstrap -0.122 [-0.141, -0.103], R2 ahead in 6/6.

### B-11c - Pythia-1.4B on the fixed estimator (2026-10-06, Kaggle T4) - **THE ORDERING HOLDS; EVERY FAILURE IS UNDER-FITTED**

Prereg `docs/preregistration-b11c-pythia14b-indep.md`. Same 50 units as B-11s (layer 12,
3,200 steps, 2 restarts) with `--independent-units`. 101 s/unit on one T4. Artifacts: the
rows, the summary, and 50 direction files under `data/b11/`. The Kaggle output zip contained a
copy of itself, which was deleted on import. Analysis by `analyse_b14.py --no-model` against
B-11s; output in `results/b11c_analysis.json`.

| | B-11s (coupled) | **B-11c (fixed)** |
|---|---|---|
| pass (alignment >= 0.95) | 36/50 | **31/50** |
| failures: wrong basin / under-fitted | 0 / 14 | **0 / 19** |
| AUC held-out R2 | 0.996 | **0.973** |
| AUC restart agreement | 0.746 | **0.847** |
| restart minus R2 (DeLong) | -0.250, p=0.00019 | **-0.126, p=0.032**, bootstrap [-0.255, -0.017] |
| AUC route agreement (ground-truth-free) | not recorded | **0.910** |

1. **Pre-registered endpoints.**
   - The arm is a pooling component, not a standalone test, as filed.
   - The DeLong p of 0.032 is not matched by the label-permutation control, where 7.4% of
     permuted differences are as large. At 19 failures, DeLong is the more liberal of the two
     (see §6), so this arm's own significance is marginal.
   - McNemar against B-11s: 43/50 agree. 6 units pass only under the coupled rule and 1 only
     under the fixed one, exact p = 0.125.
   - The direction matches B-12: shared stopping trained units longer and passed more of them.
2. **Every failure is under-fitted.** This is the filed branch. At 1.4B the failure class
   is an optimisation-budget property: none of the 19 converged (held-out R2 <= 0.99), as in
   B-11s. The paper reports the 1.4B arm as under-fitting, not silent failure.
3. **First ground-truth-free method-agreement number.** Route agreement (|cos| between the
   direct and cascade directions) scores AUC 0.910. That is between restart agreement (0.847)
   and held-out R2 (0.973).
4. **Re-pooled** with B-11c replacing B-11s (`analyse_cross_arm.py`): random effects
   **-0.140 [-0.180, -0.100]**, I2 = 0, arm bootstrap -0.128 [-0.151, -0.104], R2 ahead in 6/6.

**Exploratory, from already-unblinded runs, and the reason for B-14 Addendum 2.** Split by
failure class, held-out R2's advantage comes almost entirely from under-fitted failures. On
B-1b, R2 has AUC 0.992 against restart's 0.793 for under-fitted failures, but 0.776 against
0.729 for the 12 wrong-basin ones. On B-8, wrong-basin gives 0.555 against 0.477. The
wrong-basin R2 AUC is attenuated by construction, since the class is defined by R2 > 0.99;
the restart AUC is not. So the converged-but-wrong core is close to invisible to every
ground-truth-free check measured so far.

### Cross-arm analyses on archived runs (2026-10-06, offline) - **R2 AHEAD IN 6/6 ARMS, POOLED -0.152; ABOUT 10% OF UNITS NO RESTART COUNT REACHES**

`experiments/analyse_cross_arm.py`, output `results/cross_arm_analysis.json`. No model, no
fitting. Every arm here used the coupled estimator; B-14 will replace the GPT-2 L6 row.

**1. Cross-arm meta-analysis of restart agreement minus held-out R2.** One arm per
(model, layer) condition, so no two arms share units. Arms with fewer than 5 failures are
listed and not pooled.

| arm | n | failures | AUC restart | AUC R2 | diff | p |
|---|---|---|---|---|---|---|
| GPT-2 L6 (B-1b) | 300 | 52 | 0.778 | 0.942 | -0.164 | 1.8e-07 |
| Pythia-160m L6 (B-2b) | 300 | 13 | 0.899 | 0.993 | -0.094 | 0.11 |
| GPT-2 L2 (B-7) | 50 | 25 | 0.789 | 0.950 | -0.162 | 0.0089 |
| GPT-2 L10 (B-7) | 50 | 5 | 0.729 | 0.818 | -0.089 | 0.22 |
| GPT-Neo-125m L10 (B-8) | 100 | 41 | 0.582 | 0.715 | -0.133 | 0.0086 |
| Pythia-1.4b L12, 3200 steps (B-11s) | 50 | 14 | 0.746 | 0.996 | -0.250 | 0.00019 |
| *Pythia-70m (2 failures, not pooled)* | 50 | 2 | 0.865 | 0.615 | +0.250 | 0.43 |
| *Pythia-410m (2 failures, not pooled)* | 50 | 2 | 0.771 | 1.000 | -0.229 | 0.28 |

DerSimonian-Laird random effects over the 6 pooled arms: **-0.152, 95% CI [-0.192, -0.112],
I2 = 0.00**. A bootstrap over whole arms gives mean -0.148 [-0.194, -0.110]. R2 is ahead in
6/6 pooled arms. The gap is remarkably homogeneous across models, layers and families,
although the arms' absolute AUCs vary widely (R2 from 0.72 to 1.00). This is the right form
for the paper's "across every arm" claim: one estimate with an arm-level interval, not a
vote count. The DeLong SE for the 5-failure arm is rough.

**2. B-4, the restart-count curve (descriptive).** AUC of restart agreement built from the
first k of 5 stored restarts, against the 5-restart fit's failure label:

| arm | failures | k=2 | k=3 | k=4 | k=5 | held-out R2 |
|---|---|---|---|---|---|---|
| B-7 L6 | 7 | 0.834 | 0.781 | 0.774 | 0.817 | 0.944 |
| B-12 control | 6 | 0.735 | 0.705 | 0.655 | 0.761 | 0.837 |

More restarts do **not** make restart agreement a better failure predictor here. The curve
is flat within noise and sits below R2 at every k. The theory's predicted rise with k is
not seen. Its predicted ceiling below R2 is. With 6-7 failures this is descriptive only.

**3. A share of units no restart count reaches.** With a two-class model (a fraction pi
never recovered; the rest succeed per restart with probability q; best-of-k passes if any
restart succeeds):

| pair of runs | pass at k=2 | pass at k=5 | pi | q | one-class model predicts at k=5 |
|---|---|---|---|---|---|
| E0.1 gate -> B-1 (n=100) | 0.77 | 0.91 | **0.080** | 0.60 | 0.975 |
| B-12 b32 -> control (n=50) | 0.80 | 0.88 | **0.118** | 0.70 | 0.982 |

A single shared per-restart success rate overpredicts k=5 by 7-10 points in both pairs. The
data fit a class of roughly **8-12% of units that more restarts do not fix.** "Run more
restarts" is therefore not a remedy for silent failure. The paired B-12 table also shows the
idealisation breaking: 2 units pass at k=2 and fail at k=5, which best-of-k cannot produce if
selection always picked a succeeding restart. Selection is by held-out R2 between routes, so
it does not.

### Audit of every run to date (2026-10-05) - **THE 4 OCT WRITE-UPS CARRY THREE INDEPENDENCE ERRORS**

Every Study 1 jsonl was re-scored from disk (`align_selected`, both pass rules, all four
signal AUCs, DeLong restart-minus-R2), the Study 2 plants were re-summarised against their
pre-registration, and the 4 Oct code was read line by line. The headline survives. Several
4 Oct readings do not, and they share one cause: **units were treated as independent where
they are not, or as identical where they are different.**

**What survives, recomputed.** B-1b: 52 failures of 300, AUC held-out R2 **0.942** vs
restart agreement **0.778**, DeLong diff -0.164, **p = 1.8e-07**. Held-out R2 beats restart
agreement in **every Study 1 arm that has more than two failures** (B-1, B-1b, B-2b, B-7 x3,
B-8, B-10 x4, B-12 x3, B-11 1.4b x2). The only reversal is Pythia-70m, which has two
failures. Many arms reuse B-1b's units, so this is consistency, not 15 replications.

**Correction 1 - B-7's legs are not paired. The 4 Oct "paired" reading is withdrawn.**
`rng.choice(3072)` draws the same *indices* at every layer, but neuron 1234 at layer 2 and
neuron 1234 at layer 6 are different neurons with different weights. Index-matching pairs
nothing. The check: under independence, with pass rates 0.50 / 0.86 / 0.90 (align rule),
the expected pattern counts match what was observed almost exactly.

| L2 L6 L10 | observed | expected if independent |
|---|---|---|
| fail pass pass | 20 | 19.4 |
| pass pass pass | 19 | 19.4 |
| fail fail pass | 3 | 3.2 |
| fail fail fail | 0 | 0.3 |

So "no unit fails at every depth" and "depth changes *which* units fail" are what three
independent draws produce, not findings. What B-7 does show is unpaired: pass rate rises
from 0.48 (gate rule, Wilson about [0.35, 0.62]) at L2 to 0.86 and 0.90, a real depth
effect between independent samples of units. The retraction of "two regimes" in commit
3e3593b stands; the reasoning offered for it does not. `analyse_b7_depth.py` and
`tests/test_analyse_b7_depth.py` assert the pairing, so they guard an invalid design.

**Correction 2 - batching couples units through early stopping, not only through
initialisation.** In `fit_batch`, `since` resets whenever *any* unit improves, and the whole
batch stops only when *none* has improved for `patience` steps. A unit's training length
therefore depends on its batch-mates. The docstring says the opposite. Measured on a
synthetic 16-unit set with `--per-neuron-seed` on: the batch ran **3,200** optimiser steps
(both restarts to the cap); the same units fitted alone stopped at as few as **952**. That is
a second coupling mechanism, independent of the stacked-randn one that per-neuron seeding
fixed, and it predicts exactly what B-12 now shows: smaller batches stop earlier and pass
less (below).

**Correction 3 - the "20% restart-only floor" is confounded with batch composition.** B-7 L6
and B-1b share their first 50 units in the same order. The first 32 sat in an identical
first batch in both runs, so they differ only in restart count: **4/32 flip (12.5%)**. The
last 18 were a batch of 18 in B-7 and part of a batch of 32 in B-1b, so they also differ in
initialisation and stopping: **6/18 flip (33%)**. The restart-only floor is nearer 12% than
20%, and B-12's amended criterion imports the larger, confounded figure.

**Correction 4 - B-12's verdict compares unlike quantities.** `total_flips` sums flips over
*both* small-batch comparisons (b8 and b1 vs b32) and tests the sum against the flips of
*one* control comparison. With two arms each sitting at the floor, the verdict fires FAIL.
`tests/test_b12_verdict.py` only ever builds one comparison, so it cannot see this. The
verdict line must not be quoted; compare each arm to the control separately.

**Correction 5 - B-10's 8k leg does not reproduce B-1b.** It was documented as the
consistency check against the primary arm. On the same 50 units it passes **33** against
B-1b's **41** (40/50 verdicts agree). Cause, from the code: `collect` truncates to
`max_tokens` *before* shuffling, so B-1b's 8k is the first 8,000 tokens of the stream, while
B-10's 8k is a random 8,000 of the first 16,000. A different token sample changes 10 of 50
verdicts. **Token sample is a fourth variance component**, next to restarts, batch and
device. B-10 also labels failure by alignment alone, while the gate adds `k2_gain`.

**Correction 6 - B-10's commit message overstates significance.** "Significantly so where
powered" lists 2k at p = 0.061. That is not significant. The arms at or above 20 failures are
2k (p = 0.061) and 4k (p = 0.017); 8k (p = 0.001) carries 17 failures and is flagged
underpowered by the script itself. Direction is consistent at every N; significance is
clean at one powered arm.

**Correction 7 - S1-2's criterion drifted before the run.** The script docstring says the
criterion is judged on the `cascade` arm; the code judges it on the best of all deflation
arms, including `resp_only`, which was added after it measured higher. There is no
`preregistration-s1-2*.md`. Response deflation subtracts the unit-coefficient linear term
`(X @ basis).sum(1)`, not the fitted nonlinearity, so it is not a deflation in the usual
sense. S1-2 must be filed properly before it runs: cascade arm primary as originally
written, best-of-arms exploratory.

**Correction 8 - B-8 claims more than it isolates.** "Only the training corpus differs"
is too strong: GPT-Neo also alternates local and global attention and was trained with
different hyperparameters. The defensible claim is "model family". The ground truth itself
was re-verified on GPT-Neo on 5 Oct: `s . w + b` reproduces `c_fc` with relative error
3e-07 and correlation 1.0000000000 at L6 and L10. The "layer-6 correlation 0.698" quoted
in §7.8 is the linear correlation with the *post-GELU* response, not a ground-truth check.
The L6 null rests on **4 units** and a median alignment of 0.0026, below the random-direction
median of about 0.024. Below-chance is not what an ordinary hard unit looks like. It needs
a 20-unit run with per-unit diagnostics before it is called a property of the model.
Separately: **the flag itself is weaker on GPT-Neo** (AUC held-out R2 **0.715**, restart
0.582). The headline 0.94 is a GPT-2 number.

**Correction 9 - Study 3's reproduction figure.** C40 found the pre-registered scorer gives
**6.7%**, not the 10.0% in `s3-results.md`. Several downstream documents
(`PROJECT_NOTES_COMPLETE.md` §5.2, the review deck) still quote 10.0% and the 27% / 7%
readout swing that C40 showed matches no computable rule. They need the C40 numbers.

**Missing or orphan artifacts found.**
- C57 cannot be run: no extracted-vector `.npz` exists anywhere in the repo.
- `experiments/b10_required_n.py`, `run_queue_detached.sh`, `caliper_queue.cmd` and all
  `b12_*` results were untracked on 5 Oct.
- `results/b1_stability_pythia.jsonl` holds 3 orphan rows from the abandoned B-2s.
- `data/s3/g_refit_steer_placeholder.jsonl` is byte-identical in size to
  `g_refit_steer_norm1.jsonl` (90,438). It is either a duplicate or misnamed.
- Uncommitted edits to `PROJECT_NOTES_COMPLETE.md` (81 lines). About 50 other "modified"
  files are line-ending churn only.

### B-12 control arm lands (2026-10-05 11:16) - **FAIL BY THE FILED RULE, BUT BATCH FLIPS ARE INDISTINGUISHABLE FROM RESTART FLIPS**

`results/b12_ctl_b032_r5.jsonl`: the same 50 units at batch 32, 5 restarts instead of 2,
position-keyed per-neuron seeding as in the other arms. 44/50 pass (0.88, against 40/50 at
2 restarts), 49.6 s/unit. The report (`results/b12_batch_invariance.json`, rewritten 11:17
under the per-arm verdict by the relaunched queue) reads:

| comparison | flips | rate | median \|d\| | max \|d\| |
|---|---|---|---|---|
| **control: restarts 2 -> 5, batch 32** | **8/50** | 0.16 | 0.0033 | 0.749 |
| batch 8 vs batch 32 | 10/50 | 0.20 | 0.0068 | 0.864 |
| batch 1 vs batch 32 | 9/50 | 0.18 | 0.0055 | 0.608 |

**Filed verdict: FAIL**, because the worst arm (10) exceeds the floor (8). The rule as filed
compares two counts without a test, and the margin is one or two units on n=50. Paired on
units against the control's flips, batch 8 has 6 flips the control lacks and the control has
4 that batch 8 lacks, exact binomial p = 0.75. For batch 1 the split is 6 vs 5, p = 1.0.
**Batch-size flips are statistically indistinguishable from restart-count flips at this n.**
This reading is post hoc and labelled as such; the filed verdict stands as recorded.

What B-12 can and cannot say. It shows that on the coupled estimator, changing batch size
moves verdicts about as much as changing restart count: 16-20% of units at n=50. It cannot
attribute anything to float reduction order. Every arm here still shares early stopping
across the batch and seeds by position (§6b, 5 Oct), so batch size changes training length
and initialisation too. The question B-12 was filed to answer moves to B-14, which removes
both couplings.

Side result: the restart 2 -> 5 change raises the pass rate (40 -> 44), as B-1 did (77 ->
91 at n=100). Restart count is a lever on recovery, not only on the agreement signal.

**Watcher incident.** `caliper-after-b12` exited with code -1 some time between 09:59 (a
sleep event in the System log) and 10:34, when the queue was relaunched. Cause not
identified; the machine did not reboot (last boot 1 Oct). It never started B-14, and the
CPU sat idle from 11:17 to 13:17. Restarted by hand at 13:17: B-12 report rewritten (exit 0)
and **B-14 started at 13:17.** It died within minutes, and the task exited with 0xC000013A, the console-close / Ctrl+C status. The task is Interactive, so it opens a visible bash window, and closing that window kills the run. The -1 this morning was most likely the same thing. Switching to a windowless S4U logon needs admin rights (access denied), so the task now starts bash through `powershell -WindowStyle Hidden ... Start-Process -WindowStyle Hidden -Wait`. **B-14 restarted at 13:26 (PID 19284)** with no window. The stdout log stays empty until Python flushes its buffer; progress is the row count of `results/b14_primary_gpt2_indep.jsonl`.

### B-12 - batch invariance with per-neuron seeding (2026-10-05, control arm still running) - **SEEDING DID NOT CLOSE IT**

GPT-2 L6, the first 50 units of B-1b's pool, 2 restarts, 1600 steps, `--per-neuron-seed`.

| batch | pass (align) | flips vs batch 32 | pass only in the small batch | pass only at 32 | median \|d align\| |
|---|---|---|---|---|---|
| 32 | 40/50 | — | — | — | — |
| 8 | 36/50 | **10/50** | 3 | 7 | 0.0068 |
| 1 | 37/50 | **9/50** | 3 | 6 | 0.0055 |

Batch 8 against batch 1: 11 flips. Only **7 of 50** units give an identical alignment (to
1e-6) at batch 8 and batch 32, although every unit's initial parameters are now bitwise
identical across batch sizes. Float reduction order alone would leave most units identical to
several decimals. **The residual is the coupled early stopping** (Correction 2), and its
direction fits: units pass more often in the large batch, where batch-mates keep training
alive.

Two consequences. First, the B-12 filing reads a FAIL as "reduction order in the batched
GEMM". That branch is not identified while stopping is coupled. Second, the B-1b headline
was produced with coupled stopping, at batch 32 with partial final batches. **It must be
re-run after the stopping fix** before it goes in a paper. Batch-32 coupling plausibly
trained units *longer*, so the corrected failure rate may rise.

The control arm (`b12_ctl_b032_r5`, 5 restarts at batch 32) failed once at 07:36 with exit
-1 after 24 s and was relaunched by the queue at 07:54. When it lands, read each small-batch
arm against it separately, not the summed verdict.

> **Correction, 5 Oct 2026 (later the same morning).** The sentence above saying every
> unit's initial parameters are "now bitwise identical across batch sizes" is wrong.
> `--per-neuron-seed` seeds by *position in the batch*, not by neuron (§6b, 5 Oct). Units at
> matching positions share a start, and the identical-output units are exactly those
> (positions 1, 2, 3, 6, 37, 38). B-12's flips therefore come from two couplings,
> position-keyed initialisation and shared stopping, not from stopping alone. The B-12
> design cannot isolate either from reduction order. B-14 runs with both removed.

### B-10 - required-N per signal (2026-10-04) - **R2 LEADS AT EVERY N; THE 8k LEG IS NOT B-1b**

`results/b10_required_n.jsonl`, GPT-2 L6, 50 units (B-1b's first 50), 2 restarts, one
16,000-token stimulus with 2k/4k/8k/16k nested prefixes. Failure = alignment < 0.95.

| tokens | pass | failures | AUC R2 | AUC restart | AUC disagree | restart - R2 | p |
|---|---|---|---|---|---|---|---|
| 2,000 | 0.16 | 42 | 1.000 | 0.938 | 0.711 | -0.063 | 0.061 |
| 4,000 | 0.56 | 22 | 0.985 | 0.847 | 0.679 | -0.138 | **0.017** |
| 8,000 | 0.66 | 17 | 0.925 | 0.724 | 0.842 | -0.201 | 0.001 (underpowered) |
| 16,000 | 0.80 | 10 | 0.770 | 0.718 | 0.793 | — | underpowered |

Recovery rises with data: at least 16k tokens for a usable pass rate at this restart count.
Held-out R2's AUC *falls* as failures get rarer. At 16k, disagreement nominally leads, inside
overlapping CIs. The 2k AUC of 1.000 with a degenerate bootstrap CI [1.0, 1.0] reflects 42
failures against 8 passes; it is not a precision claim. The 8k leg passes 33 of the 50 units
B-1b passes 41 of (Correction 5).

### B-8 and B-7 - flag AUCs per arm (2026-10-04, added 5 Oct)

The 4 Oct write-ups gave pass rates only. AUCs, failure = alignment < 0.95:

| arm | failures | AUC R2 | AUC restart | AUC disagree | restart - R2, p |
|---|---|---|---|---|---|
| GPT-2 L2 (B-7) | 25 | 0.950 | 0.789 | 0.861 | -0.162, 0.0089 |
| GPT-2 L6 (B-7, 5 restarts) | 7 | 0.944 | 0.817 | 0.850 | -0.126, 0.039 |
| GPT-2 L10 (B-7) | 5 | 0.818 | 0.729 | **0.951** | -0.089, 0.22 |
| GPT-Neo-125m L10 (B-8) | 41 | **0.715** | 0.582 | 0.546 | -0.133, 0.0086 |

The ranking R2 > restart holds everywhere. Its *size* does not transfer: on GPT-Neo every
signal is weak, and the flag that is "free and good" on GPT-2 discriminates at 0.72.

### C56 / C58 - Study 2 P1b and the concept-bank Gram matrix (2026-09-08, back-filled 5 Oct) - **FAIL AS FILED**

Never written up here; recorded only in `PIVOTS.md` P7 and a script docstring. Gemma-3-27B,
plant at L37, 8 real concept vectors, 16 prompts, version `2026-09-08c`. Primary cell 0.40 of
the 36,245 concept-token norm, per `preregistration-s2-p1b.md`.

| alpha-frac | median recovery | median null A (other concept) | median null B (random) | plants beating null A | median steered prompts /16 |
|---|---|---|---|---|---|
| 0.20 | 0.1544 | 0.1257 | 0.0068 | 4/8 | 2 |
| **0.40 (primary)** | **0.1557** | **0.1870** | 0.0066 | **3/8** | 6.5 |
| 0.60 | 0.5795 | 0.4913 | 0.0085 | 4/8 | 7.5 |

Criterion (6/8 plants beat null A **and** median recovery > 0.30): **FAIL** on both legs. The
manipulation check passed. Recovery sits 20-80x above the random null at every strength, so
difference-of-means lands in concept space, but it is no closer to the planted concept than to
another concept. C58 (`data/s2/gram.config.json`): the bank's median off-diagonal |cos| is
**0.4216**, p90 0.733, max 0.852. Null A is the inter-concept floor, which is why it is so
high. The 0.60 cell is reported, not used: it was excluded from the primary in advance.

### B-2b - the primary endpoint on a second family (2026-09-10) - **ORDERING REPLICATES, SIGNIFICANCE DOES NOT**

`results/b1b_primary_pythia.jsonl`. Pythia-160m, layer 6, n=300, 2 restarts, local CPU,
same protocol as B-1b. Ran 20:35 to 07:21.

| | GPT-2 small | Pythia-160m |
|---|---|---|
| passing | 248/300 = 83%, Wilson [0.780, 0.865] | **287/300 = 96%**, Wilson [0.927, 0.975] |
| median / min alignment | 0.9942 / 0.0508 | 0.9990 / 0.1354 |
| failures | 52 | **13** |

**Signal AUCs - the rank order is identical on both families:**

| signal | GPT-2 | Pythia |
|---|---|---|
| held-out R2 | **0.942** | **0.993** |
| restart agreement | 0.778 | 0.899 |
| disagreement | 0.752 | 0.854 |
| r2_spread | 0.704 | 0.793 |

**DeLong on the primary endpoint:**

| | delta AUC | z | p |
|---|---|---|---|
| GPT-2 | -0.1638 | -5.221 | **1.78e-07, significant** |
| Pythia | -0.0938 | -1.579 | **0.114, not significant** |

**Read this as the filing requires, not as we would like.** Addendum 1 committed in advance:
*"If n = 300 still yields fewer than 30 failures, the comparison is reported as underpowered
and inconclusive rather than extended a second time on a third guess."* **Pythia has 13.**
So the Pythia arm is **underpowered and inconclusive on significance**, and it is not
re-run at a larger n to chase a p-value.

**What does replicate, and it is not nothing.** All four signals rank in the same order on
both families, with held-out R2 first and restart agreement second by a clear margin. That
is a qualitative replication of the ordering; it is not a second significant test.

**The supportable claim, stated exactly:** *on GPT-2, restart agreement is significantly
dominated by held-out R2 (p < 1e-06); on Pythia the same ordering appears but the arm is
underpowered at 13 failures.* Anything stronger overstates it.

**Why Pythia is underpowered by construction, not by accident.** Its failure rate is 4%
against GPT-2's 17%, so n=300 buys 13 failures where GPT-2 buys 52. Reaching ~40 failures
would need n ~ 1000, roughly 36 hours locally. **The filing says do not**, and that is the
right call - a second significant test would be nice, but chasing it after seeing p = 0.114
is exactly the behaviour pre-registration exists to prevent.

**Stable-and-wrong replicates in kind:** 6 of 52 on GPT-2, 1 of 13 on Pythia.

---


### B-0 extended to n=50 (2026-09-10) - **A BIGGER FINDING THAN THE DEVICE ONE: batch composition changes the answer**

`data/b11/device_equivalence_n50.json`. Extended to settle whether 5-of-16 was a rate. It
is not, and the attempt to measure it uncovered something better.

**The headline number, first.** 4 of 50 units flip pass/fail between CPU and CUDA, Wilson
[0.032, 0.188], max |d alignment| **0.9189** against a median of 0.000136. GPU speedup
1.87x. Verdict FAIL, as at n=16.

**Then the contradiction.** The B-0 draw is nested - `choice(3072, 100)[:n]` - so the 16
units are a subset of the 50. Yet the flip sets barely overlap:

| | flipped units |
|---|---|
| n=16 | 100, 268, 1034, **1759**, 2723 |
| n=50 | 85, 1503, 1520, **1759** |

One unit in common. The same unit's CUDA alignment moves with n: **1759 goes 0.9960 ->
0.3137**, 1034 goes 0.7881 -> 0.9257, 2723 goes 0.9980 -> 0.8034.

**Batch size alone changes the answer, on BOTH devices.** Same 16 units, same seeds, same
device, only the number of *other* units in the batch differs:

| device | median \|d\| | max \|d\| | units changing pass/fail side |
|---|---|---|---|
| CPU | 0.005158 | 0.299429 | **5 of 16** |
| CUDA | 0.010560 | 0.682346 | 2 of 16 |

**On CPU with fixed seeds this cannot be floating point. It is the initialisation.**
`caliper/batched.py:30` draws `torch.randn(n, d, k, generator=g)` - one tensor of shape
(n, d, k) from a single generator - so **neuron i's starting point depends on how many
neurons are in the batch**. Different init, different basin, different answer. Deterministic
and fully reproducible; nothing to do with hardware.

**The claim this supports, and it is stronger than the device one:**

> The direction an estimator recovers for a unit depends on which other units were fitted
> alongside it. Batching for speed - which is standard practice - silently changes the
> scientific result.

**Three consequences.**

1. **B-0's two runs are not comparable to each other** and the 5/16 vs 4/50 discrepancy is
   explained, not mysterious. Within a single run, both devices share n and therefore share
   the init, so **the device comparison inside one run stands**.
2. **A caveat on our own data, disclosed.** `e01_gate.py` chunks by `--batch 32`, so a run
   is internally consistent - but **B-1 resumed at 92/100**, and its final 8 units were
   fitted in a batch of 8 rather than inside a batch of 32. Their init therefore differs
   from a fresh run. B-1b and B-2b ran straight through and are unaffected.
3. **This is arguably a defect worth fixing**, by seeding per neuron rather than drawing one
   stacked tensor. **Not fixed now** - it would change every number already collected, for
   the same reason `fit_cascade`'s device was left alone mid-ladder.

---


### B-11 steps check - the scale effect is an OPTIMISER ARTEFACT (2026-09-10) - **SCALE CLAIM WITHDRAWN**

Executes the branch pre-committed in `preregistration-b11-pythia-ladder.md` before the
ladder ran. `data/b11/b11_pythia-{14b,410m}_s3200.jsonl`, Kaggle T4, paired on the same 50
units per rung (the draw depends on `--neurons` and the seed, not on `--steps`).

| rung | 1600 steps | 3200 steps | discordant | McNemar |
|---|---|---|---|---|
| **pythia-1.4b** | 31/50 (62%) | **36/50 (72%)** | **5 fail->pass, 0 pass->fail** | p = 0.0625 |
| pythia-410m (control) | 48/50 (96%) | 48/50 (96%) | 1 each way | p = 1.0 |

**The filing anticipated four outcomes and this is a fifth: partial attenuation.** 1.4b rose
but nowhere near 96%. Taken alone that is ambiguous. **The free diagnostic settles it.**

**Not one failing unit at 1.4b has converged.**

| | failing units, median held-out R2 | passing units | failures with R2 > 0.99 |
|---|---|---|---|
| 1.4b @1600 | 0.899 | 0.997 | **0 of 19** |
| 1.4b @3200 | 0.752 | 0.997 | **0 of 14** |
| 410m @1600 | 0.907 | 0.99979 | 0 of 2 |

**This is decisive, and it withdraws the scale claim.** A converged-but-wrong unit has high
held-out R2 and a wrong direction - that is the silent failure this whole project is about,
and GPT-2 has them at R2 ~ 0.9999. **1.4b has none.** Its failures are units the optimiser
never fitted at all, and held-out R2 flags them loudly: **AUC 0.980 at 1600 steps and 0.996
at 3200.**

**So 1.4b's failures are not silent failures. They are under-fitting**, which is detectable,
attributable and fixable - a different phenomenon that happens to land on the same side of
the 0.95 bar.

**What is reported.** Both budgets side by side, with 1.4b labelled under-optimised at both.
**No scale claim is made in either direction.** The residual 24-point gap is not evidence of
a scale effect, because 3200 steps has not converged either - the surviving failures have
median R2 0.752, which *fell* from 0.899 as the five easiest were fixed.

**What the project gains instead**, and it is worth more than the withdrawn claim:

1. **A fixed step budget silently under-fits wider models, and reads as a scale effect.**
   Anyone running one interpretability protocol across a model suite at a fixed budget will
   see this and may report it as scale. That is a real methodological contribution.
2. **Held-out R2 passes an independent test.** It was calibrated on GPT-2 against silent
   failures; here it detects a completely different failure mode at AUC 0.996 without
   retuning.
3. The pre-registered sceptical check did exactly its job. **The exciting result did not
   survive, and it was filed in advance that it would not be reported until it had been
   attacked.**

**Secondary endpoint mostly not evaluable.** Per-rung signal AUCs need failures, and the
n=50 amendment leaves 70m/160m/410m with **2 failures each** - their AUCs (0.615 to 1.000)
are noise. Only 1.4b has enough, and there held-out R2 dominates as everywhere else.

---


### B-1b - THE PRIMARY ENDPOINT RESOLVES (2026-09-09) - **RESTART AGREEMENT IS DOMINATED, p < 0.0001**

Local CPU, GPT-2 L6, **n=300, 2 restarts**, 08:46-20:35, 141.7 s/unit.
`results/b1b_primary_gpt2.jsonl`. Protocol: `preregistration-b1-stability-calibration.md`
+ Addendum 1.

**52 failures of 300 (17%)** - inside the 36-142 the power work requires, so this is the
first adequately powered arm in the series.

| signal | AUC |
|---|---|
| **held-out R2** | **0.942** |
| restart agreement (`stability`) | 0.778 |
| two-route disagreement | 0.752 |
| `r2_spread` | 0.704 |

**Primary endpoint, DeLong on correlated ROC curves, same units:**
`restart agreement - held-out R2 = -0.164, z = -5.22, p < 0.0001`.

**This is the pre-registered outcome "stability AUC significantly below held-out R2".**
The field's default reliability check is significantly worse than a number the fit already
computes for free.

**What a threshold costs, which is the practitioner-facing result:**

| catch rate | held-out R2 discards | restart agreement discards |
|---|---|---|
| 50% | **0 of 248** | 40 of 248 (16%) |
| 70% | 8 of 248 (3%) | 62 of 248 (25%) |
| 80% | 24 of 248 (10%) | 90 of 248 (36%) |
| 90% | 52 of 248 (21%) | 118 of 248 (48%) |

At 50% catch, held-out R2 costs **nothing**; restart agreement costs 40 good units. The
gap is not a rounding difference in AUC, it is the difference between a usable rule and an
expensive one.

**6 of 52 failing units look stable (>0.95)** - the stable-and-wrong class exists at 2
restarts, where B-1's 5-restart arm found none of 9.

**Finding 2 is confirmed and now properly powered.** The earlier readings (C13's 0.906
vs 0.802, B-1's 0.913 vs 0.845) pointed the same way on too few failures; this one
settles it.

---

### B-0 - THE DEVICE IS A CONFOUND (2026-09-09) - **FAIL, and it is a finding**

Kaggle T4, 16 units, 2 restarts, same seeds on both devices.
`data/b11/device_equivalence.json`.

| | |
|---|---|
| units flipping pass/fail | **5 of 16 (31%)** |
| max abs d alignment | **0.6615** (tolerance 0.01) |
| median abs d alignment | 0.00068 |
| GPU speedup | 1.29x |

**The median unit agrees to four decimal places and the tail disagrees by 0.66.** That is
the pre-registered prediction confirmed: failing units sit near basin boundaries by
construction, and that is exactly where floating-point reduction order flips the answer.

**Reportable claim: estimator failure classification is hardware-dependent.** Reproducing
an interpretability result on different hardware may reproduce the aggregate and not
reproduce *which units failed*.

**Consequences, all already in force:** the ladder is internally GPU-consistent and stands
on its own; the local arms are internally CPU-consistent; **the two are never pooled**, and
the 160m-rung-vs-C53 cross-check is now expected to differ rather than agree.

---

### B-11 - PYTHIA SCALE LADDER (2026-09-09) - **RISES WITH SCALE; NOT REPORTABLE UNTIL THE STEPS CHECK RUNS**

Kaggle T4, one device, n=50 per rung, 2 restarts, depth-matched to 0.5.

| rung | d_model | pass | rate | Wilson 95% | median | min |
|---|---|---|---|---|---|---|
| 70m | 512 | 48/50 | 96% | [0.865, 0.989] | 0.9970 | 0.0982 |
| 160m | 768 | 48/50 | 96% | [0.865, 0.989] | 0.9989 | 0.1563 |
| 410m | 1024 | 48/50 | 96% | [0.865, 0.989] | 0.9994 | 0.9145 |
| **1.4b** | 2048 | **31/50** | **62%** | **[0.482, 0.741]** | 0.9878 | 0.0256 |

**The filing anticipated this exact outcome and forbids reporting it yet:**

> if the failure rate rises with scale, do not report it yet. Wider models may simply be
> harder to fit at a fixed 1600 steps. **Re-run the top rung at 3200 steps and report
> both.** A rate that falls with more steps is an optimiser artefact, not a scale effect.

**The artefact reading is currently the more likely one.** Three rungs sit at *exactly*
48/50 across a 2x width range and then only the widest breaks. 1.4b is d_model 2048 - 4x
the width of 70m - fitted with the same 1600 steps and the same width-64 bottleneck. That
is what an under-optimised fit looks like, not obviously what a scale effect looks like.

**Required next run: `--model EleutherAI/pythia-1.4b --layer 12 --d-mlp 8192 --neurons 50
--steps 3200`, on GPU, reported beside the 1600-step result whichever way it comes out.**

Also noted: 410m's min alignment is **0.9145** - it has no severe failures at all - while
its neighbours have minima of 0.098 and 0.156. Aggregate invisibility is not monotone in
scale.

---


### B-1 - restart agreement calibrated, 5 restarts, n=100 (2026-09-09) - **SENSITIVITY ARM, UNDERPOWERED AS FILED**

Ran 22:56 to 08:46 across a machine sleep at 02:54 that cost five hours and no rows.
`results/b1_stability_gpt2.jsonl`, GPT-2 small, layer 6, 8000 tokens, 1600 steps,
**5 restarts**, the C13/C47 unit draw.

| | value | C13 at 2 restarts |
|---|---|---|
| passing | **91/100**, Wilson [0.838, 0.952] | 77/100 |
| median alignment | 0.9943 | 0.9933 |
| **min alignment** | **0.8115** | **0.1445** |

**Signal AUCs, all on the same 100 units:**

| signal | AUC |
|---|---|
| held-out R2 | **0.913** |
| restart agreement (`stability`) | 0.845 |
| two-route disagreement | 0.760 |
| `r2_spread` | 0.648 |

**Read this arm as underpowered, because it is.** 9 failures. Addendum 1 was filed before
the run finished, on exactly this basis: comparing 0.845 against 0.913 with 9 positives is
not a comparison. The primary arm (2 restarts, n=300) decides.

**Three things the arm does establish.**

1. **More restarts is a better optimiser, and the filing already said this is not a
   finding.** 77/100 to 91/100, and the severe failures disappear - min alignment 0.1445
   becomes 0.8115. C13's aggregate-invisibility case (a median of 0.9933 hiding a unit at
   0.1445) is much weaker here: 0.9943 hiding 0.8115.
2. **At 5 restarts, restart agreement is NOT useless, and my earlier framing was wrong.**
   0 of 9 failing units look stable at the 0.95 threshold. The "stable and wrong" cases in
   the n=8 pilot came from a different configuration. The honest claim is narrower than the
   one I started with: restart agreement trails held-out R2 here, on 9 failures, and
   whether that gap is real is not yet known.
3. **Held-out R2 leads in every arm measured so far** - 0.906 at C13, 0.913 here. The
   incumbent recommendation survives this arm without being confirmed by it.

**What this arm cannot answer, by construction.** Secondary 4 - does a cheap 2-restart
stability estimate match the 5-restart one - needs the per-restart pairs, and B-1's rows
predate `stability_pairs`. B-2 onward carry the column.

---


*(Newest first. Append; never rewrite.)*

### C54 / C55 - Study 2 P1 and P2, planted random directions (2026-09-08) - **VOID**

Archived to `data/s2/`. P1: 24 rows, 8 random unit plants x 3 strengths, extraction at
the plant layer. P2: 40 rows, same plants at 40% of norm, extraction at layers 37, 39,
43, 47, 55. Both ran to completion at `2026-09-08b`.

**P1 - recovery sits below the null at every strength.**

| %norm | median recovery | median null | rec > null |
|---|---|---|---|
| 10% | 0.0064 | 0.0128 | 3/8 |
| 20% | 0.0086 | 0.0117 | 2/8 |
| 40% | 0.0078 | 0.0100 | 3/8 |

Above null on **8 of 24** rows, worse than a coin flip. Max recovery anywhere 0.0264
against a pre-registered bar of 0.30 - **11x below**. Flat across a 4x strength range,
which rules out the obvious "too weak" explanation.

**P2 - flat at the null at every depth, and no rise anywhere.**

| layer | offset | median recovery | median null | rec > null |
|---|---|---|---|---|
| 37 | +0 | 0.0079 | 0.0100 | 3/8 |
| 39 | +2 | 0.0085 | 0.0102 | 3/8 |
| 43 | +6 | 0.0080 | 0.0111 | 3/8 |
| 47 | +10 | 0.0065 | 0.0108 | 3/8 |
| 55 | +18 | 0.0073 | 0.0067 | 4/8 |

16 of 40 above null - exactly chance. Max recovery at any depth 0.0198. P2 ran only
because the notebook was headless and executes every cell; it was expected to be
uninterpretable and it is. Its one use was as a free negative control, and it serves
that: there is no signal at the plant layer, and none appears anywhere downstream.

**VERDICT: VOID, NOT FAILED. The fault is the design, and it is mine.**

The pre-registration permits reporting this as a finding - "a deployed extraction
pipeline cannot recover a planted cause" - and it should not be reported that way.

The plant is a **random unit direction**. The persona pipeline works because a *trait*
direction makes the model produce trait-consistent text, and difference-of-means over
that text recovers the direction. A random direction has no natural representation in
the model, so the text it produces carries no consistent signal to re-read. **The
pipeline was asked for something that cannot happen.**

My filing justified random as removing a confound: *"a concept vector would confound
whether the pipeline recovers what was planted with whether the model has a natural
representation of the trait."* That reasoning removed the confound by removing the
mechanism the pipeline depends on. The total absence of signal - below null at every
strength and every depth, not merely weak - is what a mis-specified test looks like
rather than a failing instrument.

This is the same category as C27-C30 on Qwen: a null measured through an instrument that
could not have produced a signal.

**The corrected run**, filed separately before it goes: plant a **real concept vector**
that C48 showed demonstrably steers Gemma at 40% of norm (10/30 semantic against a 3/30
baseline), and score recovery against a different concept's vector as the null. That
confounds slightly - the model does have a representation of the concept - but it tests
something achievable, which this did not.


### C53 - E0.1 gate on a second model family, Pythia-160m (2026-09-08) - **REPLICATES, AT A DIFFERENT RATE**

The run outline v3 made the top priority, since Study 1 leads the paper and "one model
family" is its obvious objection. Pre-registered at `docs/preregistration-e01-pythia.md`,
filed before the run. Everything held to C13 except the model: layer 6 of 12, 8,000
tokens, 2 restarts, 1600 steps, batch 32, same unit-draw procedure, same pass criterion.
100 units, 86.0 s/neuron, 8,601 s. `results/e01_gate_pythia.jsonl`.

**PRIMARY - FAIL, at a materially different rate.**

| | GPT-2 (C13) | **Pythia-160m** |
|---|---|---|
| pass rate | 77/100 | **93/100** |
| Wilson 95% | [0.6785, 0.8416] | **[0.8625, 0.9657]** |
| verdict vs the 0.90 bar | FAIL | **FAIL** |

**Both models fail the pre-registered criterion, so the phenomenon replicates across
families.** But the intervals **do not overlap**, so the magnitude is genuinely
model-dependent: 23% silent failure on GPT-2 against 7% on Pythia.

**Per the filed rule this drops the "roughly a quarter" framing.** The claim becomes a
range - the estimator fails silently on 7-23% of units depending on the model - and both
rates are reported side by side. The GPT-2 figure is not retired.

**The failures are equally invisible, and on Pythia they are worse when they happen.**

| | GPT-2 | Pythia |
|---|---|---|
| median alignment, all units | 0.9933 | **0.9994** |
| median alignment, failures only | 0.8885 | **0.7941** |
| **minimum alignment** | 0.1445 | **0.0761** |

Pythia looks *better* by any aggregate anyone would report - a median of 0.9994 would be
called excellent - while containing a unit recovered at 0.0761. Fewer failures, more
severe. That strengthens rather than weakens the aggregate-invisibility argument: the
median gets *further* from the truth as the failure rate drops.

**SECONDARY 1 - the flag transfers, and improves.** This is the paper's deliverable, so
it was the endpoint most at risk:

| detector | GPT-2 | Pythia |
|---|---|---|
| two-route disagreement | 0.810 | **0.975** |
| held-out R2 | 0.915 | **0.995** |

Both rise. The ground-truth-free check does not depend on the model family, and on the
model where failures are rarer and more extreme it is close to perfect.

**One honest caveat on that.** Part of the improvement is likely that Pythia's failures
are more severe (median 0.794 against 0.889, minimum 0.076 against 0.145), and extreme
failures are easier to separate. The AUC gain is therefore not purely a statement about
the detector - it is partly a statement about what it was asked to detect. **Do not
report 0.995 as "the flag is better on Pythia" without that sentence.**

**What this buys the paper.** The lead result survives its most obvious objection. The
phenomenon is not a GPT-2 artifact, the aggregate-invisibility argument holds on both,
and the deliverable works on both. What has to change is the headline number - a range,
not a quarter.


### C50 / C51 / C52 - A-14, the validated-steering window (2026-09-08) - **OUTCOME B2**

The run Addendum 4 was filed for, before it ran. Gemma-3-27B, layer 37, refit vectors,
`--alpha-frac 0 0.30 0.40 0.50 0.60` of the 36,245 concept-token norm. Real, random and
span, 300 rows each. `data/s3/gw_forced*.jsonl`.

**Neutral framing** (the arm with interpretable magnitudes):

| alpha | %norm | real | random | span |
|---|---|---|---|---|
| 0 | 0% | 0.1882 | 0.1882 | 0.1882 |
| 10,873 | 30% | **0.1009** | 0.2575 | 0.2085 |
| 14,498 | 40% | 0.1238 | 0.1851 | 0.2166 |
| 18,122 | 50% | 0.1688 | 0.1557 | 0.3039 |
| 21,747 | 60% | 0.2445 | 0.2624 | 0.2815 |

**FILED PRIMARY - real vs random, neutral, pooled across 30-60%, paired within
strength:** real 0.1595 against random 0.2152, Wilcoxon W=3034, **p = 0.119**, real
lower on 72 of 120 pairs.

**OUTCOME B2: no significant difference.** The suppression does not survive into the
window where the vectors demonstrably steer.

**What the filed rule says to do, quoted from Addendum 4:** *"The C45/C46 headline is
then reported as confined to strengths at which the vectors do not measurably steer,
which is a substantial weakening, and Paper A's central claim becomes the
readout-dependence and the framing results rather than A3."* That is now the position.

**Per-strength, pre-specified as secondary.** Only the bottom of the window survives, and
it is the strength closest to the unvalidated region:

| %norm | neutral p | introspective p |
|---|---|---|
| 30% | **0.045** | **0.002** |
| 40% | 0.289 | 0.339 |
| 50% | 0.903 | 0.670 |
| 60% | 0.792 | 0.685 |

**And the direction reverses.** Real climbs with strength (0.1009 to 0.2445 neutral,
0.0648 to 0.3009 introspective) while random stays roughly flat. They converge around
40-50% and by 60% real is at or above random. The A3 effect is a **low-strength
phenomenon that inverts**, not a property of concept vectors.

**The below-baseline suppression is also gone.** At 1-10% of the norm (C45/C46) a real
vector pushed neutral P(YES) below the no-injection baseline at p=0.004-0.009, down on
23-24 of 30 concepts. Here:

| %norm | real | vs baseline 0.1882 | down on |
|---|---|---|---|
| 30% | 0.1009 | p=0.516 | 16/30 |
| 40% | 0.1238 | p=0.685 | 15/30 |
| 50% | 0.1688 | p=0.952 | 15/30 |
| 60% | 0.2445 | p=0.237 | 12/30 |

Chance. Both halves of the C45/C46 result are confined to strengths at which C48 showed
semantic steering running at 2/30 and 5/30 against a 3/30 baseline.

**Secondary 1 - C49 holds.** Span is indistinguishable from random at every strength
here too (p = 0.44, 0.47, 0.13, 0.79), so the off-manifold account stays refuted. That
finding does not depend on the window.

**What this settles.** Study 3 has no stable empirical claim about what detection is, at
any strength, on this model. Three reversals, and the third confined to a range where the
instrument is not working. **Outline v3 already put Study 3 last as a five-knob
methods chapter; this run confirms that was the right call rather than a defensive one.**

The five knobs are untouched by this: readout method, scoring rule, extraction position,
strength normalisation, prompt framing. None of them depended on A3 being true.


### Literature scout 2, 8 September 2026 - **PERSONA VALIDATION IS OPEN; AGENT DRIFT IS NOT**

Second scouting pass, on agent personas and AGI-adjacent framings, asked before
committing chapter two.

**Agent persona drift is crowded and behavioural.** ContextEcho (2605.24279, drift in
long agentic-coding sessions), SPASM (2604.09212, multi-turn identity failures),
Quantifying Agent Drift (2601.04170), Measuring What Persists (2606.21843, geometric
framework for agent identity), FinPersona-Bench (2606.31522), plus a CHI 2026 temporal
stability paper. All measure drift from outputs. Benchmarks in this space are cheap to
produce and several groups are producing them; we would be the fifth entrant with no
advantage. **Not a direction.**

**AGI framings: nothing actionable.** No measurable gap that a capstone on free-tier
compute can close.

**The gap that is open, and it is Study 2.** Persona vectors are validated two ways in
the literature: by **steering effect**, and by **correlation with finetuning-induced
shifts** (r = 0.76-0.97, Anthropic 2507.21509). Auditing tools are already built on top -
2607.13162 audits open-weight models with them, 2605.13329 traces them through
pretraining. **Nobody plants a known direction and measures whether the extraction
pipeline recovers it.**

**Venkatesh & Kurapath (2602.06801) makes this sharper, and it is a gift rather than a
scoop.** They show orthogonal perturbations achieve near-equivalent steering efficacy, so
behavioural equivalence classes are large. That means **"it steers, therefore it is the
right direction" is demonstrably invalid** - and it is the primary validation the
persona-vector literature relies on. Our motivation section now exists as someone else's
citable result.

**Consequence for the plan.** Study 2's blocker was the K>=2 degeneracy (C14), but that
blocks only **P4**, the multitrait-multimethod matrix. **P1 (recovery at the plant layer)
and P2 (recovery versus depth) need only K=1**, which the estimator handles. The decisive
experiments are unblocked.

Stakes are also higher than in August: persona vectors are moving into production
monitoring while their extraction has never been checked against truth.

**Ordering.** Study 1 still leads - free ground truth, no competitors. **Study 2 replaces
Study 3 as chapter two**, and the through-line becomes: free ground truth at the unit
level, planted ground truth at the trait level, the same instrument failing both ways,
and steering-based validation unable to tell. Study 3 drops to a short section.

**Caution recorded.** Study 2 needs a >=7B instruct model on Kaggle, and this week cost
four sessions to loader and protocol bugs. P1 is a positive control: if it does not
recover a planted direction at the plant layer, stop and fix rather than proceed. Budget
two sessions and treat the first as likely lost.


### Literature scout, 8 September 2026 - **STUDY 3 IS CROWDED; STUDY 1 IS NOT**

Not a run. A deliberate scouting pass before committing to the methods-paper pivot,
because the pivot was being chosen on internal evidence alone.

**What is already published in Study 3's space**

| paper | what it holds |
|---|---|
| Detecting the Disturbance, arXiv 2512.12411 | **Binary injection-detection is artifactual** - global logit shifts bias toward YES regardless of question content, control correlation r=0.999. Llama-3.1-8B, layers 0-30 |
| Lederman & Mahowald, arXiv 2603.05414 | Emergent introspection is **content-agnostic**; models default to high-frequency guesses like "apple" |
| Venkatesh & Kurapath, arXiv 2602.06801 | Steering vectors are **non-identifiable**; orthogonal perturbations achieve near-equivalent efficacy. Feb 2026, revised Apr |
| Can LLMs Introspect? A Reality Check, arXiv 2605.26242 | **Published at COLM 2026** |
| Attention-guided feature learning, arXiv 2602.00333 | Token position for extraction **already known to matter enormously**, including chat-template positions |

**Consequences, stated plainly**

1. **Our A2 result is scooped twice.** "Detection is content-free" is Lederman & Mahowald
   and 2512.12411. It was never ours.
2. **2512.12411 extracts by averaging over all prompt tokens.** They never hit our
   template-tail bug because their method sidesteps it, and their control is cleaner than
   ours: same injection, a factual question with a known NO answer, r=0.999. They did in
   one control what took us C18-C49.
3. **The extraction-position finding is not novel as "position matters."** That is known.
   What may remain is the sharper form - a vector passing every reported health check
   while carrying nothing - but it is a narrower claim than it looked yesterday.

**What survives in Study 3**, each checked against 2512.12411 specifically:

- readout comparison, first-token vs generated text (they use logits only and say so)
- introspective vs neutral prompt framing (they do not vary framing at all)
- scoring-rule sensitivity, 6.7% vs 33.3% with the FPR flipping 0% to 17%
- cross-model norm incomparability, 138x between Gemma and Qwen at the concept position
- norm-matched random and span controls - their control varies the *question*, not the *vector*

**What nobody is doing.** A targeted search for prior work using an MLP neuron's weight
column as free ground truth to score a direction-finding estimator returned neuroscience
MID papers and LLM weight-space papers, and nothing joining them. **Study 1's territory
is uncrowded**, its ground truth is free rather than planted, and we already hold: 77/100
recovery with a pre-registered failure, failures that are confidently wrong (restart
agreement 0.85-0.99), a ground-truth-free flag at AUC 0.915, characterised ruggedness, a
measured seed lottery, and a repair that failed its own criterion and was reported.

**A free lead from the paper that scooped us.** 2512.12411 finds *genuine* partial
introspection in differential tasks - 88% localising which sentence was injected, 83%
comparing injection strengths - but only at **early layers L0-L5**. We tested binary
detection at layer 37 of 62. If Study 3 is ever to carry a positive result, that is the
experiment. It would confirm their finding rather than establish ours.


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

> **Study 3 addendum, 7 Oct (S-4):** under paired TOST (margin 0.10), real vs random at alpha 6 and at
> 32768 is *inconclusive*, not equivalent. Only C20 at alpha 6 (prompt framing) and real vs span
> at 30% of norm are equivalent. Random is significantly above real at 30% of norm. See §4, S-4.
>
> **STATE OF THE FINDINGS, 6 Oct 2026 (B-14 landed) - supersedes the 5 Oct note below.**
> On the fixed estimator the primary endpoint stands: restart agreement 0.792 vs held-out R2
> 0.919, diff -0.127, p = 1.8e-04, permutation p = 0.001. Pooled over six conditions it is
> -0.125 [-0.165, -0.084]. **But the advantage is entirely the under-fitted class.** On the 16
> converged-wrong failures, both checks are tied at about 0.78. Restart agreement adds nothing
> beyond unit-difficulty baselines (0.825 -> 0.831) and R2 does (-> 0.892). "Method
> disagreement" is withdrawn (ground-truth-dependent). Route agreement is the replacement and
> exists only from B-11c on (0.910 there). See §4, B-14.
>
> **STATE OF THE FINDINGS, 5 Oct 2026 - read before quoting anything below.** Study 1's
> primary endpoint (B-1b, p = 1.8e-07) stands as measured, and the ranking held-out R2 >
> restart agreement holds in every arm with more than two failures. Four qualifiers now
> travel with it: (i) every Study 1 number so far was fitted with batch-coupled early
> stopping, so B-1b must be re-run after the fix before it is a paper number; (ii) pass/fail
> verdicts move with restart count (~12%), batch composition (~20-33%), token sample (10/50)
> and device, so single pass rates are soft; (iii) the flag's strength is family-dependent
> (AUC 0.94 on GPT-2 L6, 0.72 on GPT-Neo L10); (iv) depth findings are unpaired. Study 2 P1b
> failed as filed. Study 3's reproduction figure is 6.7% (C40), not 10.0%, and the only
> validated-vector Gemma measurement (C45/C46) came out A3, random above real. Detail in §4,
> "Audit of every run to date".

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

- **Does a content-free vector also remove the self-report disclaimer? (added 6 Oct, from
  APERTURE A-G1)** In APERTURE's R3, "As an AI, I don't experience thoughts" appeared in 8/8
  responses at alpha <= 0.5 and in 0/16 at alpha >= 1; Qwen refused 30/30 at baseline in
  C30. If random vectors also remove it, a model's apparent willingness to introspect under
  injection is part of the perturbation alarm. S-2 tests it.
- **Is concept identification separable from derailment? (added 6 Oct)** APERTURE's G1 found
  "exact" identification almost only at high KL. No CALIPER run records per-trial KL, so
  CALIPER's identification numbers cannot yet be split by coherence. S-0 adds the KL meter.

- **Coupled early stopping in `fit_batch` (added 5 Oct, standing, unfixed).** Patience is
  shared across the batch, so a unit's training length depends on its batch-mates. Every
  batched Study 1 run carries it. Fix: per-unit patience with an active mask, and freeze
  stopped units by restoring their parameters after `opt.step()`. Zeroing their gradients is
  not enough, because Adam's momentum keeps moving them. Then a test that a unit fitted alone
  equals the same unit fitted inside a shuffled batch.
- **Token sample is a variance component (added 5 Oct).** Same units, same config, a
  different 8k token draw: 10/50 verdicts change (B-10 vs B-1b). `collect` truncates before
  shuffling, so "8k tokens" means different things in different scripts.
- **Is the failure class real or under-training?** If fixing the stopping rule moves many
  B-1b failures to passes, part of the "silent failure" rate is an optimiser artefact, as at
  1.4b. Each failure should be classified as under-trained (R2 still rising) or wrong basin
  (R2 converged near 1 at a wrong direction) after the re-run.
- **GPT-Neo L6 below-chance alignment (0.0026 on 4 units).** Unexplained; ground truth
  verified exact there on 5 Oct. Needs n >= 20 and per-unit diagnostics.
- **Why is the flag weak on GPT-Neo (AUC 0.715)?** Unknown. If it is the heavy failure rate
  (41/100) or the family, the headline recommendation needs a family qualifier.

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
| 2026-10-07 | **THE CASCADE ROUTE WAS NEVER REPRODUCIBLE: its head initialisation came from torch's global RNG.** `_Bottleneck` seeded the direction `v` from a generator but built its MLP head with `nn.Linear`, which draws from the process-wide RNG. So every single-unit `fit` (the cascade route, its polish, `fit_deflate`) depended on everything the process had fitted before. **Found by the B-15 seed smoke test:** unit 2262 reproduced B-14's direct route exactly (align_direct 0.0137, stability 0.8569), but its cascade gave 0.930 against B-14's 0.976 and flipped the verdict. Repeated calls in one process at the same seed and 10 threads gave 0.941, then 0.559.
- **Scope:** every run's `align_cascade`, and every `align_selected`/`r2_k1` where the cascade won. The direct route (`fit_batch`) and restart agreement were always seeded. The cascade draws were still random draws from the intended distribution, so no result is biased, but none is reproducible to the digit.
- **Fix:** the head is built inside `torch.random.fork_rng` from the same seed, leaving the global state untouched. The test fails on the old code and passes on the new.
- **Runs:** B-2c and B-8b (running) use the old initialisation; T-SAE and B-15 use the fixed one, as the B-15 prereg note records.
- **Also seen:** at a fixed seed, 3 vs 10 threads changed the cascade alignment (0.885 vs 0.941). Thread count is a separate, smaller reproducibility source and is now recorded | `caliper/estimator.py`, `tests/test_estimator.py` |
| 2026-10-06 | **B-2c and B-8b launched 08:40 under Task Scheduler (`caliper-rerun-fixed`, hidden window).** B-2c (Pythia-160m L6, n=300) runs first, then B-8b (GPT-Neo-125M L10, n=100), both with `--independent-units`; about 17 h in total. The progress log is `results/rerun_fixed.log`, with per-run logs beside it. B-2c's stimulus was collected and 300/300 units are alive. The old `caliper-after-b12` task was removed | `docs/preregistration-b2c-b8b-reruns.md`, `experiments/rerun_fixed.sh` |
| 2026-10-06 | **B-14 SETTLES THE RE-RUN SCOPE: the reduced path.** B-14's gap (-0.127) lies inside B-1b's bootstrap interval, and 256/300 verdicts agree (McNemar p = 0.45). Re-run on the fixed estimator, with directions saved and route agreement: B-2c (Pythia-160m L6, n=300), B-8b (GPT-Neo L10, n=100) and B-15 (replicates). B-7's depth legs and the other coupled arms go to an appendix as coupled-estimator replications. **The paper's central claim is reframed around Addendum 2:** held-out R2 catches under-fitting, and no check catches the converged-wrong class | §4 B-14 |
| 2026-10-06 | **B-14 ADDENDUM 2 FILED BEFORE UNBLINDING (230/300 rows): failure-class-stratified AUCs.** Exploratory splits of already-unblinded arms show held-out R2's advantage comes from under-fitted failures. On converged-but-wrong failures, both checks are near chance (B-8: R2 0.555, restart 0.477). B-14 will report AUCs by class, with a sensitivity to the class boundary. If the pattern holds, the paper's claim becomes: the checks catch under-fitting, none catches the converged wrong basin, and here is that class's size | `docs/preregistration-b14-addendum-2.md` |
| 2026-10-06 | **APERTURE's notebook re-read in full and folded into this one.**
- **§3:** gains a follow-up with every APERTURE run as A-R1 to A-R12, plus A-F1, and which have data on disk (A-R1, A-R3, A-F1).
- **Correction:** R11 is now carried by A-F1 c00.
- **Six interpretations for CALIPER:**
  - the framing control at two scales and two readouts (Paper 2);
  - identification at derailment;
  - prompt vocabulary leaking into answers;
  - the disclaimer removed by perturbation;
  - library-version drift as a fourth reproducibility source;
  - centring.
- **§6:** two new open questions.
- **§8:** APERTURE's Kaggle gotchas.
- **Elsewhere:** the run plan's S-0 gains the KL meter for impact matching and coherence bands; the citation ledger gains §23 for works cited only through APERTURE | §3 follow-up, `docs/APERTURE_INHERITANCE.md` §5, `docs/RUN_PLAN_L2_L3.md` |
| 2026-10-06 | **APERTURE AUDITED FOR WHAT CALIPER CAN USE.**
- **Evidence:** F1's archived reference cell is the replacement for R11, and its c04 flag shows the introspective prompt's vocabulary being emitted as the answer.
- **Adopted:** APERTURE's vector recipe (whole-sentence means against same-category negatives) as a live arm; its stability/probe/steering gate as a health-check set to calibrate; residual centring; the gamma prior-null estimator for identification beyond default words; the affect-confound stratification; E13's Full-minus-Context-only control; seed twins and meta-d' for Part B.
- **New hypothesis from R3 pilot data:** the perturbation, not the concept, removes the 'As an AI' disclaimer (4/4 at alpha <= 0.5, 0/4 at alpha >= 1).
- **New optional run:** S-13, an Assistant-Axis sweep bridging L2 and L3.
- **Warning:** the name CALIPER is taken by an LLM probing method (arXiv 2606.04915); the released bench needs another name | `docs/APERTURE_INHERITANCE.md`, `docs/RUN_PLAN_L2_L3.md` |
| 2026-10-06 | **"METHOD DISAGREEMENT" WAS NEVER GROUND-TRUTH-FREE. Every disagreement AUC to date is withdrawn as a practitioner's check.** `e01_gate.py` (and `b10_required_n.py`) record `disagreement = |align_direct - align_cascade|`, and both terms are alignments to the true w. No practitioner could compute it. The affected numbers are B-1b's 0.752, C35's 0.802, C38's 0.810, and the B-7, B-8, B-10 and B-12 disagreement AUCs, plus every 'held-out R2 beats disagreement' sentence. It lost to R2 anyway despite seeing the answer, which is a curiosity, not a result. **Fix:** `e01_gate.py` now also records `route_agreement = |cos(direct direction, cascade direction)|`, which is ground-truth-free. `analyse_b14.py` labels `disagreement` as ground-truth-dependent and scores `route_agreement` whenever rows carry it. B-14 started before the fix, so it has no `route_agreement`. Since runs now save their directions, the comparison can be recomputed for any future run. In Paper 1, method disagreement is reported only from runs that carry `route_agreement`, and §5 finding 2's comparison with disagreement is withdrawn. The `analyse_b14.py` edit was made before it ran on B-14 and does not touch the primary or any pre-registered secondary computation | `experiments/e01_gate.py`, `experiments/analyse_b14.py` |
| 2026-10-06 | **APERTURE MERGED INTO CALIPER; L2 AND L3 PLANNED IN FULL.** The two programmes merge at the paper level. APERTURE keeps its repo and is used as a library for L3, and its runs are registered here under the prefix A-. Plan of record: `docs/RUN_PLAN_L2_L3.md`.
- **L2 runs:** T-0 harness; T-1 exact-estimand precision control; **T-2 exact readout bench (primary)**; T-3 whitened P1b rescore; T-4 trained-in directions (Gate B); T-5 and T-6 optional.
- **L3 runs:** Part A, the instrument audit for Paper 2: S-0 apparatus merge, S-1 dead-vector precision/position ablation, S-2 small-model audit, S-3 finishing APERTURE F1 under its frozen prereg, S-4 reanalysis. Part B, exact-truth self-reports for Paper 3: S-5 to S-10 plus S-12 human grading.
- **Gates:** A on 31 Dec, B in mid-March. All new science ends by March 2027.
- **Publication collision with APERTURE F8 (§6):** resolved by the merge | `docs/RUN_PLAN_L2_L3.md`, `docs/PIVOTS.md` P11 |
| 2026-10-05 | **B-14 ANALYSIS SCRIPT WRITTEN AND TESTED ON B-1b, BEFORE B-14 IS OPENED.** `experiments/analyse_b14.py` implements the prereg and Addendum 1. On B-1b it reproduces the primary exactly: restart 0.778, R2 0.942, diff -0.164, p = 1.8e-07, bootstrap 95% [-0.224, -0.106]. With alignments scrambled across units it is null (p = 0.67). The ground-truth signal scores AUC 1.0, and the label-shuffle permutation test gives p < 0.001. **Deviation from Addendum 1 items 1-2:** the identifiable-direction and functional labels need the fitted directions, and `e01_gate.py` archives only alignments. The script reports this instead of skipping it, and substitutes the per-unit 1/gamma cosine ceiling with a sensitivity rerun (B-1b: no unit limited, median ceiling 0.99976). Future runs should save directions. **Development findings on B-1b (already unblinded, so reportable):** (a) the nuisance baselines alone (active fraction, z_mean, kurtosis, GELU first-order coefficient) reach cross-validated AUC 0.844, rising to **0.907 with held-out R2** and to **0.841 with restart agreement**. Restart agreement adds nothing beyond unit difficulty, and held-out R2 does. (b) 40 of B-1b's 52 failures have held-out R2 <= 0.99 (under-fitted) and 12 are wrong-basin. Most of the old failure class is not stable-and-wrong, which makes B-14 the decisive run. Tests: `tests/test_analyse_b14.py` (3) | `experiments/analyse_b14.py` |
| 2026-10-05 | **B-14 ADDENDUM 1 FILED BEFORE UNBLINDING; citation ledger consolidated; two construct-validity findings.** (1) The addendum adds secondary labels and analyses: the identifiable-direction label, functional correlation, threshold sweep, PR-AUC, nuisance baselines with incremental AUC, calibration, and permuted-label and cheating-signal controls. The script is developed on B-1b, then run once on B-14. (2) **LayerNorm-null direction:** s . (1/gamma) is constant, so w's component along 1/gamma is not identifiable. In GPT-2 small the median share of |w|^2 along it is about 5e-4. Units whose share limits cosine below 0.95: L2 4, L6 0, L10 1 (of 3072). B-1b is unaffected. B-7 L2 units 1825 (ceiling 0.917, scored 0.814) and 666 (ceiling 0.894, scored 0.872) were counted as failures; rescoring is pending. (3) **Held-out tokens share sequences with training tokens:** `collect` shuffles rows before `fit_batch` takes its first 20% as test. Sequence-level splitting is a P1 fix for future runs. (4) `docs/CITATIONS.md` grows from 56 to 459 citation rows, with a new MEM level (170 rows) and a verify-before-submission list. (5) Also noted: `activations.py` hard-codes GELU and `_mlp_in` returns `up_proj`, so a SwiGLU run needs the gate+up span and the right activation before B-16 | `docs/preregistration-b14-addendum-1.md`, `docs/CITATIONS.md` |
| 2026-10-05 | **B-14 IS CHAINED BEHIND THE QUEUE BY A ONE-SHOT TASK.** Task Scheduler task `caliper-after-b12` (registered 08:50, started 08:51) runs `experiments/after_b12.sh`. The script waits until no process is running `run_queue`, `b12_batch_invariance` or `e01_gate`. If the control arm has 50 rows, it reruns `b12_batch_invariance.py` so the report carries the per-arm verdict. It then starts B-14 to `results/b14_primary_gpt2_indep.jsonl`, with progress in `results/after_b12.log`. Unlike the queue task, it is allowed to start and keep running on battery. B-14 is resumable, so an interruption costs only the unit in flight. Delete the task once B-14 is done: `schtasks /delete /tn caliper-after-b12 /f` | `experiments/after_b12.sh`, `experiments/after_b12.cmd` |
| 2026-10-05 | **B-13 LANDED: `--independent-units`.** `fit_batch` gains `unit_ids` (seed by neuron id) and `per_neuron_stop` (a per-unit patience counter; a unit stops recording once its own patience runs out). Both opt-in, so completed runs keep their meaning. `e01_gate.py --independent-units` turns both on and writes `independent_units` into the summary. Verified both ways on the test's 5-unit set: under the old rule, alone vs in a shuffled batch gives worst agreement **0.9728** and \|dR2\| 0.108; under the new one **1.000000** and 6e-07. Three new tests in `tests/test_batched.py`: the old seeding is position-keyed, alone equals in-batch, and the shared rule runs a unit longer. Bundle rebuilt | `caliper/batched.py`, `experiments/e01_gate.py`, `tests/test_batched.py` |
| 2026-10-05 | **`--per-neuron-seed` WAS NEVER UNIT SEEDING. IT SEEDS BY POSITION IN THE BATCH.** `_per_neuron_stack` used `range(n)`, so a unit's stream depended on where it sat in its batch, not on which neuron it was. In B-12 the units that came out identical at batch 8 and batch 32 sit mostly at matching in-batch positions (1, 2, 3, 6, 37, 38). The 4 Oct B-12 docstring's "init is now bitwise identical across batch sizes" holds only for those positions. So B-12's arms carry two couplings, position-keyed init and shared stopping, and cannot isolate reduction order | `caliper/batched.py` docstring, §4 B-12 |
| 2026-10-05 | **B-12's verdict is now judged per arm against the floor.** The summed version is kept as `n_flips_total`, with `n_flips_worst_arm` and per-arm `within_floor` beside it. The new test fails on the old logic (a two-arm case at the floor read FAIL) and passes on the new. The running B-12 process loaded the old script, so after its control arm lands, rerun `b12_batch_invariance.py` (all arms skip as complete) to rewrite the report | `experiments/b12_batch_invariance.py`, `tests/test_b12_verdict.py` |
| 2026-10-05 | **S1-2 PUT ON HOLD IN THE QUEUE.** The queue would have started it straight after B-12, without a filed prereg. Editing a running bash script is unsafe, so the script was renamed `experiments/s1_2_deflation.py.hold`. The queue's S1-2 step will then fail cleanly and the queue stops. Rename it back to run it. The prereg is now filed, and the code judges the cascade arm as filed, with best-of-arms printed as exploratory | `docs/preregistration-s1-2-deflation.md` |
| 2026-10-05 | **B-14 FILED: B-1b re-run with `--independent-units`.** Same 300 units, tokens, restarts and steps. Primary DeLong endpoint unchanged; decision table, failure classification (wrong basin = held-out R2 > 0.99) and McNemar against B-1b fixed in advance | `docs/preregistration-b14-primary-rerun.md` |
| 2026-10-05 | **B-7 analysis now reports the independence null.** `analyse_b7_depth.py` prints the expected count per pattern under independent layers next to the observed count. Its test now asserts the counts sit within 3 of the expectation and that the shared indices are not read as pairing. `results/b7_depth_structure.txt` regenerated (it had been committed as a binary, UTF-16 file) | `experiments/analyse_b7_depth.py`, `tests/test_analyse_b7_depth.py` |
| 2026-10-05 | **NO NEW STUDY 1 NUMBER GOES IN THE PAPER UNTIL THE STOPPING FIX LANDS AND B-1b IS RE-RUN.** The 5 Oct audit found batch-coupled early stopping in `fit_batch`; B-12 shows it moves 9-10 of 50 verdicts even with per-neuron seeding. Queue order after B-12's control arm: stopping fix + invariance test, then B-1b re-run (B-13), then everything else | §4 audit, §7.8 status 5 Oct |
| 2026-10-05 | **S1-2 IS NOT RUN UNTIL IT HAS A FILED PRE-REGISTRATION.** The criterion drifted from the cascade arm to best-of-arms in code, with no prereg file. File `preregistration-s1-2-deflation.md` with the cascade arm primary as originally written; resp_only and best-of-arms exploratory | §4 audit, Correction 7 |
| 2026-10-05 | **B-7 IS REPORTED UNPAIRED.** Index-matched units across layers are different neurons, and the pattern counts match independence. The paired analysis and its test are kept in the repo as a record but not cited | §4 audit, Correction 1 |
| 2026-10-05 | **B-12's summed verdict is not quoted.** Each small-batch arm is compared to the control on its own | §4 audit, Correction 4 |
| 2026-10-05 | **The positive-control figure is 6.7% everywhere.** C40's correction never propagated to `PROJECT_NOTES_COMPLETE.md` or the deck | §4 audit, Correction 9 |
| 2026-10-04 | **THE BUNDLE PARITY TEST COULD NOT SEE A MISSING MODULE, AND THE GAP WAS ONE IMPORT AWAY FROM FIRING.** `SHIPPED` is an explicit 8-file list, so adding a helper and importing it from a bundled script leaves all eight files byte-identical to source — every hash test passes — while the bundle is unrunnable, because the import fails only on Kaggle. Adding `planted_units` today would have done exactly that, had the script been on the manifest. New check walks each bundled file's AST and asserts every first-party import resolves inside the bundle, deciding locality from the **source** tree: deciding it from the bundle classifies as ours only the modules already shipped, which are precisely the ones that cannot be missing. That first version passed against a deliberately broken bundle, and only passed-and-failed-both-ways counts as a working tripwire | `tests/test_kaggle_bundle.py` |
| 2026-10-04 | **B-12's FILED CRITERION WAS STRICTER THAN THE INSTRUMENT'S OWN REPEATABILITY, SO IT WAS AMENDED BEFORE THE RUN.** It demanded zero pass/fail flips across batch sizes, but B-7 L6 (r=5) against B-1b L6 (r=2) on the same 50 units gives 10 of 50 verdict flips (20%) from restart count alone - and batch size perturbs float32 reduction order, which is no smaller a perturbation. Under the filed criterion, ordinary restart noise would have been scored as "flips remain", firing the branch that claims silent failure depends on estimator implementation. B-12 now runs a fourth arm at the reference batch size with a different restart count and judges batch flips against control flips, reporting the external 20% alongside. `--control-restarts 0` restores the strict version. **A criterion stricter than the measurement noise is unfalsifiable, not strict** | `experiments/b12_batch_invariance.py`, `tests/test_b12_verdict.py` |
| 2026-10-04 | **S1-2 GAINED A MULTIPLICATIVE ARM, AND ITS 0.80 CRITERION IS DELIBERATELY NOT EXTENDED TO IT.** The deflation result is coupling-dependent - response-only beats the projected method by 0.05 additive (0.995 vs 0.945) but 0.42 multiplicative (0.930 vs 0.506) - so additive-only would have characterised the effect only where it is smallest and omitted the case the paper's explanation rests on. The pre-registered 0.80 bar stays on the additive cells, because the 0.5213 joint baseline it is defined against comes from `e03_required_n.json`, which planted additive units; the archived baseline is written as `null` on multiplicative rows so it cannot be misread. The plant moved to `experiments/planted_units.py` so the gated structure can be tested, since the whole justification is a claim about that function | `experiments/planted_units.py`, `experiments/s1_2_deflation.py`, `tests/test_planted_units.py` |
| 2026-10-04 | **B-7'S THREE LEGS ARE PAIRED, AND CHECKING THAT OVERTURNED THE RESULT I HAD ALREADY PUSHED.** The unit draw depends only on the seed and `d_model`, both fixed across layers, so L2/L6/L10 ran the *same 50 units* - the failure sets are comparable within unit. They barely overlap: **0 units fail at all three layers**, 21 of 26 L2 failures pass at both L6 and L10 (median align 0.836 -> 0.995), and 3 units pass at L2 and fail at L10. So the depth effect is not "shallow layers are noisier" and the aggregate 0.48/0.86/0.90 is an average over near-disjoint populations. **Failure is a property of the (unit, layer) pair.** An earlier write-up in this same session read the rates as "two regimes rather than a gradient" from the overlapping L6/L10 intervals; that was wrong, and it was wrong because the write-up did not check whether the legs were paired. Corrected in README, notebook and PAPER_STRATEGY, with the retraction kept visible | notebook §7.8 Tier 3, `experiments/analyse_b7_depth.py` |
| 2026-10-04 | **PASS/FAIL IS NOT REPRODUCIBLE AT FIXED LAYER: 20% OF VERDICTS FLIP ON RESTART COUNT ALONE.** B-7 L6 at 5 restarts and B-1b L6 at 2 restarts share all 50 units and agree on 40 (3 units fail in both, 4 fail only at 5 restarts, 6 fail only at 2). This bounds how finely any single pass rate can be read, and it is why B-8's family contrast is argued as a 31-point gap rather than a marginal one. It also means a unit must not be described as permanently good or permanently broken | `experiments/analyse_b7_depth.py` |
| 2026-10-04 | **THE KAGGLE BUNDLE HAD GONE STALE A THIRD TIME, AND NOTHING ABOUT IT LOOKS WRONG.** `kaggle/bundle/` ships copies of `caliper/`; it was missing `per_neuron_seed` and `fit_deflate`, so any Kaggle run would have executed a different estimator from the one the notebook reports as scored. Two of the three staleness events predate `kaggle/build_bundle.py`, whose own docstring records them. Rebuilt, and pinned with `tests/test_kaggle_bundle.py`, which hashes each shipped file against source and was verified to FAIL on injected one-line drift before being trusted | `tests/test_kaggle_bundle.py` |
| 2026-09-09 | **PAPER STRATEGY FILED, built around a claim-evidence map.** Nine claims are supported today; four need runs that are running or queued. **Four tempting claims are explicitly ruled out**, including *restart agreement does not work* - it reaches AUC 0.778 and catches most failures, so the supportable claim is **dominated, not useless**. An earlier framing in this project said useless and that was wrong | `docs/PAPER_STRATEGY.md` |
| 2026-10-04 | **DEFLATION'S STIMULUS PROJECTION FAILED ITS OWN CHECK, AND THE DEFAULT WAS WRONG.** `fit_deflate`'s stated rationale was that response-only deflation leaves f(X v1) − X v1 in the residual, so the next rank-1 search rediscovers v1 and the method degenerates to "a robust rank-1 fit run twice". Measured, response-only scores HIGHER under both couplings: additive 0.995 vs full 0.945, multiplicative 0.930 vs full 0.506. The rediscovery argument is a hypothesis that failed, the projection COSTS accuracy, and the docstring now says so. It is kept only because the flag is what S1-2 varies. Under multiplicative coupling the gap is large because y depends on the gated direction through the gate itself - removing the stimulus removes a factor the response needs | `caliper/estimator.py`, `tests/test_estimator.py` |
| 2026-10-04 | **`kill -0` IS BLIND TO WINDOWS PIDS FROM MSYS, SO EVERY EARLIER QUEUE GUARD WAS INERT.** Measured: `kill -0` reports "not alive" for a running Windows python AND for an absurd PID, because MSYS keeps its own process table. Every prior version of the run-queue lock used `kill -0`, which means it never once detected a live queue - it only appeared to work when a stale lock happened to point at an MSYS PID. `tasklist` reads the real Windows process table and separates live from dead correctly. Liveness probes now use `tasklist` | `experiments/run_queue.sh` |
| 2026-10-04 | **A QUEUE LOCK DOES NOT GUARD A RUN, AND THE GAP REPRODUCED THE BUG IT WAS WRITTEN TO PREVENT.** When Task Scheduler aborted the chain (0x8007042B) the bash chain died but its python child survived as an orphan, still appending to the output file. A queue started at that moment saw 31 rows against a target of 100, called the run incomplete, and launched a SECOND python against the same file - which is the original double-write incident, recreated by its own guard. Each run now also claims its own output file and records the launching PID, checked before the row count is trusted, because the row count is precisely the thing that cannot distinguish "not started" from "already running" | `experiments/run_queue.sh` |
| 2026-10-04 | **A ZERO-ROW RUN IS NOT A STUCK RUN, AND I READ ONE AS A HANG FOR AN HOUR.** B-7 L10 sat at 0/50 across several restarts while I kept killing and relaunching it. Two separate mistakes: my manual relaunches ran without `export PYTHONPATH=.`, so they died instantly on `ModuleNotFoundError: No module named 'caliper'` while the queue's own runs (which set it) were fine - three of them, concurrently, which is why the log interleaved; and a 50-neuron batch writes nothing until it finishes, which took 164 min. Before concluding a run is stuck, check the log for a traceback and check CPU, not just the row count. `exit 127` in the queue log means MY `Stop-Process`, not a crash | `experiments/run_queue.sh` |
| 2026-09-09 | **PAPER STRATEGY FILED, built around a claim-evidence map.** Nine claims are supported today; four need runs that are running or queued. **Four tempting claims are explicitly ruled out**, including *restart agreement does not work* - it reaches AUC 0.778 and catches most failures, so the supportable claim is **dominated, not useless**. An earlier framing in this project said useless and that was wrong | `docs/PAPER_STRATEGY.md` |
| 2026-09-09 | **B-0's 5-of-16 is a demonstration of EXISTENCE, not a rate.** 31% on n=16 is too thin to quote as an estimate. Either extend to n=50 (~1h) or state existence only - which is enough for the claim we want, that failure classification *can* be hardware-dependent | `docs/PAPER_STRATEGY.md` section 4 |
| 2026-09-09 | **Three highest-severity rejection risks named with answers**: only small models (abstract, not appendix); only MLP units reading their own layer (unfixable, state it up front); and *where is the method* - which is precisely why TMLR and the E&D track are the targets and a methods main track is not | `docs/PAPER_STRATEGY.md` section 5 |
| 2026-09-09 | **Drafting order fixed: claim-evidence map, then figures and tables, then methods, then introduction, abstract last.** Write in the order that fails fastest - if the operating-characteristic table and the ladder table do not carry the argument alone, prose will not save them | `docs/PAPER_STRATEGY.md` section 7 |
| 2026-09-09 | **CITATION LEDGER CREATED, with a verification level on every row.** 60+ sources, marked FULL (main text read) / ABS (abstract or search summary only) / BIB (formalised). **Never cite a specific claim from an ABS row** - that is the rule the August audit exists to enforce, after it killed two novelty claims that abstracts had not revealed | `docs/CITATIONS.md` |
| 2026-09-09 | **TIER 3 REFRAMED: B-7 and B-8 become variance components, not reviewer-proofing.** `2604.11581` shows naive CIs run **40-60% too narrow** by ignoring design-choice variance; `2607.19386` finds methodological variance exceeds architectural variance. Our Wilson intervals have that flaw. **We have five of six components already run** - device, restarts, scale, steps, and the two queued - so the paper can report a design-sensitivity-corrected interval | notebook section 7.8 |
| 2026-09-09 | **VENUE CORRECTED: NeurIPS D&B is now the Evaluations & Datasets track**, whose scope explicitly includes *work that analyzes failure modes of existing benchmarks or evaluation practices* and welcomes negative results. That is a description of this paper. **NeurIPS 2026 closed (May 2026 deadline); NeurIPS 2027 E&D approx May 2027**, which lands on the 'science done by May 2027' constraint. TMLR stays the near-term target | scout 9 Sep |
| 2026-09-09 | **ADOPT NIST AI 800-3 (Feb 2026), the official standard for benchmark statistical rigour.** Cheap - we already do Wilson intervals and pre-registration. Two things to add: the **benchmark accuracy vs generalized accuracy** distinction (300 units estimate a 3,072-unit population) and intervals over point scores throughout | scout 9 Sep |
| 2026-09-09 | **B-0 must be positioned against the FP non-associativity literature, not claimed as novel against it.** `2408.05148` and `2511.00025` already establish that numbers vary across hardware. **Our claim is narrower and sharper: a qualitative classification flips** - which units count as failures - with median agreement 0.00068 against a tail of 0.66 | `docs/CITATIONS.md` section 6 |
| 2026-09-09 | **GAP FOUND: we report AUC and operating points but no reliability diagram.** The failure-prediction literature (`2303.02970`, `2403.02886`) expects one - does a signal value of X correspond to failure probability Y? About 20 lines of offline analysis, and reviewers in this area look for it | `docs/CITATIONS.md` section 7 |
| 2026-09-09 | **Kaggle run sheets now ship ONE self-contained cell.** The steps check was handed over as three cells assuming setup had already run; under Save-and-Run-All it had not, cwd was still `/kaggle/working`, and both runs died instantly. Setup in a separate cell is setup that gets skipped | `NEXT_SESSION_STEPS_CHECK.md` |
| 2026-09-09 | **The packaging guard paid off on its first outing.** `assert rows` refused to archive an empty directory. The same situation two sessions earlier produced a 49-byte zip that looked like a successful download until opened - a silent failure turned into a loud one | notebook section 8 |
| 2026-09-09 | **STEPS CHECK QUEUED, executing a branch pre-committed before the ladder ran.** 1.4b at 3200 steps (~4.3h), plus **410m at 3200 as a control** (~2.1h). The control is the addition: re-running only 1.4b cannot separate *more steps helps everything* from *more steps helps the wide model specifically*, and only the second supports the under-optimisation reading | `NEXT_SESSION_STEPS_CHECK.md` |
| 2026-09-09 | **The steps comparison is PAIRED - McNemar, not two proportions.** The unit draw depends only on `--neurons` and the seed, not on `--steps`, so the 3200-step run scores the same 50 units. Note the contrast with `--neurons`, which changes the whole draw | `NEXT_SESSION_STEPS_CHECK.md` |
| 2026-09-09 | **Readings fixed before the numbers exist, all four branches.** Including the one where the artefact reading wins: the scale claim is withdrawn, and the paper instead gains a methodological point - **a fixed step budget silently under-fits wider models and would have been read as a scale effect**. Both step budgets are reported side by side either way; the 1600-step table stays with its numbers intact | `NEXT_SESSION_STEPS_CHECK.md` |
| 2026-09-09 | **THE HEADLINE LANDS: restart agreement is significantly dominated by held-out R2.** B-1b, n=300, 52 failures (inside the required 36-142), AUC 0.778 vs 0.942, **DeLong z = -5.22, p < 0.0001**. At 50% catch, held-out R2 discards **0** good units and restart agreement discards 40. The field's default check is worse than a number the fit already computes for free | notebook B-1b |
| 2026-09-09 | **B-0 FAILS: the device is a confound, and it is a reportable finding.** 5 of 16 units flip pass/fail between CPU and GPU on identical seeds; max abs d alignment **0.6615** against a median of 0.00068. The prediction is confirmed exactly - failing units sit near basin boundaries, which is where reduction order flips the answer. **Claim: estimator failure classification is hardware-dependent** | `data/b11/device_equivalence.json` |
| 2026-09-09 | **B-11 rises with scale (4%, 4%, 4%, 38%) and is NOT REPORTABLE until the steps check runs.** The filing pre-committed this branch precisely because it is the exciting one. Three rungs at exactly 48/50 across a 2x width range, then only the widest breaks, at 4x the width of 70m on the same 1600 steps - that is what an under-optimised fit looks like. **Required: 1.4b re-run at 3200 steps** | `preregistration-b11-pythia-ladder.md` |
| 2026-09-09 | **GPU speedup measured at 1.29x**, confirming the 1.1x estimate from the 70m rung. B-7 and B-8 stay local permanently; there is no case for moving comparative runs to Kaggle | B-0 |
| 2026-09-09 | **TWO FILINGS CARRIED A FALSE NESTING CLAIM; both struck through in place.** `choice(3072, size=N)` is not a prefix relation in N - n=50 shares **1 unit of 50** with the first 50 of n=100, and n=100 shares **6 of 100** with n=300. B-1 and B-1b are **near-disjoint independent draws**, not nested arms | b1-addendum-1, b11-addendum-1 |
| 2026-09-09 | **The endpoints survive; one write-up rule changes.** Each arm computes its own incumbents on its own units, so the within-arm DeLong comparisons are untouched. What changes: **C13's AUCs (0.906, 0.802) are context, never a comparator** | notebook section 8 |
| 2026-09-09 | **The 70m ladder rung is CONTAMINATED - delete, do not resume.** Checkpoint keys on unit id, so re-running at `--neurons 50` against a file written at `--neurons 100` appended a second unit set; the progress line read `79/50`. A counter past its target is the tell | NEXT_SESSION_LADDER.md |
| 2026-09-09 | **THE BOTTLENECK IS `fit_cascade`, AND IT IS STRUCTURAL: 24x the work of `fit_batch`, unbatched, on CPU.** Per batch of 32 units, `fit_batch` runs 6,400 steps batched across all 32; `fit_cascade` runs 720 grid probes plus 5x800 polish **per unit** = 151,040 steps one unit at a time. `e01_gate.py:72` omits `device=`, so it takes `fit()`'s cpu default. **The GPU accelerates about 4% of the job** - which is the whole explanation for the measured 1.1x | `experiments/e01_gate.py:72` |
| 2026-09-09 | **B-11 AMENDED TO 50 UNITS PER RUNG.** At 100 the four rungs need ~8.4h against a 12h cap with the largest last; at 50 they need ~4.6h and all four land. Filed **before any rung completed** - no alignment, pass rate or AUC had been seen for any rung, so the amendment rests on timing alone. Committed with it: a null trend at n=50 is reported as **underpowered and inconclusive**, never as "scale-invariant" | `preregistration-b11-addendum-1-n50.md` |
| 2026-09-09 | **`fit_cascade`'s device default is deliberately NOT fixed mid-ladder.** Passing `device=` would speed future runs and is worth doing, but changing numerics between rungs would put a **code version inside the scaling curve** - the exact class of confound this ladder was built to avoid. Fix it after the ladder, not during | `preregistration-b11-addendum-1-n50.md` |
| 2026-09-09 | **THE GPU IS WORTH ~1.1x FOR THIS WORKLOAD, NOT 10x - measured, and it answers B-0's speedup question for free.** pythia-70m (d_model 512) ran at **38.6 s/unit on a T4**; local GPT-2 (d_model 768) runs at 63.7 s/unit on 10 CPU threads. Width-adjusted that is 57.9 vs 63.7 - a 1.10x speedup | ladder session log |
| 2026-09-09 | **Cause found: `fit_cascade` never gets the device.** `e01_gate.py:72` calls it without `device=device`, so it inherits `fit()`'s `device="cpu"` default. Only `fit_batch` (lines 66, 68) uses the GPU. The cascade is a 720-point grid search plus polish **per neuron**, and it runs on Kaggle's 4-core CPU - weaker than this machine's 10 threads, which is why the GPU barely wins | `experiments/e01_gate.py:72` |
| 2026-09-09 | **CONSEQUENCE: there is no longer a case for moving B-7 or B-8 to Kaggle even if B-0 passes.** The confound risk was priced against a large speedup that does not exist. They stay local, where the CPU is faster anyway. B-0 drops from "the run that unlocks Kaggle" to a cheap methodological curiosity worth running only if a session is idle | notebook §7.8 |
| 2026-09-09 | **Ladder reordered: 1.4b runs BEFORE 410m.** Projected remaining is 8.4 h against a 12 h cap, so the last rung is at risk. 1.4b is the rung that cannot run locally (5.6 GB fp32 against 2.6 GB free) and the one that answers "only small models"; **410m fits locally at ~1.6 GB**, so losing 410m to a timeout is recoverable and losing 1.4b is not | `NEXT_SESSION_LADDER.md` |
| 2026-09-09 | **B-0 WAS COSTED WRONG AND CERTIFIED THE WRONG CONFIG.** It defaulted to 5 restarts / 24 units, so on Kaggle's 4-core CPU (~2.5x slower than local) the **CPU arm alone runs ~90 min with the T4 idle** - GPU quota spent on CPU work. Worse, nothing else runs at 5 restarts: Addendum 1 moved the primary arms and the ladder to **2**. Defaults are now 2 restarts / 16 units, which certifies the config actually in use and costs ~10x less | `kaggle_device_equivalence.py` |
| 2026-09-09 | **The ladder does not depend on B-0 and should run first when a session is short.** The ladder is internally device-consistent by construction, so it carries its own result; B-0 only decides whether *future comparative* runs may move to Kaggle. Cell order in the sheet is convenience, not dependency, and the sheet now says so | `NEXT_SESSION_LADDER.md` |
| 2026-09-09 | **B-11 SESSION 2 ALSO PRODUCED NOTHING** - Cell 1 succeeded, then every run failed with `ModuleNotFoundError: No module named 'caliper'`. `!python` is a subprocess: it inherits cwd but not `sys.path`, and Python adds the *script's* directory, never the bundle root. Fixed with `os.environ["PYTHONPATH"] = root` plus an in-cell subprocess import test. The local queue never had this bug because `run_queue.sh` starts with `export PYTHONPATH=.`; the sheet was written without carrying it across | notebook §8 |
| 2026-09-09 | **Setup cells must prove what they assert.** Both failed ladder sessions had a Cell 1 that reported success on something it had not tested - first the mount path, then subprocess importability. Cell 1 now asserts the files exist AND runs a subprocess import under the exact condition the runs use. The shipped cell is rehearsed against a simulated mount at the same nesting depth before it ships | `NEXT_SESSION_LADDER.md` |
| 2026-09-09 | **B-11 SESSION 1 PRODUCED NOTHING - a run-sheet bug, not a science result.** Cell 1 hard-coded `glob("/kaggle/input/*/")[0]`; the bundle mounted a level deeper (`/kaggle/input/datasets/santoshcheethirala/`), so every command failed on a missing path. The ladder loop still printed all four rung headers, so **the log looks like four runs happened**. Nothing is recorded as a ladder result and the run is re-queued unchanged | notebook §8 |
| 2026-09-09 | **Run sheets are now validated before a session is spent on them** - `kaggle/check_sheets.py` parses every python cell in `kaggle/*.md` and reports magic cells as skipped rather than passed. 29 cells parse, 4 skipped, 0 broken. This is the second run-sheet failure in the project; the first cost four rounds and ~3h, and both were visible before the session started | `kaggle/check_sheets.py` |
| 2026-09-09 | **B-11 FILED: the Pythia scale ladder goes to Kaggle, and it is the right run for a GPU session.** Pythia publishes 70m/160m/410m/1.4b **trained on identical data in identical order** - the only suite where scale varies with corpus and curriculum fixed. So this is not "we also tried a bigger model", it is silent-failure rate as a controlled function of scale | `preregistration-b11-pythia-ladder.md` |
| 2026-09-09 | **Why the ladder is not device-confounded: every rung runs on the same GPU**, so the scaling curve is internally consistent and its claim needs no reference to the local numbers. Absolute rates are NOT pooled with the local n=300 arms and the device is named beside every number. The layer sweep and GPT-Neo runs stay local because they ARE comparative - B-7 against layer 6, B-8 against GPT-2 | `NEXT_SESSION_LADDER.md` |
| 2026-09-09 | **The 160m rung is a free device cross-check.** C53 ran the same protocol locally on CPU (93/100 at 2 restarts), so the GPU rung is directly comparable at n=100. Weaker than B-0's paired per-unit test and does not replace it, but a large discrepancy would be informative and is reported either way | `preregistration-b11-pythia-ladder.md` |
| 2026-09-09 | **Pre-committed trap: if the failure rate RISES with scale, do not report it yet.** Wider models may simply be harder to fit at a fixed 1600 steps. The check is to re-run the top rung at 3200 steps and report both - a rate that falls with more steps is an optimiser artefact, not a scale effect. Filed before the run so the sceptical check cannot be skipped if the exciting outcome appears | `preregistration-b11-pythia-ladder.md` |
| 2026-09-09 | **B-9a dropped from the local queue.** 410m becomes a ladder rung instead; running it locally too would produce a number poolable with neither set | `run_queue.sh` |
| 2026-09-09 | **QUEUE DIED OVERNIGHT AT 02:54 - the machine slept.** No traceback, parent shell gone: killed, not crashed. ~5 of 9 hours wasted, 0 rows lost because Checkpoint resumed B-1 at 92/100. The global CLAUDE.md already warned that the laptop is often closed and local scheduled work cannot be relied on; this is that cost, measured. Mitigation is `powercfg /change standby-timeout-ac 0`, which Santosh must run - a system setting is not mine to change | notebook §8 |
| 2026-09-09 | **B-1 ADDENDUM 1 FILED: the design was underpowered, and the primary arm moves to `restarts=2, n=300`.** Disclosed in the filing: written after seeing 64 of 100 rows, with the interim AUCs reproduced. Nothing changes about the bar, units, signals, endpoint or decision rule - only sample size and a second arm. **The decisive reason is fairness, not power**: the incumbent AUCs (R2 0.906, disagreement 0.802) were measured at 2 restarts, so scoring a 5-restart stability against them compares signals under different optimisers | `preregistration-b1-addendum-1-power.md` |
| 2026-09-09 | **CORRECTION: the failure-rate drop from 22% to ~9% is NOT a finding.** The original filing anticipated it in writing - "the pass rate is therefore expected to move, and a changed pass rate is not a finding here" - because 5 restarts is a better optimiser than 2. An earlier session note called it a real finding; that was wrong. The *power consequence* is real and is what Addendum 1 acts on | `preregistration-b1-addendum-1-power.md` |
| 2026-09-09 | **DeLong's test, not overlapping intervals**, for every AUC comparison. Two AUCs on the same units are correlated because both come from the same fits, so independent intervals would overstate the evidence for a difference. Benjamini-Hochberg across the signal set. Power reference: 36-142 cases per group for Delta AUC 0.10 (Obuchowski-McClish / Hanley-McNeil), against ~10 expected failures at n=100 | `preregistration-b1-addendum-1-power.md` |
| 2026-09-09 | **`Fit.stability_pairs` added - secondary 4 was not computable from what B-1 saved.** The filing asks whether a cheap 2-restart stability estimate agrees with the 5-restart one, but the row stored only the median pairwise alignment, discarding the pairs. Now every restart subsample is recoverable offline. B-1's rows lack the column; B-2 onward carry it | `caliper/estimator.py`, `e01_gate.py` |
| 2026-09-09 | **B-1 (5 restarts, n=100) is re-labelled the SENSITIVITY arm, not the primary.** It is not dropped for being underpowered - it answers two things the primary cannot: whether a better optimiser reduces silent failure, and whether expensive stability beats cheap stability. Both arms are reported with their n and restart count stated | `run_queue.sh` |
| 2026-09-08 | **KAGGLE DROPPED, everything runs locally.** Removes the device confound outright rather than measuring it, so **B-0 is unnecessary, not skipped** - its script and sheet stay in the tree for the day a GPU is needed. Cost: the programme is serial on 10 threads, ~30h wall clock across the remaining runs | `run_queue.sh` |
| 2026-09-08 | **B-9 BECOMES A PYTHIA SCALE LADDER rather than one big model.** Pythia publishes 70m/160m/410m/1b/1.4b trained on identical data in identical order, so running one protocol across rungs gives a **controlled scaling curve for the failure rate** - a finding, not just a rebuttal to "only small models". 410m is queued; **1.4b is not**, because it needs ~5.6GB in fp32 against 16GB total with 2.6GB free under load, and it gets its own run when the machine is idle | notebook §7.8 |
| 2026-09-08 | **Serial queue, one run at a time, deliberately.** 10 threads are saturated by a single gate run, so concurrency halves each without finishing anything sooner, and doubles peak RAM against a 16GB ceiling that already binds the scale runs. Every run resumes via Checkpoint, so killing the queue and rerunning continues from the last completed unit | `run_queue.sh` |
| 2026-09-08 | **KAGGLE IS AVAILABLE AND THE CODE ALREADY SUPPORTS IT** - `pick_device` resolves cuda, `fit_batch` takes a device, `Checkpoint` was written for the 12h session cap, and the corpus is cached prose so internet can stay off. What was NOT ready: the bundle was two files stale and **missing `batched.py` entirely**, which the gate cannot start without. Hand-copying went stale twice, so the build is now `kaggle/build_bundle.py`, which prints what changed | `kaggle/build_bundle.py` |
| 2026-09-08 | **B-0 FILED: the device is treated as a confound until measured.** The study is about WHICH UNITS FAIL, and failing units sit near basin boundaries by construction - exactly where floating-point reduction order can flip the answer. A table reporting GPT-2 23% / Pythia 7% with one device each would carry a hardware term inside its headline. C31 is the precedent: measure the control, do not assume it | `NEXT_SESSION_B0_DEVICE.md` |
| 2026-09-08 | **A B-0 FAIL would be a finding, not a wasted run** - estimator failure classification would be hardware-dependent, meaning a reproduction on different hardware may not reproduce which units failed. Both branches are pre-registered and both are reportable | `NEXT_SESSION_B0_DEVICE.md` |
| 2026-09-08 | **Kaggle CPU sessions are the confound-free fallback.** They draw no GPU quota and are device-consistent with B-1/B-2 running locally. Honest arithmetic: Kaggle gives 4 cores against this machine's 10, so ~11h where local takes 4.4h - only just inside a 12h session, and Checkpoint resumes rather than restarts. Parallelism across sessions is the win, not per-run speed | notebook §7.8 Tier 2b |
| 2026-09-08 | **DELONG FILED AS THE PRIMARY TEST, before B-1 produced a row.** The parent B-1 filing said "significantly below held-out R2" without naming a test - an open door, since several tests give several answers and the choice could be made after seeing numbers. Both signals are scored on the SAME units so the ROC curves are correlated; independent intervals would be the wrong comparison. Implemented in-repo, 8 unit tests, no new dependency | `preregistration-b1-addendum-1.md` §1 |
| 2026-09-08 | **B-1 MAY BE UNDERPOWERED, and that is filed in advance rather than discovered.** Published power figures: dAUC=0.10 needs 36-142 cases, dAUC=0.02 needs 909-3,709. A synthetic check during implementation put the SE of a dAUC at 0.07-0.11 for uncorrelated signals at n=100/25 failures. **Rule fixed now: if \|dAUC\| < 0.10 the result is reported as UNDERPOWERED, not as a null - "no significant difference" is forbidden - and triggers one extension to n=300** | `preregistration-b1-addendum-1.md` §3 |
| 2026-09-08 | **Multiplicity fixed: exactly ONE primary comparison** (restart agreement vs held-out R2 on GPT-2), uncorrected because it is a single pre-specified test. Every other signal and everything on Pythia is secondary and BH-corrected across the family, reported with raw and adjusted p | `preregistration-b1-addendum-1.md` §2 |
| 2026-09-08 | **A FREE ANALYSIS the parent filing missed: the restart-count curve.** B-1 stores 5 restarts; practitioners run 2-3. Stability can be recomputed at 2/3/4/5 from data already collected, at zero extra compute, giving the operating characteristic of the check AS ACTUALLY USED. If it only works at 5 restarts that is a different recommendation from working at 2. Subsampled by first-k in recorded order, not a favourable subset | `preregistration-b1-addendum-1.md` §4 |
| 2026-09-08 | **SCOUT: 2602.13450 "Inference From Random Restarts" validates the premise and does not scoop it.** It says the restart heuristic "lacks a formal inferential foundation, despite its widespread use" - but it is Bayesian theory on an econometric solver, not empirical calibration against ground truth, and never measures whether restart agreement predicts correctness. **It also supplies a mechanism for our pilot:** uniqueness concentrates polynomially while basin size concentrates exponentially, so restart agreement measures BASIN SIZE, not correctness - and a large basin can be the wrong basin. That is "stable and wrong" with independent theory behind it | scout, 8 Sep |
| 2026-09-08 | **RUN PROGRAMME OF RECORD written as §7.8, superseding the A-series.** Five tiers, ~30 CPU-hours on the critical path. Three runs (B-8 family, B-9 scale, B-10 required-N) exist only because a scout named a reviewer objection we had not planned for; B-9 answers the strongest one, "only 124M and 160M models" | notebook §7.8 |
| 2026-09-08 | **README corrected: "restart agreement gives no warning at all" was too strong.** The n=8 pilot shows it separates the severe failures cleanly (passes agree to 0.0003) and misses subtle ones (2 of 4 failures agree to within 0.05 while wrong). Replaced with the pilot numbers and a pointer to the running measurement, in whichever direction it falls | README §What Phase 0 found |
| 2026-09-08 | **LATENT INCONSISTENCY FOUND in our own reported numbers.** The notebook reports pass rates under E0.1's filed gate criterion (`align > 0.95 AND k2_gain < 0.01`: GPT-2 77/100, Pythia 93/100) and AUCs under an alignment-only label (`s1_flag_roc.py`, 0.906 / 0.995). Both are correct for their own question and neither says which it is. Caught by writing a fresh analysis that scored alignment only and returned 94/100 where the run said 93 - the same conflation C43 already cost us once | `b1_signal_calibration.py` |
| 2026-09-08 | **THE BENCH SCORES "recovery", NOT the gate criterion, and the reason is circularity.** k2_gain is one of the candidate signals AND a term in the gate label, so under that label it predicts itself. The bench asks "did the estimator find the right direction" - alignment alone. The script takes `--gate` to reproduce E0.1's number and excludes k2_gain as a signal when it does | `b1_signal_calibration.py` |
| 2026-09-08 | **VENUE: TMLR is the primary target, not a top-tier main track.** The grade depends on a real acceptance, so the objective is certainty rather than prestige. TMLR accepts on correctness and clarity and explicitly NOT on novelty or significance - rolling, no deadline, indexed. A careful methods paper with honest negatives and pre-registration is what it exists for. NeurIPS D&B second, workshops as backup. Main track is low odds at 124M/160M | `PIVOTS.md` P10 |
| 2026-09-08 | **CUT FROM THE FLAGSHIP: Study 2 and Study 3.** Both are real work. Study 3 is a solid result in a fast-crowding area; Study 2 is a negative that a published behavioural paper already reached. Including weak studies weakens a paper. They become separate outputs or nothing | `PIVOTS.md` P5, P7, P9 |
| 2026-09-08 | **TWO LIMITATIONS GO IN THE ABSTRACT, not the appendix.** Only 124M and 160M models; ground truth holds only for a unit reading its own layer and transfer is an untestable assumption. Reviewers forgive a stated limit far more readily than a discovered one | `PROPOSAL_III_STATUS.md` §3 |
| 2026-09-08 | **FRAMING COMMITTED: the measurement problem is identical, NOT the systems.** The mentor responded to "neuroscience and models are similar / model psychology". That is the hook and it works, but P3 already killed one analogy-shaped headline in this project. We do not claim models resemble brains. We claim a unit responds to something, you cannot ask it what, so you work backwards - and neuroscience has forty years of method for exactly that. Same appeal, no attack surface | `PIVOTS.md` P9 |
| 2026-09-08 | **P1b REFRAMED by the literature check.** 2602.06801 reaches the same non-identifiability conclusion from behavioural evidence and states explicitly it does NOT plant known directions and attempt recovery. P1b is the ground-truth complement to a published behavioural result, not a failed study. Value recovered only because the check was done afterwards - do it first next time | `PIVOTS.md` P7 |
| 2026-09-08 | **OPEN, HIGHEST VALUE, FREE: nobody has read the rubric's definition of "published".** Preprint, workshop, or peer-reviewed only? Accepted or submitted? The entire schedule depends on it and it is currently a guess. Added as question 4 to the mentor one-pager | `PROPOSAL_III_ONE_PAGER.md` |
| 2026-09-08 | **PIVOT LOG CREATED.** Ten direction changes across Jul-Sep 2026, each with date, cause, and cost. Six of ten were caused by checking before committing (literature pass, an unrun control, a manipulation check) and were cheap; the expensive ones came from outside or from an un-stress-tested framing | `PIVOTS.md` |
| 2026-09-08 | **REPOSITIONED: the bench is Proposal III's instrument paper, not a separate project.** The mentor selected Proposal III on 18 Aug 2026 - population-level measurement of LM units, tuning curves plus pairwise maximum entropy, imported from systems neuroscience. APERTURE is Proposal I and was not selected. The bench was being scoped as a standalone artifact; it is in fact III's Paper 1 and Phase D, already substantially run | `PROPOSAL_III_STATUS.md` |
| 2026-09-08 | **III's own spec named the estimators we already scored.** `capstone-III-neuro-method-map` calls Maximally Informative Dimensions the crown jewel and spike-triggered covariance "the baseline that fails". E0.5 ran both against free ground truth months earlier: STC 0/30, MID-style bottleneck 26/30, with Bussgang's constant as the stated mechanism. The prediction in the proposal is confirmed, not assumed | notebook C3/C7, E0.5 |
| 2026-09-08 | **The weight column supersedes InterpBench for III's Phase D.** III's spec proposed calibrating unit characterisation against compiled circuits. The weight column is exact AND natural - the units are hard for reasons the model chose in training, not reasons we introduced. Its capability table marks "ground-truth validation of unit characterisation" as Open; that row is now closed | `PROPOSAL_III_STATUS.md` §3 |
| 2026-09-08 | **Single-unit calibration is a PREREQUISITE for III's population layer, not a detour.** Pythia's median alignment is 0.9994 while the same population holds a unit recovered at 0.0761. Tuning curves and pairwise maxent fitted over a population would inherit that error invisibly, and population structure would not be separable from estimation error. Systems neuroscience establishes single-unit estimator error before inferring population structure, for this exact reason | `PROPOSAL_III_STATUS.md` §5 |
| 2026-09-08 | **APERTURE: date-stamp, do not develop.** Preprint 1 (the neutral-framing control) needs no new compute and a full-text check confirms it is unreported in the current introspection debate, so scoop risk is the highest of any asset. It is preliminary work for an unselected proposal. Put to the mentor as a question rather than decided unilaterally | `PROPOSAL_III_STATUS.md` §9 |
| 2026-09-08 | **B-1 LAUNCHED** at `--restarts 5`, GPT-2, n=100, `results/b1_stability_gpt2.jsonl`. First launch died in 45s on `ModuleNotFoundError: caliper` - running a script inside `experiments/` puts that directory on `sys.path`, not the repo root. Relaunched with `PYTHONPATH=.`. Cheap because it failed loudly and immediately | run log |
| 2026-09-08 | **THE DELIVERABLE IS A CALIBRATION BENCH FOR RELIABILITY CHECKS, not a benchmark of methods.** Method benchmarking is crowded (ObserverBench, MIB, AxBench). The scout found the same hole from three sides: reliability signals validated against behaviour or reconstruction but **never against a known-correct direction**; ground truth for direction recovery that is always planted or compiled (FRR, Linear Representation Bench, InterpBench, Tracr); harnesses that rank methods but carry no ground-truth-free failure detector. Free exact ground truth on a real model closes all three | `docs/BENCH_SPEC.md` |
| 2026-09-08 | **B-1 FILED (`preregistration-b1-stability-calibration.md`) and UNBLOCKED BY A TWO-LINE CHANGE.** `Fit.stability` — restart agreement, the field's default check — has been computed on every fit since the estimator was written and never written to a row. Patches landed and smoke-tested in `caliper/batched.py`, `caliper/estimator.py`, `experiments/e01_gate.py`. **The headline run needs ~4.4 h of local CPU and no new idea** | prereg b1 |
| 2026-09-08 | **B-1's pilot is disclosed in the filing, and it moderates the claim I first made.** I had said restart agreement was near-useless on our failures. `s1_seed_lottery.json` (n=8, run for C43) says otherwise: passing units are stable to within 0.0003, and of four failures **two are stable to within 0.05 while being wrong**, two are plainly unstable. The defensible expectation is **catches the severe failures, misses the subtle ones** — moderate AUC, poor operating point at high specificity. Recorded before the run so it cannot be retrofitted | prereg b1 |
| 2026-09-08 | **All three B-1 outcomes are pre-registered as results.** If stability matches held-out R2 the finding becomes cost-benefit (5 fits vs free); if it *beats* R2 the deliverable's recommendation changes and **Finding 2 is amended**. A bench whose headline only works in one direction is an advocacy exercise | prereg b1 |
| 2026-09-08 | **B-1 will move the pass rate off 77/100 and that is NOT a finding.** 5 restarts is a better optimiser than 2. C13's gate stands at its own filed configuration; B-1 is not a re-run of the gate. Committed in advance so the temptation to quote a nicer pass rate never arises | prereg b1 |
| 2026-09-08 | **P1b IS NOT SCOOPED — it is the ground-truth complement to a published behavioural result.** [Non-Identifiability of Steering Vectors (2602.06801)](https://arxiv.org/html/2602.06801v4) reaches P1b's conclusion on Llama-3.1-8B and Qwen2.5-3B using **behavioural evidence only**, and states explicitly that it does not plant known directions and attempt recovery. That is exactly what P1b did. Reframes P1b from our failed study into corroboration by a different route, and revives the `2026-09-08f` re-run — behind the CPU work, since it needs a GPU | notebook 7.8 |
| 2026-09-08 | **The stated limit goes in the abstract, not the appendix.** Free ground truth holds for an MLP unit reading its own layer. It does not extend to residual-stream features, SAE latents, or persona directions, and the bench cannot test whether its calibrations transfer there. A reviewer will raise this; better that we raise it first | `docs/BENCH_SPEC.md` |
| 2026-09-08 | **`Fit.restarts` was documented as holding `test_r2` per restart; it holds subspaces.** Stale comment since the estimator was written, corrected while adding `r2_restarts`. Nothing downstream was wrong — `stability` always did a subspace comparison — but the docstring would have misled the next reader into scoring the wrong thing | `caliper/estimator.py` |
| 2026-09-08 | **P1b FILED AND BUILT (`preregistration-s2-p1b.md`, `2026-09-08c`).** Plants real concept vectors instead of random ones, null A is a *different concept's* vector (conservative - concept vectors share structure), null B a random direction reported alongside. **The manipulation check is the key addition and it gates the primary**: the run now logs every generation and reports the semantic steering rate, so a plant that never reached the text is distinguishable from an extraction that failed. That distinction is exactly what C54/C55 could not make | prereg s2-p1b |
| 2026-09-08 | **P1b is a NECESSARY-CONDITION test, and the framing is committed in advance.** A concept direction is the easy case - the model already represents it - so a PASS does not establish that persona extraction recovers arbitrary traits. A FAIL with the manipulation check passed is the substantive negative. The filing also commits that a third void reconsiders the study rather than re-running on another guess | prereg s2-p1b |
| 2026-09-08 (C55) | **P2 confirms P1 is void rather than weak: flat at the null at every depth (37, 39, 43, 47, 55), 16/40 above null which is exactly chance, no rise anywhere, max 0.0198.** Total absence of signal at the plant layer AND downstream is the signature of a mis-specified test, not a failing instrument | notebook C54/C55 |
| 2026-09-08 (C53) | **STUDY 1 REPLICATES ON A SECOND FAMILY: Pythia-160m 93/100, Wilson [0.863, 0.966], FAIL against the same 0.90 bar.** Both models fail, so the silent-failure phenomenon is not a GPT-2 artifact - **but the intervals do not overlap**, so the rate is model-dependent. **Per the filed rule, "roughly a quarter" is dropped for a range: 7-23% depending on the model**, both reported side by side | notebook C53 |
| 2026-09-08 (C53) | **THE FLAG TRANSFERS AND IMPROVES - the deliverable survives its biggest risk.** Disagreement AUC 0.810 to 0.975, held-out R2 0.915 to 0.995. **Caveat that must travel with it:** Pythia's failures are more severe (median 0.794 vs 0.889, min 0.076 vs 0.145) and extreme failures are easier to separate, so part of the gain is about what the detector was asked to detect | notebook C53 |
| 2026-09-08 (C53) | **Aggregate invisibility gets STRONGER as the failure rate drops.** Pythia's median alignment is 0.9994 - excellent by any reported aggregate - while containing a unit recovered at 0.0761. Fewer failures, more severe, and the median moves further from the truth | notebook C53 |
| 2026-09-08 (P1) | **STUDY 2 P1 IS VOID, NOT FAILED - and the fault is my design.** Recovery of a planted random direction sits BELOW the null at every strength (medians 0.0064/0.0086/0.0078 vs nulls 0.0128/0.0117/0.0100), above null on only 8/24 rows, max 0.0264 against a 0.30 bar, and flat across a 4x strength range. **But a random direction has no natural representation, so the text it produces carries no consistent signal to re-read** - the pipeline was asked for something that cannot happen. The pre-registration permits reporting this as a finding; it should not be. Corrected P1 plants a real concept vector that C48 showed steers, filed as a new run | notebook C53; prereg s2-p1 |
| 2026-09-08 (C50-52) | **A-14 RETURNS B2: THE A3 EFFECT DOES NOT SURVIVE THE VALIDATED WINDOW.** Filed primary - real vs random, neutral, pooled 30-60% - gives p=0.119, real lower on 72/120. Only 30% is significant, and it sits at the edge of the unvalidated region. **The below-baseline suppression is also gone** (p=0.24-0.95, down on 12-16 of 30, i.e. chance). Both halves of C45/C46 are confined to strengths where C48 measured steering at chance | notebook C50-52 |
| 2026-09-08 | **The A3 effect INVERTS with strength.** Real climbs 0.1009 to 0.2445 (neutral) while random stays flat; they converge near 40-50% and by 60% real is at or above random. So it is a low-strength phenomenon, not a property of concept vectors | notebook C50-52 |
| 2026-09-08 | **Study 3 has no stable empirical claim at any strength on this model.** Per Addendum 4's filed B2 clause, Paper A's centre of gravity is the readout-dependence and framing results. Outline v3 already demoted Study 3 to a five-knob methods chapter - **this run confirms that was correct rather than defensive.** None of the five knobs depended on A3 | Addendum 4; outline v3 |
| 2026-09-08 | **C49 survives the window: span is indistinguishable from random at every strength (p=0.13-0.79).** The off-manifold account stays refuted, and that finding does not depend on the strength range | notebook C50-52 |
| 2026-09-08 (scout 2) | **STUDY 2 REPLACES STUDY 3 AS CHAPTER TWO.** Persona vectors are validated only by steering effect and finetuning correlation (r=0.76-0.97, 2507.21509), and auditing tools are already built on them (2607.13162, 2605.13329). **Nobody plants a known direction and measures recovery.** Venkatesh & Kurapath 2602.06801 shows orthogonal perturbations steer near-equivalently, so steering-based validation is demonstrably invalid - our motivation, in someone else's citable paper | notebook scout 2 |
| 2026-09-08 | **Study 2 is UNBLOCKED.** The K>=2 degeneracy (C14) blocks only P4, the multitrait matrix. P1 (recovery at the plant layer) and P2 (recovery vs depth) need only K=1. The November go/no-go can be answered by two Kaggle sessions rather than by generalising the cascade | notebook scout 2 |
| 2026-09-08 | **Agent persona drift REJECTED as a direction.** ContextEcho, SPASM, agent-drift quantification, geometric identity frameworks, FinPersona-Bench and a CHI 2026 paper all landed in 2026, all behavioural. Cheap to produce, several groups producing them, no advantage for us. AGI framings: no measurable gap on free-tier compute | notebook scout 2 |
| 2026-09-08 (scout) | **PAPER ORDER FLIPS: STUDY 1 LEADS, STUDY 3 BECOMES A CHAPTER.** A literature scan found Study 3's space contested - binary detection already shown artifactual by arXiv 2512.12411 (r=0.999 control) and content-agnostic by Lederman & Mahowald, steering vectors already shown non-identifiable by Venkatesh & Kurapath, and a COLM 2026 reality-check paper published. **Our A2 result was never ours.** A targeted search for weight-column-as-ground-truth calibration of a direction estimator returned nothing - **Study 1 is uncrowded and its ground truth is free rather than planted** | notebook scout |
| 2026-09-08 | **The extraction-position finding is weaker than it looked.** Token position for concept extraction is already documented as mattering enormously (arXiv 2602.00333), and 2512.12411 avoids the bug entirely by averaging over all prompt tokens. What may survive is the narrow form - a vector passing every reported health check while carrying nothing - not "position matters" | notebook scout |
| 2026-09-08 | **Adopt averaging over prompt tokens as the extraction default**, following 2512.12411. It is more robust than either the template tail (C31, dead vectors) or the concept-token read (C48, weak steering on Gemma). Not yet run | notebook scout |
| 2026-09-08 | **If Study 3 is ever to carry a positive result it must move to differential tasks at early layers.** 2512.12411 reports 88% sentence-localisation and 83% strength-comparison at L0-L5, against our binary detection at L37/62. That experiment would confirm their finding, not establish ours - so it is optional, and it is not the paper | notebook scout |
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

### 7.9 Levels 2 and 3, with APERTURE merged in: the plan of record from 6 Oct 2026

The full plan lives in `docs/RUN_PLAN_L2_L3.md`, not here. That file holds the shared design
for every level, the APERTURE inventory, each run's reference, dials, checks, criterion,
failure branch and cost, the calendar, the gates and ownership. Runs get their own §3 rows
and §4 entries here as they land, as for the B-series.

| ID | Run | Paper | Compute | Gate / prereg |
|---|---|---|---|---|
| T-0 | Trait harness | 3 | CPU | gates T-2, T-4 |
| T-1 | Exact-estimand precision control | 3 | CPU | prereg before run |
| **T-2** | **Exact readout bench (primary L2)** | 3 | CPU | pilot -> prereg; **Gate A, 31 Dec** |
| T-3 | Whitened P1b rescore | 3 | Kaggle ~2 h | optional |
| T-4 | Trained-in trait directions | 3 | Kaggle 10-20 h | **Gate B, mid-March** |
| S-0 | Apparatus merge + steering gate + impact-matched random | 2 | eng. | gates all S runs |
| **S-1** | Dead-vector precision x position ablation | 2 | Kaggle 6-10 h | prereg before run |
| **S-2** | Small-model instrument audit | 2 | Kaggle ~15 h | prereg before run |
| **S-3** | A-F1, finish APERTURE confound hardening | 2 | Kaggle ~10 h | frozen prereg (mirror) |
| S-4 | Study 3 reanalysis (TOST, exact CIs, dose-response) | 2 | CPU | offline |
| S-5 to S-9 | Exact-truth self-report tasks | 3 | Kaggle ~13 h | prereg per task |
| S-10 | PLANTED verbalizer calibration | 3 | Kaggle ~8 h | needs `readouts.py` |
| S-12 | Human grading + kappa | 2, 3 | labour | gates judge-scored numbers |

### 7.8 The bench run programme (B-series) — the plan of record, Sep–Oct 2026

Supersedes 7.1's A-series as the active queue. A-1 to A-8 belonged to Paper A under the
introspection framing; that paper is now cut from the flagship (see 6b, 8 Sep). Everything
below is CPU-only and free unless marked otherwise.

**Design rule carried from 7.1, and it now has teeth.** Every run states its question, its
criterion, its cost, and its failure branch. Three runs below exist *only* because a
literature scout named a reviewer objection we had not planned for.

**Critical path to a submittable paper:** B-1 → B-2 → B-3 → (B-6 if triggered) → B-8 → B-9
→ draft. About 30 CPU-hours total, which is three or four overnight runs.

> **STATUS AND NEXT QUEUE, 5 Oct 2026 - supersedes the Tier 0 table below, which is stale
> (B-1 "RUNNING", B-2 "QUEUED").**
>
> | run | state |
> |---|---|
> | B-0, B-0b, B-1, B-1b, B-2b, B-11, B-11s, B-7, B-8, B-10 | done, archived |
> | B-2s | abandoned (3 orphan rows) |
> | B-3 spec sheet / B-4 restart curve / B-5 combination | not run as separate artifacts; B-1b's analysis covers B-3's core |
> | B-6 | not triggered (B-1b went straight to n=300) |
> | B-9 (third scale point) | not run |
> | B-12 | three arms done, control arm running since 07:54 |
> | S1-2 | not run; blocked on a filed prereg (6b, 5 Oct) |
>
> **Next queue, in order. Each needs its criterion filed before it runs.**
>
> 1. **B-13 · Stopping fix + invariance test (code, ~1 day, no long compute).** Per-unit
>    patience and an active mask in `fit_batch`. Stopped units are frozen by restoring their
>    parameters after each `opt.step()`, so Adam's momentum cannot move them. Add a test that
>    unit i alone equals unit i in a shuffled batch of 8 to within 1e-5, with fp64 and
>    deterministic algorithms on. Rebuild the Kaggle bundle. *Failure branch:* if units still
>    differ, the residual really is reduction order, and B-12's original branch applies.
> 2. **B-14 · B-1b re-run on the fixed estimator (n=300, GPT-2 L6, ~12 h CPU).** Same
>    units, restarts, tokens and seeds. Primary endpoint unchanged (DeLong restart vs R2),
>    with a paired cluster bootstrap on the AUC difference beside it. For every failure,
>    record whether held-out R2 was still rising at stop (under-trained) or converged
>    (wrong basin). *Failure branch:* if the gap shrinks below 0.10, B-1b's p-value was partly
>    a stopping artefact, and the paper says so.
> 3. **B-15 · Replicates for soft labels (3 replicates x 100 units, GPT-2 L6, ~14 h).**
>    Different per-unit seeds and token draws. Report each unit's failure fraction,
>    test-retest ICC of every signal, and the share of boundary units near 0.95. Turns the
>    verdict noise into a measured quantity rather than a caveat.
> 4. **S1-2 · Deflation gate, filed properly (synthetic, ~4 h).** Add two arms before
>    running: whitening plus a moment / AGOP warm start, and response deflation of the
>    *fitted* g1(v1 . x) with backfitting. Report the minimum principal-angle cosine beside
>    the mean.
> 5. **B-16 · SwiGLU K=2 ground truth on SmolLM2-135M (~6 h CPU).** Each gated unit reads
>    exactly two directions (gate and up rows times the RMSNorm gain). This is the first real,
>    not planted, K=2 test, and the cheapest answer to "only GPT-2-class models".
> 6. **B-17 · GPT-Neo L6, n=20 with per-unit diagnostics (~1 h).** Settles whether the
>    below-chance 0.0026 is a model property or a pipeline fault.
> 7. **B-18 · One model at or above 0.5B on the headline comparison** (Qwen2.5-0.5B or
>    OLMo-1B, n=100, Kaggle, steps scaled to width per B-11s). Answers the scale objection
>    without repeating B-11's under-fitting.
> 8. **B-19 · New ground-truth-free signals scored on B-14's fits (offline, minutes):** R2 on
>    active tokens only, rank-2 minus rank-1 R2 gap, active-token count over d, and a residual
>    test for a missing direction. On-thesis: more checks calibrated against the same truth.
>
> **Not before the above:** SAE encoder rows as a second substrate; PolyPythias seed spread;
> Study 3 as its own short paper with an impact-matched control (Ferrara 2608.20569 now
> covers the norm-matched-random result on 8 other models, and our own C45/C46 A3 agrees with
> it).

---

#### Tier 0 — in flight

| run | model | config | cost | state |
|---|---|---|---|---|
| **B-1** | GPT-2 small | n=100, L6, 5 restarts | ~4.4 h | **RUNNING** |
| **B-2** | Pythia-160m | n=100, L6, 5 restarts | ~4.4 h | **QUEUED** behind B-1, auto-starts at 100 rows |

Criteria in `preregistration-b1-stability-calibration.md` + `preregistration-b1-addendum-1.md`.

---

#### Tier 1 — offline, zero compute, runs in seconds once Tier 0 lands

**B-3 · The spec sheet.** `b1_signal_calibration.py`. Recovery rate with a Wilson
interval; every signal's AUC, sensitivity at 10% false alarm, and good units discarded at
70% catch; DeLong against the incumbent with BH correction on the secondary family.
*Script written and tested (8 tests, `tests/test_delong.py`).*

**B-4 · The restart-count curve.** AUC of restart agreement recomputed at 2, 3, 4, 5
restarts by subsampling what B-1 already stored. **Free.** Practitioners commonly run two
or three; if the check only works at five, that is a materially different recommendation
and the paper must say which. Pre-specified as descriptive, not tested (addendum 1 §4).

**B-5 · Signal combination.** Logistic on {held-out R2, disagreement, restart agreement},
nested 5-fold CV, compared to R2 alone by cross-validated AUC.
*Failure branch, and it is a good outcome:* if the combination does not beat R2 alone,
that is the finding — the free signal is sufficient — and the practical recommendation gets
simpler, not weaker.

---

#### Tier 2 — conditional, triggered by B-3

**B-6 · Extend to n=300 on GPT-2.** **Triggered if and only if |dAUC| < 0.10** between
restart agreement and held-out R2. Authorised in advance by addendum 1 so it cannot be a
reaction to a disappointing p-value. ~13 h.

*Why it is likely to fire.* A synthetic check during implementation put the standard error
of a dAUC at roughly 0.07-0.11 for uncorrelated signals at n=100 with 25 failures. Our
signals are computed from the same fits and so are correlated, which shrinks that
substantially — but the published power figures are blunt: dAUC = 0.10 needs 36-142 cases,
dAUC = 0.02 needs 909-3,709. **n=100 can resolve a large gap and cannot resolve a small
one, and the paper says so either way.**

**At most one extension.** If n=300 still cannot resolve it, the honest conclusion is that
the two signals are close enough that the choice is about cost — and cost already favours
held-out R2, which is free, over restart agreement, which is 5x the compute.

---

#### Tier 3 REFRAMED - the variance decomposition. B-7 and B-8 are now load-bearing.

**Filed 9 September 2026 after the scout.** `2604.11581` (Hidden Measurement Error) shows
naive confidence intervals run **40-60% too narrow** because they count sampling noise and
ignore *researcher design choices*. Our Wilson intervals have exactly that flaw: they count
unit-sampling noise and ignore device, layer, family, scale, step budget and restart count.

`2607.19386` (Building Fast, Evaluating Slow) independently finds **methodological variance
exceeds architectural variance** across autointerp metrics. The two together say the same
thing: the design choices are bigger than the thing being measured, and nobody budgets for
them.

**We can budget for them, because we have already run most of the components.**

| component | run | status | measured |
|---|---|---|---|
| device | B-0 | **done** | **5/16 units flip pass/fail** |
| restart count | B-1 vs B-1b | **done** | 77/100 at r=2 -> 91/100 at r=5 |
| scale | B-11 ladder | **done** | 4 rungs, 96/96/96/62% |
| step budget | steps check | running | 1600 vs 3200 on 1.4b + 410m control |
| **model family** | **B-8** | queued | GPT-Neo-125m, same shape as GPT-2, different corpus |
| **layer** | **B-7** | queued | L2 / L6 / L10 |

**This changes what B-7 and B-8 are for.** They were reviewer-proofing - answers to
objections nobody had raised. They are now **variance components in a total-error budget**,
and the paper reports a design-sensitivity-corrected interval beside the naive Wilson one.

**Nobody has asked what fraction of a reported interpretability number is design choice
rather than signal.** We can answer it with runs that are five-sixths paid for.

**Priority unchanged, reasoning changed:** B-7 and B-8 still run after the primary arms, but
they are no longer optional and are not cut if time is short.

---

#### Tier 2b — B-0, the device-equivalence check. Gates every Kaggle GPU run.

**Question.** Does fitting on CUDA give the same answer as fitting on CPU?

**Why it is not pedantic.** The study is about *which units fail*. A failing unit is one
where the optimiser landed in the wrong basin, so failing units sit near basin boundaries
by construction - exactly the population where floating-point reduction order can flip the
outcome. CPU and GPU do not agree bitwise. A table reporting "GPT-2 23%, Pythia 7%" with
one device each has a hardware term inside its headline number.

**Criterion, filed before the run.** PASS = no unit changes pass/fail side AND
max |d alignment| < 0.01 -> B-series may mix devices. FAIL = any flip, or the tolerance
exceeded -> every compared set shares one device and the device is reported beside every
number.

**A FAIL is a finding, not a wasted run:** estimator failure classification would be
hardware-dependent, which means anyone reproducing an interpretability result on different
hardware may not reproduce *which* units failed. That is squarely what this bench is for.

**Cost** ~40 min, 24 units, both devices, GPU session. Sheet:
`kaggle/NEXT_SESSION_B0_DEVICE.md`. Script: `experiments/kaggle_device_equivalence.py`.

**It also returns the measured speedup**, so the rest of the programme is scheduled
against a number rather than a hope. Below ~3x, GPU is not worth the confound risk even
on a PASS.

---

#### Tier 3 — reviewer-proofing. Each one answers a specific predictable objection.

**B-7 · Layer sweep.** GPT-2 layers 2, 6, 10, n=50 each. ~6.6 h.
*Objection answered:* "you looked at one layer."
*Question:* is silent failure depth-dependent? This is not only defensive — Proposal III's
tuning curves will be measured across layers, so the depth profile of the instrument's
error is needed before that work starts.
*Failure branch:* if the failure rate swings wildly with depth, the headline becomes a
range over depth as well as over models, and the single-layer numbers must be relabelled.

> **RESULT — landed 4 Oct 2026, all three legs complete (150/150 units).** Silent failure
> is **strongly depth-dependent in aggregate**, and the paired structure shows the aggregate
> is misleading about the mechanism.
>
> | layer | n | pass | pass rate | Wilson 95% | median align | min align | methods disagree >0.05 | s/unit |
> |---|---|---|---|---|---|---|---|---|
> | L2 | 50 | 24 | 0.48 | 0.348–0.615 | 0.9541 | 0.4793 | 0.58 | 123.9 |
> | L6 | 50 | 43 | 0.86 | 0.738–0.931 | 0.9935 | 0.2707 | 0.32 | 143.2 |
> | L10 | 50 | 45 | 0.90 | 0.786–0.957 | 0.9943 | 0.8526 | 0.24 | 197.0 |
>
> **These three legs are PAIRED, and that is the whole point.** The unit draw depends only on
> the seed and `d_model`, both fixed across layers, so all three legs ran the *same 50 units*
> — verified, the unit keys are identical across the three files. The failure sets can
> therefore be compared within unit, not only between samples.
>
> **They barely overlap. From `experiments/analyse_b7_depth.py`:**
>
> ```
> L2 fail 26/50  ->  21 of those 26 PASS at both L6 and L10 (median align 0.836 -> 0.995)
> fail at all three layers  : 0/50
> pass at L2 but FAIL at L10: 3/50   (median align 0.998 at L2, 0.873 at L10)
> pass everywhere            : 18/50
> ```
>
> **So the depth effect is NOT "shallow layers are noisier."** No unit fails at every depth.
> Depth does not repair a fixed set of broken units — it changes *which* units fail. The
> 0.48 → 0.90 rise is real, but it is an average over near-disjoint failure sets, and the
> reverse case exists: three units are cleanly recovered at L2 (0.998) and fail at L10 (0.873).
>
> **Correct statement: failure is a property of the (unit, layer) pair, not of either alone.**
> Any sentence of the form "layer N has a failure rate of X" describes a population whose
> membership shifts with N, so depth-stratified numbers must be quoted per layer with that
> caveat rather than aggregated into a trend.
>
> **Pass/fail is also not perfectly reproducible at fixed layer.** B-7 L6 at 5 restarts and
> B-1b L6 at 2 restarts share all 50 units and agree on **40 of 50 verdicts (80%)** — 3 units
> fail in both, 4 fail only at 5 restarts, 6 fail only at 2. **20% of verdicts flip on restart
> count alone.** That bounds how finely any single pass rate can be read, and is why the B-8
> family contrast is argued as a 31-point gap rather than a small one.
>
> Secondary: method disagreement tracks the same instability (0.58 / 0.32 / 0.24). The four
> rank-1 routes agree far more often in deep layers — consistent with shallow-layer
> directions being genuinely less determined, but equally consistent with the L2 units being
> a different population. It cannot separate those two readings, so neither is claimed.
>
> **Consequence for the headline.** Every prior number in this notebook — B-1, B-1b, B-2b,
> and the whole Study-2 gate — is measured at **layer 6**. B-8 shows layer-6 GPT-2 is the
> favourable end of the family axis. The single-layer figures must be relabelled "layer 6"
> wherever they appear.
>
> Cost note: the sweep ran 150 units in ~8.7 h wall against a ~6.6 h estimate. Deeper layers
> cost more per unit (123.9 → 197.0 s/unit, +59%). Estimate the deepest layer first in future
> sweeps.
>
> **Correction, filed the same day.** An earlier write-up of this result read the pass rates
> as "two regimes rather than a gradient", on the strength of the overlapping L6/L10
> intervals. **The paired analysis refutes that.** The intervals overlap because the L6 and
> L10 failure sets are nearly the same *size*, not because the layers behave alike — only 3
> units fail at L6 while passing at L10. The deep regime is stable; the shallow regime is a
> different population, not a noisier sample of the same one. The first reading was made
> without checking whether the legs were paired, which is the actual lesson.

**B-8 · Third model family.** GPT-Neo-125m (different training corpus, same scale), n=100,
L6. ~4.4 h.
*Objection answered:* "two models is not a family effect."
*Failure branch:* if the third family behaves unlike both, the claim narrows from "models"
to "these models" and the paper says so.

> **RESULT — landed 4 Oct 2026 (100/100 units).** The run is at **layer 10, not layer 6**,
> and that substitution is part of the finding rather than a convenience.
>
> **The planned L6 arm does not exist in the data.** A matched 4-unit smoke at L6 gave
> median alignment **0.0026 with 0/4 passing** — chance. The same smoke on GPT-2 L6 gave
> 0.9872 with 3/4 passing, so the harness, the bar, and the code were all fine; GPT-Neo
> simply does not admit recoverable directions at L6 at this bar. Hook placement was
> verified independently (the MLP input hook sits at exactly ln 2 layers up) and the
> layer-6 correlation was 0.698, in line with the layers that do work. **A null at a layer
> where the same estimator recovers 0.99 on GPT-2 is a property of the model, not a bug**,
> and it is reported as such rather than quietly re-run at a layer that works.
>
> **Final arm, GPT-Neo-125m L10, depth-matched to the GPT-2 L10 leg:** n=100, 59 passing,
> pass rate **0.59** (Wilson 0.492–0.681), median alignment **0.9586**, min 0.0344,
> methods disagreeing >0.05 in **0.51** of units, 48.4 s/unit.
>
> **The failure branch fired too, and this is the load-bearing result.** GPT-Neo at L10
> passes at 0.59 while GPT-2 at the *same layer, same depth, same bar* passes at 0.90. The
> intervals do not overlap (0.492–0.681 vs 0.786–0.957). **Silent failure is a model-family
> effect, not just a model effect or a depth effect** — same scale, same depth, different
> corpus, 31 points of pass rate on a like-for-like comparison. That is a stronger and
> stranger claim than "two models is not a family effect": it says the corpus is the axis
> that matters most for a benchmark built on these models, and that "models" as a category
> carries a 31-point spread inside a single layer.
>
> The family contrast is now the cleanest large effect in the notebook, precisely because it
> holds layer fixed. **Do not rank it against the B-7 depth span.** The depth figure
> (0.48 → 0.90, 42 points) is *not* a like-for-like contrast — B-7's legs run near-disjoint
> failure sets, so that 42 is an aggregate over shifting populations, not a measured
> difference between two comparable groups. Comparing 31 to 42 would be a category error.
>
> The honest consequence is that **the instrument's error rate is corpus-dependent, and
> the paper's headline range must be reported across model families, not just models.**
> GPT-2 is the favourable case. Any generalisation to "models" has to carry GPT-Neo's 0.59.
> Its high method disagreement (0.51) matches GPT-2 *L2*, but the B-7 paired analysis showed
> disagreement tracks population difficulty without identifying the cause — a harder
> population and a less-determined direction produce the same signature. Neither is claimed
> here.
>
> Cost note: 48.4 s/unit at 2 restarts — roughly half the GPT-2 cost per unit despite
> being a different tokenizer, so the ~4.4 h estimate was sound.

**B-9 · Scale.** Pythia-1.4b, n=50, layer 12 of 24. d_model 2048 against 768, so the fit is
roughly 2.7x wider. CPU overnight, or Kaggle GPU if it drags.
*Objection answered:* **"only 124M and 160M models."** This is the single strongest
objection to the paper and the one most likely to decide a review.
*Failure branch:* if it will not run on CPU in a night, it moves to Kaggle rather than
being dropped. If the failure rate at 1.4B is near zero, that is itself a result — the
problem shrinks with scale — and it must be reported as readily as the opposite.

**B-10 · Required-N per signal.** Token sweep 2k/4k/8k/16k at n=50. ~8 h.
*Objection answered:* "your operating point is arbitrary."
*Question:* does each signal's AUC stabilise, and does the **ranking** of signals change
with data? A signal that wins only at 8k tokens is a different recommendation from one that
wins everywhere.

---

#### Tier 4 — what makes it a bench rather than a paper

**B-11 · The adapter interface and the spec-sheet CLI.** A third party supplies an
estimator behind a small interface; the bench returns the spec sheet. Without this it is a
paper with tables. With it, it is a thing other people can run — which is the axis
NeurIPS D&B and TMLR actually reward. No compute.

---

#### Tier 5 — the bridge to the rest of Proposal III (sem 6)

**T-1 · Generalise the cascade to K>1.** **Blocking.** Joint estimation currently
degenerates to K=1 at every sample size. The population layer cannot start until this is
fixed, and it has been open since C14.

**T-2 · Tuning curves** over a controlled stimulus battery, cut to two variables by the
August audit — now sitting on an instrument whose error rate is known.

**T-3 · Pairwise maximum entropy** on a population of units. The August research pass
verified this has zero existing language-model applications; the near-miss builds Ising
couplings *into* an architecture, which is the opposite direction.

---

#### Cut, and why

- **Study 2 (planted personas)** — P1 void, P1b a negative that a published behavioural
  paper (2602.06801) already reached. Real work; not flagship material.
- **Study 3 (introspection readout)** — solid result, fast-crowding area.
- **A-1 to A-8** — belonged to Paper A's framing.

Both cut studies remain reportable as separate outputs. Including weak studies weakens a
paper.

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

> **AMENDED 4 Oct 2026, before the run: a second coupling is added, and the criterion is
> not.** The deflation finding is coupling-dependent — response-only beats the projected
> method by 0.05 under additive coupling (0.995 vs 0.945) but by **0.42** under
> multiplicative (0.930 vs 0.506) — so running additive alone would characterise the effect
> only where it is smallest, and the paper's *explanation* for why projecting the stimulus
> hurts rests on the multiplicative case. Both couplings now run.
>
> The pre-registered 0.80 criterion is **still judged on the additive cells only**, because
> the 0.5213 joint baseline it is defined against comes from `e03_required_n.json`, which
> planted additive units. Judging it on the multiplicative cells would compare against a
> number from a different generative model. The multiplicative cells are reported for
> mechanism, and the archived baseline is recorded as `null` on those rows rather than
> quoted, so it cannot be misread.
>
> The stimulus-projection rationale that used to sit in this section — *"the projection is
> not optional, because response-only leaves f(Xv₁) − Xv₁ for the next rank-1 search to
> find"* — was **tested and refuted**, and is now corrected in the script docstring. The
> arms `cascade`/`plain` are retained only because this flag is what the run varies;
> `resp_only` is the arm the evidence supports.

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

> **B-12's criterion is AMENDED 4 Oct 2026, before the run, from "zero flips" to a
> measured floor.** B-12 was filed with the criterion *"ZERO units change pass/fail side
> between any two batch sizes."* That is not a bar this instrument can be held to.
> Comparing B-7 layer 6 at 5 restarts against B-1b layer 6 at 2 restarts, on the same 50
> units, **10 of 50 verdicts flip (20%) with nothing changed but the restart count.** Batch
> size perturbs float32 reduction order, which is not a smaller perturbation than changing
> restarts — so under the filed criterion, ordinary restart noise would have been reported
> as "flips remain", which triggers the much larger claim that silent failure depends on how
> the estimator was implemented.
>
> The fix is to measure the floor inside the run rather than import it from another
> experiment: a fourth arm repeats the reference batch size at a different restart count, on
> the same units in the same session. Batch-size flips are judged against control flips, and
> the external 20% is reported beside it so the two can be compared instead of one being
> trusted. `--control-restarts 0` restores the strict zero-flip criterion.
>
> **A criterion stricter than the instrument's own repeatability is not a strong criterion,
> it is an unfalsifiable one** — it cannot distinguish a residual defect from the estimator
> doing what it already does. Worth checking every filed criterion against a measured noise
> floor before the run rather than after.

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

### 7.8 The bench — B-1 to B-5, the calibration deliverable (Sep 2026, local CPU)

**Filed 8 September 2026.** Spec: `docs/BENCH_SPEC.md`. Pre-registration for B-1/B-2:
`docs/preregistration-b1-stability-calibration.md`.

**What this line of work is.** Not a benchmark of direction-finding methods — that
space is crowded (ObserverBench, MIB, AxBench). A **calibration bench for the
ground-truth-free reliability checks** the field uses to decide whether to trust a
result. The 8 September scout found the same hole from three sides: reliability
signals validated against behaviour or reconstruction but never against a
known-correct direction; ground truth for direction recovery that is always planted
or compiled; harnesses that rank methods but carry no failure detector. Free exact
ground truth on a real model closes it.

**Why it costs almost nothing.** Every ingredient is already on disk. E0.5 is a
four-method scoreboard against free ground truth (STA 0/30, decorrelated STA 1/30,
STC 0/30, fitted bottleneck 26/30). Two substrates are wired and run (GPT-2 77/100,
Pythia-160m 93/100). Three signals are already scored. The missing one is the
important one and it was being computed and thrown away.

---

#### **B-1 · Restart-agreement calibration, GPT-2 — THE HEADLINE** ⚠ unblocked, not run

**Question.** Restart agreement is the default reliability check in this literature.
Does it predict whether the recovered direction is *correct*? Unanswerable without a
known-correct direction; answerable here.

**State — code change landed 8 Sep, run not started.** `Fit.stability` has existed
at `caliper/estimator.py:29` since the estimator was written. `fit_batch` populated
it on every fit. `e01_gate.py` never wrote it to the row. Three patches, all landed
and smoke-tested:

- `caliper/batched.py` — `per_restart_r2`, so held-out R2 per restart survives.
- `caliper/estimator.py` — `Fit.r2_spread`; `fit()` now records `r2_restarts`; the
  stale comment claiming `restarts` held test_r2 (it held subspaces) is corrected.
- `experiments/e01_gate.py` — records `stability`, `r2_spread`, `n_restarts`.

**Config.** `--restarts 5`, everything else the C13/C47 configuration: same 100
units, layer 6, 8000 tokens, 1600 steps, bar `align > 0.95`.

**Cost.** C13 ran 63.7 s/neuron at 2 restarts. The fit dominates, so 5 restarts is
about 2.5x: **~4.4 h**, one overnight local CPU run, resumable via `Checkpoint`.

**Endpoint.** AUC of `stability` against the AUC of held-out R2 (0.906) on the same
100 units, with a bootstrap interval on the *difference*, not two bare numbers.

**Pilot already on disk, disclosed in the filing.** `results/s1_seed_lottery.json`,
8 units at 10 seeds, run for a different purpose (C43). Passing units are stable to
within 0.0003; of four failing units, **two are stable to within 0.05 while being
wrong** (n527 median align 0.9163 spread 0.0474; n1625 0.9131 / 0.0658), and two are
plainly unstable (n1503 spread 0.9169; n2023 0.3309). Expectation recorded before
the run: **moderate AUC, poor operating point at high specificity** — it catches the
severe failures and misses the subtle ones. n=8 across two hand-picked groups is not
a sample and licenses no claim.

**Failure branches — all three are results, none is a failed run.**

| outcome | reading | what ships |
|---|---|---|
| `stability` AUC < 0.70, significantly below R2 | The field's default check does not discriminate failures on this substrate | The headline. A widely used check calibrated for the first time and found wanting, with a free better alternative in hand |
| `stability` AUC inside R2's interval | Equivalent detectors | Cost-benefit result: restart agreement costs 5 fits, held-out R2 costs zero. Reported without corrective framing |
| `stability` AUC significantly above R2 | The incumbent recommendation is wrong | The bench recommends restart agreement, **and notebook Finding 2 is amended.** Say so plainly |

**Anticipated confound, committed in advance.** 5 restarts is a better optimiser than
2, so the pass rate will move off 77/100. **A changed pass rate is not a finding
here** — C13's gate result stands at its own filed configuration, and B-1 is not a
re-run of the gate.

---

#### **B-2 · The same on Pythia-160m** — transfer

Same five signals, same criteria, C53's 100 units. ~4.4 h.

**The caveat travels with it, unchanged from C53:** Pythia's failures are more severe
(median 0.794 vs 0.889, min 0.076 vs 0.145) and extreme failures are easier to
separate, so any gain is partly about what the detector was asked to detect. If
`stability` improves on Pythia by the same margin R2 and disagreement did
(0.906 -> 0.995, 0.802 -> 0.975), that is severity, not transfer, and the write-up
says so.

**Failure branch.** If the ranking of signals *flips* between families, the bench
cannot recommend one signal and must report per-substrate spec sheets instead. That
weakens the deliverable's headline and strengthens its honesty; it is reported, not
worked around.

---

#### **B-3 · Does stability add anything on top of held-out R2?** — offline, free

Logistic on the signal set, 5-fold CV, cross-validated AUC against R2 alone. C38
already built the CV machinery for thresholds.

**Why it is separate from B-1.** A signal that is *redundant* is a different finding
from one that is *uninformative*, and only this run separates them. If stability
adds nothing on top of R2 but is a fine detector alone, the recommendation is still
"use R2, it is free" — but for a different reason, and the paper must not conflate
them.

**Failure branch.** With 22 failures in 100 units, a three-predictor logistic is at
the edge of what the data supports. If CV intervals are too wide to separate the
models, report that the run was underpowered and give the interval — do not report a
point estimate as if it settled anything.

---

#### **B-4 · Required-N per signal** — offline, reuses `e03_required_n`

At what N does each signal's AUC stabilise? The bench's spec sheet has a required-N
row and it should be per-signal, not global. C41 established that the binding
quantity is **events, not positions**, and that C37's original attribution was wrong;
that correction carries into this run's design.

---

#### **B-5 · The spec-sheet generator** — one script, no new compute

`experiments/bench_report.py`: reads the gate rows, emits the section 4.4 table for a
(method, signal, substrate) triple. Recovery with Wilson CI, signal AUC with
interval, TPR at fixed FPR, discarded-good at 73% catch, random-direction floor,
required-N.

**The operating-characteristic columns are the deliverable, not the AUC.** C35 is the
precedent: held-out R2 and disagreement both had high AUC, and the reason to prefer
R2 was that at 73% catch it discarded 5 good units against disagreement's 21. A
practitioner needs that number; an AUC does not give it to them.

---

#### **Not in v1, and why**

- **A third model family.** Only after B-1/B-2 land. Breadth before the calibration
  is measured is breadth for its own sake.
- **SAE latents as a substrate.** No free ground truth there — that is the whole
  point of section 3's stated limit. Wanting it does not create it.
- **P1b re-run at `2026-09-08f`.** Still worth doing, now for a better reason than
  before: the 8 Sep scout found
  [Non-Identifiability of Steering Vectors (2602.06801)](https://arxiv.org/html/2602.06801v4),
  which reaches P1b's conclusion **on behavioural evidence only** and explicitly does
  not plant a known direction and attempt recovery. P1b is the ground-truth
  complement to a published behavioural result, not a scooped study. GPU, so it
  queues behind the CPU work.

---

## 8. Gotchas solved (so we never lose the time again)

### Kaggle gotchas inherited from APERTURE (recorded 6 Oct 2026)

Solved once in APERTURE (`projects/mirror/docs/LAB_NOTEBOOK.md` §1 and §7). Recorded here
because CALIPER's L3 runs now use the same apparatus.

- **`%pip install` of a GitHub main.zip serves a cached old archive.** Use
  `--no-cache-dir --force-reinstall --no-deps`, or the run silently uses last week's code.
- **`git+https://` installs hang** on a credential prompt. Install from the
  `/archive/refs/heads/main.zip` URL instead.
- **TransformerLens rebuilds weights at full precision and doubles CPU RAM** on load. It caps at
  about 6B on Kaggle (Gemma-2-9B fp16 dies after loading 100%, with no CUDA error). Above about
  6B, use the HF backend with 8-bit bitsandbytes.
- **For Gemma-2-2B under TransformerLens:** `from_pretrained_no_processing` with float16.
  `from_pretrained` OOMs 30 GB of RAM; float32 OOMs the T4 on the 256k-vocab unembedding.
- **A verdict line that tests one side of a CI** prints "includes 0" for an interval that
  excludes 0 negatively (APERTURE R11, fixed). It is the same class as B-12's verdict
  arithmetic: verdict logic needs tests in both directions.
- **Greedy decoding is not reproducible across library versions under 8-bit quantisation.**
  Same seeds changed 5 of 192 answers (A-F1 c00 vs A-R11). Record library versions in every
  config, and score on statistics robust to a few flips.

### Setup in its own cell is setup that gets skipped (2026-09-09)

Third Kaggle session lost to the same class of problem, and the cheapest one to prevent.

The steps check was handed over as three cells - two runs and a packaging step - on the
assumption that the bundle setup cell had already been run. In a fresh notebook under
Save-and-Run-All there is no already-run kernel state to inherit, so the working directory
was still `/kaggle/working` and both runs died instantly:

```
python3: can't open file '/kaggle/working/experiments/e01_gate.py'
```

**Rule.** A Kaggle run sheet ships **one self-contained cell**: locate the bundle, set
`PYTHONPATH`, assert the GPU, then run. Never a run cell that depends on a setup cell
above it, because the dependency is invisible and the failure is instant and total.

**What did work, and is worth keeping.** The packaging cell's `assert rows, "nothing to
package"` fired correctly and refused to build an archive from an empty directory. Two
sessions earlier the same situation produced a 49-byte zip that looked like a successful
download until it was opened. **The guard converted a silent failure into a loud one**, which
is the entire reason it was added.


### Changing --neurons changes the WHOLE draw, it does not extend it (2026-09-09)

`np.random.default_rng(0).choice(3072, size=N, replace=False)` is not a prefix relation in
N. Drawing without replacement at a different size gives a **different sequence**, not a
longer one:

| pair | shared units |
|---|---|
| first 50 of n=100, vs n=50 | **1 of 50** |
| n=100, vs first 100 of n=300 | **6 of 100** |

Two consequences, one cosmetic and one serious.

**Cosmetic.** Two filings claimed the larger draw extends the smaller and that the arms are
nested. Both were false and are struck through in place. The endpoints survive, because
each arm computes its own incumbents on its own units - but C13's AUCs become **context,
never a comparator**, since quoting them beside a different draw is a cross-sample
comparison.

**Serious.** `Checkpoint` keys on the unit id, so re-running with a different `--neurons`
against an existing output file **appends a second, different unit set to it**. The Kaggle
70m rung resumed 30 rows written under `--neurons 100`, began filling the n=50 draw into
the same file, and its progress line read `79/50`. That file mixes two draws and is not a
valid rung.

**Rules.**
1. `--neurons` is part of a run's identity, like the model and the layer. Changing it means
   a **new output file**, never a resume.
2. A progress counter reading past its target (`79/50`) is the tell: the checkpoint holds
   rows the current draw never asked for.
3. To make draws genuinely nested, permute the pool once and slice:
   `rng.permutation(d_mlp)[:n]`. **Not changed mid-ladder**, for the same reason
   `fit_cascade`'s device is not - it would put a code version inside the scaling curve.


### `!python` on Kaggle is a subprocess - it inherits cwd, not sys.path (2026-09-09)

Second failed ladder session, same sheet, different cause. Cell 1 worked perfectly this
time - it found the bundle, printed the tree, said `bundle OK` - and then every run died:

```
ModuleNotFoundError: No module named 'caliper'
```

`sys.path.insert(0, root)` changes the **notebook kernel's** path. `!python
experiments/e01_gate.py` starts a **new process**, which inherits the working directory
but not `sys.path`. Python then puts the *script's* directory (`experiments/`) on the
path - never the bundle root - so the package is invisible. Cell 1 reporting success is
what makes it expensive: the diagnostics all pass and the failure appears four cells
later.

The local queue never had this bug because `run_queue.sh` opens with
`export PYTHONPATH=.`. The Kaggle sheet was written without carrying that across.

**Rule.** In any notebook that launches work with `!python` or `subprocess`, set the
environment, not just the path:

```python
os.environ["PYTHONPATH"] = root
```

and then **prove it in the same cell**, under the exact condition the runs use:

```python
p = subprocess.run([sys.executable, "-c", "import caliper, caliper.batched"],
                   capture_output=True, text=True)
assert p.returncode == 0, "a subprocess cannot import caliper: " + p.stderr
```

A setup cell that cannot fail is not a setup cell. Both ladder sessions died because Cell 1
declared success on something it had not tested.

**`check_sheets.py` earned itself immediately.** The first attempt at this fix shipped an
f-string whose escape had been mangled into a real newline; the validator caught it before
a third session was spent. The shipped cell is now also rehearsed end to end against a
simulated mount at the same nesting depth, so what ships is what was tested.


### Never hard-code the Kaggle mount depth - locate the bundle by a marker file (2026-09-09)

Cost: one wasted session. Cheap only by luck - the whole notebook failed in 25 seconds
instead of after an hour of fitting.

The B-11 ladder session produced **nothing**. Cell 1 did:

```python
root = glob.glob("/kaggle/input/*/")[0]   # WRONG
```

and the log's ninth line said exactly why:

```
/kaggle/input/datasets/ ['santoshcheethirala']
```

The bundle mounted a level deeper than the sheet assumed, so `root` became
`/kaggle/input/datasets/`, and every command after it failed with
`can't open file '.../experiments/e01_gate.py': No such file or directory`. The ladder loop
still printed all four rung headers, so the log *looks* like four runs happened. It is a
run sheet that fails silently and looks successful.

**Rule.** Find the bundle by a file that must exist inside it, assert loudly when it does
not, and print what is actually mounted:

```python
hits = glob.glob("/kaggle/input/**/caliper/estimator.py", recursive=True)
assert hits, "bundle not found - is the dataset attached?"
root = os.path.dirname(os.path.dirname(hits[0]))
```

Then assert every file the run needs before starting, so a stale bundle fails in Cell 1
rather than mid-run.

**And check the sheets before spending a session on them.** `python kaggle/check_sheets.py`
parses every ```python block in `kaggle/*.md` and reports cells using IPython magic as
*skipped*, not passed. This is the second run-sheet failure in this project - the first cost
four rounds and about three hours - and both were visible before the session started.


### A local overnight run is only as reliable as the machine's sleep settings (2026-09-09)

Cost: about five hours of wall clock, zero data.

The B-series queue was launched at 22:56 and died at 02:54 with B-1 at 92 of 100 units.
**No traceback**, and the parent shell was gone too - the signature of the process being
killed rather than crashing. The machine slept.

`Checkpoint` did its job: restarting the queue printed `B-1 starting (92/100 done)` and the
remaining 8 units cost 20 minutes instead of redoing four hours. **Nothing was lost except
time**, and only because every run in this project is resumable by default.

The global project index already carried the warning - "the laptop is often closed:
anything scheduled that must run reliably is a CLOUD routine, not a local one." This is
that warning arriving as a bill.

**Rule.** Before starting an overnight local run, either disable sleep on AC
(`powercfg /change standby-timeout-ac 0`, reversed with `... 30` afterwards) or accept that
progress is checkpointed and the queue needs restarting in the morning. Check the queue log
for an `exit` line: its absence means the shell died, not the run.


### Rows-on-disk is not a liveness check, and pgrep lies on Git Bash (2026-09-08)

**Corrected 4 October 2026 — the rule below was wrong and I have since replaced it. Read
this section as the story of how it was wrong, because the replacement is unintuitive.**

Cost: two duplicate runs and about half an hour, inside ten minutes.

A serial queue was launched while B-1 was already running. Its guard asked "does this
run's output file already have 100 rows?" - it had zero, because the first batch of 32
had not finished writing - so the queue started a **second identical B-1 against the same
output file**. Rows-on-disk tells you what *finished*. It never tells you what is running.

The obvious fix was worse. `pgrep -f "experiments/e01_gate.py"` matches nothing under Git
Bash on Windows, which cannot see a Windows process command line, so it returned no match,
the guard passed, and a **third** run started. A guard that fails silently is worse than
no guard, because it is trusted.

`kill` alone did not stop the duplicate either - it needed `kill -9`, and the parent shell
had to go with it.

**The rule as first written, and now known to be inert:**

```bash
LOCK="results/.queue.lock"
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then exit 1; fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT
```

**`kill -0` does not work here either, and for a subtler reason than `pgrep`.** MSYS keeps
its own process table and cannot see native Windows processes. Measured on this machine:

| probe | live Windows PID 17876 | absurd PID 999999 |
|---|---|---|
| `kill -0` | "not alive" | "not alive" |
| `tasklist` | found | missed |

`kill -0` reports *not alive* for both, so it never once detected a live owner. It only
appeared to work in testing when a stale lock happened to point at an MSYS PID — which is
to say, it was right by accident. `tasklist` reads the real Windows process table and
separates live from dead correctly:

```bash
alive () { tasklist //FI "PID eq $1" 2>/dev/null | grep -q "[0-9] $1 " }
```

**And the lock is not enough on its own, because a lock guards a QUEUE, not a RUN.** When
Task Scheduler aborted the chain (0x8007042B, ERROR_PROCESS_ABORTED) the bash chain died
but its python child survived as an orphan, still appending to the output file. A queue
started at that moment saw 31 rows against a target of 100, called B-8 incomplete, and
launched a **second python against the same file** — the original incident, recreated by
the guard written to stop it. Each run now also claims its own output file and records the
launching PID, and that claim is checked *before* the row count is consulted, because the
row count is exactly the thing that cannot tell "not started" from "already running".

And verify a guard by **running it in the state where it must refuse**, not by reading it.
Three broken versions of this guard looked correct. The claim-guard harness exercises four
cases — live owner must refuse, dead owner must proceed, empty claim must proceed, absent
claim must proceed — and the first version failed the live-owner case while passing the
other three.

### A run that writes nothing is not a hung run (2026-10-04)

Cost: about an hour of killing and relaunching a healthy job.

B-7's GPT-2 L10 leg sat at **0/50 rows** across several restarts and I read that as a hang.
Two separate mistakes stacked:

1. **My manual relaunches ran without `export PYTHONPATH=.`**, so they died instantly on
   `ModuleNotFoundError: No module named 'caliper'`. The queue's own runs set it and were
   fine — three of them at once, which is why the log showed interleaved model loads.
2. **A batch writes nothing until it finishes.** 50 neurons at 8k tokens took 164 minutes
   to produce the first row, which is entirely normal here.

So "0 rows" was reported by processes that had already exited. Two diagnostics would have
settled it in seconds instead of an hour:

```bash
grep -E "Traceback|ModuleNotFound" results/<run>.log   # did it even import?
tasklist //FI "PID eq <pid>"                            # is it still consuming CPU?
```

**Rule.** A row count is not a liveness check *in either direction* — it cannot tell you a
run is finished, and it cannot tell you a run is stuck. Check the log for a traceback and
check the process for CPU, then check the row count last. And in the queue log, **`exit 127`
means something killed the python from outside** (my `Stop-Process` did this repeatedly);
it is not a crash signature and should not be read as one.


### A truncated pytest run reports dots, and they look like a pass (2026-09-08)

The suite takes about 16 minutes — `test_estimator.py` and `test_batched.py` fit real
models. It does not fit a foreground call, and a backgrounded `timeout 900 pytest
tests -q` came back with nine dots and a zero exit. Nine dots is not nine passes: the
suite collects **14**, pytest never reached its summary line, and the dots were
progress output frozen at the wall.

I quoted "9 passed" and pushed on it. The change was in fact fine — 11 passed in the
two patched files, 3 in `test_runtime.py`, 14/14 — but that was established after the
push, not before.

**Rule.** A pytest result is the summary line (`N passed in Ms`), never the dots. If
there is no summary line, the run did not finish. Run the suite as two calls —
`test_estimator.py test_batched.py` (~16 min, background) and `test_runtime.py`
(~25 s) — and read both summaries.


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

### A sweep over layers may be paired without anyone noticing, and it changes the answer
B-7 swept L2/L6/L10 and the aggregate said "two regimes, not a gradient" — shallow layers
just noisier. That reading was pushed before anyone asked whether the three legs ran the
same units. They did: the draw depends only on the seed and `d_model`, both fixed across
layers. Paired, the failure sets turn out to be near-disjoint — **0 of 50 units fail at all
three**, 21 of 26 L2 failures pass at both deeper layers, and 3 fail in the reverse
direction. The aggregate was averaging over populations that were not the same population.

The tell was available in the data all along: identical unit keys across files, and an L2→L6
jump large enough to look like a regime change. **Standing rule: before interpreting a
multi-leg sweep, check whether the legs are paired (same ids, same draw) and compare the
per-unit overlap, not just the rates.** Overlapping confidence intervals are not evidence
that two groups are alike — here they overlap because the failure sets are the same
*size*, and it was the wrong conclusion that hid the real finding.

Second lesson, same run: the same comparison also answers what restart count does. Two runs
at the same layer differing only in restarts agreed on 40/50 verdicts, so **20% of verdicts
flip on restart count alone**. Nothing else in the notebook bounds the noise floor this
cheaply, and it should be measured for every gate before its pass rate is quoted as a
property of the units rather than of the draw.

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
