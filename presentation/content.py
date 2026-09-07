"""All slide text for the Review 1 deck, in one place.

Wording marked CANON is quoted verbatim from the repository and must not be
paraphrased when edited:
  - PROBLEM_STATEMENT  -> docs/sem5-review-plan.md ("use verbatim, every review")
  - NOVELTY            -> docs/review1-notes.md section 6
  - DO_NOT_CLAIM       -> docs/review-reference.md section 35
  - RESOURCES          -> docs/review1-notes.md section 5

Review 1 scope: problem, prior work, use cases, workflow, plan, references.
NO RESULTS. docs/sem5-review-plan.md is explicit: "Do not say: anything about
Study 3's results, the arXiv preprint, or the 77/100." build_deck.py enforces
this with a text guard that fails the build.
"""

# ---------------------------------------------------------------- title slide

COURSE = "UE23CS320A – Capstone Project Approval"
TITLE = "CALIPER: Calibrating Interpretability Readouts for Language Models"
SHORT_TITLE = "CALIPER: Calibrating Interpretability Readouts for Language Models"

PROJECT_ID = "[PROJECT ID]"      # format encodes the guide's initials - ask
GUIDE = "[GUIDE NAME]"           # docs refer to the mentor only as "she"

TEAM = [
    "SANTOSH CHEETHIRALA",
    "NIVAS REDDY DANDU",
    "C. SREE KRISHNA KOUSHIK",
    "MARREDDY RUSHI ESWAR REDDY",
]
TEAM_SRN = [
    "PES1UG24CS127",
    "PES1UG24AM075",
    "PES1UG24CS128",
    "PES1UG24AM158",
]
FOOTER = "_".join(TEAM)

# ------------------------------------------------------------------- outline

OUTLINE = [
    "Introduction",
    "Problem Statement",
    "Work Flow",
    "Scope and Feasibility study",
    "Applications / Use cases",
    "Background Study",
    "Project Timeline and Team Effort Plan",
    "References",
]

# -------------------------------------------------------------- introduction

INTRO = [
    "A language model is made of billions of numbers. Almost none of them "
    "have a known meaning.",
    "Interpretability is the field that tries to read those numbers — to say "
    "“this unit responds to X”, “this direction is the model’s honesty "
    "trait”, “the model can report on its own internal state”.",
    "Every one of those is a reading, produced by an instrument.",
    "And an instrument is only worth what its calibration is worth.",
]

INTRO_ANALOGY = [
    "A thermometer is trusted because someone once put it in ice water and in "
    "boiling water, and checked that it read 0 and 100.",
    "Only then was it used on a patient, where the true temperature is not "
    "known in advance.",
    "Nobody has done this for the tools that read language models.",
    "We build the ice water and the boiling water.",
]

# CANON - docs/sem5-review-plan.md, "use verbatim, every review"
PROBLEM_STATEMENT = (
    "Interpretability tools claim to read a model’s mind: which direction a "
    "neuron responds to, which vector encodes a personality trait, whether the "
    "model can report on its own internal states. None of these tools has ever "
    "been checked against a case where the right answer was already known. "
    "Neuroscience solved this a century ago with the preparation — the squid "
    "axon, the sea slug, the worm — a system simple enough that a new "
    "instrument could be tested against a known answer before being trusted on "
    "an unknown one. We build preparations for language models at three levels "
    "— unit, trait, self — and use them to measure how often each reading "
    "tool is right, how often it is silently wrong, and what it takes to tell "
    "the difference."
)

PROBLEM_TAGLINE = (
    "Does the mind-reading kit work? "
    "Building preparations for language models."
)

# three tools, three levels, zero checks
LEVELS = [
    ("UNIT", "Subspace / probe search",
     "“This neuron responds to concept X”",
     "Weights are known, but never used as truth"),
    ("TRAIT", "Persona / steering vectors",
     "“This vector is the model’s honesty trait”",
     "Validated by steering effect only"),
    ("SELF", "Introspective prompting",
     "“The model detects an injected thought”",
     "Controlled against no injection only"),
]

# --------------------------------------------------------- neuroscience frame

PREPARATION = [
    "In biology a preparation is the system you choose because it makes an "
    "otherwise impossible measurement tractable.",
    "The squid giant axon was picked because it is wide enough to put an "
    "electrode inside. Aplysia, because it has large countable neurons. "
    "C. elegans, because every one of its 302 neurons could be mapped.",
    "In each case the instrument was proven where the answer was knowable "
    "before it was trusted where it was not.",
    "A language model is the first system where all three of our questions "
    "have a knowable answer.",
]

GROUND_TRUTH = [
    "For an MLP neuron, the pre-activation is exactly w · s — the dot product "
    "of its own input weight column with its layer’s residual stream.",
    "So the neuron’s true direction is not estimated. It is read off the "
    "weights.",
    "This is true by construction. It is not a discovery about models — it is "
    "a test bed, and the contribution is noticing it can be used as one.",
    "At the trait and self levels there is no free truth, so we plant it: "
    "inject a known direction or a known state, then ask the tool to find it.",
]

# ------------------------------------------------------------------ workflow

PIPELINE = [
    ("Stimulus", "Public-domain text drives the model"),
    ("Capture", "Record activations at a chosen layer"),
    ("Readout", "Run the tool under test"),
    ("Compare", "Score against the known answer"),
    ("Calibrate", "Null, required-N, interval, disagreement"),
    ("Report", "One error rate per tool, per level"),
]

WORKFLOW_NOTE = (
    "Every study reports the same four quantities, so the three levels read as "
    "one instrument characterised at three depths rather than three separate "
    "projects."
)

# --------------------------------------------------------------------- scope

SCOPE_IN = [
    "Three levels of readout: unit, trait, self.",
    "Open-weight models only — GPT-2 small for the unit level, one ≥ 27B "
    "instruct model for trait and self.",
    "English text, public-domain corpora.",
    "Readouts that are already published. We test instruments, we do not "
    "invent them.",
]

SCOPE_OUT = [
    "No claim about machine consciousness. The terms are introspective "
    "accuracy and self-report validity.",
    "No training-time work — every experiment is inference only.",
    "No proprietary or API-only models; results must be reproducible by anyone.",
    "Not a benchmark leaderboard. We measure error rates, not rankings.",
]

# --------------------------------------------------------------- feasibility

FEASIBILITY = [
    ("Model availability",
     "GPT-2 small, Gemma3-27B-it and Qwen2.5-32B-Instruct are all open-weight "
     "and free to obtain."),
    ("Ground truth",
     "Free at the unit level — it is already in the weights. Planted at the "
     "trait and self levels using published injection methods."),
    ("Compute",
     "Unit level runs on a laptop CPU. Large-model work runs on Kaggle’s free "
     "tier, 2× Tesla T4, 30 GPU-hours per week. The pipeline checkpoints and "
     "resumes."),
    ("Methods",
     "The estimators are decades old and published; the injection and persona "
     "methods are public. Nothing needs to be invented before measurement can "
     "start."),
]

FEASIBILITY_TAG = "No personal data. No proprietary data. No paid compute."

# ------------------------------------------------------------------ use cases
# NOTE: the source text in review1-notes.md section 3 embeds Study 3 results.
# Those are stripped here - Review 1 states the motivation, not the evidence.

USE_CASES = [
    ("AI safety audits",
     "Regulators and labs increasingly check for hidden traits — deception, "
     "sycophancy, bias — by reading a model’s internals. Every such audit "
     "rests on a readout whose error rate nobody has measured."),
    ("Trusting a model’s self-report",
     "Models are asked “are you being honest?” and the answer is treated as "
     "evidence. Whether that report tracks the model’s actual state has never "
     "been tested against a planted state."),
    ("Model debugging",
     "When a model misbehaves, engineers hunt for the responsible direction. A "
     "tool that points at the wrong one sends them down the wrong path, and "
     "gives no sign it has done so."),
    ("Persona and character products",
     "Shipped models carry tuned personalities controlled by persona vectors. "
     "Whether such a vector captures the trait or a mixture of unrelated "
     "things has never been checked against a planted case."),
    ("Reproducibility for the field",
     "Every experiment runs on a free account. Code, data and pre-registration "
     "dates are published, so anyone can repeat the measurement."),
    ("Foundational",
     "Neuroscience learned the hard way that reading tools need preparations. "
     "Transferring that discipline is the intellectual contribution."),
]

# ----------------------------------------------------------------- literature

LIT_STRANDS = [
    ("Unit level", 12, "Tuning curves and probes",
     "Rich methods, no ground-truth validation"),
    ("Trait level", 11, "Persona and steering vectors",
     "Validated by effect, never by recovery"),
    ("Self level", 9, "Introspection and self-report",
     "Controlled against no injection only"),
    ("Neuroscience", 8, "Estimators and calibration",
     "Mature discipline, never transferred"),
]

LIT_UNIT = [
    ("Maximally Informative Dimensions",
     "Sharpee, Rust & Bialek (2004), Neural Computation 16(2)",
     "Introduced the information-theoretic estimator for what a neuron "
     "responds to; the method we adapt to language-model units."),
    ("Equivalence of MID and maximum likelihood",
     "Williamson, Sahani & Pillow (2015), PLoS Comput Biol 11(4)",
     "Showed MID is maximum likelihood for a linear-nonlinear model, which "
     "makes it a rank-K bottleneck regression — the form we implement."),
    ("Zoom In: An Introduction to Circuits",
     "Olah et al. (2020), Distill",
     "Established tuning-curve style analysis of individual units in vision "
     "models. Prior art we differ from: vision, and no ground-truth check."),
    ("Sparse coding and population sparseness",
     "Willmore & Tolhurst (2001), Network 12(3)",
     "Lifetime kurtosis is not population sparseness — a caution we inherit "
     "when characterising unit selectivity."),
]

LIT_TRAIT = [
    ("Persona vectors",
     "Chen et al. (2025), arXiv:2507.21509",
     "Extracts trait directions by difference of means and steers behaviour "
     "with them; validated by steering effect, never by recovering a known "
     "direction."),
    ("Steering language models with activation engineering",
     "Turner et al. (2023), arXiv:2308.10248",
     "Established activation addition as a control method; the family of "
     "readouts our trait level puts under test."),
    ("Convergent and discriminant validation",
     "Campbell & Fiske (1959), Psychological Bulletin 56(2)",
     "The multitrait–multimethod matrix, the psychology standard for "
     "validating a trait measure. Level 2 adopts it and adds a truth column."),
    ("Construct validity in psychological tests",
     "Cronbach & Meehl (1955), Psychological Bulletin 52(4)",
     "Defines what it means for an instrument to measure the construct it "
     "claims to; the question we ask of persona vectors."),
]

LIT_SELF = [
    ("Emergent introspective awareness",
     "Macar, Yang, Wang, Wallich, Ameisen & Lindsey (2026), Anthropic",
     "Reports that a model can detect an injected concept in its own "
     "activations. Controlled against no injection; the control we add is an "
     "injection with no content."),
    ("Telling more than we can know",
     "Nisbett & Wilson (1977), Psychological Review 84(3)",
     "People confabulate reasons for their own behaviour. The direct parallel "
     "for model self-reports, and never measurable against a planted cause."),
    ("Consciousness in artificial intelligence",
     "Butlin et al. (2023), arXiv:2308.08708",
     "Fourteen indicator properties; introspective accuracy is the only one "
     "where the internal state can be constructed and the report scored."),
]

LIT_NEURO = [
    ("Spike-triggered average and covariance",
     "Schwartz, Pillow, Rust & Simoncelli (2006), J Vision 6(4)",
     "The classical closed-form estimators. We run them as baselines to show "
     "which assumptions fail on a residual stream."),
    ("Bussgang’s theorem and linear recovery",
     "Bussgang (1952), MIT RLE Technical Report 216",
     "Guarantees that ridge regression recovers a direction under Gaussian "
     "stimuli — an assumption the residual stream violates."),
    ("Calibration before trust",
     "Systems-neuroscience practice, 1939–present",
     "Null models, required sample size, and instrument validation on a "
     "preparation. The package we transfer wholesale."),
]

# ------------------------------------------------------- differences / novelty

COLLISIONS = [
    ("Tuning curves on vision models",
     "Distill Circuits (2020–21)",
     "They characterise units in vision networks. We work on language models "
     "and, unlike them, score every reading against a known direction."),
    ("Number representation tuning",
     "Cacioli (2026)",
     "Overlapping method, different target. Their question is what a unit "
     "encodes; ours is whether the instrument that answers that question is "
     "correct."),
    ("Introspection with a no-injection control",
     "Macar et al. (2026)",
     "Same model, same layer, same prompt. They control against no injection; "
     "we control against an injection carrying no content."),
]

# CANON - docs/review1-notes.md section 6
NOVELTY = [
    "Free ground truth at the unit level — an MLP neuron’s true direction is "
    "its own weight column, so a real model can serve as its own preparation.",
    "The content-free control at the self level — controlling against an "
    "injection with no content, rather than against no injection.",
    "The transfer of the calibration discipline — null, required sample size, "
    "pre-registered interval and disagreement flag as one package.",
]

# CANON - docs/review-reference.md section 35, trimmed to four for the slide
DO_NOT_CLAIM = [
    ("We do not say the neuroscience methods “don’t work.”",
     "They do not transfer here, and we name the two assumptions that break."),
    ("We do not say persona vectors are wrong.",
     "We have not tested them yet. Same family of method, never checked."),
    ("We do not claim the validity framing is new.",
     "About twelve 2026 papers already use it. The preparation is what we add."),
    ("We do not claim a model can or cannot introspect.",
     "We claim the published protocol cannot distinguish introspection from "
     "noticing a disturbance."),
]

NOT_NEW = (
    "Tuning curves have been done on vision models. The neuroscience "
    "estimators are decades old. Persona vectors and concept injection are "
    "published methods. We are testing them, not inventing them."
)

# ------------------------------------------------------------------- timeline

GANTT = [
    ("Literature survey and framing", 0.0, 1.6),
    ("Unit-level preparation", 0.6, 2.2),
    ("Calibration protocol", 1.4, 2.4),
    ("Self-level preparation", 2.0, 3.6),
    ("Trait-level preparation", 3.4, 5.2),
    ("Cross-level analysis", 4.6, 6.4),
    ("Thesis and defence", 6.2, 8.0),
]
GANTT_TICKS = ["Sep 26", "Nov 26", "Jan 27", "Mar 27", "May 27",
               "Sep 27", "Nov 27", "Mar 28", "May 28"]

REVIEWS = [
    ("Review 1", "Early Sep", "Problem, team, literature survey"),
    ("Review 2", "Late Sep", "Requirements and system design"),
    ("Review 3", "Late Oct", "Initial prototype, unit level"),
    ("Review 4", "Mid Nov", "Prototype on a real question; sem 6 plan"),
]

TRACKS = [
    ("Estimator and integration",
     "SANTOSH CHEETHIRALA",
     "Readout implementation, the measurement pipeline, cross-level "
     "integration"),
    ("Literature and framing",
     "[MEMBER B]",
     "The paper matrix, the neuroscience method map, related-work sections"),
    ("Second-model replication",
     "[MEMBER C]",
     "Independent replication of every self-level result on a second open "
     "model"),
    ("Calibration and benchmarking",
     "[MEMBER D]",
     "Planted-latent networks, false-discovery rates, statistics. CPU only"),
]

STAGES = [
    ("Weeks 1–4", "Problem definition, proposal audit, mentor selection"),
    ("Weeks 5–10", "Literature survey across four strands"),
    ("Weeks 11–14", "Requirements, system design, calibration protocol"),
    ("Weeks 15–17", "Initial prototype and review preparation"),
]

# ------------------------------------------------------------------ resources

RESOURCES = [
    ("GPT-2 small", "HuggingFace, open, 2019",
     "Runs on a laptop CPU; neuron directions readable from the weights"),
    ("Gemma3-27B-it", "Google, open weights, via Kaggle",
     "The model used in the published introspection work"),
    ("Qwen2.5-32B-Instruct", "Alibaba, open weights",
     "Ungated; second model for robustness"),
    ("Project Gutenberg", "Public domain",
     "Stimulus text; no personal data, no licensing issues"),
    ("PyTorch, Transformers", "Open source", "Model loading and hooks"),
    ("NumPy, SciPy", "Open source", "The estimator and all statistics"),
    ("Kaggle, 2× Tesla T4", "Free tier",
     "Every large-model experiment; 30 GPU-hours per week"),
]

# ----------------------------------------------------------------- references

REFERENCES = [
    "Sharpee, T., Rust, N. C., & Bialek, W. (2004). Analyzing neural responses "
    "to natural signals: maximally informative dimensions. Neural Computation, "
    "16(2), 223–250.",
    "Williamson, R. S., Sahani, M., & Pillow, J. W. (2015). The equivalence of "
    "information-theoretic and likelihood-based methods for neural "
    "dimensionality reduction. PLoS Computational Biology, 11(4), e1004141.",
    "Olah, C., Cammarata, N., Schubert, L., Goh, G., Petrov, M., & Carter, S. "
    "(2020). Zoom In: An Introduction to Circuits. Distill, 5(3).",
    "Willmore, B., & Tolhurst, D. J. (2001). Characterizing the sparseness of "
    "neural codes. Network: Computation in Neural Systems, 12(3), 255–270.",
    "Chen, R., et al. (2025). Persona vectors: monitoring and controlling "
    "character traits in language models. arXiv preprint arXiv:2507.21509.",
    "Turner, A. M., et al. (2023). Steering language models with activation "
    "engineering. arXiv preprint arXiv:2308.10248.",
    "Campbell, D. T., & Fiske, D. W. (1959). Convergent and discriminant "
    "validation by the multitrait–multimethod matrix. Psychological "
    "Bulletin, 56(2), 81–105.",
    "Cronbach, L. J., & Meehl, P. E. (1955). Construct validity in "
    "psychological tests. Psychological Bulletin, 52(4), 281–302.",
    "Nisbett, R. E., & Wilson, T. D. (1977). Telling more than we can know: "
    "verbal reports on mental processes. Psychological Review, 84(3), 231–259.",
    "Butlin, P., et al. (2023). Consciousness in artificial intelligence: "
    "insights from the science of consciousness. arXiv:2308.08708.",
    "Schwartz, O., Pillow, J. W., Rust, N. C., & Simoncelli, E. P. (2006). "
    "Spike-triggered neural characterization. Journal of Vision, 6(4), 484–507.",
]

# ---------------------------------------------------------- any other info

OTHER_INFO = [
    "The problem was selected on 18 August 2026 from a set of seven candidate "
    "proposals. The mentor chose this one for its neuroscience framing.",
    "One earlier framing was audited and killed before any work was committed "
    "to it — a published theorem and two prior papers made the claim "
    "untenable. Reporting that is part of the method.",
    "Every experiment is pre-registered: the criterion is written down and "
    "committed before the run, so a result cannot be tuned after the fact.",
    "No personal data. No proprietary data. No paid compute.",
]
