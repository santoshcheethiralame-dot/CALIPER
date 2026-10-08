"""Level 2: trait directions and the checks used to vouch for them (T-0 harness).

The level-2 question is the level-1 question asked of trait vectors: when a trait direction
is extracted from contrastive prompts, which ground-truth-free check notices that it is
wrong? The reference (docs/RUN_PLAN_L2_L3.md, T-2) is exact at the final layer. A trait
expressed as a choice between answer-token sets A and B is read by

    w = P(g * (mean_{a in A} U_a - mean_{b in B} U_b))

U is the unembedding, g the final norm gain, and P removes the final norm's null direction
(the all-ones vector for LayerNorm, which centres its input; nothing for RMSNorm).
`readout_direction` computes it and `tests/test_traits.py` checks it against the model's own
logits.

Implemented here:
- the exact reference;
- the LayerNorm-null projection;
- DiffMean;
- the synthetic planted-trait fixture.

Everything else is a typed stub that raises NotImplementedError. The xfail tests in
tests/test_traits.py define what "done" means for each one (T-0, run plan section 2). Rules
from P1/P1b that bind every implementation:
1. Every reference is a direction the model provably reads or writes.
2. Every cosine is reported raw and whitened, against a null built in the same space.
"""
from dataclasses import dataclass, field

import numpy as np
import torch


# ----------------------------------------------------------------------------------------
# Model plumbing


def final_norm(model):
    """The norm applied to the last residual stream before the unembedding."""
    for path in ("transformer.ln_f", "gpt_neox.final_layer_norm", "model.norm"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise ValueError(f"no final norm found on {type(model).__name__}")


def unembedding(model):
    """(vocab, d_model) unembedding matrix. GPT-2 ties it to the input embedding."""
    return model.get_output_embeddings().weight.detach().float().cpu().numpy()


def centres_input(norm):
    """LayerNorm subtracts the mean (so the all-ones direction is unreadable); RMSNorm does
    not. Decided from the module itself, not the model name."""
    name = type(norm).__name__
    return isinstance(norm, torch.nn.LayerNorm) or ("LayerNorm" in name and "RMS" not in name)


# ----------------------------------------------------------------------------------------
# The exact reference


def ln_null(d):
    """Unit all-ones direction: the component a centring norm removes from its input."""
    return np.ones(d) / np.sqrt(d)


def project_out(v, u):
    u = u / np.linalg.norm(u)
    return v - (v @ u) * u


def readout_direction(model, tokens_a, tokens_b):
    """Exact residual-stream direction that reads "A rather than B" at the final layer.

    With z = norm(x), the mean logit of set A minus that of set B is
    (mean U_A - mean U_B) . (g * z_hat) + const, where z_hat is the centred (LayerNorm) or
    raw (RMSNorm) input divided by its per-token scale. As a function of x, up to that
    positive per-token scale, the readout is w . x, with w = P(g * dU). The all-ones
    component of w is unreadable through a LayerNorm and is removed.
    """
    U = unembedding(model)
    norm = final_norm(model)
    g = norm.weight.detach().float().cpu().numpy()
    du = U[list(tokens_a)].mean(0) - U[list(tokens_b)].mean(0)
    w = g * du
    if centres_input(norm):
        w = project_out(w, ln_null(len(w)))
    return w


def readout_logit_gap(model, tokens_a, tokens_b, x):
    """Mean logit of A minus mean logit of B, computed by the model's own final norm and
    unembedding from pre-norm residuals x (n, d). The quantity w must reproduce."""
    norm = final_norm(model)
    U = torch.as_tensor(unembedding(model))
    with torch.no_grad():
        z = norm(torch.as_tensor(x, dtype=norm.weight.dtype)).float()
        logits = z @ U.T
    return (logits[:, list(tokens_a)].mean(1) - logits[:, list(tokens_b)].mean(1)).numpy()


# ----------------------------------------------------------------------------------------
# Traits and extraction


@dataclass
class Trait:
    """A trait as a choice between two answer-token sets, with the contexts that elicit it.

    Token sets are trait-specific on purpose: A/B multiple choice would give every trait the
    same final-layer direction (run plan, T-2)."""
    name: str
    family: str                          # sentiment | yes-no | topic | language-register
    tokens_a: list
    tokens_b: list
    templates: list = field(default_factory=list)


def diffmean(acts_a, acts_b):
    """Difference of class means, the DiffMean / CAA / persona-vector estimator."""
    return np.asarray(acts_a).mean(0) - np.asarray(acts_b).mean(0)


def planted_fixture(n=512, d=64, snr=4.0, anisotropy=0.0, seed=0):
    """Synthetic activations with a known planted trait direction.

    Class A = noise + (snr / 2) w, class B = noise - (snr / 2) w, with unit w. With
    `anisotropy` > 0, the noise covariance has a spread spectrum (exp(-anisotropy * i)), the
    regime where raw and whitened cosines part ways (P1b's lesson).
    Returns (acts_a, acts_b, w, noise_cov).
    """
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(d)
    w /= np.linalg.norm(w)
    scales = np.exp(-anisotropy * np.arange(d))
    basis, _ = np.linalg.qr(rng.standard_normal((d, d)))
    cov = (basis * scales**2) @ basis.T

    def draw(sign):
        z = rng.standard_normal((n, d)) * scales
        return z @ basis.T + sign * (snr / 2) * w

    return draw(+1), draw(-1), w, cov


# ----------------------------------------------------------------------------------------
# Stubs: T-0 work still to do. Each has an xfail test in tests/test_traits.py.


def contrastive_prompts(trait, n, seed=0):
    """Prompts where the model predicts trait.tokens_a vs tokens_b, from trait.templates.
    Returns (prompts_a, prompts_b). T-1 needs a fully enumerable grid of these."""
    raise NotImplementedError("T-0: contrastive prompt generation")


def logistic_probe_direction(acts_a, acts_b, l2=1.0):
    """Weight vector of an L2-regularised logistic probe, A vs B."""
    raise NotImplementedError("T-0: logistic probe route")


def lda_direction(acts_a, acts_b, shrinkage=None):
    """Whitened (LDA) direction: Sigma^-1 (mu_a - mu_b), with optional shrinkage."""
    raise NotImplementedError("T-0: LDA route")


def whiten(v, cov, eps=1e-6):
    """Map a direction into the whitened space of `cov`. Must be idempotent on an already
    whitened covariance (tests/test_traits.py)."""
    raise NotImplementedError("T-0: covariance whitening")


def causal_inner_product(model):
    """Park et al.'s causal inner product, Cov(U)^-1 from the unembedding, for comparing
    directions in the space the model reads them in."""
    raise NotImplementedError("T-0: causal inner product")


def split_half_cosine(acts_a, acts_b, route=diffmean, seed=0):
    """Agreement check: |cos| between directions from two random halves of the prompts."""
    raise NotImplementedError("T-0: split-half agreement check")


def bootstrap_cosine(acts_a, acts_b, route=diffmean, n_boot=100, seed=0):
    """Agreement check: median |cos| between bootstrap-resampled directions and the full
    direction."""
    raise NotImplementedError("T-0: bootstrap agreement check")


def heldout_probe_accuracy(direction, acts_a, acts_b):
    """Functional check: accuracy of sign(x . direction - threshold) on held-out prompts,
    with the threshold fitted on the training split."""
    raise NotImplementedError("T-0: held-out probe check")


def separation_snr(direction, acts_a, acts_b):
    """Functional check: class separation along the direction, (mu_a - mu_b) / pooled sd
    of the projections."""
    raise NotImplementedError("T-0: separation check")


def save_direction(path, direction, **meta):
    """Write a direction with its provenance (trait, route, layer, prompts, seed) as .npz.
    Every run saves its directions (the C57 lesson)."""
    raise NotImplementedError("T-0: direction saving")
