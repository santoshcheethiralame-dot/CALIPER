"""Level-2 harness: the exact readout reference, the planted fixture, and the T-0 contract.

The passing tests check what is implemented. The xfail tests are the team's definition of
done for the T-0 stubs. Remove each xfail when its function lands.
"""
import numpy as np
import pytest
import torch

from caliper import traits


@pytest.fixture(scope="module")
def gpt2():
    from caliper.activations import load_model
    return load_model("gpt2")


def final_residuals(model, tok, text):
    """Pre-final-norm residuals, captured on the norm's input."""
    grab = {}
    h = traits.final_norm(model).register_forward_pre_hook(
        lambda m, args: grab.__setitem__("x", args[0].detach()))
    try:
        with torch.no_grad():
            out = model(**tok(text, return_tensors="pt"))
    finally:
        h.remove()
    return grab["x"][0].float().numpy(), out.logits[0].float().numpy()


def test_readout_direction_is_exact_on_gpt2(gpt2):
    model, tok = gpt2
    a = tok.encode(" good")[:1] + tok.encode(" great")[:1]
    b = tok.encode(" bad")[:1] + tok.encode(" awful")[:1]
    x, logits = final_residuals(model, tok, "The film was long, and by the end I thought it")
    gap_model = logits[:, a].mean(1) - logits[:, b].mean(1)
    # The module-level reconstruction matches the model's own forward pass...
    assert np.allclose(traits.readout_logit_gap(model, a, b, x), gap_model, atol=1e-3)
    # ...and w reproduces it: gap = (w . x) / sigma(x) + dU . beta.
    w = traits.readout_direction(model, a, b)
    norm = traits.final_norm(model)
    U = traits.unembedding(model)
    du = U[a].mean(0) - U[b].mean(0)
    sigma = np.sqrt(x.var(1) + norm.eps)
    pred = (x @ w) / sigma + du @ norm.bias.detach().numpy()
    assert np.max(np.abs(pred - gap_model)) < 1e-3 * np.abs(gap_model).max() + 1e-4
    assert np.corrcoef(pred, gap_model)[0, 1] > 0.999999


def test_readout_direction_has_no_null_component(gpt2):
    model, tok = gpt2
    w = traits.readout_direction(model, tok.encode(" yes")[:1], tok.encode(" no")[:1])
    assert abs(w @ traits.ln_null(len(w))) < 1e-6 * np.linalg.norm(w)


def test_diffmean_recovers_planted_trait():
    a, b, w, _ = traits.planted_fixture(n=4000, d=64, snr=4.0)
    v = traits.diffmean(a, b)
    assert abs(v @ w) / np.linalg.norm(v) > 0.99


def test_fixture_anisotropy_spreads_the_spectrum():
    _, _, _, flat = traits.planted_fixture(anisotropy=0.0)
    _, _, _, steep = traits.planted_fixture(anisotropy=0.1)
    ev = np.linalg.eigvalsh
    assert ev(flat).max() / ev(flat).min() < 1.01
    assert ev(steep).max() / ev(steep).min() > 100


# ---- the T-0 contract: xfail until implemented ----------------------------------------

todo = pytest.mark.xfail(raises=NotImplementedError, strict=True,
                         reason="T-0 stub; remove this mark when implemented")


@todo
def test_lda_recovers_planted_trait_under_anisotropy():
    a, b, w, _ = traits.planted_fixture(n=4000, anisotropy=0.1)
    v = traits.lda_direction(a, b)
    assert abs(v @ w) / np.linalg.norm(v) > 0.95


@todo
def test_logistic_probe_recovers_planted_trait():
    a, b, w, _ = traits.planted_fixture(n=4000)
    v = traits.logistic_probe_direction(a, b)
    assert abs(v @ w) / np.linalg.norm(v) > 0.95


@todo
def test_whitening_is_idempotent():
    _, _, w, cov = traits.planted_fixture(anisotropy=0.1)
    once = traits.whiten(w, cov)
    assert np.allclose(traits.whiten(once, np.eye(len(w))), once)


@todo
def test_agreement_checks_run_on_a_two_trait_toy():
    a, b, _, _ = traits.planted_fixture(n=200)
    assert 0.0 <= traits.split_half_cosine(a, b) <= 1.0
    assert 0.0 <= traits.bootstrap_cosine(a, b, n_boot=20) <= 1.0


@todo
def test_functional_checks_run_on_a_two_trait_toy():
    a, b, w, _ = traits.planted_fixture(n=200)
    assert traits.heldout_probe_accuracy(w, a, b) > 0.9
    assert traits.separation_snr(w, a, b) > 1.0


@todo
def test_contrastive_prompts_are_balanced():
    t = traits.Trait("sentiment", "sentiment", [1], [2], ["I thought the film was {}"])
    pa, pb = traits.contrastive_prompts(t, n=8)
    assert len(pa) == len(pb) == 8
