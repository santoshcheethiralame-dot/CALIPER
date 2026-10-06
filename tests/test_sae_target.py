"""SAE latents as a reference standard: the response must be exactly the latent's ReLU
activation, and its pre-activation must be exactly linear in the stimulus along the
mean-removed encoder column. If either fails, the transfer experiment has no ground truth.
"""

import numpy as np
import pytest
import torch

pytest.importorskip("safetensors")


@pytest.fixture(scope="module")
def probe():
    from caliper.activations import collect, load_model, sample_corpus
    from caliper.sae import load_sae
    try:
        sae = load_sae(6)
    except Exception as exc:  # no network and no cache
        pytest.skip(f"SAE weights unavailable: {exc}")
    model, tok = load_model("gpt2")
    latents = np.array([775, 4210, 9657])
    p = collect(model, tok, sample_corpus(n_docs=20, seed=0), layer=6, neurons=latents,
                max_tokens=1000, seed=0, sae=sae)
    return p, sae, latents


def test_stimulus_is_mean_centred(probe):
    p, _, _ = probe
    assert np.abs(p.stimulus.mean(1)).max() < 1e-4


def test_response_is_the_latent_activation(probe):
    p, sae, latents = probe
    z = (p.stimulus - sae["b_dec"]) @ sae["W_enc"][:, latents] + sae["b_enc"][latents]
    assert np.allclose(p.response, np.maximum(z, 0), atol=1e-4)


def test_reference_direction_is_exact(probe):
    p, sae, latents = probe
    z = (p.stimulus - sae["b_dec"]) @ sae["W_enc"][:, latents] + sae["b_enc"][latents]
    for j in range(len(latents)):
        proj = p.stimulus @ p.weights[:, j]
        assert np.corrcoef(proj, z[:, j])[0, 1] > 0.999999
        assert abs(p.weights[:, j].mean()) < 1e-6   # the identifiable part only
