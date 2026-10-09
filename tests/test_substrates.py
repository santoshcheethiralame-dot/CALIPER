"""Flagship substrates: the recorded response must be an exact function of the recorded stimulus
and the reference direction, or the substrate has no ground truth.

    F-3  OPT ReLU neurons: relu(s . w + b), w the fc1 row
    F-6  unembedding rows: logit_t = s . W_U[t]
The GPT-2 MLP path must stay as it was."""
import numpy as np
import pytest
import torch

from caliper.activations import _blocks, _mlp_in, collect, load_model

TEXTS = ["The river ran past the old mill, and the miller watched it every morning. " * 6]


def _have(name):
    from huggingface_hub import try_to_load_from_cache
    return try_to_load_from_cache(name, "config.json") is not None


def _check_mlp(name, layer, act):
    m, t = load_model(name)
    neurons = np.array([3, 41, 300])
    p = collect(m, t, TEXTS, layer=layer, neurons=neurons, max_tokens=200, seq_len=64, batch_size=1)
    b = _mlp_in(_blocks(m)[layer]).bias.detach().numpy()[neurons]
    pre = p.stimulus @ p.weights + b
    assert np.abs(act(torch.as_tensor(pre)).numpy() - p.response).max() < 1e-4
    return p


@pytest.mark.skipif(not _have("gpt2"), reason="gpt2 not cached")
def test_gpt2_mlp_unchanged():
    _check_mlp("gpt2", 6, torch.nn.functional.gelu)


@pytest.mark.skipif(not _have("gpt2"), reason="gpt2 not cached")
def test_gpt2_unembed_identity():
    m, t = load_model("gpt2")
    tokens = np.array([11, 262, 1000, 30000])
    p = collect(m, t, TEXTS, layer=0, neurons=tokens, max_tokens=200, seq_len=64, batch_size=1,
                target="unembed")
    assert p.weights.shape == (768, 4)
    assert np.abs(p.stimulus @ p.weights - p.response).max() < 1e-3
    # The stimulus is what the head reads: the model's own logits for the tokens are
    # reproduced up to row order, which collect shuffles.
    ids = torch.tensor([t(TEXTS[0])["input_ids"][:64]])
    with torch.no_grad():
        logits = m(ids).logits[0, 1:][:, torch.as_tensor(tokens)].numpy()
    assert np.allclose(np.sort(logits, 0), np.sort(p.response[:63], 0), atol=1e-3)


@pytest.mark.skipif(not _have("facebook/opt-125m"), reason="opt-125m not cached")
def test_opt_relu_identity():
    p = _check_mlp("facebook/opt-125m", 6, torch.relu)
    assert (p.response == 0).mean() > 0.3          # ReLU units are sparse


@pytest.mark.skipif(not _have("google/gemma-3-270m"), reason="gemma-3-270m not cached")
def test_gemma3_glu_identity():
    m, t = load_model("google/gemma-3-270m")
    layer, neurons = 6, np.array([5, 77, 900])
    p = collect(m, t, TEXTS, layer=layer, neurons=neurons, max_tokens=200, seq_len=64,
                batch_size=1, target="glu")
    assert p.weights.shape == (m.config.hidden_size, 2, 3)
    mlp = _blocks(m)[layer].mlp
    got = {}
    h = mlp.down_proj.register_forward_pre_hook(lambda _m, a: got.__setitem__("x", a[0].detach()))
    try:
        with torch.no_grad():
            m(torch.tensor([t(TEXTS[0])["input_ids"][:64]]))
    finally:
        h.remove()
    real = got["x"][0, 1:, :][:, torch.as_tensor(neurons)].float().numpy()
    assert np.allclose(np.sort(real, 0), np.sort(p.response[:63], 0), atol=1e-4)
    g, u = p.weights[:, 0, :], p.weights[:, 1, :]
    act = mlp.act_fn
    pred = (act(torch.as_tensor(p.stimulus @ g)) * torch.as_tensor(p.stimulus @ u)).numpy()
    assert np.abs(pred - p.response).max() < 1e-4
