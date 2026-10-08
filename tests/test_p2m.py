"""P2-M's weight reader: the two tensors read straight from safetensors must equal the loaded
model's, so the logit-lens score is computed on the real unembedding and final norm."""
import glob
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "experiments"))
SNAP = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--hf-internal-testing--tiny-random-LlamaForCausalLM/"
    "snapshots/*/"))
pytestmark = pytest.mark.skipif(not SNAP, reason="tiny Llama not in the HF cache")


def test_reader_matches_model():
    import torch
    from transformers import AutoModelForCausalLM
    import analyse_p2m as m
    src = m.Local(SNAP[0])
    t = m.tensors(src, m.pick)
    unemb, norm = m.pick(list(t))
    model = AutoModelForCausalLM.from_pretrained(SNAP[0], dtype=torch.float32)
    assert np.allclose(t[unemb], model.lm_head.weight.detach().numpy(), atol=1e-6)
    assert np.allclose(t[norm], model.model.norm.weight.detach().numpy(), atol=1e-6)
    direction, ids = m.answer_direction(src)
    assert direction.shape == (model.config.hidden_size,)
    assert ids["yes"] and ids["no"] and not set(ids["yes"]) & set(ids["no"])


def test_bf16_decode():
    import analyse_p2m as m
    x = np.array([1.0, -2.5, 3.140625], dtype=np.float32)
    raw = (x.view(np.uint32) >> 16).astype(np.uint16).tobytes()
    assert np.allclose(m._decode(raw, "BF16", [3]), x)
