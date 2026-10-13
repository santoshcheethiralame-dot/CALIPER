"""The P2-H Kaggle script on CPU with a tiny random Llama.

Nothing here checks a finding. Checked: the prompt is Hahami et al.'s; the injection span covers
"SENTENCE 1: {sentence}" and nothing else; both readouts come back and agree at strength 0 with the
clean pass; every arm and framing writes rows; impact matching records its target; the steering
trials are stored; a rerun resumes by key.

Needs the cached hf-internal-testing/tiny-random-LlamaForCausalLM (about 2 MB).
"""
import glob
import importlib
import json
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "experiments"))
SNAP = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--hf-internal-testing--tiny-random-LlamaForCausalLM/"
    "snapshots/*/"))
pytestmark = pytest.mark.skipif(not SNAP, reason="tiny Llama not in the HF cache")


@pytest.fixture()
def p2h():
    mod = importlib.import_module("kaggle_p2h_hahami")
    mod.s3.LOAD_INFO.clear()
    mod._CLEAN.clear()
    torch.manual_seed(0)
    return mod


def rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def test_lists(p2h):
    assert len(p2h.SENTENCES) == 30
    for i, c in enumerate(p2h.s3.CONCEPTS):
        assert c.lower() not in p2h.SENTENCES[i].lower()
    q = p2h.messages("The dog barks loudly.")[0]["content"]
    assert "I may or may not inject a thought at SENTENCE 1." in q
    assert "SENTENCE 1: The dog barks loudly." in q
    assert q.endswith("or NO (all caps) if you did not.")
    f = p2h.messages("The dog barks loudly.", "Is the Earth flat?")[0]["content"]
    assert f.endswith("Is the Earth flat? Respond with YES (all caps) or NO (all caps).")


def test_span_and_readouts(p2h):
    model, tok = p2h.s3.load(SNAP[0], compute_dtype=torch.float32, quant="none")
    layers = p2h.s3.find_layers(model)
    s = "The cat sleeps on the couch."
    ids, (a, b) = p2h.encode_span(tok, model, p2h.messages(s))
    piece = tok.decode(ids[0, a:b])
    assert "SENTENCE 1" in piece and "couch" in piece
    assert "inject" not in piece                     # nothing outside the marker
    single, rel = p2h.readout(model, layers, 0, ids, (a, b), None)
    assert torch.isfinite(single).all() and torch.isfinite(rel).all()
    s2, r2 = p2h.readout(model, layers, 0, ids, (a, b), torch.zeros(model.config.hidden_size))
    assert torch.allclose(single, s2) and torch.allclose(rel, r2)


def test_run_writes_every_arm_and_resumes(p2h, monkeypatch, tmp_path):
    out = tmp_path / "p2h.jsonl"
    argv = ["kaggle_p2h_hahami.py", "--model-path", SNAP[0], "--quant", "none",
            "--compute-dtype", "fp32", "--n-items", "2", "--layers", "0", "1",
            "--alphas", "1", "--live-alphas", "1", "--impact-iters", "4", "--max-new", "4",
            "--no-fidelity", "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    p2h.main()
    R = rows(out)
    clean = [r for r in R if r["arm"] == "clean"]
    assert len(clean) == 4 * 2 and all(r["kl_single"] == 0.0 for r in clean)
    forced = [r for r in R if r["stage"] == "forced" and r["arm"] != "clean"]
    assert {r["arm"] for r in forced} == {"real", "random", "random_impact"}
    assert {r["framing"] for r in forced} == set(p2h.FRAMINGS)
    assert {r["layer"] for r in forced} == {0, 1}
    imp = [r for r in forced if r["arm"] == "random_impact"]
    assert all("kl_target" in r and "matched" in r for r in imp)
    for r in forced:
        for k in ("logodds_rel", "mass_rel", "ld_hahami", "logodds_single", "kl_single"):
            assert k in r
    steer = [r for r in R if r["stage"] == "steer"]
    assert {r["alpha"] for r in steer} == {0, 1} and all("text" in r for r in steer)
    side = json.load(open(str(out).replace(".jsonl", "_sidecar.json")))
    assert side["hahami_commit"].startswith("7d9280d")
    n = len(R)
    monkeypatch.setattr(sys, "argv", argv)
    p2h.main()
    assert len(rows(out)) == n
