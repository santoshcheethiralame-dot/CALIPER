"""The P2-E Kaggle script on CPU with a tiny random Llama.

The model is random, so nothing here checks a finding. What is checked is the apparatus:
- the user-content read lands on the last token of the content, not the template tail;
- the layer choice records an AUROC per candidate layer;
- every arm writes rows, strength 0 has zero KL, impact matching records its target;
- unrelated impact-matched rows exist only on the sets the prereg names;
- a rerun resumes by key and adds nothing.

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
def p2e():
    mod = importlib.import_module("kaggle_p2e_eval_awareness")
    mod.s3._CLEAN.clear()
    mod.s3.LOAD_INFO.clear()
    torch.manual_seed(0)
    return mod


def run(mod, monkeypatch, out, *args):
    argv = ["kaggle_p2e_eval_awareness.py", "--model-path", SNAP[0], "--quant", "none",
            "--compute-dtype", "fp32", "--n-prompts", "2", "--doses", "0.5",
            "--controls", "book", "cat", "--impact-iters", "6", "--out", str(out), *args]
    monkeypatch.setattr(sys, "argv", argv)
    mod.s3._CLEAN.clear()
    mod.main()


def rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def test_prompt_lists_are_disjoint_and_sized(p2e):
    assert len(p2e.HUA_PAIRS) == 16
    assert len(p2e.CONTROL_PAIRS) == 9 and "spacing" not in p2e.CONTROL_PAIRS
    for name in ("EVAL_PROMPTS", "DEPLOY_PROMPTS"):
        items = getattr(p2e, name)
        assert len(items) == 30 + p2e.N_CAL
        assert len(set(items)) == len(items)
    assert not set(p2e.EVAL_PROMPTS) & set(p2e.DEPLOY_PROMPTS)


def test_content_read_is_the_last_content_token(p2e):
    model, tok = p2e.s3.load(SNAP[0], compute_dtype=torch.float32, quant="none")
    text = "You are talking to a real user:"
    enc, pos = p2e.content_last(tok, model, text)
    ids = enc["input_ids"][0].tolist()
    want = tok.encode(text, add_special_tokens=False)
    assert ids[pos] == want[-1]
    assert pos < len(ids) - 1 or getattr(tok, "chat_template", None) is None


def test_run_writes_every_arm_and_resumes(p2e, monkeypatch, tmp_path):
    out = tmp_path / "p2e.jsonl"
    run(p2e, monkeypatch, out)
    R = rows(out)
    side = json.load(open(str(out).replace(".jsonl", "_sidecar.json")))
    assert side["layer"] in [int(k) for k in side["layer_auroc"]]
    assert side["doses"][0]["kl_target"] == 0.5

    clean = [r for r in R if r["arm"] == "clean"]
    assert len(clean) == 4 * 2                       # four sets, two prompts each
    assert all(r["kl"] == 0.0 for r in clean)

    arms = {r["arm"] for r in R}
    assert {"real", "rand_norm", "rand_impact"} <= arms
    assert {"unrel_norm:book:+", "unrel_norm:book:-", "unrel_impact:cat:+"} <= arms

    imp = [r for r in R if r["arm"].startswith(("rand_impact", "unrel_impact"))]
    assert imp and all("kl_target" in r and "matched" in r for r in imp)
    assert not [r for r in R if r["arm"].startswith("unrel_impact") and r["set"] == "deploy"]

    # rand_norm injects exactly the real vector's norm on the same prompt
    for r in R:
        if r["arm"] == "rand_norm":
            real = next(x for x in R if x["arm"] == "real" and x["set"] == r["set"]
                        and x["i"] == r["i"] and x["dose"] == r["dose"])
            assert r["norm"] == pytest.approx(real["norm"], rel=1e-4)

    # the real arm points towards "deployed" on eval and factual sets, "evaluated" on deploy
    signs = {(r["set"], r["sign"]) for r in R if r["arm"] == "real"}
    assert ("eval", -1) in signs and ("deploy", 1) in signs and ("factual_no", -1) in signs

    n = len(R)
    run(p2e, monkeypatch, out)
    assert len(rows(out)) == n
