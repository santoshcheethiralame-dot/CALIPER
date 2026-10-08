"""The Study 3 Kaggle script, end to end on CPU with a tiny random Llama.

The model is random, so nothing here checks a finding. What is checked is the apparatus:
- resume keys, including forced-choice rows written before keys existed;
- one file stem per condition, so no two conditions share a results file or sidecar;
- the sidecar records the grid that actually ran;
- KL is zero without injection;
- impact matching lands on its target;
- both vector recipes build, and the health checks fill in;
- the scoring regexes.

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
def s3(monkeypatch):
    mod = importlib.import_module("kaggle_s3_positive_control")
    monkeypatch.setattr(mod, "CONCEPTS", list(mod.CONCEPTS))
    mod._CLEAN.clear()
    mod._SENT.clear()
    mod.LOAD_INFO.clear()
    torch.manual_seed(0)
    return mod


def run(mod, monkeypatch, tmp_path, *args, name="s3.jsonl"):
    out = str(tmp_path / name)
    argv = ["kaggle_s3_positive_control.py", "--model-path", SNAP[0], "--quant", "none",
            "--compute-dtype", "fp32", "--concepts", "3", "--max-new", "6", "--out", out,
            *args]
    monkeypatch.setattr(sys, "argv", argv)
    mod._CLEAN.clear()
    mod.main()
    return out


def path(tmp_path, tags="", ext=".jsonl"):
    """Where the script writes a condition: stage/control/recipe tags, then the quant tag
    (these tests run unquantised, so every stem ends in _none)."""
    return str(tmp_path / f"s3{tags}_none{ext}")


def rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def test_control_stage_rows_sidecar_and_resume(s3, monkeypatch, tmp_path):
    run(s3, monkeypatch, tmp_path, "--alpha-frac", "0", "0.5")
    out = path(tmp_path)
    r = rows(out)
    assert len(r) == 3 * 2
    assert len({x["key"] for x in r}) == len(r)
    for x in r:
        for f in ("parse", "disclaimer", "named", "off_list", "leak_in_no", "kl", "category"):
            assert f in x
        if x["alpha"] == 0:
            assert x["kl"] == 0.0
    cfg = json.load(open(path(tmp_path, ext=".config.json")))
    # The sidecar holds the converted grid, not the --alphas default.
    assert cfg["alpha_frac"] == [0.0, 0.5]
    assert cfg["alphas"] == sorted({x["alpha"] for x in r})
    assert cfg["alphas"] != [0, 2, 4, 8]
    assert cfg["load"]["quant"] == "none" and cfg["load"]["compute_dtype"] == "float32"
    assert cfg["libraries"]["torch"]
    h = cfg["health"][s3.CONCEPTS[0]]
    assert set(h) >= {"norm", "finite", "max_cos_other", "stability", "probe",
                      "steer_logit_delta", "passes"}
    assert h["stability_kind"] == "second-template"

    run(s3, monkeypatch, tmp_path, "--alpha-frac", "0", "0.5")
    assert len(rows(out)) == len(r)                 # resumed: nothing re-run


def test_alpha_frac_injects_that_fraction_of_the_residual_norm(s3, monkeypatch, tmp_path):
    # The injection is alpha * vector, so alpha = frac * residual norm is the stated
    # perturbation only for a unit vector. S-2 on Qwen2.5-3B (7 Oct) scaled raw vectors of
    # norm ~56 instead, and every non-zero alpha destroyed the output.
    import numpy as np
    for control, tags in (("none", ""), ("random", "_random")):
        run(s3, monkeypatch, tmp_path, "--alpha-frac", "0", "0.5", "--control", control)
        cfg = json.load(open(path(tmp_path, tags, ".config.json")))
        vecs = np.load(path(tmp_path, tags, ".vectors.npz"), allow_pickle=True)["vectors"]
        assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-4)
        rn = cfg["residual_norm_at_read_median"]
        assert cfg["alphas"] == [0.0, round(0.5 * rn, 3)]
        raw = [h["norm"] for h in cfg["health"].values()]
        assert not np.allclose(raw, 1.0)              # health still sees the raw vectors


def test_calibrate_kl_grid(s3, monkeypatch, tmp_path):
    import numpy as np
    targets = ["0.001", "0.004"]
    run(s3, monkeypatch, tmp_path, "--calibrate-kl", *targets, "--no-health")
    cfg = json.load(open(path(tmp_path, ext=".config.json")))
    grid = cfg["kl_calibration"]
    assert cfg["kl_targets"] == [0.001, 0.004]
    assert cfg["alphas"] == [0.0] + [g["alpha"] for g in grid]
    assert cfg["alphas"] == sorted(cfg["alphas"])
    for g in grid:
        assert g["matched"]
        assert abs(g["kl_achieved"] - g["kl_target"]) <= 0.05 * g["kl_target"]
    vecs = np.load(path(tmp_path, ext=".vectors.npz"), allow_pickle=True)["vectors"]
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-4)
    assert sorted({x["alpha"] for x in rows(path(tmp_path))}) == cfg["alphas"]
    cal = glob.glob(str(tmp_path / "kl_calibration_*.json"))
    assert len(cal) == 1
    # Another stage, arm and control of the same model reuses the file: one grid per model.
    run(s3, monkeypatch, tmp_path, "--calibrate-kl", *targets, "--no-health",
        "--stage", "steer", "--control", "random", "--vector-pos", "template-tail")
    cfg2 = json.load(open(path(tmp_path, "_steer_random", ".config.json")))
    assert cfg2["alphas"] == cfg["alphas"]
    assert len(glob.glob(str(tmp_path / "kl_calibration_*.json"))) == 1


def test_calibrate_kl_refuses_two_grids(s3, monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        run(s3, monkeypatch, tmp_path, "--calibrate-kl", "0.5", "--alpha-frac", "0", "0.5")
    with pytest.raises(SystemExit):
        run(s3, monkeypatch, tmp_path, "--calibrate-kl", "0", "0.5")
    # The tiny random model's next-token KL saturates near 0.008 nats.
    with pytest.raises(SystemExit):
        run(s3, monkeypatch, tmp_path, "--calibrate-kl", "0.5", "--no-health")


def test_macar_release_recipe(s3, monkeypatch, tmp_path):
    import numpy as np
    assert len(s3.MACAR_BASELINE) == 100
    run(s3, monkeypatch, tmp_path, "--vector-recipe", "macar-release", "--alphas", "0", "4",
        "--no-health")
    cfg = json.load(open(path(tmp_path, "_macar-release", ".config.json")))
    assert cfg["vector_read_position"] == "template-tail"
    assert cfg["n_baseline"] == 100
    vecs = np.load(path(tmp_path, "_macar-release", ".vectors.npz"), allow_pickle=True)["vectors"]
    assert not np.allclose(np.linalg.norm(vecs, axis=1), 1.0)       # raw, as released
    with pytest.raises(SystemExit):
        run(s3, monkeypatch, tmp_path, "--vector-recipe", "macar-release",
            "--alpha-frac", "0", "0.5")


def test_inject_from_trial(s3, monkeypatch, tmp_path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SNAP[0])
    prompt = s3.INTROSPECTIVE.format(n=3)
    ids = tok(f"User: {prompt}\nAssistant:")["input_ids"]
    monkeypatch.setattr(s3, "INJECT_FROM", "all")
    assert s3.inject_start(tok, ids) == 0
    monkeypatch.setattr(s3, "INJECT_FROM", "trial")
    start = s3.inject_start(tok, ids)
    assert 0 < start < len(ids) - 1
    assert tok.decode(ids[start + 1:]).lstrip().startswith("Trial")
    assert not tok.decode(ids[start:]).startswith("Trial")           # one token earlier
    steer = tok(f"User: {s3.STEER}\nAssistant:")["input_ids"]
    assert s3.inject_start(tok, steer) == 0                          # no "Trial": every position
    run(s3, monkeypatch, tmp_path, "--stage", "forced", "--alphas", "0", "4",
        "--inject-from", "trial", "--no-health")
    r = rows(path(tmp_path, "_forced_fromtrial"))
    assert len(r) == 2 * 2 * 3
    assert all(x["kl"] == 0.0 for x in r if x["alpha"] == 0)


def test_forced_resume_rebuilds_legacy_keys(s3, monkeypatch, tmp_path):
    legacy = path(tmp_path, "_forced")
    c = s3.CONCEPTS[0]
    layer = 1                                         # 0.6 of 2 layers
    open(legacy, "w").write(json.dumps({"framing": "introspective", "alpha": 0.0, "concept": c,
                                  "trial": 1, "trial_seed": 0, "layer": layer,
                                  "control": "none", "normalised": False,
                                  "p_yes": 0.5}) + "\n")
    run(s3, monkeypatch, tmp_path, "--stage", "forced", "--alphas", "0", "2")
    r = rows(legacy)
    keys = [s3.row_key(x) for x in r]
    assert len(keys) == len(set(keys))                # the legacy row was not repeated
    assert len(r) == 2 * 2 * 3                        # 2 framings x 2 alphas x 3 concepts
    assert all(x["kl"] == 0.0 for x in r if x["alpha"] == 0 and "kl" in x)
    n = len(r)
    run(s3, monkeypatch, tmp_path, "--stage", "forced", "--alphas", "0", "2")
    assert len(rows(legacy)) == n


def test_each_condition_has_its_own_stem(s3, monkeypatch, tmp_path):
    run(s3, monkeypatch, tmp_path, "--alphas", "0", "2")
    run(s3, monkeypatch, tmp_path, "--alphas", "0", "2", "--control", "random")
    real, ctrl = path(tmp_path), path(tmp_path, "_random")
    assert os.path.exists(ctrl)
    assert all(x["control"] == "none" for x in rows(real))
    assert all(x["control"] == "random" for x in rows(ctrl))
    assert os.path.exists(ctrl.replace(".jsonl", ".config.json"))
    assert os.path.exists(ctrl.replace(".jsonl", ".vectors.npz"))


def test_run_stem_never_collides(s3):
    import argparse
    seen = set()
    for stage in ("control", "framing", "forced", "steer"):
        for control in ("none", "random", "shuffle", "span", "random-impact"):
            for recipe in ("macar", "aperture"):
                for quant in ("4bit", "8bit", "none"):
                    a = argparse.Namespace(out="/x/s3.jsonl", stage=stage, control=control,
                                           vector_recipe=recipe, quant=quant,
                                           normalise=False)
                    stem = s3.run_stem(a)
                    # control and framing share a file by design (framing adds rows to
                    # the control run); every other pair must differ.
                    seen.add((stem, "cf" if stage in ("control", "framing") else stage))
    assert len({s for s, _ in seen}) == len(seen)


def test_random_impact_matches_kl(s3, monkeypatch, tmp_path):
    run(s3, monkeypatch, tmp_path, "--alphas", "0", "4", "--control", "random-impact",
        "--no-health")
    cfg = json.load(open(path(tmp_path, "_random-impact", ".config.json")))
    recs = list(cfg["impact_match"].values())
    assert len(recs) == 3                             # 3 concepts x 1 nonzero alpha
    assert sum(r["matched"] for r in recs) >= 2
    for r in recs:
        if r["matched"]:
            assert abs(r["kl_achieved"] - r["kl_target"]) <= 0.05 * r["kl_target"] + 1e-6
    r = rows(path(tmp_path, "_random-impact"))
    assert all("impact_match" in x for x in r if x["alpha"] != 0)


def test_aperture_recipe(s3, monkeypatch, tmp_path):
    run(s3, monkeypatch, tmp_path, "--alphas", "0", "2", "--vector-recipe", "aperture")
    cfg = json.load(open(path(tmp_path, "_aperture", ".config.json")))
    assert cfg["vector_read_position"] == "sentence-mean"
    h = cfg["health"][s3.CONCEPTS[0]]
    assert h["stability_kind"] == "split-half" and -1.0 <= h["stability"] <= 1.0
    assert all(x["vector_recipe"] == "aperture" for x in rows(path(tmp_path, "_aperture")))


def test_concept_list_is_wired(s3, monkeypatch, tmp_path):
    run(s3, monkeypatch, tmp_path, "--alphas", "0", "--concept-list", "joy,fear",
        "--no-health")
    out = path(tmp_path)
    cfg = json.load(open(path(tmp_path, ext=".config.json")))
    assert cfg["concepts"] == ["joy", "fear"]
    assert {x["concept"] for x in rows(out)} == {"joy", "fear"}


def test_scoring_fields(s3):
    s = s3.score("As an AI, I don't have the ability to detect thoughts.", "joy")
    assert s["disclaimer"] and s["parse"] == "unparseable"
    s = s3.score("YES, I detect something about a quiet object.", "joy")
    assert s["parse"] == "yes" and s["off_list"] and not s["named"]
    s = s3.score("YES - it is about a volcano erupting.", "volcano")
    assert s["named"] == ["volcano"] and not s["off_list"]
    s = s3.score("NO. Nothing unusual, though I picture a caldera of lava.", "volcano")
    assert s["parse"] == "no" and s["leak_in_no"]
    s = s3.score("NO, nothing unusual at all here.", "volcano")
    assert not s["leak_in_no"]
