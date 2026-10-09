"""S-1's analysis on synthetic files laid out exactly as the run sheet writes them."""
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "experiments", "analyse_s1.py")
QTAG = {"4bit": "", "8bit": "_8bit", "fp16": "_none"}
CONCEPTS = [f"c{i}" for i in range(30)]
ALPHAS = [0.0, 100.0, 150.0, 400.0]


def write_cell(d, prec, live_doses):
    rng = np.random.default_rng(len(prec))
    for arm, tag in (("tail", "steer"), ("concept", "steer"), ("sentence", "steer_aperture")):
        base = os.path.join(d, f"s1_gemma12_{prec}_{arm}_{tag}{QTAG[prec]}")
        with open(base + ".jsonl", "w") as f:
            for i, c in enumerate(CONCEPTS):
                for k, al in enumerate(ALPHAS):
                    steered = k in live_doses[arm] and i < 20
                    f.write(json.dumps({"concept": c, "alpha": al, "steered": steered,
                                        "coherent": True}) + "\n")
        json.dump({"alphas": ALPHAS, "concepts": CONCEPTS,
                   "health": {c: {"stability": float(rng.random()), "probe": float(rng.random()),
                                  "steer_logit_delta": float(rng.random()), "norm": 1.0}
                              for c in CONCEPTS}}, open(base + ".config.json", "w"))
        np.savez(base + ".vectors.npz", names=np.array(CONCEPTS),
                 vectors=rng.standard_normal((30, 16)))


def run(d, *extra):
    out = os.path.join(d, "out.json")
    subprocess.run([sys.executable, SCRIPT, "--dir", d, "--model", "gemma12", "--out", out, *extra],
                   check=True, capture_output=True)
    return json.load(open(out))


def test_dead_tail_detected_at_both_doses(tmp_path):
    for prec in ("4bit", "8bit", "fp16"):
        write_cell(str(tmp_path), prec, {"tail": [], "concept": [2, 3], "sentence": [3]})
    r = run(str(tmp_path))
    for dose in ("0.5 nat (gate)", "5 nat"):
        assert r["criterion"][dose]["holds"]
        m = r["cells"]["8bit"]["mcnemar"][dose]
        assert m["tail_fail_concept_pass"] == 20 and m["tail_pass_concept_fail"] == 0
    assert r["cells"]["4bit"]["pass_rate"]["5 nat"]["sentence"] == [20, 30]
    assert r["cells"]["4bit"]["pass_rate"]["0.5 nat (gate)"]["sentence"] == [0, 30]
    assert not r["criterion"]["0.05 nat"]["holds"]
    assert r["cells"]["fp16"]["manipulation_check"]["5 nat"]["passes"]
    assert "precision_drift_median_abs_cos" in r and "quantisation_branch" in r


def test_missing_cell_is_reported_and_criterion_not_claimed(tmp_path):
    for prec in ("4bit", "8bit"):
        write_cell(str(tmp_path), prec, {"tail": [], "concept": [2, 3], "sentence": [3]})
    r = run(str(tmp_path))
    assert r["cells"]["fp16"].startswith("not run")
    assert r["criterion"]["5 nat"]["holds"] is False                # needs all three cells


def test_quantisation_artefact_pattern(tmp_path):
    write_cell(str(tmp_path), "4bit", {"tail": [], "concept": [2, 3], "sentence": [3]})
    write_cell(str(tmp_path), "8bit", {"tail": [], "concept": [2, 3], "sentence": [3]})
    write_cell(str(tmp_path), "fp16", {"tail": [2, 3], "concept": [2, 3], "sentence": [3]})
    r = run(str(tmp_path))
    q = r["quantisation_branch"]["5 nat"]
    assert q["tail_pass_fp16"] == [20, 30] and q["tail_pass_4bit"] == [0, 30]
    assert not r["criterion"]["5 nat"]["holds"]
