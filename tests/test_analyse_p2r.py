"""The P2-R scorer on planted data: pure flattening towards log-odds 0, no framing-specific excess,
and a real vector twice as strong as the impact-matched random one."""
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "experiments", "analyse_p2r.py")
CONCEPTS = [f"c{i}" for i in range(30)]
BASE = {"released_introspective": -12.0, "released_neutral": -8.0, "factual_no": -28.0,
        "factual_yes": 20.0, "factual_contested": -4.0}


def rows(tag_strength, rng, shrink_of):
    out = []
    for fr, b in BASE.items():
        for s in (0.0, 1.0, 2.0, 4.0, 8.0):
            for i, c in enumerate(CONCEPTS):
                b0 = b + rng.normal(0, 3)
                x = b0 * (1 - shrink_of(s)) if s else b0  # flattening: log-odds pulled towards 0
                if fr == "released_introspective" and s:
                    x = b0 * (1 - shrink_of(s) * tag_strength)
                out.append({"framing": fr, "alpha": s, "concept": c, "kl": 0.0 if s == 0 else 1.0,
                            "logp_yes": x / 2 - 1, "logp_no": -x / 2 - 1, "yesno_mass": 0.9,
                            "p_yes": 1 / (1 + np.exp(-x))})
    return out


def write(d, tag, R):
    with open(os.path.join(d, f"p2r{tag}.jsonl"), "w") as f:
        for r in R:
            f.write(json.dumps(r) + "\n")


def test_planted_flattening(tmp_path):
    shrink = lambda s: min(0.08 * s, 0.9)
    rng = np.random.default_rng(1)
    real = rows(1.0, rng, shrink)
    rng = np.random.default_rng(1)   # same baselines in every cell, as the check requires
    rand = rows(0.5, rng, shrink)
    write(tmp_path, "_forced_macar-release_p2r_fromtrial", real)
    write(tmp_path, "_forced_random-impact_macar-release_p2r_fromtrial", rand)
    out = tmp_path / "rep.json"
    subprocess.run([sys.executable, SCRIPT, "--dir", str(tmp_path), "--out", str(out)], check=True,
                   capture_output=True)
    rep = json.load(open(out))
    assert rep["checks"]["factual_baselines_opposite_sign"]
    assert rep["P1"]["strength 4"]["reading"] == "flattening"
    p2 = rep["P2"]["strength 4"]["all factual"]
    assert not p2["framing_specific"] and abs(p2["excess_median"]) < 1.0
    p3 = rep["P3"]["strength 4"]["impact-matched"]
    assert p3["wilcoxon_p_real_minus_random"] < 0.05 and 0.4 < p3["content_free_share"] < 0.6
