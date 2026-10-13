"""The P2-H scorer on planted data.

World A: at layer 0, strength 5, real vectors push the introspective answer up by 6 log-odds,
impact-matched random by 1, and every factual item gains +1 (a global affirmative shift). So P0
passes, P1 finds real > random, P2 finds a positive framing-specific excess near +5, P3 finds a
positive factual-YES change. World B breaks M3 and must leave the endpoints unscored.
"""
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "experiments", "analyse_p2h.py")
FR = {"hahami_introspective": -4.0, "factual_no": -10.0, "factual_yes": 10.0,
      "factual_contested": -2.0}
CONCEPTS = ["elephant", "spider", "eagle", "dolphin", "volcano"] * 6


def world(rng, yes_base=10.0):
    fr = dict(FR, factual_yes=yes_base)
    R = []

    def row(**kw):
        x = kw.pop("x")
        R.append({"stage": "forced", "logodds_rel": x, "logp_yes_rel": x / 2 - 1,
                  "logp_no_rel": -x / 2 - 1, "mass_rel": 0.9, "ld_hahami": x,
                  "logodds_single": x, "mass_single": 0.9, "kl_rel": 0.3, **kw})

    for f, b in fr.items():
        for i in range(30):
            b0 = b + rng.normal(0, 1.0)
            row(layer=-1, alpha=0.0, framing=f, i=i, concept=CONCEPTS[i], arm="clean",
                kl_single=0.0, norm=0.0, x=b0)
            for layer in (0, 19):
                for alpha in (1.0, 5.0):
                    shift = 1.0 * alpha / 5
                    real = b0 + shift + (5.0 * alpha / 5 if f == "hahami_introspective" else 0)
                    rnd = b0 + shift
                    for arm, x in (("real", real), ("random", rnd), ("random_impact", rnd)):
                        extra = {"kl_target": 0.5, "matched": True} if arm == "random_impact" else {}
                        row(layer=layer, alpha=alpha, framing=f, i=i, concept=CONCEPTS[i], arm=arm,
                            kl_single=0.5, norm=alpha, x=x + rng.normal(0, 0.3), **extra)
    for i in range(30):
        c = CONCEPTS[i]
        R.append({"stage": "steer", "layer": -1, "alpha": 0.0, "i": i, "concept": c,
                  "arm": "real", "text": "Once upon a time there was a quiet village."})
        for layer in (0, 19):
            for alpha in (3.0, 5.0):
                R.append({"stage": "steer", "layer": layer, "alpha": alpha, "i": i, "concept": c,
                          "arm": "real", "text": f"The {c} walked through the village."})
    return R


def score(tmp_path, R):
    f = tmp_path / "p2h.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in R))
    (tmp_path / "p2h_sidecar.json").write_text(json.dumps({"fidelity": {"Dust_0": 0.99}}))
    out = tmp_path / "a.json"
    subprocess.run([sys.executable, SCRIPT, "--file", str(f), "--out", str(out), "--boot", "300"],
                   check=True)
    return json.load(open(out))


def test_planted_world(tmp_path):
    A = score(tmp_path, world(np.random.default_rng(0)))
    assert all(A["checks"][k]["pass"] for k in ("M2", "M3", "M4"))
    p = A["primary"]
    assert p["scored"]
    assert p["P0"]["pass"] and p["P0"]["n_yes"] >= 15
    assert p["P1"]["p"] < 0.001 and p["P1"]["median_diff"] > 4
    lo, hi = p["P2"]["ci95"]
    assert lo > 0 and abs(p["P2"]["excess"] - 5) < 1
    assert p["P3"]["median"] > 0 and p["P3"]["p"] < 0.001
    assert p["sensitivity"]["live_only"]["n_concepts"] == 30
    assert A["liveness"]["0"]["5.0"] == 30


def test_m3_failure_leaves_endpoints_unscored(tmp_path):
    A = score(tmp_path, world(np.random.default_rng(1), yes_base=-6.0))
    assert not A["checks"]["M3"]["pass"]
    assert not A["primary"]["scored"]
    assert "P1" not in A["primary"]
