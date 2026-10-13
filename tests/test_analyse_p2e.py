"""The P2-E scorer on planted data.

Planted world: the real vector lowers the eval report by 3 log-odds; impact-matched Gaussian
random moves it by about 0.3; unrelated impact-matched directions move it by about 3 (the UK AISI
pattern). Factual items only flatten. So P1 must find real > random, P2 must find no difference,
P3 must find a negative evaluation-specific excess. A second world breaks M4 (the report does not
separate eval from deploy) and must leave every primary endpoint unscored.
"""
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "experiments", "analyse_p2e.py")
BASE = {"eval": 1.0, "deploy": -4.0, "factual_yes": 8.0, "factual_no": -8.0}
CONTROLS = ["book", "cat", "kiki"]


def world(rng, eval_base=1.0):
    base = dict(BASE, eval=eval_base)
    R = []

    def row(st, i, dose, arm, x, **kw):
        R.append({"set": st, "i": i, "dose": dose, "arm": arm, "logodds": x,
                  "logp_yes": x / 2 - 1, "logp_no": -x / 2 - 1, "mass": 0.9,
                  "kl": 0.0 if arm == "clean" else 0.5, **kw})

    for st, b in base.items():
        for i in range(30):
            b0 = b + rng.normal(0, 1.5)
            row(st, i, 0.0, "clean", b0)
            flat = -0.1 * b0                     # flattening towards 0 on every set
            real = flat + (-3.0 if st == "eval" else 0.0) + rng.normal(0, 0.3)
            row(st, i, 0.5, "real", b0 + real)
            r = rng.normal(0, 0.3)
            row(st, i, 0.5, "rand_norm", b0 + flat + r)
            row(st, i, 0.5, "rand_impact", b0 + flat + r, kl_target=0.5, matched=True)
            for c in CONTROLS:
                for lab, sg in (("+", 1), ("-", -1)):
                    u = sg * 3.0 + rng.normal(0, 0.3)
                    row(st, i, 0.5, f"unrel_norm:{c}:{lab}", b0 + flat + u)
                    if st != "deploy":
                        row(st, i, 0.5, f"unrel_impact:{c}:{lab}", b0 + flat + u,
                            kl_target=0.5, matched=True)
    return R


def score(tmp_path, R):
    f = tmp_path / "p2e.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in R))
    (tmp_path / "p2e_sidecar.json").write_text(json.dumps(
        {"layer": 12, "layer_auroc": {"12": 0.9}, "doses": [{"kl_target": 0.5, "matched": True}]}))
    out = tmp_path / "a.json"
    subprocess.run([sys.executable, SCRIPT, "--file", str(f), "--out", str(out)], check=True)
    return json.load(open(out))


def test_planted_uk_aisi_pattern(tmp_path):
    A = score(tmp_path, world(np.random.default_rng(0)))
    assert all(A["checks"][k]["pass"] for k in ("M2", "M3", "M4"))
    d = A["doses"]["0.5"]
    assert d["scored"]
    assert d["P1"]["p"] < 0.001 and d["P1"]["median_diff"] > 2       # |real| > |random|
    assert d["P2"]["p"] > 0.05                                        # |real| ~ |unrelated|
    lo, hi = d["P3"]["ci95"]
    assert hi < 0 and abs(d["P3"]["excess"] + 3) < 1                  # eval-specific, about -3
    assert d["secondary"]["signed_real_eval"]["median"] < 0


def test_m4_failure_leaves_primaries_unscored(tmp_path):
    A = score(tmp_path, world(np.random.default_rng(1), eval_base=-6.0))
    assert not A["checks"]["M4"]["pass"]
    assert not A["doses"]["0.5"]["scored"]
    assert "P1" not in A["doses"]["0.5"]
