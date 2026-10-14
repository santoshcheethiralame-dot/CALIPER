"""Score P2-N (docs/preregistration-p2n-qwen35-current.md) once, when every file is complete.

    python experiments/analyse_p2n.py      # writes results/p2n_analysis.json

For each model (qwen35_2b primary, qwen35_08b secondary): the unchanged S-2 scorer on the filed
substring label, and on the strict gate (a copy of the steer rows rescored with
rescore_steering_gate.rescore); then the dose comparison of P2 from the forced rows: median
next-token KL per fraction and arm over both framings, the quantity of P2-D's dose table.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from rescore_steering_gate import rescore  # noqa: E402

SRC = ROOT / "results/p2n"
MODELS = ["qwen35_2b", "qwen35_08b"]
ARMS = ["concept_steer_none", "tail_steer_none", "sentence_steer_aperture_none"]
FORCED = {"concept": "concept_forced_none", "tail": "tail_forced_none",
          "sentence": "sentence_forced_aperture_none", "random": "random_forced_random_none"}
FRACS = ["0.25", "0.5", "1"]


def rows(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []


def s2(d, model):
    out = d / f"{model}_s2.json"
    r = subprocess.run([sys.executable, str(ROOT / "experiments/analyse_s2.py"), "--dir", str(d), "--model", model,
                        "--quant", "none", "--out", str(out)], capture_output=True, text=True, cwd=ROOT)
    if r.returncode:
        return {"error": r.stderr[-800:]}
    a = json.load(open(out))
    return {k: a[k] for k in ("vectors", "live", "dead", "live_by_arm", "scored", "auc", "predictions") if k in a}


def main():
    rep = {}
    for m in MODELS:
        if not (SRC / f"s2_{m}_concept_steer_none.jsonl").exists():
            rep[m] = "not run"
            continue
        strict = ROOT / "results/gate_strict" / f"p2n_{m}"
        if strict.exists():
            shutil.rmtree(strict)
        shutil.copytree(SRC, strict, ignore=lambda d, names: [n for n in names if not n.startswith(f"s2_{m}_")])
        for arm in ARMS:
            f = strict / f"s2_{m}_{arm}.jsonl"
            new = rescore(rows(f))
            f.write_text("".join(json.dumps(r) + "\n" for r in new), encoding="utf-8")
        # P2-D's quantity: forced-stage next-token KL, median over both framings, per arm. Rows carry the
        # absolute alpha; the three nonzero alphas of a file are the fractions 0.25, 0.5 and 1.0 in order.
        kl = {}
        for arm, stem in FORCED.items():
            R = rows(SRC / f"s2_{m}_{stem}.jsonl")
            nz = sorted({r["alpha"] for r in R} - {0})
            if not R or len(nz) != 3:
                kl[arm] = None
                continue
            kl[arm] = {f: float(np.median([r["kl"] for r in R if r["alpha"] == a])) for f, a in zip(FRACS, nz)}
        ratio = kl["random"]["0.5"] / kl["concept"]["0.5"] if kl["random"] and kl["concept"] else None
        rep[m] = {"filed_substring_label": s2(SRC, m), "strict_label": s2(strict, m), "median_kl": kl,
                  "random_over_concept_kl_at_0.5": ratio}
    a, b = rep.get("qwen35_2b"), rep.get("qwen35_08b")
    if isinstance(a, dict) and isinstance(b, dict) and a["median_kl"]["concept"] and b["median_kl"]["concept"]:
        ka, kb = a["median_kl"]["concept"]["0.25"], b["median_kl"]["concept"]["0.25"]
        fold = max(ka, kb) / min(ka, kb) if min(ka, kb) > 0 else None
        r2 = a["random_over_concept_kl_at_0.5"]
        rep["P2"] = {"concept_kl_at_0.25_2b": ka, "concept_kl_at_0.25_08b": kb, "fold_difference": fold,
                     "random_over_concept_2b": r2,
                     "holds": bool(fold and fold > 2 and r2 and r2 > 1)}
    json.dump(rep, open(ROOT / "results/p2n_analysis.json", "w"), indent=1, default=float)
    print(json.dumps({m: (v if isinstance(v, str) else {k: v[k] for k in v if k != "median_kl"})
                      for m, v in rep.items()}, indent=1, default=float)[:3000])


if __name__ == "__main__":
    main()
