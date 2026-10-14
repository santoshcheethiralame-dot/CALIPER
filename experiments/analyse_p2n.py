"""Score P2-N (docs/preregistration-p2n-qwen35-current.md) once, when every file is complete.

    python experiments/analyse_p2n.py      # writes results/p2n_analysis.json

For each model (qwen35_2b primary, qwen35_08b secondary): the unchanged S-2 scorer on the filed
substring label, and on the strict gate (a copy of the steer rows rescored with
rescore_steering_gate.rescore); then the dose comparison of P2 from the forced rows (median
next-token KL per fraction, real arms and the norm-matched random arm).
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
        kl = {}
        for arm, stem in FORCED.items():
            R = rows(SRC / f"s2_{m}_{stem}.jsonl")
            kl[arm] = {f"{a:g}": float(np.median([r["kl"] for r in R if r["alpha"] == a and r["framing"] == "introspective"]))
                       for a in sorted({r["alpha"] for r in R}) if a} if R else None
        real = [kl[a] for a in ("concept", "tail", "sentence") if kl.get(a)]
        ratio = None
        if kl.get("random") and real and "0.5" in kl["random"]:
            med_real = float(np.median([k["0.5"] for k in real]))
            ratio = kl["random"]["0.5"] / med_real if med_real else None
        rep[m] = {"filed_substring_label": s2(SRC, m), "strict_label": s2(strict, m), "median_kl_introspective": kl,
                  "random_over_real_kl_at_0.5": ratio}
    a, b = rep.get("qwen35_2b"), rep.get("qwen35_08b")
    if isinstance(a, dict) and isinstance(b, dict):
        ka = np.median([v["0.25"] for k, v in a["median_kl_introspective"].items() if k != "random" and v])
        kb = np.median([v["0.25"] for k, v in b["median_kl_introspective"].items() if k != "random" and v])
        rep["P2"] = {"kl_at_0.25_2b": float(ka), "kl_at_0.25_08b": float(kb),
                     "fold_difference": float(max(ka, kb) / min(ka, kb)) if min(ka, kb) > 0 else None,
                     "random_over_real_2b": a["random_over_real_kl_at_0.5"]}
    json.dump(rep, open(ROOT / "results/p2n_analysis.json", "w"), indent=1, default=float)
    print(json.dumps({m: (v if isinstance(v, str) else {k: v[k] for k in v if k != "median_kl_introspective"})
                      for m, v in rep.items()}, indent=1, default=float)[:3000])


if __name__ == "__main__":
    main()
