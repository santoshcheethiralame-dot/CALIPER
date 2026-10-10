"""F-1b scorer (docs/preregistration-f1b-identifiable.md), written before any F-1b data.

    PYTHONPATH=. python experiments/analyse_f1b.py      # writes results/f1b_analysis.json

Every verdict is on the identifiable label (1/gamma of the layer norm feeding the MLP removed from
fit and reference). Primary, on population A: targeted against reseed, exact one-sided McNemar on
paired verdicts (prediction: targeted passes more, p < 0.05). Secondary: targeted against natural
(two-sided), random against targeted, per-arm pass counts in every population, harm on population
C, and each arm's share on 1/gamma.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))
from analyse_f1 import identifiable, mcnemar_one_sided  # noqa: E402
from analyse_ln_null import null_direction  # noqa: E402

ARMS = ("targeted", "random", "natural", "reseed")
MODEL = {"neo10": ("EleutherAI/gpt-neo-125M", 10), "gpt2": ("gpt2", 6), "neo6": ("EleutherAI/gpt-neo-125M", 6)}
PASS = 0.95


def two_sided(a, b):
    a_only, b_only = int(np.sum(a & ~b)), int(np.sum(b & ~a))
    n = a_only + b_only
    p = float(min(1.0, 2 * binom.cdf(min(a_only, b_only), n, 0.5))) if n else 1.0
    return {"a_only": a_only, "b_only": b_only, "p_two_sided": p}


def main():
    sel = json.load(open(ROOT / "results/f1b_units.json"))
    score = {}
    for g, d in sel["groups"].items():
        u, _ = null_direction(*MODEL[g.split("_")[0]])
        for arm in ARMS:
            path = ROOT / f"results/f1b_{g}_{arm}.jsonl"
            if not path.exists():
                continue
            for l in open(path, encoding="utf-8"):
                if not l.strip():
                    continue
                r = json.loads(l)
                if str(r["_key"]) in d["units"]:
                    al, share = identifiable(ROOT / f"results/f1b_{g}_{arm}_dirs/n{r['_key']}.npz", r["picked"], u)
                    score.setdefault((g, r["_key"]), {"population": d["units"][str(r["_key"])]})[arm] = (al, share)
    rep = {"units_scored": len(score)}
    for pop in ("A", "B", "C"):
        ks = [k for k, v in score.items() if v["population"] == pop and all(a in v for a in ARMS)]
        if not ks:
            continue
        ok = {a: np.array([score[k][a][0] >= PASS for k in ks]) for a in ARMS}
        res = {"complete": len(ks), **{f"{a}_pass": int(ok[a].sum()) for a in ARMS},
               "median_null_share": {a: float(np.median([score[k][a][1] for k in ks])) for a in ARMS}}
        if pop == "A":
            mc = mcnemar_one_sided(ok["targeted"], ok["reseed"])
            res["primary_targeted_vs_reseed"] = {**mc, "holds": bool(mc["p_one_sided"] < 0.05)}
            res["targeted_vs_natural"] = two_sided(ok["targeted"], ok["natural"])
            res["random_vs_targeted"] = two_sided(ok["random"], ok["targeted"])
            res["natural_vs_reseed"] = two_sided(ok["natural"], ok["reseed"])
        if pop == "C":
            res["harm"] = {a: int((~ok[a]).sum()) for a in ARMS}
        rep[f"population {pop}"] = res
    rep["per_unit"] = {f"{g}:{u}": {a: [round(x, 4) for x in v[a]] if a in v else None for a in ARMS}
                       | {"population": v["population"]} for (g, u), v in score.items()}
    print(json.dumps({k: v for k, v in rep.items() if k != "per_unit"}, indent=1))
    json.dump(rep, open(ROOT / "results/f1b_analysis.json", "w"), indent=1)


if __name__ == "__main__":
    main()
