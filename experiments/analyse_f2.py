"""F-2 scorer (docs/preregistration-f2-active.md), written before any F-2 data.

    PYTHONPATH=. python experiments/analyse_f2.py      # writes results/f2_analysis.json

The active arm (2,000 top-responding fresh tokens out of 20,000) against F-1b's arms on the same
units, every verdict on the identifiable label as in analyse_f1b.py. Primary, on F-1b's population B
(under-fitted units): active against natural, exact one-sided McNemar (prediction: active passes
more, p < 0.05). Secondary: active against reseed and against random, and harm on population C.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))
from analyse_f1 import identifiable, mcnemar_one_sided  # noqa: E402
from analyse_f1b import MODEL, PASS, two_sided  # noqa: E402
from analyse_ln_null import null_direction  # noqa: E402

ARMS = ("active", "natural", "reseed", "random", "targeted")


def path(g, arm):
    return ROOT / (f"results/f2_{g}_active" if arm == "active" else f"results/f1b_{g}_{arm}")


def main():
    sel = json.load(open(ROOT / "results/f1b_units.json"))
    score = {}
    for g, d in sel["groups"].items():
        u, _ = null_direction(*MODEL[g.split("_")[0]])
        for arm in ARMS:
            p = path(g, arm)
            if not p.with_suffix(".jsonl").exists():
                continue
            for l in open(p.with_suffix(".jsonl"), encoding="utf-8"):
                if not l.strip():
                    continue
                r = json.loads(l)
                pop = d["units"].get(str(r["_key"]))
                if pop in ("B", "C"):
                    al, _ = identifiable(Path(f"{p}_dirs") / f"n{r['_key']}.npz", r["picked"], u)
                    score.setdefault((g, r["_key"]), {"population": pop})[arm] = al
    rep = {"units_scored": sum("active" in v for v in score.values())}
    for pop in ("B", "C"):
        ks = [k for k, v in score.items() if v["population"] == pop and all(a in v for a in ARMS)]
        if not ks:
            continue
        ok = {a: np.array([score[k][a] >= PASS for k in ks]) for a in ARMS}
        res = {"complete": len(ks), **{f"{a}_pass": int(ok[a].sum()) for a in ARMS},
               "median_alignment": {a: float(np.median([score[k][a] for k in ks])) for a in ARMS}}
        if pop == "B":
            mc = mcnemar_one_sided(ok["active"], ok["natural"])
            res["primary_active_vs_natural"] = {**mc, "holds": bool(mc["p_one_sided"] < 0.05)}
            res["active_vs_reseed"] = two_sided(ok["active"], ok["reseed"])
            res["active_vs_random"] = two_sided(ok["active"], ok["random"])
        else:
            res["harm"] = {a: int((~ok[a]).sum()) for a in ARMS}
        rep[f"population {pop}"] = res
    rep["per_unit"] = {f"{g}:{u}": {a: round(v[a], 4) if a in v else None for a in ARMS}
                       | {"population": v["population"]} for (g, u), v in score.items()}
    print(json.dumps({k: v for k, v in rep.items() if k != "per_unit"}, indent=1))
    json.dump(rep, open(ROOT / "results/f2_analysis.json", "w"), indent=1)


if __name__ == "__main__":
    main()
