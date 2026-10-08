"""F-1: does interventional data repair converged-wrong fits? (docs/preregistration-f1-interventional.md)

    python experiments/analyse_f1.py                         # the filed runs, once
    python experiments/analyse_f1.py --pattern "<dir>/pilot_{arm}.jsonl" --tags b15a   # development

Units and their base labels come from results/f1_units_<tag>.json. Pass = alignment >= 0.95.
Primary, on the base converged-wrong units pooled over both tags: the share passing under the
targeted arm (prediction >= 21 of 26), and targeted against natural by an exact one-sided
McNemar test (prediction p < 0.05). Both must hold.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("targeted", "random", "natural")
PASS = 0.95


def load(path):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        return {}
    return {json.loads(l)["_key"]: json.loads(l) for l in open(p, encoding="utf-8") if l.strip()}


def mcnemar_one_sided(a_pass, b_pass):
    """Exact one-sided McNemar: is arm a's pass rate above arm b's on paired units?"""
    a_only = int(np.sum(a_pass & ~b_pass))
    b_only = int(np.sum(b_pass & ~a_pass))
    n = a_only + b_only
    p = float(binom.sf(a_only - 1, n, 0.5)) if n else 1.0
    return {"a_only": a_only, "b_only": b_only, "p_one_sided": p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="results/f1_{tag}_{arm}.jsonl")
    ap.add_argument("--tags", nargs="+", default=["b15a", "b8b"])
    ap.add_argument("--out", default="results/f1_analysis.json")
    a = ap.parse_args()

    groups = {"converged_wrong": [], "under_fitted": [], "passing_draw": []}
    fits = {arm: {} for arm in ARMS}
    for tag in a.tags:
        units = json.load(open(ROOT / f"results/f1_units_{tag}.json"))
        for g in groups:
            groups[g] += [(tag, u) for u in units[g]]
        for arm in ARMS:
            for k, r in load(a.pattern.format(tag=tag, arm=arm)).items():
                fits[arm][(tag, k)] = r

    def passed(arm, keys):
        return np.array([fits[arm][k]["align_selected"] >= PASS for k in keys])

    rep = {"missing": {arm: [f"{t}:{u}" for g in groups.values() for t, u in g
                             if (t, u) not in fits[arm]] for arm in ARMS}}
    cw = [k for k in groups["converged_wrong"] if all(k in fits[arm] for arm in ARMS)]
    rep["converged_wrong_scored"] = len(cw)
    if cw:
        t, n_, r_ = passed("targeted", cw), passed("natural", cw), passed("random", cw)
        mc = mcnemar_one_sided(t, n_)
        rep["primary"] = {
            "targeted_repaired": int(t.sum()), "of": len(cw),
            "natural_repaired": int(n_.sum()), "random_repaired": int(r_.sum()),
            "targeted_vs_natural": mc,
            "holds": bool(t.sum() >= 21 and mc["p_one_sided"] < 0.05)
            if len(cw) == 26 else "not scored: incomplete"}
        rep["secondary_arms"] = {"targeted_vs_random": mcnemar_one_sided(t, r_),
                                 "random_vs_natural": mcnemar_one_sided(r_, n_)}
    for g in ("under_fitted", "passing_draw"):
        keys = [k for k in groups[g] if all(k in fits[arm] for arm in ARMS)]
        rep[g] = {"scored": len(keys),
                  **{f"{arm}_pass": int(passed(arm, keys).sum()) for arm in ARMS}}
    if rep["passing_draw"]["scored"]:
        keys = [k for k in groups["passing_draw"] if all(k in fits[arm] for arm in ARMS)]
        rep["harm_targeted"] = {"failed": int((~passed("targeted", keys)).sum()),
                                "of": len(keys), "prediction": "at most 2 of 40"}
    rep["per_unit"] = {f"{t}:{u}": {"group": g, **{arm: fits[arm].get((t, u), {}).get(
        "align_selected") for arm in ARMS}} for g, ks in groups.items() for t, u in ks}
    print(json.dumps({k: v for k, v in rep.items() if k != "per_unit"}, indent=1))
    out = Path(a.out)
    json.dump(rep, open(out if out.is_absolute() else ROOT / out, "w"), indent=1)


if __name__ == "__main__":
    main()
