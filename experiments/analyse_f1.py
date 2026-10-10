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


# Addendum 1 (filed 9 Oct, before any F-1 row): every verdict also scored with 1/gamma of the layer
# norm feeding the MLP projected out of the fit and of w, and each fit's share on 1/gamma, per arm.
BASE = {"b15a": ("gpt2", 6, "results/b15a_fitseed1"),
        "b8b": ("EleutherAI/gpt-neo-125M", 10, "results/b8b_gptneo125m_indep")}


def identifiable(npz, picked, u):
    z = np.load(npz)
    w = z["w"].astype(np.float64)
    v = z[picked][:, 0].astype(np.float64)
    v /= np.linalg.norm(v)
    P = lambda x: x - u * (u @ x)
    pv, pw = P(v), P(w)
    return float(abs(pv @ pw) / (np.linalg.norm(pv) * np.linalg.norm(pw))), float((u @ v) ** 2)


def addendum_1(a, groups, fits):
    import sys
    sys.path.insert(0, str(ROOT / "experiments"))
    from analyse_ln_null import null_direction
    base_rows = {tag: load(f"{stem}.jsonl") for tag, (_, _, stem) in BASE.items()}
    score = {}
    for (tag, unit) in [k for g in groups.values() for k in g]:
        model, layer, stem = BASE[tag]
        u, _ = null_direction(model, layer)
        b = base_rows[tag].get(unit)
        if b is None:
            continue
        rec = {"base": identifiable(ROOT / f"{stem}_dirs/n{unit}.npz", b["picked"], u)}
        for arm in ARMS:
            r = fits[arm].get((tag, unit))
            if r is not None:
                rec[arm] = identifiable(ROOT / f"results/f1_{tag}_{arm}_dirs/n{unit}.npz", r["picked"], u)
        score[(tag, unit)] = rec
    out = {}
    for g, ks in groups.items():
        ks = [k for k in ks if k in score and all(arm in score[k] for arm in ARMS)]
        if not ks:
            continue
        res = {"scored": len(ks),
               "base_pass_identifiable": int(sum(score[k]["base"][0] >= PASS for k in ks)),
               **{f"{arm}_pass_identifiable": int(sum(score[k][arm][0] >= PASS for k in ks)) for arm in ARMS},
               "median_null_share": {s: float(np.median([score[k][s][1] for k in ks]))
                                     for s in ("base", *ARMS)}}
        if g == "converged_wrong":
            still = [k for k in ks if score[k]["base"][0] < PASS]
            res["still_wrong_on_identifiable_label"] = len(still)
            res["of_those_repaired"] = {arm: int(sum(score[k][arm][0] >= PASS for k in still)) for arm in ARMS}
            if still:
                t = np.array([score[k]["targeted"][0] >= PASS for k in still])
                n_ = np.array([score[k]["natural"][0] >= PASS for k in still])
                res["targeted_vs_natural_on_those"] = mcnemar_one_sided(t, n_)
        out[g] = res
    out["per_unit"] = {f"{t}:{u}": {s: [round(x, 4) for x in v] for s, v in rec.items()}
                       for (t, u), rec in score.items()}
    return out


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
    rep["addendum_1_identifiable"] = addendum_1(a, groups, fits)
    rep["per_unit"] = {f"{t}:{u}": {"group": g, **{arm: fits[arm].get((t, u), {}).get(
        "align_selected") for arm in ARMS}} for g, ks in groups.items() for t, u in ks}
    print(json.dumps({k: v for k, v in rep.items() if k != "per_unit"}, indent=1))
    out = Path(a.out)
    json.dump(rep, open(out if out.is_absolute() else ROOT / out, "w"), indent=1)


if __name__ == "__main__":
    main()
