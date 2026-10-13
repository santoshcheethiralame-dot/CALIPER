"""Score P2-H once, as filed (docs/preregistration-p2h-hahami-llama.md).

    python experiments/analyse_p2h.py --file <unzipped>/p2h_llama8b.jsonl

Checks M2-M4 first; the primary cell (layer 0, strength 5, released readout) is scored only if they
pass. Every other cell, both readouts and Hahami et al.'s own metric are descriptive.
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

from rescore_steering_gate import hits

INTRO = "hahami_introspective"
FACTUAL = ("factual_no", "factual_yes", "factual_contested")
PRIMARY = (0, 5.0)


def wilcoxon(x):
    x = np.asarray(x, float)
    if not np.any(x):
        return {"n": len(x), "median_diff": 0.0, "p": 1.0}
    return {"n": len(x), "median_diff": float(np.median(x)),
            "p": float(stats.wilcoxon(x, alternative="two-sided").pvalue)}


def excess(fx, fy, ix, iy):
    slope, icpt = np.polyfit(fx, fy, 1)
    return float(np.median(np.asarray(iy) - (icpt + slope * np.asarray(ix))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--out", default="results/p2h_analysis.json")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()

    R = [json.loads(l) for l in open(a.file) if l.strip()]
    side_path = a.file.replace(".jsonl", "_sidecar.json")
    side = json.load(open(side_path)) if os.path.exists(side_path) else {}
    forced = [r for r in R if r["stage"] == "forced"]
    clean = {(r["framing"], r["i"]): r for r in forced if r["arm"] == "clean"}
    cell = defaultdict(dict)                 # (layer, alpha, arm, framing) -> {i: row}
    for r in forced:
        if r["arm"] != "clean":
            cell[(r["layer"], r["alpha"], r["arm"], r["framing"])][r["i"]] = r

    def base(f, key="logodds_rel"):
        return np.array([clean[k][key] for k in sorted(clean) if k[0] == f])

    checks = {}
    mass = {f: float(np.median(base(f, "mass_rel"))) for f in (INTRO, *FACTUAL)}
    checks["M2"] = {"median_mass": mass, "pass": all(m >= 0.5 for m in mass.values())}
    fy, fn = float(np.median(base("factual_yes"))), float(np.median(base("factual_no")))
    checks["M3"] = {"factual_yes": fy, "factual_no": fn, "pass": fy > 0 > fn}
    imp0 = [r for (L, al, arm, f), rows in cell.items() if L == 0 and arm == "random_impact"
            and f == INTRO for r in rows.values()]
    share = float(np.mean([bool(r.get("matched")) for r in imp0])) if imp0 else 0.0
    checks["M4"] = {"matched_share_layer0": share, "pass": share >= 0.9}
    ok = all(c["pass"] for c in checks.values())

    # Liveness, strict rule: steered at strength 5 beyond the concept's alpha-0 text.
    steer = [r for r in R if r["stage"] == "steer"]
    t0 = {r["i"]: r["text"] for r in steer if r["alpha"] == 0}
    liveness, live = defaultdict(dict), defaultdict(set)
    for r in steer:
        if r["alpha"]:
            on = bool(hits(r["text"], r["concept"]) - hits(t0.get(r["i"], ""), r["concept"]))
            key = str(r["layer"])
            liveness[key][str(r["alpha"])] = liveness[key].get(str(r["alpha"]), 0) + int(on)
            if on and r["alpha"] == 5.0:
                live[r["layer"]].add(r["i"])

    def change(layer, alpha, arm, f, key="logodds_rel"):
        return {i: r[key] - clean[(f, i)][key] for i, r in cell[(layer, alpha, arm, f)].items()}

    def endpoints(layer, alpha, keep=None, key="logodds_rel"):
        """P0-P3 at one cell; `keep(i)` restricts concepts (sensitivity analyses)."""
        keep = keep or (lambda i: True)
        real = {i: v for i, v in change(layer, alpha, "real", INTRO, key).items() if keep(i)}
        ids = sorted(real)
        if len(ids) < 5:
            return {"n_concepts": len(ids), "note": "too few concepts"}
        inj = cell[(layer, alpha, "real", INTRO)]
        out = {"n_concepts": len(ids)}
        n_yes = int(sum(inj[i][key] > 0 for i in ids))
        med = float(np.median([real[i] for i in ids]))
        out["P0"] = {"median_change": med, "n_yes": n_yes,
                     "pass": med >= 2 and n_yes >= int(np.ceil(len(ids) / 2))}
        rnd = change(layer, alpha, "random_impact", INTRO, key)
        out["P1"] = wilcoxon([real[i] - rnd[i] for i in ids if i in rnd])
        nrm = change(layer, alpha, "random", INTRO, key)
        out["real_vs_norm_random"] = wilcoxon([real[i] - nrm[i] for i in ids if i in nrm])
        fch = {f: change(layer, alpha, "real", f, key) for f in FACTUAL}

        def pooled(sel):
            fx, fyv, ix, iy = [], [], [], []
            for i in sel:
                for f in FACTUAL:
                    if i in fch[f]:
                        fx.append(clean[(f, i)][key])
                        fyv.append(fch[f][i])
                ix.append(clean[(INTRO, i)][key])
                iy.append(real[i])
            return fx, fyv, ix, iy

        est = excess(*pooled(ids))
        rng = np.random.default_rng(0)
        boots = [excess(*pooled(rng.choice(ids, len(ids), replace=True))) for _ in range(a.boot)]
        ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        out["P2"] = {"excess": est, "ci95": ci, "specific": not (ci[0] <= 0 <= ci[1])}
        fyc = [fch["factual_yes"][i] for i in ids if i in fch["factual_yes"]]
        out["P3"] = {**wilcoxon(fyc), "median": float(np.median(fyc))}
        return out

    L, al = PRIMARY
    prim = {"layer": L, "alpha": al, "scored": ok}
    if ok:
        prim.update(endpoints(L, al))
        answering = {i for i, r in cell[(L, al, "real", INTRO)].items() if r["mass_rel"] >= 0.5}
        prim["sensitivity"] = {
            "answering_items": endpoints(L, al, keep=lambda i: i in answering),
            "live_only": endpoints(L, al, keep=lambda i: i in live[L])}

    cells = {}
    for (layer, alpha) in sorted({(k[0], k[1]) for k in cell}):
        c = {"released": endpoints(layer, alpha),
             "single_pass": endpoints(layer, alpha, key="logodds_single")}
        c["ld_hahami_median"] = {f: {arm: float(np.median(list(change(layer, alpha, arm, f, "ld_hahami").values())))
                                     for arm in ("real", "random", "random_impact")
                                     if cell[(layer, alpha, arm, f)]}
                                 for f in (INTRO, *FACTUAL)}
        c["kl_single_median"] = {arm: float(np.median([r["kl_single"] for r in cell[(layer, alpha, arm, INTRO)].values()]))
                                 for arm in ("real", "random", "random_impact") if cell[(layer, alpha, arm, INTRO)]}
        rel, sgl = change(layer, alpha, "real", INTRO), change(layer, alpha, "real", INTRO, "logodds_single")
        c["released_minus_single"] = float(np.median([rel[i] - sgl[i] for i in rel if i in sgl]))
        cells[f"L{layer}_a{alpha}"] = c

    hahami_r = {}
    for layer in sorted({k[0] for k in cell}):
        al_s = sorted({k[1] for k in cell if k[0] == layer})
        xi = [np.mean(list(change(layer, s, "real", INTRO, "ld_hahami").values())) for s in al_s]
        xn = [np.mean(list(change(layer, s, "real", "factual_no", "ld_hahami").values())) for s in al_s]
        hahami_r[str(layer)] = float(np.corrcoef(xi, xn)[0, 1]) if len(al_s) > 2 else None

    out = {"file": a.file, "checks": checks, "primary": prim, "cells": cells,
           "liveness": liveness, "hahami_r_intro_vs_factual_no": hahami_r,
           "fidelity": side.get("fidelity"), "baseline_ld_intro_median":
           float(np.median(base(INTRO, "ld_hahami"))) if clean else None,
           "load": side.get("load")}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("checks: " + ", ".join(f"{k} {'pass' if v['pass'] else 'FAIL'}" for k, v in checks.items()))
    for k in ("P0", "P1", "P2", "P3"):
        print(f"{k} @ L{L} a{al}: {prim.get(k, 'not scored')}")


if __name__ == "__main__":
    main()
