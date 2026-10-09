"""S-1: are dead vectors a read-position effect or a precision artefact?
(docs/preregistration-s1-dead-vectors.md, Amendments 1-4; P2-G secondary)

    python experiments/analyse_s1.py --dir results/s1_gemma12 --model gemma12
    python experiments/analyse_s1.py --dir results/s1_gemma27 --model gemma27 --precisions 4bit

File names are the run sheet's: s1_<model>_<prec>_<arm>.jsonl with the stage tag the script
adds (_steer), the recipe tag for the sentence arm (_aperture), and the quant tag (none for
4-bit, _8bit, _none for fp16).

A vector passes at a dose if its steer-stage trial is steered there and not at alpha 0. Doses
are read by grid position: 1 = 0.05 nats, 2 = 0.5 nats (the filed gate), 3 = 5 nats
(co-primary, Amendment 4). Per 12B precision cell, exact one-sided McNemar on the 30 concepts
(tail fails while concept passes, against the reverse); cells Fisher-combined. The criterion
holds at a dose if every cell's difference is in the predicted direction and the combined p
is below 0.025.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binom, chi2, rankdata

ROOT = Path(__file__).resolve().parents[1]
ARMS = {"tail": "tail_steer", "concept": "concept_steer", "sentence": "sentence_steer_aperture"}
QTAG = {"4bit": "", "8bit": "_8bit", "fp16": "_none"}
DOSES = {"0.05 nat": 1, "0.5 nat (gate)": 2, "5 nat": 3}
ALPHA = 0.025


def auc(score, live):
    score, live = np.asarray(score, float), np.asarray(live, bool)
    m, n = live.sum(), (~live).sum()
    if m == 0 or n == 0:
        return None
    r = rankdata(score)
    return float((r[live].sum() - m * (m + 1) / 2) / (m * n))


def stem(d, model, prec, arm):
    return Path(d) / f"s1_{model}_{prec}_{arm}"


def load_cell(d, model, prec):
    """Per arm: config, steer rows, unit vectors; None if the cell did not run."""
    out = {}
    for arm, tag in ARMS.items():
        base = f"{stem(d, model, prec, arm)}_{tag.split('_', 1)[1]}{QTAG[prec]}"
        rows_p, cfg_p, vec_p = Path(base + ".jsonl"), Path(base + ".config.json"), Path(base + ".vectors.npz")
        if not rows_p.exists():
            return None
        rows = [json.loads(l) for l in open(rows_p, encoding="utf-8") if l.strip()]
        cfg = json.load(open(cfg_p))
        z = np.load(vec_p, allow_pickle=True)
        vecs = {str(n): v / np.linalg.norm(v) for n, v in zip(z["names"], z["vectors"])}
        out[arm] = {"rows": rows, "cfg": cfg, "vecs": vecs}
    return out


def passes(cell, arm, dose):
    cfg, rows = cell[arm]["cfg"], cell[arm]["rows"]
    al = sorted(cfg["alphas"])
    pos = {a: i for i, a in enumerate(al)}
    by = {}
    for r in rows:
        by.setdefault(r["concept"], {})[pos[r["alpha"]]] = r
    return {c: bool(v[dose]["steered"]) and not bool(v[0]["steered"])
            for c, v in by.items() if dose in v and 0 in v}


def coherence(cell, dose):
    n = k = 0
    for arm in ARMS:
        al = sorted(cell[arm]["cfg"]["alphas"])
        for r in cell[arm]["rows"]:
            if r["alpha"] == al[dose]:
                n += 1
                k += bool(r.get("coherent"))
    return {"coherent": k, "of": n, "passes": n > 0 and k >= n / 2}


def mcnemar(tail, concept):
    cs = sorted(set(tail) & set(concept))
    b = sum((not tail[c]) and concept[c] for c in cs)     # tail dead, concept live
    c_ = sum(tail[c] and (not concept[c]) for c in cs)
    n = b + c_
    p = float(binom.sf(b - 1, n, 0.5)) if n else 1.0
    return {"concepts": len(cs), "tail_fail_concept_pass": b, "tail_pass_concept_fail": c_,
            "p_one_sided": p, "predicted_direction": b > c_}


def shared_share(cell, live):
    keys, V = [], []
    for arm in ARMS:
        for c, v in cell[arm]["vecs"].items():
            keys.append((arm, c))
            V.append(v)
    V = np.array(V, float)
    m = V.mean(0)
    s = (V @ (m / np.linalg.norm(m))) ** 2
    lab = [live.get(k) for k in keys]
    ok = [i for i, x in enumerate(lab) if x is not None]
    return auc(s[ok], np.array([lab[i] for i in ok]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--precisions", nargs="+", default=["4bit", "8bit", "fp16"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = {"model": a.model, "cells": {}, "criterion": {}}
    cells = {p: load_cell(a.dir, a.model, p) for p in a.precisions}
    for prec, cell in cells.items():
        if cell is None:
            rep["cells"][prec] = "not run (e.g. stopped by the finiteness probe)"
            continue
        r = {"pass_rate": {}, "mcnemar": {}, "manipulation_check": {}, "health_auc_at_gate": {},
             "p2g_shared_share_auc": {}}
        for dname, dose in DOSES.items():
            pa = {arm: passes(cell, arm, dose) for arm in ARMS}
            r["pass_rate"][dname] = {arm: [sum(v.values()), len(v)] for arm, v in pa.items()}
            r["mcnemar"][dname] = mcnemar(pa["tail"], pa["concept"])
            r["manipulation_check"][dname] = coherence(cell, dose)
            live = {(arm, c): x for arm, v in pa.items() for c, x in v.items()}
            r["p2g_shared_share_auc"][dname] = shared_share(cell, live)
            if dose == 2:
                for stat, key in (("stability", "stability"), ("probe", "probe"),
                                  ("logit steering", "steer_logit_delta"), ("norm", "norm")):
                    xs, ls = [], []
                    for arm in ARMS:
                        h = cell[arm]["cfg"].get("health", {})
                        for c, x in pa[arm].items():
                            if c in h and h[c].get(key) is not None:
                                xs.append(h[c][key])
                                ls.append(x)
                    r["health_auc_at_gate"][stat] = auc(xs, ls)
        rep["cells"][prec] = r
    run = [p for p in a.precisions if cells[p] is not None]
    for dname in DOSES:
        ps = [rep["cells"][p]["mcnemar"][dname] for p in run]
        if not ps:
            continue
        fisher = float(chi2.sf(-2 * sum(np.log(max(x["p_one_sided"], 1e-300)) for x in ps), 2 * len(ps)))
        rep["criterion"][dname] = {
            "cells": run, "fisher_p": fisher, "alpha": ALPHA,
            "all_in_predicted_direction": all(x["predicted_direction"] for x in ps),
            "holds": bool(all(x["predicted_direction"] for x in ps) and fisher < ALPHA
                          and len(run) == 3)}
    if {"4bit", "fp16"} <= set(run):
        rep["quantisation_branch"] = {
            dname: {"tail_pass_fp16": rep["cells"]["fp16"]["pass_rate"][dname]["tail"],
                    "tail_pass_4bit": rep["cells"]["4bit"]["pass_rate"][dname]["tail"]}
            for dname in DOSES}
    if len(run) > 1:
        drift = {}
        for arm in ARMS:
            pairs = [(p, q) for i, p in enumerate(run) for q in run[i + 1:]]
            drift[arm] = {f"{p}-{q}": float(np.median([abs(cells[p][arm]["vecs"][c] @ cells[q][arm]["vecs"][c])
                                                       for c in cells[p][arm]["vecs"]
                                                       if c in cells[q][arm]["vecs"]]))
                          for p, q in pairs}
        rep["precision_drift_median_abs_cos"] = drift
    print(json.dumps(rep, indent=1))
    out = Path(a.out or f"results/s1_{a.model}_analysis.json")
    json.dump(rep, open(out if out.is_absolute() else ROOT / out, "w"), indent=1)


if __name__ == "__main__":
    main()
