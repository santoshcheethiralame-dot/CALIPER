"""Score D-3 (docs/preregistration-d3-token-scaling.md) once, when every budget's 150 rows exist.

    PYTHONPATH=. python experiments/analyse_d3.py                  # writes results/d3_analysis.json
    PYTHONPATH=. python experiments/analyse_d3.py --dev            # development run on B-15a / B-15b

Primary: on the identifiable label (1/gamma of ln_2 removed from fit and reference), Delta(T) =
AUC(restart agreement) - AUC(held-out R2) over all failures vs passes at each budget T, and the
change Delta(16k) - Delta(8k), with a paired bootstrap over units (the same units at every budget).
The lead shrinks if that interval lies above 0. Fewer than 10 identifiable failures at 8k or 16k:
underpowered.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))
import analyse_ln_null as ln  # noqa: E402
from analyse_review_round1 import auc  # noqa: E402

BUDGETS = (2000, 4000, 8000, 16000)
STEM = "results/d3_gpt2_l6_t{}"
N, MIN_FAIL, NB = 150, 10, 2000


def units(stem):
    R = [json.loads(l) for l in open(ROOT / f"{stem}.jsonl", encoding="utf-8") if l.strip()]
    u, _ = ln.null_direction("gpt2", 6)
    P = lambda x: x - u * (u @ x)
    out = {}
    for r in R:
        z = np.load(ROOT / f"{stem}_dirs/n{r['_key']}.npz")
        w = z["w"].astype(np.float64)
        v = z[r["picked"]][:, 0].astype(np.float64)
        pv, pw = P(v), P(w)
        out[r["_key"]] = {"ident": abs(pv @ pw) / (np.linalg.norm(pv) * np.linalg.norm(pw)),
                          "raw": abs(v @ w) / (np.linalg.norm(v) * np.linalg.norm(w)),
                          "r2": r["r2_k1"], "restart": r["stability"]}
    return out


def delta(d, keys, lab):
    fail = np.array([d[k][lab] < ln.PASS for k in keys])
    if not 0 < fail.sum() < len(fail):
        return None
    st, r2 = np.array([d[k]["restart"] for k in keys]), np.array([d[k]["r2"] for k in keys])
    return auc(-st, fail) - auc(-r2, fail)


def ci(xs):
    xs = [x for x in xs if x is not None]
    return [float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))] if len(xs) > NB // 2 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true", help="B-15a as 8k and B-15b as 16k; no row-count check")
    a = ap.parse_args()
    stems = ({8000: "results/b15a_fitseed1", 16000: "results/b15b_corpusseed1"} if a.dev
             else {t: STEM.format(t) for t in BUDGETS})
    D = {t: units(s) for t, s in stems.items()}
    if not a.dev and any(len(d) != N for d in D.values()):
        raise SystemExit({t: len(d) for t, d in D.items()})
    keys = sorted(set.intersection(*[set(d) for d in D.values()]))
    rng = np.random.default_rng(0)
    boots = [rng.choice(keys, len(keys)) for _ in range(NB)]
    rep = {"dev": a.dev, "units": len(keys), "by_budget": {}}
    for t, d in D.items():
        row = {}
        for lab, name in (("ident", "identifiable"), ("raw", "as filed")):
            fail = np.array([d[k][lab] < ln.PASS for k in keys])
            r2 = np.array([d[k]["r2"] for k in keys])
            row[name] = {"failures": int(fail.sum()), "converged_wrong": int((fail & (r2 > ln.WB)).sum()),
                         "under_fitted": int((fail & (r2 <= ln.WB)).sum()), "delta": delta(d, keys, lab),
                         "delta_ci95": ci([delta(d, list(b), lab) for b in boots])}
        rep["by_budget"][str(t)] = row
    for lab, name in (("ident", "identifiable"), ("raw", "as filed")):
        d8, d16 = D[8000], D[16000]
        diffs = []
        for b in boots:
            x, y = delta(d16, list(b), lab), delta(d8, list(b), lab)
            diffs.append(None if x is None or y is None else x - y)
        point = delta(d16, keys, lab) - delta(d8, keys, lab)
        nf = min(rep["by_budget"]["8000"][name]["failures"], rep["by_budget"]["16000"][name]["failures"])
        c = ci(diffs)
        reading = ("underpowered" if nf < MIN_FAIL else "lead shrinks" if c and c[0] > 0 else
                   "lead grows" if c and c[1] < 0 else "no change detected")
        rep[f"change 8k to 16k, {name}"] = {"delta_16k_minus_8k": point, "ci95_paired": c, "reading": reading}
    out = ROOT / ("results/d3_dev.json" if a.dev else "results/d3_analysis.json")
    json.dump(rep, open(out, "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in rep.items() if k.startswith("change")}, indent=1, default=float))


if __name__ == "__main__":
    main()
