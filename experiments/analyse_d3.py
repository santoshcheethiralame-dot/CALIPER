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


def certification(d, keys):
    """D-3 Addendum 1: rule A (held-out R2 >= 0.995) against rule B (restart agreement >= 0.97) on
    the identifiable label. E1 = yield_A - yield_B (share of correct fits accepted); E2 = FCR_A - FCR_B
    (share of accepted fits that are wrong). Paired unit bootstrap, 2,000 resamples, default_rng(20261012)."""
    ok = np.array([d[k]["ident"] >= ln.PASS for k in keys])
    acc = {"A": np.array([d[k]["r2"] >= 0.995 for k in keys]), "B": np.array([d[k]["restart"] >= 0.97 for k in keys])}

    def stats(ix):
        o, y, f = ok[ix], {}, {}
        for r, a in acc.items():
            a = a[ix]
            y[r] = (a & o).sum() / o.sum() if o.sum() else np.nan
            f[r] = (a & ~o).sum() / a.sum() if a.sum() else np.nan
        return y["A"] - y["B"], f["A"] - f["B"]
    e1, e2 = stats(np.arange(len(keys)))
    rng = np.random.default_rng(20261012)
    b1, b2, dropped = [], [], 0
    for _ in range(NB):
        x, y = stats(rng.integers(0, len(keys), len(keys)))
        b1.append(x)
        if np.isnan(y):
            dropped += 1
        else:
            b2.append(y)
    ci1 = [float(np.nanpercentile(b1, 2.5)), float(np.nanpercentile(b1, 97.5))]
    ci2 = [float(np.percentile(b2, 2.5)), float(np.percentile(b2, 97.5))] if b2 else None
    wrong = int(sum((a & ~ok).sum() for a in acc.values()))
    if wrong < 5 or ci2 is None:
        r2 = "underpowered (fewer than 5 accepted-wrong fits across both rules)"
    else:
        r2 = "non-inferior" if ci2[1] < 0.03 else ("R2 worse" if ci2[0] > 0 else "inconclusive")
    return {"units": len(keys), "correct": int(ok.sum()),
            "accepted": {r: int(a.sum()) for r, a in acc.items()},
            "accepted_wrong": {r: int((a & ~ok).sum()) for r, a in acc.items()},
            "E1_yield_diff": float(e1), "E1_ci95": ci1, "E1_reading": "holds" if ci1[0] > 0 else "does not hold",
            "E2_fcr_diff": float(e2), "E2_ci95": ci2, "E2_dropped_resamples": dropped, "E2_reading": r2}


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
    rep["addendum 1, certification at 16k"] = certification(D[16000], keys)
    out = ROOT / ("results/d3_dev.json" if a.dev else "results/d3_analysis.json")
    json.dump(rep, open(out, "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in rep.items() if k.startswith("change")}, indent=1, default=float))


if __name__ == "__main__":
    main()
