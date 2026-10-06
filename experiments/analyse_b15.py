"""B-15: the error budget. How far do verdicts and signals move when one source of randomness
changes and nothing about the unit does? (docs/preregistration-b15-replicates.md)

    python experiments/analyse_b15.py                     # the three filed arms vs B-14
    python experiments/analyse_b15.py --arm b7l6=results/b7_layer06_gpt2.jsonl \
        --restart-arm results/b7_layer06_gpt2.jsonl --out results/b15_dev.json   # development

Endpoints, as filed:
1. Per arm: the verdict flip rate against B-14 on shared units, with an exact CI, at the 0.95
   bar and swept from 0.90 to 0.98.
2. Per arm: AUC(restart) - AUC(held-out R2) with DeLong, beside B-14's own gap on the same
   units.
3. Test-retest: ICC(A,1) between B-14 and each arm, for alignment and each signal.
4. B-15a restart curve:
   - restart-agreement AUC from the first k of 5 stored restarts;
   - the two-class "unreachable units" fit on the within-run direct-route pass rates at
     k = 2 and k = 5, read from the saved restarts, so no pairing across runs is needed.
5. B-15c: the shift in held-out R2 under the sequence-level split, and whether R2's AUC falls
   below restart agreement's. That is the one directional claim the prereg makes.

Developed on B-14 against B-7 L6 (coupled, 5 restarts, no saved directions), which exercises
everything except the within-run pass@k.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).parent))
from analyse_cross_arm import restart_curve, two_class  # noqa: E402
from b1_signal_calibration import PASS, delong, roc  # noqa: E402

SIGNALS = {"held-out R2": "r2_k1", "restart agreement": "stability",
           "route agreement": "route_agreement"}
BARS = (0.90, 0.92, 0.95, 0.97, 0.98)
FILED = {"B-15a (fit seed 1, 5 restarts)": "results/b15a_fitseed1.jsonl",
         "B-15b (corpus seed 1)": "results/b15b_corpusseed1.jsonl",
         "B-15c (sequence split)": "results/b15c_seqsplit.jsonl"}


def load(path):
    return {r["_key"]: r for r in (json.loads(l) for l in open(path, encoding="utf-8"))
            if "_key" in r}


def clopper_pearson(k, n, a=0.05):
    lo = beta.ppf(a / 2, k, n - k + 1) if k else 0.0
    hi = beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return [round(float(lo), 4), round(float(hi), 4)]


def icc_a1(x, y):
    """ICC(A,1): two-way random effects, absolute agreement, single measurement."""
    m = np.column_stack([x, y]).astype(float)
    n, k = m.shape
    grand = m.mean()
    msr = k * ((m.mean(1) - grand) ** 2).sum() / (n - 1)
    msc = n * ((m.mean(0) - grand) ** 2).sum() / (k - 1)
    sse = ((m - m.mean(1, keepdims=True) - m.mean(0, keepdims=True) + grand) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def gap(rows):
    """AUC(restart) - AUC(R2) at the 0.95 bar, DeLong p."""
    fail = [r["align_selected"] < PASS for r in rows]
    if not 0 < sum(fail) < len(fail):
        return None
    sa = [-r["stability"] for r in rows]
    sb = [-r["r2_k1"] for r in rows]
    d = delong(sa, sb, fail)
    return {"failures": int(sum(fail)), "auc_restart": round(d[0], 4),
            "auc_r2": round(d[1], 4), "diff": round(d[2], 4), "p": float(d[5])}


def compare(ref, arm):
    shared = sorted(set(ref) & set(arm))
    R = [ref[u] for u in shared]
    A = [arm[u] for u in shared]
    out = {"shared_units": len(shared), "flips": {}}
    for bar in BARS:
        f = sum((r["align_selected"] >= bar) != (a["align_selected"] >= bar)
                for r, a in zip(R, A))
        out["flips"][str(bar)] = {"flips": int(f), "rate": round(f / len(shared), 4),
                                  "ci95": clopper_pearson(f, len(shared))}
    out["gap_reference"] = gap(R)
    out["gap_arm"] = gap(A)
    if out["gap_reference"] and out["gap_arm"]:
        out["gap_change"] = round(out["gap_arm"]["diff"] - out["gap_reference"]["diff"], 4)
    out["icc"] = {"alignment": round(icc_a1([r["align_selected"] for r in R],
                                            [a["align_selected"] for a in A]), 4)}
    for name, f in SIGNALS.items():
        if all(f in r for r in R) and all(f in a for a in A):
            out["icc"][name] = round(icc_a1([r[f] for r in R], [a[f] for a in A]), 4)
    out["median_r2_shift"] = round(float(np.median([a["r2_k1"] - r["r2_k1"]
                                                    for r, a in zip(R, A)])), 6)
    return out


def within_run_passes(path, ks=(2, 3, 4, 5)):
    """Direct-route pass rate when only the first k restarts exist: best held-out R2 among
    them, scored against w. Needs the saved directions (`<out>_dirs/n<id>.npz`)."""
    dirs = Path(path.replace(".jsonl", "_dirs"))
    if not dirs.exists():
        return None
    rates = {k: [] for k in ks}
    for f in sorted(dirs.glob("n*.npz")):
        d = np.load(f)
        w = d["w"] / np.linalg.norm(d["w"])
        rs, r2 = d["direct_restarts"], d["direct_r2_restarts"]
        if len(rs) < max(ks):
            continue
        for k in ks:
            best = rs[int(np.argmax(r2[:k]))].reshape(-1)
            rates[k].append(abs(best @ w) / np.linalg.norm(best) >= PASS)
    if not rates[ks[0]]:
        return None
    out = {f"pass@{k}": round(float(np.mean(v)), 4) for k, v in rates.items()}
    out["units"] = len(rates[ks[0]])
    out["two_class"] = two_class(out["pass@2"], out["pass@5"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="results/b14_primary_gpt2_indep.jsonl")
    ap.add_argument("--arm", action="append", default=None,
                    help="name=path; repeatable. Default: the three filed B-15 arms")
    ap.add_argument("--restart-arm", default=FILED["B-15a (fit seed 1, 5 restarts)"])
    ap.add_argument("--split-arm", default=FILED["B-15c (sequence split)"])
    ap.add_argument("--out", default="results/b15_analysis.json")
    a = ap.parse_args()
    arms = dict(x.split("=", 1) for x in a.arm) if a.arm else FILED
    ref = load(a.reference)
    rep = {"reference": a.reference, "arms": {}}
    for name, path in arms.items():
        if not Path(path).exists():
            rep["arms"][name] = "missing"
            continue
        rep["arms"][name] = compare(ref, load(path))
    if Path(a.restart_arm).exists():
        rep["restart_curve"] = restart_curve(a.restart_arm)
        rep["within_run_passes"] = within_run_passes(a.restart_arm)
    if Path(a.split_arm).exists():
        c = rep["arms"].get(next((n for n, p in arms.items() if p == a.split_arm), ""), {})
        g = c.get("gap_arm") if isinstance(c, dict) else None
        if g:
            rep["split_verdict"] = (
                "R2 BELOW restart agreement under a sequence-level split: the paper's "
                "recommendation fails under a clean split (headline limitation)"
                if g["auc_r2"] < g["auc_restart"] else
                "R2 stays ahead of restart agreement under a sequence-level split")
    Path(a.out).write_text(json.dumps(rep, indent=1))
    for name, c in rep["arms"].items():
        if c == "missing":
            print(f"{name}: not run yet")
            continue
        f95 = c["flips"]["0.95"]
        print(f"{name}: {c['shared_units']} shared, flips@0.95 {f95['flips']} "
              f"({f95['rate']:.1%}, {f95['ci95']}), gap ref {c['gap_reference'] and c['gap_reference']['diff']}"
              f" -> arm {c['gap_arm'] and c['gap_arm']['diff']}, ICC align {c['icc']['alignment']}")
    for k in ("restart_curve", "within_run_passes", "split_verdict"):
        if k in rep:
            print(k, rep[k])
    print("wrote", a.out)


if __name__ == "__main__":
    main()
