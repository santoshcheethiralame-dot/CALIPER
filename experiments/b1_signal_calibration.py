"""B-1/B-2 - score every ground-truth-free reliability signal against the answer key.

The bench's central question. A practitioner without ground truth has to decide which
recovered directions to trust, and the field offers several checks for this: rerun from
several random starts and see whether the answers agree, look at held-out fit, compare
two estimation routes, test whether a second dimension helps. None of them has been
scored against a known-correct direction, because doing that requires the ground truth
whose absence is the reason the check exists.

The weight column supplies that ground truth, so every one of them can be scored here.

Criteria for B-1 are pre-registered in docs/preregistration-b1-stability-calibration.md
and were filed before the run started.

Usage:
    python experiments/b1_signal_calibration.py results/b1_stability_gpt2.jsonl [more.jsonl ...]

Reads only. No compute, no GPU, no model.
"""
import json
import sys
from statistics import mean

PASS = 0.95  # the pre-registered alignment bar, unchanged since E0.1
K2_BAR = 0.01

# Two different questions, and conflating them has already cost this project once (C43).
#
#   "recovery"  - did the estimator find the RIGHT DIRECTION?  alignment > 0.95 alone.
#   "gate"      - E0.1's filed pass criterion, which ALSO required the unit to be well
#                 described by one dimension: alignment > 0.95 AND k2_gain < 0.01.
#
# The bench asks the first. E0.1 asked the second, and its reported pass rates (GPT-2
# 77/100, Pythia 93/100) are gate numbers. The AUCs already in the notebook are
# alignment-labelled. Both are correct for their own question; reporting them beside each
# other without saying which is which is what invites the error.
#
# k2_gain is EXCLUDED as a signal under the gate label, because it appears in that label
# and would be predicting itself.

# Every signal is oriented so that HIGHER means MORE SUSPECT, which is what makes the
# AUCs comparable. A signal a practitioner reads as "good" is negated here.
SIGNALS = {
    "-held-out R2":      lambda r: -r["r2_k1"],
    "disagreement":      lambda r: r["disagreement"],
    "k2 gain":           lambda r: r["k2_gain"],
    "-restart agreement": lambda r: -r["stability"],
    "restart R2 spread": lambda r: r["r2_spread"],
}


def roc(scores, labels):
    """ROC points and AUC by the Mann-Whitney identity. labels: 1 = failure.

    Ties contribute 0.5, which matters here: restart agreement saturates at 1.0 on
    well-behaved units and a tie-blind AUC would flatter it.
    """
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return [], float("nan")
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    auc = wins / (len(pos) * len(neg))
    pts = [(t,
            sum(1 for s, y in zip(scores, labels) if y and s >= t) / len(pos),
            sum(1 for s, y in zip(scores, labels) if not y and s >= t) / len(neg))
           for t in sorted(set(scores))]
    return pts, auc


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def spec_sheet(path, criterion="recovery"):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if criterion == "gate":
        fail = [not (r["align_selected"] > PASS and r["k2_gain"] < K2_BAR) for r in rows]
    else:
        fail = [r["align_selected"] < PASS for r in rows]
    n, nf = len(rows), sum(fail)

    lo, hi = wilson(n - nf, n)
    print(f"\n{'=' * 78}\n  {path}\n{'=' * 78}")
    bar = (f"alignment > {PASS} AND k2_gain < {K2_BAR}  (E0.1 gate criterion)"
           if criterion == "gate" else f"alignment > {PASS}  (direction recovery)")
    print(f"  recovered   {n - nf}/{n}   Wilson 95% [{lo:.3f}, {hi:.3f}]")
    print(f"  criterion   {bar}")
    if nf:
        worst = min(r["align_selected"] for r in rows)
        med = sorted(r["align_selected"] for r in rows)[n // 2]
        print(f"  median alignment {med:.4f}   worst unit {worst:.4f}")
        print(f"  -> the median is what a paper reports; the worst unit is in the same population")
    restarts = {r.get("n_restarts") for r in rows}
    print(f"  restarts per fit: {sorted(x for x in restarts if x is not None) or 'not recorded'}")

    if not nf or nf == n:
        print("  degenerate: no contrast to score signals against")
        return

    print(f"\n  {'signal':<22}{'AUC':>7}{'TPR@FPR=.10':>13}{'good lost @70% catch':>22}")
    print("  " + "-" * 62)
    scored = []
    for name, fn in SIGNALS.items():
        if criterion == "gate" and "k2" in name:
            print(f"  {name:<22}{'--':>7}   excluded: appears in the label, would predict itself")
            continue
        try:
            sc = [float(fn(r)) for r in rows]
        except (KeyError, TypeError):
            print(f"  {name:<22}{'--':>7}   not recorded in this file")
            continue
        if len(set(sc)) == 1:
            print(f"  {name:<22}{'--':>7}   constant, carries no information")
            continue
        pts, auc = roc(sc, fail)
        at10 = max((p[1] for p in pts if p[2] <= 0.10), default=0.0)
        catch = [p for p in pts if p[1] >= 0.70]
        lost = f"{min(p[2] for p in catch) * (n - nf):.0f} of {n - nf}" if catch else "unreachable"
        print(f"  {name:<22}{auc:>7.3f}{at10:>13.2f}{lost:>22}")
        scored.append((auc, name, sc))

    if not scored:
        return
    scored.sort(reverse=True)
    best_auc, best_name, _ = scored[0]
    print(f"\n  best: {best_name} at AUC {best_auc:.3f}")

    stab = [s for a, nm, s in scored if "restart" in nm]
    r2 = [(a, nm) for a, nm, s in scored if "held-out" in nm]
    if stab and r2:
        r2_auc = r2[0][0]
        gap = r2_auc - max(a for a, nm, s in scored if "restart" in nm)
        verdict = ("held-out R2 dominates restart agreement" if gap > 0.05 else
                   "restart agreement matches held-out R2" if abs(gap) <= 0.05 else
                   "restart agreement BEATS held-out R2 - Finding 2 needs amending")
        print(f"  {verdict} (gap {gap:+.3f})")

    print(f"\n  mean signal value on failures vs passes:")
    for auc, name, sc in scored:
        f_, p_ = mean(s for s, y in zip(sc, fail) if y), mean(s for s, y in zip(sc, fail) if not y)
        print(f"    {name:<22} failures {f_:>10.4f}   passes {p_:>10.4f}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        raise SystemExit(__doc__)
    crit = "gate" if "--gate" in sys.argv else "recovery"
    for p in args:
        spec_sheet(p, crit)
