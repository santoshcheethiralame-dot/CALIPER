"""S1-1 - is the disagreement flag a usable decision rule?

The gate run (C13) established that units where the two estimation routes disagree fail
more often: 87% vs 55% pass, p=7.7e-4. That is a correlation, not a rule. A practitioner
with no ground truth needs a threshold and an operating characteristic: flag at
disagreement > t, catch X% of silent failures, discard Y% of good units.

Reads results/e01_gate.jsonl. No compute, no GPU.
"""
import json
from statistics import mean

PASS = 0.95          # the pre-registered alignment bar from E0.1


def roc(scores, labels):
    """ROC points and AUC by the Mann-Whitney identity. labels: 1 = failure."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return [], float("nan")
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    auc = wins / (len(pos) * len(neg))
    pts = []
    for t in sorted(set(scores)):
        tp = sum(1 for s, y in zip(scores, labels) if y and s >= t)
        fp = sum(1 for s, y in zip(scores, labels) if not y and s >= t)
        pts.append((t, tp / len(pos), fp / len(neg)))
    return pts, auc


rows = [json.loads(l) for l in open("results/e01_gate.jsonl", encoding="utf-8") if l.strip()]
fail = [r["align_selected"] < PASS for r in rows]
n_fail, n_ok = sum(fail), len(rows) - sum(fail)
print(f"{len(rows)} units: {n_fail} failures (alignment < {PASS}), {n_ok} passes")
print(f"  failure alignments: min {min(r['align_selected'] for r,f in zip(rows,fail) if f):.4f}, "
      f"max {max(r['align_selected'] for r,f in zip(rows,fail) if f):.4f}")

print("\n=== candidate predictors, all ground-truth-free ===")
cands = {
    "disagreement": [r["disagreement"] for r in rows],
    "-r2_k1": [-r["r2_k1"] for r in rows],
    "k2_gain": [r["k2_gain"] for r in rows],
}
aucs = {}
for name, sc in cands.items():
    _, auc = roc(sc, fail)
    aucs[name] = auc
    fm = mean(s for s, y in zip(sc, fail) if y)
    om = mean(s for s, y in zip(sc, fail) if not y)
    print(f"  {name:<15} AUC {auc:.3f}   mean on failures {fm:>9.4f}   on passes {om:>9.4f}")

print("\n=== disagreement as a decision rule: operating points ===")
sc = cands["disagreement"]
pts, auc = roc(sc, fail)
print(f"  {'threshold':>10} {'caught':>8} {'sensitivity':>12} {'good units lost':>17} {'flagged':>9}")
for target in (0.50, 0.60, 0.70, 0.80, 0.90, 1.00):
    ok = [p for p in pts if p[1] >= target]
    if not ok:
        continue
    t, tpr, fpr = max(ok, key=lambda p: p[0])
    flagged = sum(1 for s in sc if s >= t)
    print(f"  {t:>10.4f} {int(round(tpr*n_fail)):>5}/{n_fail:<2} {tpr:>11.0%} "
          f"{int(round(fpr*n_ok)):>7}/{n_ok:<3} ({fpr:>4.0%}) {flagged:>8}")

json.dump({"n": len(rows), "n_fail": n_fail, "pass_bar": PASS,
           "auc": aucs,
           "roc_disagreement": [{"t": t, "tpr": a, "fpr": b} for t, a, b in pts]},
          open("results/s1_flag_roc.json", "w"), indent=2)
print("\nwrote results/s1_flag_roc.json")
