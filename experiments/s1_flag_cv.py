"""Cross-validate the flag thresholds from C35/C36.

Both analyses picked thresholds on the same 100 units they were scored on, so the
operating points are optimistic. AUC does not depend on a threshold and is unaffected;
the sensitivity/specificity pairs are. This runs 5-fold CV: pick the threshold on four
folds at a target sensitivity, measure what it achieves on the fifth.

Local, no compute.
"""
import json
import numpy as np

rows = [json.loads(l) for l in open("results/e01_gate.jsonl", encoding="utf-8") if l.strip()]
stats = json.load(open("results/s1_failure_class.json"))["per_unit"]
ids = [int(r["_key"]) for r in rows]
fail = np.array([not (r["align_selected"] > 0.95 and r["k2_gain"] < 0.01) for r in rows])

feats = {
    "held-out R2 (post-fit)":  np.array([-r["r2_k1"] for r in rows]),
    "z_mean (PRE-fit)":        np.array([-stats[str(i)]["z_mean"] for i in ids]),
    "disagreement (2 fits)":   np.array([r["disagreement"] for r in rows]),
}

def thresh_at(sc, y, target):
    """Lowest-cost threshold reaching `target` sensitivity on this data."""
    best = None
    for t in np.unique(sc):
        tp = ((sc >= t) & y).sum()
        if y.sum() and tp / y.sum() >= target and (best is None or t > best):
            best = t
    return best

rng = np.random.default_rng(0)
order = rng.permutation(len(ids))
folds = np.array_split(order, 5)

print(f"{len(ids)} units, {int(fail.sum())} failures, 5-fold CV\n")
for name, sc in feats.items():
    auc = sum((a > b) + 0.5 * (a == b)
              for a in sc[fail] for b in sc[~fail]) / (fail.sum() * (~fail).sum())
    print(f"{name}   AUC {auc:.3f}  (threshold-free, unaffected by CV)")
    print(f"  {'target':>7} | {'in-sample':>21} | {'held-out':>21}")
    print(f"  {'sens':>7} | {'sens':>9} {'cost':>10} | {'sens':>9} {'cost':>10}")
    for target in (0.5, 0.7, 0.8):
        t_all = thresh_at(sc, fail, target)
        tp_i = ((sc >= t_all) & fail).sum() / fail.sum()
        fp_i = ((sc >= t_all) & ~fail).sum() / (~fail).sum()
        tps, fps = [], []
        for f in folds:
            m = np.ones(len(ids), bool); m[f] = False
            t = thresh_at(sc[m], fail[m], target)
            if t is None:
                continue
            te_f, te_ok = fail[f], ~fail[f]
            if te_f.sum():
                tps.append(((sc[f] >= t) & te_f).sum() / te_f.sum())
            if te_ok.sum():
                fps.append(((sc[f] >= t) & te_ok).sum() / te_ok.sum())
        print(f"  {target:>7.0%} | {tp_i:>9.0%} {fp_i:>10.0%} | "
              f"{np.mean(tps):>9.0%} {np.mean(fps):>10.0%}")
    print()
