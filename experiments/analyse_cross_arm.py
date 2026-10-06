"""Three offline analyses over runs already on disk. No model, no fitting.

1. B-4, the restart-count curve. AUC of restart agreement when it is built from the first
   k restarts only, k = 2..5, using the stored pairwise alignments. The failure label is
   the 5-restart fit's, as B-4 was filed ("by subsampling what B-1 already stored").
   Only B-7 L6 and the B-12 control kept all ten pairs, 50 units each, with 7 and 6
   failures, so this is descriptive.

2. Units no restart count reaches. A two-class model: a fraction pi of units can never be
   recovered, the rest succeed per restart with probability q, and best-of-k selection
   passes a unit if any restart succeeds. Two pass rates at k=2 and k=5 on the same units
   identify (pi, q). Solved on E0.1/B-1 (77 -> 91 of 100) and B-12 (40 -> 44 of 50).
   Selection is by held-out R2 between the direct and cascade routes, so "any restart
   succeeds" is an idealisation. The paired table is printed to show where it breaks.

3. Cross-arm meta-analysis of restart-minus-R2. One arm per distinct (model, layer)
   condition, so arms do not share units. DerSimonian-Laird random-effects pooling of
   the DeLong differences, plus a bootstrap that resamples whole arms.

    python experiments/analyse_cross_arm.py --out results/cross_arm_analysis.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from b1_signal_calibration import PASS, delong, roc  # noqa: E402

PAIR_INDEX = [(i, j) for i in range(5) for j in range(i + 1, 5)]   # _pairs order, 5 restarts

# One arm per condition; none shares units with another. Arms with < 5 failures cannot
# carry a DeLong variance and are listed but not pooled.
ARMS = {
    # B-14 (fixed estimator, 6 Oct) replaces B-1b (coupled) for this condition.
    "GPT-2 L6 (B-14, fixed)": "results/b14_primary_gpt2_indep.jsonl",
    # B-2c (fixed estimator, 6 Oct) replaces B-2b (coupled).
    "Pythia-160m L6 (B-2c, fixed)": "results/b2c_pythia160m_indep.jsonl",
    "GPT-2 L2 (B-7)": "results/b7_layer02_gpt2.jsonl",
    "GPT-2 L10 (B-7)": "results/b7_layer10_gpt2.jsonl",
    "GPT-Neo-125m L10 (B-8)": "results/b8_gptneo125m.jsonl",
    # B-11c (fixed estimator, 6 Oct) replaces B-11s (coupled) for this condition.
    "Pythia-1.4b L12, 3200 steps (B-11c, fixed)": "data/b11/b11c_pythia-14b_s3200_indep.jsonl",
    "Pythia-70m (B-11)": "data/b11/b11_pythia-70m.jsonl",
    "Pythia-410m (B-11)": "data/b11/b11_pythia-410m.jsonl",
}


def load(path):
    out = {}
    for line in open(ROOT / path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            out[int(str(r["_key"]).split("_")[-1])] = r
    return out


def restart_curve(path):
    rows = [r for r in load(path).values() if len(r.get("stability_pairs", [])) == 10]
    fail = [r["align_selected"] < PASS for r in rows]
    out = {"units": len(rows), "failures": int(sum(fail))}
    for k in (2, 3, 4, 5):
        keep = [n for n, (i, j) in enumerate(PAIR_INDEX) if i < k and j < k]
        stab = [float(np.median([r["stability_pairs"][n] for n in keep])) for r in rows]
        out[f"k={k}"] = roc([-s for s in stab], fail)[1]
    out["held-out R2"] = roc([-r["r2_k1"] for r in rows], fail)[1]
    return out


def two_class(p2, p5):
    """Solve (1-pi)(1-(1-q)^k) = p_k for k = 2, 5 by bisection on q."""
    def ratio(q):
        return (1 - (1 - q) ** 5) / (1 - (1 - q) ** 2)
    target = p5 / p2
    if not 1 < target < 2.5:
        return None
    lo, hi = 1e-9, 1 - 1e-9          # ratio falls from 2.5 at q->0 to 1 at q->1
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ratio(mid) > target else (lo, mid)
    q = (lo + hi) / 2
    pi = 1 - p2 / (1 - (1 - q) ** 2)
    return {"pi_unreachable": pi, "q_per_restart": q,
            "single_class_prediction_at_k5": 1 - (1 - (1 - (1 - p2) ** 0.5)) ** 5}


def paired(path2, path5):
    a, b = load(path2), load(path5)
    shared = sorted(set(a) & set(b))
    t = {"pass@2 pass@5": 0, "pass@2 fail@5": 0, "fail@2 pass@5": 0, "fail@2 fail@5": 0}
    for u in shared:
        k = (f"{'pass' if a[u]['align_selected'] >= PASS else 'fail'}@2 "
             f"{'pass' if b[u]['align_selected'] >= PASS else 'fail'}@5")
        t[k] += 1
    return {"shared": len(shared), **t}


def dersimonian_laird(d, se):
    d, w = np.asarray(d), 1 / np.asarray(se) ** 2
    fixed = (w * d).sum() / w.sum()
    qstat = (w * (d - fixed) ** 2).sum()
    tau2 = max(0.0, (qstat - (len(d) - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    wr = 1 / (np.asarray(se) ** 2 + tau2)
    pooled = (wr * d).sum() / wr.sum()
    se_p = (1 / wr.sum()) ** 0.5
    i2 = max(0.0, (qstat - (len(d) - 1)) / qstat) if qstat > 0 else 0.0
    return {"pooled_diff": pooled, "se": se_p,
            "ci95": [pooled - 1.96 * se_p, pooled + 1.96 * se_p],
            "tau2": tau2, "Q": qstat, "I2": i2}


def meta(n_boot=5000, seed=0):
    per_arm, d, se = {}, [], []
    data = {}
    for name, path in ARMS.items():
        if not (ROOT / path).exists():
            continue
        rows = list(load(path).values())
        fail = [r["align_selected"] < PASS for r in rows]
        st = [-r["stability"] for r in rows]
        r2 = [-r["r2_k1"] for r in rows]
        res = delong(st, r2, fail)
        per_arm[name] = {"n": len(rows), "failures": int(sum(fail)),
                         "auc_restart": res[0], "auc_r2": res[1], "diff": res[2],
                         "se": res[3], "p": res[5]}
        if sum(fail) >= 5 and res[3] > 0:
            d.append(res[2]); se.append(res[3]); data[name] = (st, r2, fail)
    out = {"per_arm": per_arm, "pooled_arms": list(data), "random_effects": dersimonian_laird(d, se)}
    rng = np.random.default_rng(seed)
    names = list(data)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(len(names), len(names))
        boots.append(np.mean([d[i] for i in pick]))
    out["arm_bootstrap_mean_diff"] = {"mean": float(np.mean(d)),
                                      "ci95": [float(np.percentile(boots, 2.5)),
                                               float(np.percentile(boots, 97.5))]}
    out["arms_with_r2_ahead"] = f"{sum(x < 0 for x in d)}/{len(d)}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = {
        "restart_curve": {"B-7 L6": restart_curve("results/b7_layer06_gpt2.jsonl"),
                          "B-12 control": restart_curve("results/b12_ctl_b032_r5.jsonl")},
        "two_class": {
            "E0.1 gate -> B-1 (n=100)": {"p2": 0.77, "p5": 0.91, **(two_class(0.77, 0.91) or {})},
            "B-12 b32 -> control (n=50)": {"p2": 0.80, "p5": 0.88,
                                           **(two_class(0.80, 0.88) or {}),
                                           "paired": paired("results/b12_fix_b032.jsonl",
                                                            "results/b12_ctl_b032_r5.jsonl")},
        },
        "meta": meta(),
    }
    print("RESTART CURVE (AUC of restart agreement from the first k restarts)")
    for arm, c in rep["restart_curve"].items():
        print(f"  {arm}: {c['units']} units, {c['failures']} failures  "
              + "  ".join(f"{k} {c[k]:.3f}" for k in ("k=2", "k=3", "k=4", "k=5"))
              + f"  | R2 {c['held-out R2']:.3f}")
    print("TWO-CLASS (unreachable fraction pi, per-restart success q)")
    for arm, c in rep["two_class"].items():
        print(f"  {arm}: pi={c.get('pi_unreachable', float('nan')):.3f} "
              f"q={c.get('q_per_restart', float('nan')):.3f}  one-class model predicts "
              f"{c.get('single_class_prediction_at_k5', float('nan')):.3f} at k=5 vs observed "
              f"{c['p5']}" + (f"  paired {c['paired']}" if "paired" in c else ""))
    m = rep["meta"]
    print("CROSS-ARM (restart minus R2, DeLong per arm)")
    for arm, c in m["per_arm"].items():
        print(f"  {arm:<38} n={c['n']:>3} fail={c['failures']:>3}  restart {c['auc_restart']:.3f} "
              f"R2 {c['auc_r2']:.3f}  diff {c['diff']:+.3f}  p={c['p']:.2g}")
    re_ = m["random_effects"]
    print(f"  random effects over {len(m['pooled_arms'])} arms: {re_['pooled_diff']:+.3f} "
          f"[{re_['ci95'][0]:+.3f}, {re_['ci95'][1]:+.3f}]  I2={re_['I2']:.2f}  "
          f"arm-bootstrap mean {m['arm_bootstrap_mean_diff']['mean']:+.3f} "
          f"{[round(x, 3) for x in m['arm_bootstrap_mean_diff']['ci95']]}  "
          f"R2 ahead in {m['arms_with_r2_ahead']}")
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=2, default=float)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
