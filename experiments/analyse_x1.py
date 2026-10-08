"""X-1: cross-sample agreement as a check, and the identifiability replication.
(docs/preregistration-x1-cross-sample.md)

    python experiments/analyse_x1.py                                   # the filed run
    python experiments/analyse_x1.py --a results/b15a_fitseed1 --b results/b15b_corpusseed1 \\
        --fresh results/b15a_diagnostics_fresh.json --model gpt2 --layer 6 \\
        --out results/x1_dev.json                                     # development

Fit A and fit B fit the same units with identical settings on document-disjoint token
samples. Labels come from fit A (|cos| to w < 0.95; converged-wrong if held-out R2 > 0.99).
Cross-sample agreement is |cos| between a unit's fit-A and fit-B selected directions.

P1 (primary): on converged-wrong vs passing units, AUC(cross-sample agreement) minus
AUC(held-out R2), stratified bootstrap 95% CI (2,000 resamples). Predicted > 0, CI above 0.
Underpowered, and reported as such, if fewer than 8 converged-wrong units.
P2: >= 80% of converged-wrong fits reach stimulus-weighted alignment >= 0.99 on fresh text.
P3: median share of error energy in the bottom-1%-variance directions >= 0.70 for
converged-wrong fits and < 0.50 for under-fitted fits.
Secondary: all failures with DeLong; restart agreement in the same comparisons; labels from
fit B; the under-fitted class.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))
from analyse_review_round1 import auc, stimulus  # noqa: E402
from b1_signal_calibration import delong  # noqa: E402

PASS, WRONG_BASIN_R2 = 0.95, 0.99


def load(stem):
    return {r["_key"]: r for r in map(json.loads, open(ROOT / f"{stem}.jsonl", encoding="utf-8"))}


def unit_dir(stem, r):
    z = np.load(ROOT / f"{stem}_dirs/n{r['_key']}.npz")
    v = z[r["picked"]][:, 0]
    return v / np.linalg.norm(v), z["w"] / np.linalg.norm(z["w"])


def boot_diff(s1, s2, y, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    pi, ni = np.where(y)[0], np.where(~y)[0]
    d = []
    for _ in range(n_boot):
        i = np.concatenate([rng.choice(pi, len(pi)), rng.choice(ni, len(ni))])
        d.append(auc(s1[i], y[i]) - auc(s2[i], y[i]))
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def compare(scores, fail, cw, uf):
    out = {}
    for label, pos, keep in (("all failures", fail, np.ones_like(fail)),
                             ("converged-wrong vs pass", cw, cw | ~fail),
                             ("under-fitted vs pass", uf, uf | ~fail)):
        y = pos[keep]
        if y.sum() < 2 or (~y).sum() < 2:
            out[label] = {"n_positive": int(y.sum())}
            continue
        s = {k: v[keep] for k, v in scores.items()}
        out[label] = {"n_positive": int(y.sum()),
                      **{f"auc {k}": auc(v, y) for k, v in s.items()},
                      "cross minus R2": auc(s["cross-sample agreement"], y) - auc(s["held-out R2"], y),
                      "cross minus R2 ci95": boot_diff(s["cross-sample agreement"], s["held-out R2"], y),
                      "cross minus restart": auc(s["cross-sample agreement"], y)
                      - auc(s["restart agreement"], y),
                      "cross minus restart ci95": boot_diff(s["cross-sample agreement"],
                                                            s["restart agreement"], y)}
    res = delong(list(scores["cross-sample agreement"]), list(scores["held-out R2"]), list(fail))
    out["all failures"]["delong cross vs R2"] = {"diff": res[2], "p": res[5]}
    return out


def labels(R, ks):
    fail = np.array([R[k]["align_selected"] < PASS for k in ks])
    cw = fail & np.array([R[k]["r2_k1"] > WRONG_BASIN_R2 for k in ks])
    return fail, cw, fail & ~cw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="results/x1a_gptneo_l10")
    ap.add_argument("--b", default="results/x1b_gptneo_l10")
    ap.add_argument("--fresh", default="results/x1a_diagnostics_fresh.json")
    ap.add_argument("--model", default="EleutherAI/gpt-neo-125M")
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--out", default="results/x1_analysis.json")
    a = ap.parse_args()

    A, B = load(a.a), load(a.b)
    ks = sorted(set(A) & set(B))
    cross = np.array([abs(unit_dir(a.a, A[k])[0] @ unit_dir(a.b, B[k])[0]) for k in ks])
    rep = {"units": len(ks)}

    for which, R, other in (("labels from fit A", A, B), ("labels from fit B", B, A)):
        fail, cw, uf = labels(R, ks)
        scores = {"cross-sample agreement": -cross,
                  "held-out R2": np.array([-R[k]["r2_k1"] for k in ks]),
                  "restart agreement": np.array([-R[k]["stability"] for k in ks])}
        rep[which] = {"failures": int(fail.sum()), "converged_wrong": int(cw.sum()),
                      "under_fitted": int(uf.sum()), **compare(scores, fail, cw, uf)}

    fail, cw, uf = labels(A, ks)
    p1 = rep["labels from fit A"]["converged-wrong vs pass"]
    if cw.sum() < 8:
        rep["P1"] = f"underpowered: {int(cw.sum())} converged-wrong units (< 8)"
    else:
        rep["P1"] = {"diff": p1["cross minus R2"], "ci95": p1["cross minus R2 ci95"],
                     "holds": p1["cross minus R2 ci95"][0] > 0}

    fresh = {u["unit"]: u for u in json.load(open(ROOT / a.fresh))["units"]}
    eq = [fresh[k]["sigma_align"] >= 0.99 for k, c in zip(ks, cw) if c]
    rep["P2"] = {"converged_wrong": len(eq), "equivalent_on_fresh_text": int(sum(eq)),
                 "share": float(np.mean(eq)) if eq else None,
                 "holds": bool(eq) and float(np.mean(eq)) >= 0.80}

    S, _ = stimulus(a.model, a.layer, ks)
    ev, evec = np.linalg.eigh(np.cov(S, rowvar=False))
    low = evec[:, np.cumsum(ev) / ev.sum() <= 0.01]
    share = {}
    for cls, mask in (("converged-wrong", cw), ("under-fitted", uf)):
        v = []
        for k, m in zip(ks, mask):
            if not m:
                continue
            f, w = unit_dir(a.a, A[k])
            f = f * np.sign(f @ w) if f @ w != 0 else f
            e = f - w
            v.append(float(((low.T @ e) ** 2).sum() / (e @ e)))
        share[cls] = float(np.median(v)) if v else None
    rep["P3"] = {"median_error_share_low_variance": share,
                 "low_variance_dims": int(low.shape[1]),
                 "holds": (share["converged-wrong"] is not None and share["converged-wrong"] >= 0.70
                           and share["under-fitted"] is not None and share["under-fitted"] < 0.50)}
    print(json.dumps({k: rep[k] for k in ("units", "P1", "P2", "P3")}, indent=1, default=float))
    json.dump(rep, open(ROOT / a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
