"""Flagship substrate scorer for F-4 and F-6 (docs/preregistration-f4-gated-units.md,
docs/preregistration-f6-unembedding.md), written before their data.

    PYTHONPATH=. python experiments/analyse_substrate.py --a <fit A stem> --b <fit B stem> \
        --model <hub id> [--norm <dotted path of the norm feeding the target>] --out <json>

Fit A is the primary fit (corpus seed 0); fit B re-fits the same units on documents disjoint from
A's. Each unit's reference is w (one column, or the gate and up columns of a gated unit) and the
fit is the selected route's direction or plane. With --norm, the primary label removes that
LayerNorm's null direction u = 1/gamma from fit and reference (the flagship's label policy); the
Euclidean label is reported beside it. Without --norm (RMSNorm models) the two labels coincide.

Primary: AUC(restart agreement) - AUC(held-out R2) for failure on fit A (DeLong, stratified
bootstrap). Secondary: AUC(cross-sample agreement) - AUC(restart agreement) (the calibration
table's primary contrast, F-10); the failure rate with a Wilson interval (Gate F-B's 20-80% band);
the class split; with --norm, the fit's share on u as a check.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))
from analyse_review_round1 import auc, boot_diff  # noqa: E402
from b1_signal_calibration import delong  # noqa: E402

PASS, WB = 0.95, 0.99


def rows(stem):
    p = ROOT / f"{stem}.jsonl"
    if not p.exists():
        return {}
    return {json.loads(l)["_key"]: json.loads(l) for l in open(p, encoding="utf-8") if l.strip()}


def null_direction(model, path):
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32)
    for part in path.split("."):
        m = getattr(m, part)
    g = m.weight.detach().double().numpy()
    return (1 / g) / np.linalg.norm(1 / g)


def basis(M, u=None):
    M = np.asarray(M, np.float64).reshape(M.shape[0], -1)
    if u is not None:
        M = M - np.outer(u, u @ M)
    return np.linalg.qr(M)[0]


def align(A, B):
    return float(np.mean(np.clip(np.linalg.svd(A.T @ B, compute_uv=False), 0, 1)))


def fitted(stem, key, r):
    z = np.load(ROOT / f"{stem}_dirs/n{key}.npz")
    return z["w"], z[r["picked"]]


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(c - h), float(c + h)]


def compare(s1, s2, fail):
    if fail.sum() < 2 or (~fail).sum() < 2:
        return None
    a1, a2, d, se, z, p = delong(list(s1), list(s2), list(fail))
    return {"auc_1": a1, "auc_2": a2, "diff": d, "delong_p": p, "boot_ci95": boot_diff(s1, s2, fail)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", default=None)
    ap.add_argument("--model", required=True)
    ap.add_argument("--norm", default=None, help="e.g. transformer.ln_f for GPT-2's unembedding")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    A, B = rows(a.a), rows(a.b) if a.b else {}
    u = null_direction(a.model, a.norm) if a.norm else None
    keys = sorted(A)
    rec = []
    for k in keys:
        w, v = fitted(a.a, k, A[k])
        r = {"unit": int(k), "r2": A[k]["r2_k1"], "restart": A[k]["stability"],
             "euclidean": align(basis(v), basis(w)), "label": align(basis(v, u), basis(w, u))}
        if u is not None:
            vv = np.asarray(v, np.float64).reshape(v.shape[0], -1)
            r["null_share"] = float(np.mean((u @ (vv / np.linalg.norm(vv, axis=0))) ** 2))
        if k in B:
            _, vb = fitted(a.b, k, B[k])
            r["cross_sample"] = align(basis(v, u), basis(vb, u))
        rec.append(r)
    f = lambda key: np.array([x[key] for x in rec])
    rep = {"units": len(rec), "fit_b_units": len(B), "label": "identifiable (1/gamma removed)" if u is not None
           else "Euclidean (no LayerNorm null direction)"}
    for lab in ("label", "euclidean"):
        al = f(lab)
        fail = al < PASS
        cw = fail & (f("r2") > WB)
        res = {"failures": int(fail.sum()), "failure_rate": float(fail.mean()),
               "failure_rate_wilson95": wilson(int(fail.sum()), len(fail)),
               "in_band_20_80": bool(0.2 <= fail.mean() <= 0.8),
               "converged_wrong": int(cw.sum()), "under_fitted": int((fail & ~cw).sum()),
               "median_alignment": float(np.median(al)),
               "primary_restart_minus_r2": compare(-f("restart"), -f("r2"), fail)}
        cs = [x.get("cross_sample") for x in rec]
        if all(c is not None for c in cs):
            res["cross_sample_minus_restart"] = compare(-np.array(cs), -f("restart"), fail)
            res["cross_sample_minus_r2"] = compare(-np.array(cs), -f("r2"), fail)
        if u is not None:
            res["auc_null_share"] = auc(f("null_share"), fail)
        rep["primary label" if lab == "label" else "Euclidean label"] = res
    rep["per_unit"] = rec
    print(json.dumps({k: v for k, v in rep.items() if k != "per_unit"}, indent=1, default=float)[:3500])
    json.dump(rep, open(ROOT / a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
