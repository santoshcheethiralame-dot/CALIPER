"""Round-2 roadmap A1 (error spectrum) and A4 (does the low-variance share add beyond 1/gamma?).

    PYTHONPATH=. python experiments/analyse_a4_spectrum.py     # writes results/a4_spectrum.json

Error spectrum: the share of a failing fit's error energy in the eigen-directions carrying the
bottom 1% of stimulus variance, as filed (e = v - w) and on the identifiable label (1/gamma removed
from v and w, and the low-variance subspace taken from the stimulus covariance with 1/gamma removed).

A4: three ground-truth-free scores of a fit v:
  null share        (u . v)^2, u = 1/gamma normalised
  low share         ||L' v||^2, L = bottom-1% eigenvectors (the diagnostic as first reported)
  low share, u out  ||L'' P v||^2 / ||P v||^2, L'' from the covariance of P s, its null eigenvector
                    dropped; what the low-variance diagnostic still sees once 1/gamma is gone
Each is scored by AUC for failures and for converged-wrong fits against passes, per arm and pooled,
on both labels. If the low share adds nothing beyond the null share, the u-out version should sit
near 0.5 on the identifiable label.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from analyse_review_round1 import auc  # noqa: E402
from caliper.activations import _blocks, _mlp_ln, collect, load_model, sample_corpus  # noqa: E402

PASS, WB = 0.95, 0.99
ARMS = [  # name, stem, model, layer, corpus seed, sequence split, fp16
    ("Pythia-160m L6 (B-2c)", "results/b2c_pythia160m_indep", "EleutherAI/pythia-160m", 6, 0, False, True),
    ("GPT-Neo-125m L10 (B-8b)", "results/b8b_gptneo125m_indep", "EleutherAI/gpt-neo-125M", 10, 0, False, False),
    ("GPT-2 L6 re-fit, fit seed 1 (B-15a)", "results/b15a_fitseed1", "gpt2", 6, 0, False, False),
    ("GPT-2 L6 re-fit, corpus seed 1 (B-15b)", "results/b15b_corpusseed1", "gpt2", 6, 1, False, False),
    ("GPT-2 L6 re-fit, sequence split (B-15c)", "results/b15c_seqsplit", "gpt2", 6, 0, True, False),
    ("GPT-Neo-125m L10, new units (X-1a)", "results/x1a_gptneo_l10", "EleutherAI/gpt-neo-125M", 10, 0, False, False),
    ("GPT-Neo-125m L10, new units (X-1b)", "results/x1b_gptneo_l10", "EleutherAI/gpt-neo-125M", 10, 1, False, False),
    ("GPT-Neo-125m L6 (B-17)", "results/b17_gptneo125m_l6", "EleutherAI/gpt-neo-125M", 6, 0, False, False),
]


def bottom(C, frac=0.01, drop=None):
    ev, evec = np.linalg.eigh(C)
    if drop is not None:  # remove the eigenvector closest to the null direction
        k = int(np.argmax(np.abs(evec.T @ drop)))
        ev, evec = np.delete(ev, k), np.delete(evec, k, axis=1)
    return evec[:, np.cumsum(ev) / ev.sum() <= frac]


def arm(name, stem, model_name, layer, corpus_seed, seq_split, fp16):
    R = [json.loads(l) for l in open(ROOT / f"{stem}.jsonl", encoding="utf-8") if l.strip()]
    model, tok = load_model(model_name, dtype=torch.float16 if fp16 else torch.float32)
    p = collect(model, tok, sample_corpus(n_docs=300, seed=corpus_seed), layer=layer,
                neurons=np.array([r["_key"] for r in R]), max_tokens=8000, seed=0,
                shuffle="sequence" if seq_split else "token")
    S = np.asarray(p.stimulus, np.float64)
    g = _mlp_ln(_blocks(model)[layer]).weight.detach().double().numpy()
    u = (1 / g) / np.linalg.norm(1 / g)
    P = np.eye(len(u)) - np.outer(u, u)
    C = np.cov(S, rowvar=False)
    L, L2 = bottom(C), bottom(P @ C @ P, drop=u)
    rec = []
    for r in R:
        z = np.load(ROOT / f"{stem}_dirs/n{r['_key']}.npz")
        w = z["w"].astype(np.float64); w /= np.linalg.norm(w)
        v = z[r["picked"]][:, 0].astype(np.float64); v /= np.linalg.norm(v)
        v = v * np.sign(v @ w) if v @ w else v
        pv, pw = P @ v, P @ w
        pv_n, pw_n = pv / np.linalg.norm(pv), pw / np.linalg.norm(pw)
        pv_n = pv_n * np.sign(pv_n @ pw_n) if pv_n @ pw_n else pv_n
        e, e2 = v - w, pv_n - pw_n
        rec.append({"unit": int(r["_key"]), "raw": abs(v @ w), "ident": abs(pv_n @ pw_n),
                    "r2": r["r2_k1"], "restart": r["stability"],
                    "err_low_filed": float(((L.T @ e) ** 2).sum() / (e @ e)),
                    "err_low_ident": float(((L2.T @ e2) ** 2).sum() / (e2 @ e2)) if e2 @ e2 > 0 else 0.0,
                    "null_share": float((u @ v) ** 2), "low_share": float(((L.T @ v) ** 2).sum()),
                    "low_share_u_out": float(((L2.T @ pv_n) ** 2).sum())})
    print(f"{name}: {len(rec)} units, low dims {L.shape[1]} / {L2.shape[1]} (u out)", flush=True)
    return rec, {"low_dims": int(L.shape[1]), "low_dims_u_out": int(L2.shape[1]), "dim": int(len(u))}


def summarise(rec):
    a = {k: np.array([x[k] for x in rec]) for k in rec[0]}
    out = {}
    for lab in ("raw", "ident"):
        fail = a[lab] < PASS
        cw, uf = fail & (a["r2"] > WB), fail & (a["r2"] <= WB)
        err = a["err_low_filed" if lab == "raw" else "err_low_ident"]
        res = {"failures": int(fail.sum()), "converged_wrong": int(cw.sum()),
               "median_error_share_low": {"converged-wrong": float(np.median(err[cw])) if cw.any() else None,
                                          "under-fitted": float(np.median(err[uf])) if uf.any() else None}}
        keep = cw | ~fail
        for s in ("null_share", "low_share", "low_share_u_out"):
            res[f"auc_{s}"] = {"all_failures": auc(a[s], fail),
                               "converged_wrong_vs_pass": auc(a[s][keep], cw[keep]) if cw.sum() else None}
        hi = keep & (a["r2"] > WB)  # R1: drop passes below the class line, where R2 wins by construction
        if cw.sum() and (hi & ~cw).sum():
            res["converged_wrong_vs_pass_r2_above_0.99"] = {
                "converged_wrong": int(cw.sum()), "passes_kept": int((hi & ~cw).sum()),
                **{f"auc_{s}": auc(a[s][hi], cw[hi]) for s in ("null_share", "low_share", "low_share_u_out")},
                "auc_held_out_r2": auc(-a["r2"][hi], cw[hi]), "auc_restart": auc(-a["restart"][hi], cw[hi])}
        out["as filed" if lab == "raw" else "identifiable"] = res
    return out


def main():
    rep, pooled = {}, []
    for spec in ARMS:
        rec, meta = arm(*spec)
        rep[spec[0]] = {**meta, **summarise(rec), "units": rec}
        pooled += rec
        print(json.dumps({k: v for k, v in rep[spec[0]].items() if k != "units"}), flush=True)
    rep["pooled (all arms)"] = summarise(pooled)
    print("POOLED", json.dumps(rep["pooled (all arms)"]), flush=True)
    json.dump(rep, open(ROOT / "results/a4_spectrum.json", "w"), indent=1)


if __name__ == "__main__":
    main()
