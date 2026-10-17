"""E-3c: the standard probe on a property with a known direction, and the repairs, compared
(docs/preregistration-e3c-controlled-probe.md).

    PYTHONPATH=. python experiments/e3c_probe_controlled.py      # writes results/e3c_probe.json (scores once)

The property is "MLP readout j is in its top 10%": label = 1[w_j . s > tau_j], with w_j the j-th
column of the MLP's input weight and tau_j the 90th percentile over the training tokens. Its true
direction is w_j (modulo the layer norm's null direction 1/gamma, which no data can show). Data as
E-3: the input of the layer's MLP, 8,000 training tokens of corpus seed 0, 4,000 test tokens of unseen
corpus-seed-1 documents. Each probe variant is scored by |cos| with w_j on the identifiable label
(1/gamma removed from both; RMSNorm models have no such direction) and by test accuracy.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))
from caliper.activations import _blocks, _mlp_in_weight, _mlp_ln, load_model, sample_corpus  # noqa: E402
from caliper.oxide import alignments  # noqa: E402
from e3_probe import N_TEST, N_TRAIN, tokens  # noqa: E402

LAYERS = [  # (model, layer, kind, null direction exists)
    ("EleutherAI/gpt-neo-125M", 6, "flagged", True), ("gpt2", 3, "flagged", True),
    ("facebook/opt-125m", 5, "flagged", True), ("meta-llama/Llama-3.2-1B-Instruct", 3, "flagged", False),
    ("google/gemma-3-270m", 13, "flagged", False),
    ("EleutherAI/gpt-neo-125M", 10, "control", True), ("gpt2", 6, "control", True),
    ("facebook/opt-125m", 6, "control", True), ("meta-llama/Llama-3.2-1B-Instruct", 10, "control", False),
]
N_UNITS, SEED = 10, 20261019


def unit(v):
    return v / np.linalg.norm(v)


def variants(Xtr, y, const):
    """Probe directions in the input's own coordinates, one per variant."""
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd0 = np.where(sd > 0, sd, 1.0)                                  # scikit-learn's zero-variance rule
    out = {}

    def lr(Z, **kw):
        return LogisticRegression(max_iter=5000, **kw).fit(Z, y).coef_[0]

    out["unscaled"] = lr(Xtr)
    out["standardised"] = lr((Xtr - mu) / sd0) / sd0
    floor = np.maximum(sd0, 0.01 * np.median(sd0))
    out["scale floor (1% of median)"] = lr((Xtr - mu) / floor) / floor
    keep = np.setdiff1d(np.arange(Xtr.shape[1]), const)
    v = np.zeros(Xtr.shape[1])
    v[keep] = lr((Xtr[:, keep] - mu[keep]) / sd0[keep]) / sd0[keep]
    out["drop near-constant, then standardise"] = v
    z = out["standardised"].copy()
    z[const] = 0
    out["standardise, then zero near-constant"] = z
    out["standardised, C = 0.01"] = lr((Xtr - mu) / sd0, C=0.01) / sd0
    out["standardised, C = 100"] = lr((Xtr - mu) / sd0, C=100.0) / sd0
    out["standardised, L1"] = lr((Xtr - mu) / sd0, penalty="l1", solver="liblinear") / sd0
    p = PCA(whiten=True).fit(Xtr)
    keep_pc = p.explained_variance_ > 0
    c = lr(p.transform(Xtr)[:, keep_pc])
    out["PCA whitening"] = p.components_[keep_pc].T @ (c / np.sqrt(p.explained_variance_[keep_pc]))
    return {k: unit(v) for k, v in out.items()}


def main():
    rep = {"layers": {}}
    for name, layer, kind, has_null in LAYERS:
        model, tok = load_model(name, dtype=torch.float32)
        block = _blocks(model)[layer]
        Xtr, _ = tokens(model, tok, layer, sample_corpus(n_docs=300, seed=0), N_TRAIN)
        seen = set(sample_corpus(n_docs=300, seed=0))
        Xte, _ = tokens(model, tok, layer, [t for t in sample_corpus(n_docs=300, seed=1) if t not in seen], N_TEST)
        W = _mlp_in_weight(block).astype(np.float64)
        n = None
        if has_null:
            g = _mlp_ln(block).weight.detach().double().numpy()
            n = unit(1.0 / g)
        sd = Xtr.std(0)
        const = np.flatnonzero(sd < 0.01 * np.median(sd))
        units = np.sort(np.random.default_rng(SEED).choice(W.shape[1], N_UNITS, replace=False))
        per = {}
        for j in units:
            w = W[:, j]
            tau = np.quantile(Xtr @ w, 0.9)
            ytr, yte = Xtr @ w > tau, Xte @ w > tau
            for vname, v in variants(Xtr, ytr, const).items():
                _, ident = alignments(v[None], w[:, None], n)
                if np.corrcoef(Xtr @ v, ytr)[0, 1] < 0:
                    v = -v
                acc = float(np.mean((Xte @ v > np.quantile(Xtr @ v, 0.9)) == yte))  # the direction, thresholded as the label is
                per.setdefault(vname, []).append({"unit": int(j), "cos": float(ident[0]), "acc": acc,
                                                  "share": float(np.sum(v[const] ** 2))})
        summ = {v: {"median_cos": float(np.median([r["cos"] for r in rs])), "median_acc": float(np.median([r["acc"] for r in rs])),
                    "median_share": float(np.median([r["share"] for r in rs]))} for v, rs in per.items()}
        rep["layers"][f"{name} L{layer}"] = {"kind": kind, "near_constant": const.tolist(), "units": units.tolist(),
                                            "summary": summ, "per_unit": per}
        print(name, layer, kind, {v: (round(s["median_cos"], 3), round(s["median_acc"], 3)) for v, s in summ.items()}, flush=True)
    L = rep["layers"].values()
    F = [r for r in L if r["kind"] == "flagged"]
    C = [r for r in L if r["kind"] == "control"]
    s = lambda r, v: r["summary"][v]["median_cos"]
    rep["P1 the standard probe misses the known direction at flagged layers"] = {
        "holds": all(s(r, "standardised") <= 0.3 for r in F)}
    rep["P2 the unscaled probe finds it there"] = {"holds": all(s(r, "unscaled") >= 0.7 for r in F)}
    rep["P3 the standard probe finds it at controls"] = {"holds": all(s(r, "standardised") >= 0.7 for r in C)}
    rep["P4 floor, drop and zero repair it at flagged layers"] = {"holds": all(
        s(r, v) >= 0.7 for r in F for v in ("scale floor (1% of median)", "drop near-constant, then standardise",
                                            "standardise, then zero near-constant"))}
    print(json.dumps({k: v for k, v in rep.items() if k.startswith("P")}, indent=1))
    json.dump(rep, open(ROOT / "results/e3c_probe.json", "w"), indent=1)


if __name__ == "__main__":
    main()
