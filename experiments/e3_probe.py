"""E-3: a standard linear probe at the near-constant layers (docs/preregistration-e3-standardised-probe.md).

    PYTHONPATH=. python experiments/e3_probe.py              # E-3 -> results/e3_probe.json (fits and scores once)
    PYTHONPATH=. python experiments/e3_probe.py --set e3b    # E-3b, two current models -> results/e3b_probe.json

The practitioner's pipeline, sklearn StandardScaler + LogisticRegression (defaults, max_iter 5000),
trained to read "the current token starts with a capital letter" from the stimulus the layer's MLP
reads (the output of the norm feeding it), on 8,000 tokens of corpus seed 0, tested on 4,000 tokens of
unseen corpus-seed-1 documents. Directions are reported in the stimulus' own coordinates (coef /
scale). Compared with the same probe without the scaler and with the difference of class means.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from caliper.activations import _blocks, _mlp_ln, load_model, sample_corpus  # noqa: E402

LAYERS = [  # (model, layer, flagged by the layer map)
    ("EleutherAI/gpt-neo-125M", 6, True), ("gpt2", 3, True), ("facebook/opt-125m", 5, True),
    ("EleutherAI/gpt-neo-125M", 10, False), ("gpt2", 6, False), ("facebook/opt-125m", 6, False),
]
N_TRAIN, N_TEST, SEQ = 8000, 4000, 128


def tokens(model, tok, layer, texts, n):
    ln = _mlp_ln(_blocks(model)[layer])
    got = {}
    h = ln.register_forward_hook(lambda _m, _i, o: got.__setitem__("s", o.detach()))
    S, ids = [], []
    try:
        for t in texts:
            x = tok(t, return_tensors="pt", truncation=True, max_length=SEQ)["input_ids"]
            if x.shape[1] < 2:
                continue
            with torch.no_grad():
                model(input_ids=x)
            s = got["s"].reshape(x.shape[1], -1)  # OPT's MLP norm sees (batch * position, d)
            S.append(s[1:].float().numpy())
            ids.extend(x[0, 1:].tolist())
            if len(ids) >= n:
                break
    finally:
        h.remove()
    return np.concatenate(S)[:n].astype(np.float64), np.array(ids[:n])


def label(tok, ids):
    out = []
    for i in ids:
        s = tok.decode([int(i)]).strip()
        out.append(bool(s) and s[0].isalpha() and s[0].isupper())
    return np.array(out)


def unit(v):
    return v / np.linalg.norm(v)


def std_probe(X, y):
    p = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(X, y)
    sc, lr = p[0], p[1]
    return p, unit(lr.coef_[0] / sc.scale_)


E3B = [  # E-3b (docs/preregistration-e3b-modern-probe.md): (model, layer, kind)
    ("meta-llama/Llama-3.2-1B-Instruct", 3, "near-constant"), ("meta-llama/Llama-3.2-1B-Instruct", 10, "control"),
    ("google/gemma-3-270m", 13, "near-constant"), ("google/gemma-3-270m", 5, "exactly constant"),
]


def main():
    e3b = "--set" in sys.argv and sys.argv[sys.argv.index("--set") + 1] == "e3b"
    rep = {"layers": {}}
    for name, layer, flagged in (E3B if e3b else LAYERS):
        model, tok = load_model(name, dtype=torch.float32)
        Xtr, itr = tokens(model, tok, layer, sample_corpus(n_docs=300, seed=0), N_TRAIN)
        seen = set(sample_corpus(n_docs=300, seed=0))
        Xte, ite = tokens(model, tok, layer, [t for t in sample_corpus(n_docs=300, seed=1) if t not in seen], N_TEST)
        ytr, yte = label(tok, itr), label(tok, ite)
        sd = Xtr.std(0)
        const = np.flatnonzero(sd < 0.01 * np.median(sd))
        coords = const if len(const) else np.array([int(sd.argmin())])
        p, v_std = std_probe(Xtr, ytr)
        q = LogisticRegression(max_iter=5000).fit(Xtr, ytr)
        v_raw = unit(q.coef_[0])
        v_dm = unit(Xtr[ytr].mean(0) - Xtr[~ytr].mean(0))
        folds = [std_probe(Xtr[i], ytr[i])[1] for i, _ in KFold(5, shuffle=True, random_state=0).split(Xtr)]
        cv = [abs(folds[a] @ folds[b]) for a in range(5) for b in range(a + 1, 5)]
        r = {"model": name, "layer": layer, "flagged": flagged, "train_tokens": len(ytr), "test_tokens": len(yte),
             "positive_rate": float(ytr.mean()), "near_constant": const.tolist(),
             "acc_std": float(p.score(Xte, yte)), "acc_raw": float(q.score(Xte, yte)),
             "auc_std": float(roc_auc_score(yte, p.decision_function(Xte))),
             "auc_raw": float(roc_auc_score(yte, q.decision_function(Xte))),
             "share_std": float((v_std[coords] ** 2).sum()), "share_raw": float((v_raw[coords] ** 2).sum()),
             "cv_agreement_std": float(np.median(cv)),
             "cos_std_raw": float(abs(v_std @ v_raw)), "cos_std_diffmean": float(abs(v_std @ v_dm)),
             "cos_raw_diffmean": float(abs(v_raw @ v_dm))}
        rep["layers"][f"{name} L{layer}"] = r
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    if e3b:
        R = rep["layers"]
        ll, lc = R["meta-llama/Llama-3.2-1B-Instruct L3"], R["meta-llama/Llama-3.2-1B-Instruct L10"]
        g13, g5 = R["google/gemma-3-270m L13"], R["google/gemma-3-270m L5"]
        rep["P1 collapse at Llama-3.2-1B L3"] = {"holds": ll["share_std"] > 0.5 and ll["cos_std_raw"] < 0.3}
        rep["P2 collapse at Gemma-3-270m L13"] = {"holds": g13["share_std"] > 0.5}
        rep["P3 no collapse on exactly constant coordinates (Gemma L5)"] = {"holds": g5["share_std"] <= 0.05}
        rep["P4 control clean (Llama L10)"] = {"holds": lc["share_std"] <= 0.05 and lc["cos_std_raw"] > 0.7}
        rep["P5 accuracy unaffected"] = {"holds": all(abs(r["acc_std"] - r["acc_raw"]) <= 0.01 for r in (ll, g13))}
        print(json.dumps({k: v for k, v in rep.items() if k != "layers"}, indent=1))
        json.dump(rep, open(ROOT / "results/e3b_probe.json", "w"), indent=1)
        return
    F = [r for r in rep["layers"].values() if r["flagged"]]
    C = [r for r in rep["layers"].values() if not r["flagged"]]
    hit = [r for r in F if r["share_std"] > 0.5]
    rep["P1 accuracy unaffected"] = {"holds": all(abs(r["acc_std"] - r["acc_raw"]) <= 0.01 for r in F)}
    rep["P2 collapse at flagged layers, none at controls"] = {
        "holds": len(hit) >= 2 and all(r["share_std"] <= 0.05 for r in C)}
    rep["P3 cross-validation agrees on the collapsed direction"] = {
        "holds": bool(hit) and all(r["cv_agreement_std"] >= 0.95 for r in hit)}
    rep["P4 the standardised direction is not the raw probe's"] = {
        "holds": bool(hit) and all(r["cos_std_raw"] < 0.3 for r in hit) and all(r["cos_std_raw"] > 0.7 for r in C)}
    print(json.dumps({k: v for k, v in rep.items() if k != "layers"}, indent=1))
    json.dump(rep, open(ROOT / "results/e3_probe.json", "w"), indent=1)


if __name__ == "__main__":
    main()
