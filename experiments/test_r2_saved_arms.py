"""Test R^2 on unseen documents for every noiseless arm with saved fitted directions (exploratory, post hoc).

    python experiments/test_r2_saved_arms.py            # writes results/test_r2_saved_arms.json
    python experiments/test_r2_saved_arms.py --arms b17c # one arm (smoke test)

In these arms held-out R^2 is a validation score: the same held-out tokens stopped training, chose the
route and were reported. Here each arm's selected direction is scored on 4,000 tokens from corpus-seed-2
documents, which no arm was fitted on, through a binned link (60 equal-count bins) fitted on that arm's
own training rows, re-collected exactly as the arm collected them. Per arm, on the filed (Euclidean) and
identifiable labels: AUC(restart agreement) - AUC(test R^2), beside the arm's own comparison against
held-out R^2. Noise arms (N-1) are left out: their responses carried seeded noise.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))
import analyse_ln_null as ln  # noqa: E402
from caliper.activations import collect, load_model, sample_corpus  # noqa: E402

TEST_SEED, TEST_TOKENS, BINS, VAL = 2, 4000, 60, 0.2
ARMS = {  # tag: (stem, model, layer, corpus_seed, sequence_split, dtype)
    "b14r": ("results/b14r_primary_gpt2_dirs", "gpt2", 6, 0, False, "fp32"),
    "d1": ("results/d1_gpt2_l6_fresh", "gpt2", 6, 0, False, "fp32"),
    "b15a": ("results/b15a_fitseed1", "gpt2", 6, 0, False, "fp32"),
    "b15b": ("results/b15b_corpusseed1", "gpt2", 6, 1, False, "fp32"),
    "b15c": ("results/b15c_seqsplit", "gpt2", 6, 0, True, "fp32"),
    "b17c": ("results/b17c_gpt2_l3", "gpt2", 3, 0, False, "fp32"),
    "b17d": ("results/b17d_gpt2_l3_drop", "gpt2", 3, 0, False, "fp32"),
    "b8b": ("results/b8b_gptneo125m_indep", "EleutherAI/gpt-neo-125M", 10, 0, False, "fp32"),
    "x1a": ("results/x1a_gptneo_l10", "EleutherAI/gpt-neo-125M", 10, 0, False, "fp32"),
    "x1b": ("results/x1b_gptneo_l10", "EleutherAI/gpt-neo-125M", 10, 1, False, "fp32"),
    "b17": ("results/b17_gptneo125m_l6", "EleutherAI/gpt-neo-125M", 6, 0, False, "fp32"),
    "b17b": ("results/b17b_gptneo125m_l6_drop", "EleutherAI/gpt-neo-125M", 6, 0, False, "fp32"),
    "b2c": ("results/b2c_pythia160m_indep", "EleutherAI/pythia-160m", 6, 0, False, "fp16"),
    "b2d": ("results/b2d_pythia160m_fp32", "EleutherAI/pythia-160m", 6, 0, False, "fp32"),
}


def test_r2(v, S_tr, y_tr, S_te, y_te):
    z_tr, z_te = S_tr @ v, S_te @ v
    edges = np.quantile(z_tr, np.linspace(0, 1, BINS + 1)[1:-1])
    b_tr, b_te = np.searchsorted(edges, z_tr), np.searchsorted(edges, z_te)
    means = np.array([y_tr[b_tr == b].mean() if (b_tr == b).any() else y_tr.mean() for b in range(BINS)])
    pred = means[b_te]
    return float(1 - ((y_te - pred) ** 2).sum() / ((y_te - y_te.mean()) ** 2).sum())


def score(tag):
    stem, model_name, layer, cseed, seq, dtype = ARMS[tag]
    R = [json.loads(l) for l in open(ROOT / f"{stem}.jsonl", encoding="utf-8") if l.strip()]
    units = np.array([int(r["_key"]) for r in R])
    model, tok = load_model(model_name, dtype={"fp32": torch.float32, "fp16": torch.float16}[dtype])
    train_docs = sample_corpus(n_docs=300, seed=cseed)
    seen = set(train_docs) | set(sample_corpus(n_docs=300, seed=1 - cseed if cseed in (0, 1) else 0))
    test_docs = [t for t in sample_corpus(n_docs=300, seed=TEST_SEED) if t not in seen]
    p = collect(model, tok, train_docs, layer=layer, neurons=units, max_tokens=8000, seed=0,
                shuffle="sequence" if seq else "token")
    q = collect(model, tok, test_docs, layer=layer, neurons=units, max_tokens=TEST_TOKENS, seed=0, shuffle="token")
    S, Y = np.asarray(p.stimulus[:8000], np.float64), np.asarray(p.response[:8000], np.float64)
    n_val = int(len(S) * VAL)
    S_tr, Y_tr = S[n_val:], Y[n_val:]
    S_te, Y_te = np.asarray(q.stimulus[:TEST_TOKENS], np.float64), np.asarray(q.response[:TEST_TOKENS], np.float64)
    n, _ = ln.null_direction(model_name, layer)
    P = np.eye(len(n)) - np.outer(n, n)
    unit = lambda x: x / np.linalg.norm(x)
    out = []
    for j, r in enumerate(R):
        z = np.load(ROOT / f"{stem}_dirs/n{r['_key']}.npz")
        v = z[r["picked"]][:, 0].astype(np.float64)
        w = z["w"].astype(np.float64)
        out.append({"unit": int(r["_key"]), "test_r2": test_r2(v, S_tr, Y_tr[:, j], S_te, Y_te[:, j]),
                    "heldout_r2": float(r["r2_k1"]), "restart": float(r["stability"]),
                    "euclid": abs(unit(v) @ unit(w)), "ident": abs(unit(P @ v) @ unit(P @ w))})
    g = lambda k: np.array([x[k] for x in out])
    rep = {"units": len(out), "test_docs": len(test_docs),
           "median_test_minus_heldout": float(np.median(g("test_r2") - g("heldout_r2")))}
    for lab in ("euclid", "ident"):
        fail = g(lab) < ln.PASS
        rep[lab] = {"failures": int(fail.sum()),
                    "restart_vs_test": ln.compare(-g("restart"), -g("test_r2"), fail),
                    "restart_vs_heldout": ln.compare(-g("restart"), -g("heldout_r2"), fail)}
    return rep, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    a = ap.parse_args()
    path = ROOT / "results/test_r2_saved_arms.json"
    rep = json.load(open(path)) if path.exists() else {}
    for tag in a.arms:
        rep[tag], rows = score(tag)
        rep[tag]["rows"] = rows
        json.dump(rep, open(path, "w"), indent=1, default=float)
        c = rep[tag]["ident"]["restart_vs_test"]
        print(tag, rep[tag]["units"], "ident fails", rep[tag]["ident"]["failures"],
              "delta vs test", c and round(c["diff"], 3), "median test-heldout", round(rep[tag]["median_test_minus_heldout"], 4),
              flush=True)


if __name__ == "__main__":
    main()
