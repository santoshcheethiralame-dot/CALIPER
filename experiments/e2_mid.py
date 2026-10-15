"""E-2 fits (docs/preregistration-e2-mid.md): a second estimator family on E-1's units.

    python experiments/e2_mid.py --model gpt2 --layer 6 --units <ids> --out results/e2_mid_gpt2_l6.jsonl

Maximally informative dimensions (Sharpee et al., 2004): the direction u whose projection carries the
most mutual information about the response. Here the information is estimated by a differentiable
plug-in estimator: the projection z = S u (standardised) is soft-binned onto 32 Gaussian kernels on
[-4, 4], the response is binned into 16 equal-count bins, and I(z; y) is computed from the joint
histogram. u is fitted by Adam, five restarts from random directions, per-unit early stopping on
validation information. No network and no link function are fitted, so the estimator shares nothing
with E-1's except the data.

Data, splits, test set and recorded checks are E-1's exactly (`e1_fit.py`): 8,000 corpus-seed-0 tokens
(first 20% validation), 4,000 test tokens from unseen corpus-seed-1 documents, per-coordinate
standardisation of the stimulus, test R^2 through a 60-bin link fitted on the training rows.
"""
import argparse
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from caliper.activations import collect, load_model, sample_corpus
from caliper.runtime import Checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gpt2")
ap.add_argument("--layer", type=int, default=6)
ap.add_argument("--units", required=True)
ap.add_argument("--tokens", type=int, default=8000)
ap.add_argument("--test-tokens", type=int, default=4000)
ap.add_argument("--restarts", type=int, default=5)
ap.add_argument("--steps", type=int, default=1600)
ap.add_argument("--lr", type=float, default=0.02)
ap.add_argument("--patience", type=int, default=200)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--z-bins", type=int, default=32)
ap.add_argument("--y-bins", type=int, default=16)
ap.add_argument("--bins", type=int, default=60)
ap.add_argument("--out", required=True)
a = ap.parse_args()

TEST_FRAC = 0.2
t0 = time.time()
torch.manual_seed(0)
units = np.array([int(u) for u in a.units.split(",")])
model, tok = load_model(a.model, dtype=torch.float32)
train_docs = sample_corpus(n_docs=300, seed=0)
seen = set(train_docs)
test_docs = [t for t in sample_corpus(n_docs=300, seed=1) if t not in seen]
p = collect(model, tok, train_docs, layer=a.layer, neurons=units, max_tokens=a.tokens, seed=0, shuffle="token")
q = collect(model, tok, test_docs, layer=a.layer, neurons=units, max_tokens=a.test_tokens, seed=0, shuffle="token")
S, Y = p.stimulus[:a.tokens].astype(np.float64), p.response[:a.tokens].astype(np.float64)
St, Yt = q.stimulus[:a.test_tokens].astype(np.float64), q.response[:a.test_tokens].astype(np.float64)
n_val = int(len(S) * TEST_FRAC)
S_tr, Y_tr = S[n_val:], Y[n_val:]
mu, sd = S.mean(0), S.std(0) + 1e-6
X = torch.as_tensor((S - mu) / sd, dtype=torch.float32)
X_val, X_tr = X[:n_val], X[n_val:]
centres = torch.linspace(-4, 4, a.z_bins)
width = float(centres[1] - centres[0])
stem = a.out[:-len(".jsonl")]
dirs = Path(stem + "_dirs")
dirs.mkdir(parents=True, exist_ok=True)
np.savez_compressed(stem + "_stimulus.npz", cov=np.cov(S, rowvar=False), mean=S.mean(0))
print(f"  stimulus {S.shape}, test {St.shape} from {len(test_docs)} unseen documents ({time.time() - t0:.0f}s)",
      flush=True)


def ybins(y):
    """Equal-count response bins, edges from the training rows, applied to any rows."""
    edges = np.quantile(y[n_val:], np.linspace(0, 1, a.y_bins + 1)[1:-1])
    return torch.as_tensor(np.searchsorted(edges, y), dtype=torch.long)


def info(Xs, U, yb):
    """Plug-in I(z; y) in nats per unit for directions U (n_units x d) and response bins yb (T x n_units)."""
    Z = Xs @ U.T
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)
    K = torch.exp(-0.5 * ((Z[:, :, None] - centres) / width) ** 2)
    K = K / K.sum(-1, keepdim=True)                                          # T x n x Mz
    Yo = torch.nn.functional.one_hot(yb, a.y_bins).float()                   # T x n x My
    joint = torch.einsum("tnm,tnk->nmk", K, Yo) / len(Xs) + 1e-12
    pz, py = joint.sum(2, keepdim=True), joint.sum(1, keepdim=True)
    return (joint * (joint.log() - pz.log() - py.log())).sum((1, 2))


def test_r2(v, y_tr, y_te):
    z_tr, z_te = S_tr @ v, St @ v
    edges = np.quantile(z_tr, np.linspace(0, 1, a.bins + 1)[1:-1])
    b_tr, b_te = np.searchsorted(edges, z_tr), np.searchsorted(edges, z_te)
    means = np.array([y_tr[b_tr == b].mean() if (b_tr == b).any() else y_tr.mean() for b in range(a.bins)])
    pred = means[b_te]
    return float(1 - ((y_te - pred) ** 2).sum() / ((y_te - y_te.mean()) ** 2).sum())


def val_r2(v, y):
    """Binned R^2 on the validation rows, link fitted on the training rows: the estimator's own score,
    as E-1's validation R^2."""
    z_tr, z_va = S_tr @ v, S[:n_val] @ v
    edges = np.quantile(z_tr, np.linspace(0, 1, a.bins + 1)[1:-1])
    b_tr, b_va = np.searchsorted(edges, z_tr), np.searchsorted(edges, z_va)
    means = np.array([y[n_val:][b_tr == b].mean() if (b_tr == b).any() else y[n_val:].mean() for b in range(a.bins)])
    pred, yv = means[b_va], y[:n_val]
    return float(1 - ((yv - pred) ** 2).sum() / ((yv - yv.mean()) ** 2).sum())


ck = Checkpoint(a.out)
alive = Y.std(0) > 1e-4
todo = [i for i in range(len(units)) if alive[i] and not ck.done(int(units[i]))]
print(f"  alive {int(alive.sum())}/{len(units)}; {len(todo)} to fit", flush=True)
for start in range(0, len(todo), a.batch):
    idx = todo[start:start + a.batch]
    yb = torch.stack([ybins(Y[:, i]) for i in idx], 1)
    yb_tr, yb_val = yb[n_val:], yb[:n_val]
    restarts, vinfo = [], []
    for r in range(a.restarts):
        g = torch.Generator().manual_seed(0)
        init = torch.stack([torch.randn(X.shape[1], generator=torch.Generator().manual_seed(int(units[i]) * 1000 + r))
                            for i in idx])
        U = torch.nn.Parameter(init / init.norm(dim=1, keepdim=True))
        opt = torch.optim.Adam([U], lr=a.lr)
        best = torch.full((len(idx),), -np.inf)
        best_U = U.detach().clone()
        waited = torch.zeros(len(idx))
        active = torch.ones(len(idx), dtype=torch.bool)
        for step in range(a.steps):
            opt.zero_grad()
            (-info(X_tr, U / U.norm(dim=1, keepdim=True), yb_tr)[active].sum()).backward()
            opt.step()
            if step % 25 == 0:
                with torch.no_grad():
                    vi = info(X_val, U / U.norm(dim=1, keepdim=True), yb_val)
                    imp = (vi > best) & active
                    best[imp] = vi[imp]
                    best_U[imp] = U.detach()[imp]
                    waited = torch.where(imp, torch.zeros_like(waited), waited + 25)
                    active &= waited < a.patience
                    if not active.any():
                        break
        # back to the original stimulus frame: z = ((s - mu) / sd) . u, so the direction in s is u / sd
        V = best_U.numpy().astype(np.float64) / sd
        restarts.append(V / np.linalg.norm(V, axis=1, keepdims=True))
        vinfo.append(best.numpy())
    R = np.stack(restarts, 1)                  # n x restarts x d
    VI = np.stack(vinfo, 1)
    for j, i in enumerate(idx):
        n = int(units[i])
        sel = int(np.argmax(VI[j]))
        w = p.weights[:, i].astype(np.float64)
        te = [test_r2(R[j, r], Y_tr[:, i], Yt[:, i]) for r in range(a.restarts)]
        va = [val_r2(R[j, r], Y[:, i]) for r in range(a.restarts)]
        pairs = [abs(float(R[j, x] @ R[j, y])) for x, y in combinations(range(a.restarts), 2)]
        np.savez_compressed(dirs / f"n{n}.npz", w=w.astype(np.float32), restarts=R[j].astype(np.float32),
                            val_info=VI[j].astype(np.float32), val_r2=np.asarray(va, np.float32),
                            test_r2=np.asarray(te, np.float32), selected=sel)
        ck.record(n, {"align_euclid": round(abs(float(R[j, sel] @ w)) / float(np.linalg.norm(w)), 6),
                      "restart_agreement": round(float(np.median(pairs)), 6),
                      "val_info": round(float(VI[j, sel]), 6), "val_r2": round(va[sel], 6), "test_r2": round(te[sel], 6),
                      "n_restarts": a.restarts})
    el = time.time() - t0
    print(f"  {len(ck.rows())}/{int(alive.sum())} units  {el:.0f}s ({el / max(len(ck.rows()), 1):.1f}s each)", flush=True)
json.dump({"model": a.model, "layer": a.layer, "units": len(units), "rows": len(ck.rows()), "restarts": a.restarts,
           "steps": a.steps, "lr": a.lr, "z_bins": a.z_bins, "y_bins": a.y_bins, "test_docs": len(test_docs),
           "seconds_per_unit": round((time.time() - t0) / max(len(ck.rows()), 1), 1),
           "torch_threads": torch.get_num_threads()}, open(stem + "_summary.json", "w"), indent=1)
