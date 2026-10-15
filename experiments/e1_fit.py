"""E-1 fits (docs/preregistration-e1-fair-comparison.md): one estimator route, independent restarts,
and a test set of documents the fit never saw.

    python experiments/e1_fit.py --model gpt2 --layer 6 --units <ids> --out results/e1_gpt2_l6.jsonl

Training and validation tokens come from the corpus draw every earlier GPT-2 run used (300
documents, corpus seed 0, 8,000 tokens): the first 20% of the shuffled rows are the validation
split that drives early stopping and picks the best restart; the rest train. Test tokens come from
documents of corpus seed 1 that are not in that draw (4,000 tokens). Only the direct route (k = 1)
is fitted, with five restarts all initialised at random (no ridge warm start), per-unit seeds and
per-unit early stopping. Per unit we record:

  restart_agreement   median pairwise |cos| between the five restarts' directions
  val_r2              validation R^2 of the selected restart (the estimator's own score)
  test_r2             R^2 on the test documents of the selected direction, through a binned link
                      (60 equal-count bins) fitted on the training rows only
  r2_spread           max minus min validation R^2 across restarts

Every restart's direction and both R^2s are saved, with the covariance of the 8,000 fitted tokens for
the low-variance diagnostic (the paper's definition, analyse_a4_spectrum.bottom).
"""
import argparse
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from caliper.activations import collect, load_model, sample_corpus
from caliper.batched import fit_batch
from caliper.runtime import Checkpoint, pick_device

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gpt2")
ap.add_argument("--layer", type=int, default=6)
ap.add_argument("--units", required=True)
ap.add_argument("--tokens", type=int, default=8000)
ap.add_argument("--test-tokens", type=int, default=4000)
ap.add_argument("--restarts", type=int, default=5)
ap.add_argument("--steps", type=int, default=1600)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--bins", type=int, default=60)
ap.add_argument("--out", required=True)
a = ap.parse_args()

TEST_FRAC = 0.2
t0 = time.time()
units = np.array([int(u) for u in a.units.split(",")])
model, tok = load_model(a.model, dtype=torch.float32)
train_docs = sample_corpus(n_docs=300, seed=0)
seen = set(train_docs)
test_docs = [t for t in sample_corpus(n_docs=300, seed=1) if t not in seen]
p = collect(model, tok, train_docs, layer=a.layer, neurons=units, max_tokens=a.tokens, seed=0,
            shuffle="token")
q = collect(model, tok, test_docs, layer=a.layer, neurons=units, max_tokens=a.test_tokens, seed=0,
            shuffle="token")
if len(p.stimulus) < a.tokens or len(q.stimulus) < a.test_tokens:
    raise SystemExit(f"corpus gave {len(p.stimulus)} / {len(q.stimulus)} tokens")
S, Y = p.stimulus[:a.tokens], p.response[:a.tokens]
St, Yt = q.stimulus[:a.test_tokens], q.response[:a.test_tokens]
n_val = int(len(S) * TEST_FRAC)
S_tr, Y_tr = S[n_val:], Y[n_val:]
stem = a.out[:-len(".jsonl")]
dirs = Path(stem + "_dirs")
dirs.mkdir(parents=True, exist_ok=True)
np.savez_compressed(stem + "_stimulus.npz", cov=np.cov(S, rowvar=False), mean=S.mean(0),
                    test_docs=len(test_docs), train_docs=len(train_docs))
print(f"  stimulus {S.shape}, test {St.shape} from {len(test_docs)} unseen documents "
      f"({time.time() - t0:.0f}s)", flush=True)


def test_r2(v, y_tr, y_te):
    """R^2 on the test documents of direction v through a binned link fitted on training rows."""
    z_tr, z_te = S_tr @ v, St @ v
    edges = np.quantile(z_tr, np.linspace(0, 1, a.bins + 1)[1:-1])
    b_tr, b_te = np.searchsorted(edges, z_tr), np.searchsorted(edges, z_te)
    means = np.array([y_tr[b_tr == b].mean() if (b_tr == b).any() else y_tr.mean()
                      for b in range(a.bins)])
    pred = means[b_te]
    return float(1 - ((y_te - pred) ** 2).sum() / ((y_te - y_te.mean()) ** 2).sum())


ck = Checkpoint(a.out)
dev = pick_device("auto")
alive = Y.std(0) > 1e-4
todo = [i for i in range(len(units)) if alive[i] and not ck.done(int(units[i]))]
print(f"  alive {int(alive.sum())}/{len(units)}; {len(todo)} to fit", flush=True)
for start in range(0, len(todo), a.batch):
    idx = todo[start:start + a.batch]
    fits = fit_batch(S, Y[:, idx], k=1, n_restarts=a.restarts, steps=a.steps, test_frac=TEST_FRAC,
                     seed=0, device=dev, warm_start=False, unit_ids=units[idx], per_neuron_stop=True)
    for j, i in enumerate(idx):
        n, f = int(units[i]), fits[j]
        R = np.stack([np.asarray(r, dtype=np.float64)[:, 0] for r in f.restarts])
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        sel = int(np.argmax(f.r2_restarts))
        te = [test_r2(R[r], Y_tr[:, i], Yt[:, i]) for r in range(len(R))]
        w = p.weights[:, i].astype(np.float64)
        pairs = [abs(float(R[x] @ R[y])) for x, y in combinations(range(len(R)), 2)]
        np.savez_compressed(dirs / f"n{n}.npz", w=w.astype(np.float32), restarts=R.astype(np.float32),
                            val_r2=np.asarray(f.r2_restarts, np.float32), test_r2=np.asarray(te, np.float32),
                            selected=sel)
        ck.record(n, {
            "align_euclid": round(abs(float(R[sel] @ w)) / float(np.linalg.norm(w)), 6),
            "restart_agreement": round(float(np.median(pairs)), 6),
            "val_r2": round(float(f.r2_restarts[sel]), 6),
            "test_r2": round(te[sel], 6),
            "r2_spread": round(float(np.max(f.r2_restarts) - np.min(f.r2_restarts)), 6),
            "n_restarts": len(R),
        })
    el = time.time() - t0
    print(f"  {len(ck.rows())}/{int(alive.sum())} units  {el:.0f}s ({el / max(len(ck.rows()), 1):.1f}s each)",
          flush=True)
json.dump({"model": a.model, "layer": a.layer, "units": len(units), "rows": len(ck.rows()),
           "restarts": a.restarts, "steps": a.steps, "tokens": a.tokens, "test_tokens": a.test_tokens,
           "test_docs": len(test_docs), "seconds_per_unit": round((time.time() - t0) / max(len(ck.rows()), 1), 1),
           "torch_threads": torch.get_num_threads()},
          open(stem + "_summary.json", "w"), indent=1)
