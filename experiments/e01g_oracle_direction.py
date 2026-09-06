"""E0.1g - is the k=1 optimum actually reachable, or is the landscape the problem?

The four surviving E0.1 failures all fit near-perfectly at k=2 (R^2 0.993-0.999) while
stalling at k=1. Since ground truth is exactly one-dimensional, a k=1 fit MUST be able to
reach R^2 ~ 1. Two possibilities:

  (a) the objective at the TRUE direction is high and the optimiser cannot find it
      -> pure optimisation failure; fix the initialisation
  (b) the objective at the true direction is itself low
      -> the model class or the loss is wrong, and no initialisation saves it

Test: clamp V to the true direction, train only the nonlinearity, and read off R^2.
Then check whether the k=2 subspace already contains the true direction - if it does,
cascading down from k=2 is the fix.
"""
import json
import numpy as np
import torch
from torch import nn
from caliper.activations import collect, load_model, sample_corpus
from caliper.estimator import fit, subspace_alignment

FAILED = np.array([2977, 230, 1543, 1989])
CONTROL = np.array([824, 1561])
neurons = np.concatenate([FAILED, CONTROL])

m, tok = load_model("gpt2")
p = collect(m, tok, sample_corpus(n_docs=200, seed=0), layer=6, neurons=neurons,
            max_tokens=20000, seed=0)


def r2_at_fixed_direction(X, y, v, steps=2500, seed=0):
    """Best achievable R^2 with the direction clamped to v; only f is trained."""
    z = torch.tensor(((X @ v) - (X @ v).mean()) / (X @ v).std(), dtype=torch.float32)[:, None]
    t = torch.tensor((y - y.mean()) / y.std(), dtype=torch.float32)
    n_te = len(z) // 5
    ztr, ttr, zte, tte = z[n_te:], t[n_te:], z[:n_te], t[:n_te]
    torch.manual_seed(seed)
    f = nn.Sequential(nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
                      nn.Linear(64, 1))
    opt = torch.optim.Adam(f.parameters(), lr=3e-3)
    best = -np.inf
    for s in range(steps):
        opt.zero_grad()
        ((f(ztr).squeeze(-1) - ttr) ** 2).mean().backward()
        opt.step()
        if s % 50 == 0:
            with torch.no_grad():
                pr = f(zte).squeeze(-1)
                best = max(best, float(1 - ((pr - tte) ** 2).sum() /
                                       ((tte - tte.mean()) ** 2).sum()))
    return best


rows = []
print(f"{'neuron':>7} {'grp':>5} {'R2@true':>9} {'R2 k=1':>8} {'align k1':>9} "
      f"{'w in V2':>8} {'R2 k=2':>8}")
for i, n in enumerate(p.neurons):
    w = p.weights[:, i]
    wu = w / np.linalg.norm(w)
    y = p.response[:, i]
    grp = "fail" if int(n) in FAILED else "ok"

    oracle = r2_at_fixed_direction(p.stimulus, y, w)
    f1 = fit(p.stimulus, y, k=1, n_restarts=3, steps=2500, seed=0)
    f2 = fit(p.stimulus, y, k=2, n_restarts=3, steps=2500, seed=0)
    row = {"neuron": int(n), "group": grp,
           "r2_at_true_direction": round(oracle, 4),
           "r2_k1": round(f1.test_r2, 4),
           "align_k1": round(abs(subspace_alignment(f1.subspace, wu[:, None])), 4),
           "w_in_k2_subspace": round(abs(subspace_alignment(f2.subspace, wu[:, None])), 4),
           "r2_k2": round(f2.test_r2, 4)}
    rows.append(row)
    print(f"{n:>7} {grp:>5} {row['r2_at_true_direction']:>9.3f} {row['r2_k1']:>8.3f} "
          f"{row['align_k1']:>9.3f} {row['w_in_k2_subspace']:>8.3f} {row['r2_k2']:>8.3f}",
          flush=True)

json.dump(rows, open("results/e01g_oracle.json", "w"), indent=2)
f = [r for r in rows if r["group"] == "fail"]
print(f"\n  failing neurons: median R2 at TRUE direction = "
      f"{np.median([r['r2_at_true_direction'] for r in f]):.3f}")
print(f"                   median R2 the optimiser found = "
      f"{np.median([r['r2_k1'] for r in f]):.3f}")
print(f"                   median |w in k=2 subspace|    = "
      f"{np.median([r['w_in_k2_subspace'] for r in f]):.3f}")
