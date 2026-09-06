"""E0.1e - is GELU's non-monotone dip what defeats the estimator?

Established: ground truth is exact (corr 1.00000000), and synthetic units on the real
residual stream are recovered at 0.976-0.998 at every sparsity (E0.3a) - so there is no
information limit. Yet ~25% of real neurons fall below 0.95.

The one thing E0.3a did not hold fixed was the nonlinearity's operating range. Real
neurons have pre-activation std ~0.42-0.84 and mean ~-1, so they sit almost entirely in
GELU's non-monotone dip rather than its steep monotone arm.

This holds w, z and the stimulus at their real values and varies ONLY the nonlinearity.
If monotone nonlinearities recover and GELU does not, the dip is the cause.
"""
import json
import numpy as np
import torch
from caliper.activations import collect, load_model, sample_corpus, _blocks
from caliper.estimator import fit, subspace_alignment

FAILING = [2977, 230, 1543, 1989]
PASSING = [824, 1561]
neurons = np.array(FAILING + PASSING)

m, tok = load_model("gpt2")
p = collect(m, tok, sample_corpus(n_docs=200, seed=0), layer=6,
            neurons=neurons, max_tokens=20000, seed=0)
bias = _blocks(m)[6].mlp.c_fc.bias.detach().numpy()[neurons]

NONLIN = {
    "gelu": lambda z: torch.nn.functional.gelu(torch.tensor(z)).numpy(),  # the real one
    "relu": lambda z: np.maximum(z, 0.0),                  # monotone, same threshold
    "softplus": lambda z: np.log1p(np.exp(np.clip(z, -30, 30))),  # monotone, smooth
    "identity": lambda z: z,                               # linear: trivially recoverable
}

rows = []
for i, n in enumerate(p.neurons):
    w = p.weights[:, i]
    w_unit = w / np.linalg.norm(w)
    z = p.stimulus @ w + bias[i]
    row = {"neuron": int(n), "group": "fail" if int(n) in FAILING else "pass",
           "z_std": round(float(z.std()), 3), "z_mean": round(float(z.mean()), 3)}
    for name, f in NONLIN.items():
        a = f(z)
        g = fit(p.stimulus, a, k=1, n_restarts=3, steps=1200, seed=0)
        row[name] = round(abs(subspace_alignment(g.subspace, w_unit[:, None])), 3)
        row[f"r2_{name}"] = round(g.test_r2, 3)
    rows.append(row)
    print(f"  n{row['neuron']:<5d} [{row['group']}] z~N({row['z_mean']:+.2f},{row['z_std']:.2f})  "
          + "  ".join(f"{k}={row[k]:.3f}" for k in NONLIN), flush=True)

json.dump(rows, open("results/e01e_nonlinearity.json", "w"), indent=2)
print()
for name in NONLIN:
    v = np.array([r[name] for r in rows])
    f = np.array([r[name] for r in rows if r["group"] == "fail"])
    print(f"  {name:<10} median={np.median(v):.3f}  median(failing neurons)={np.median(f):.3f}")
