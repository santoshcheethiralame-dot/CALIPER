"""E0.1f - does warm-starting from the ridge solution close the gap?

E0.1e showed the optimiser, not the nonlinearity, is the binding constraint: with an
IDENTITY nonlinearity - where recovering w is just regression - random starts still
returned 0.970 rather than 1.000. A closed-form starting point is available for free.

Restart 0 is warm-started from the ridge direction; the remaining restarts stay random,
so restart agreement remains a meaningful diagnostic rather than a foregone conclusion.
"""
import json
import numpy as np
from caliper.activations import collect, load_model, sample_corpus
from caliper.estimator import fit, subspace_alignment

# every neuron that has ever failed, plus two controls that never did
NEURONS = np.array([2977, 230, 1543, 1989, 1859, 536, 125, 824, 1561])

m, tok = load_model("gpt2")
p = collect(m, tok, sample_corpus(n_docs=200, seed=0), layer=6, neurons=NEURONS,
            max_tokens=20000, seed=0)

rows = []
print(f"{'neuron':>7} {'cold':>7} {'warm':>7} {'delta':>8} {'k2 warm':>9}")
for i, n in enumerate(p.neurons):
    w = p.weights[:, i] / np.linalg.norm(p.weights[:, i])
    y = p.response[:, i]
    cold = fit(p.stimulus, y, k=1, n_restarts=2, steps=1200, seed=0, warm_start=False)
    warm = fit(p.stimulus, y, k=1, n_restarts=2, steps=1200, seed=0, warm_start=True)
    warm2 = fit(p.stimulus, y, k=2, n_restarts=2, steps=1200, seed=0, warm_start=True)
    ac = abs(subspace_alignment(cold.subspace, w[:, None]))
    ah = abs(subspace_alignment(warm.subspace, w[:, None]))
    rows.append({"neuron": int(n), "cold": round(ac, 4), "warm": round(ah, 4),
                 "delta": round(ah - ac, 4), "r2_warm": round(warm.test_r2, 4),
                 "k2_gain_warm": round(warm2.test_r2 - warm.test_r2, 4)})
    print(f"{n:>7} {ac:>7.3f} {ah:>7.3f} {ah-ac:>+8.3f} {rows[-1]['k2_gain_warm']:>+9.3f}",
          flush=True)

json.dump(rows, open("results/e01f_warm_start.json", "w"), indent=2)
d = np.array([r["delta"] for r in rows])
warm = np.array([r["warm"] for r in rows])
gain = np.array([r["k2_gain_warm"] for r in rows])
print(f"\n  median improvement {np.median(d):+.4f}   improved {int((d > 0.001).sum())}/{len(d)}")
print(f"  warm: median={np.median(warm):.4f}  min={warm.min():.4f}  "
      f"passing={float(np.mean((warm > 0.95) & (gain < 0.01))):.2f}")
