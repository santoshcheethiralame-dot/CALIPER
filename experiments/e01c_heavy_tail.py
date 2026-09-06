"""E0.1c - is the residual failure the objective rather than the estimator?

Four of five E0.1 failures survived a 20x larger optimisation budget, converging
reproducibly to a wrong answer with k=1 R^2 well below 1.0 - even though the ground
truth is exactly one-dimensional. That points at the loss, not the subspace search:
squared error on a heavy-tailed response is dominated by a handful of extreme events,
so the effective sample size is the event count, not the position count. This is why
the source literature counts spikes rather than stimulus presentations.

Test: refit under variance-stabilising transforms of the response. If alignment jumps,
the failure is the objective.
"""
import json
import numpy as np
from scipy import stats
from caliper.activations import load_model, sample_corpus, collect
from caliper.estimator import fit, subspace_alignment

NEURONS = np.array([230, 2977, 1561])  # two stuck (kurt 549, 66), one passing (kurt 32)

m, tok = load_model("gpt2")
p = collect(m, tok, sample_corpus(n_docs=200, seed=0), layer=6,
            neurons=NEURONS, max_tokens=20000, seed=0)

def gaussianise(y):
    return stats.norm.ppf(stats.rankdata(y) / (len(y) + 1))

rows = []
for i, n in enumerate(p.neurons):
    w = p.weights[:, i] / np.linalg.norm(p.weights[:, i])
    y = p.response[:, i]
    kurt = float(stats.kurtosis(y, fisher=False))
    thr = y.mean() + 2 * y.std()
    row = {"neuron": int(n), "kurtosis": round(kurt, 1),
           "n_events": int((y > thr).sum())}
    for name, yy in [("raw", y), ("log1p", np.log1p(y - y.min())),
                     ("rank", gaussianise(y))]:
        f = fit(p.stimulus, yy, k=1, n_restarts=2, steps=800, seed=1)
        row[f"align_{name}"] = round(
            abs(subspace_alignment(f.subspace, w[:, None])), 3)
        row[f"r2_{name}"] = round(f.test_r2, 3)
    rows.append(row)
    print(f"  n{n:<5d} kurt={kurt:>6.0f} events={row['n_events']:>5d} | "
          f"raw {row['align_raw']:.3f} (R2 {row['r2_raw']:.3f})  "
          f"log1p {row['align_log1p']:.3f} ({row['r2_log1p']:.3f})  "
          f"rank {row['align_rank']:.3f} ({row['r2_rank']:.3f})", flush=True)

json.dump(rows, open("results/e01c_heavy_tail.json", "w"), indent=2)
best = max(("raw", "log1p", "rank"),
           key=lambda t: np.mean([r[f"align_{t}"] for r in rows]))
print(f"\n  best transform on average: {best}")
print("  VERDICT:", "objective, not estimator" if best != "raw"
      else "transforms do not rescue it; look elsewhere")
