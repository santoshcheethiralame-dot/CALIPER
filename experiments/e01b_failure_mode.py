"""E0.1b - are the E0.1 failures neuron properties or optimisation failures?

The five neurons that missed the criterion were fitted with 2 restarts and 700 steps.
Before concluding anything about sparse or high-kurtosis units, re-fit them with a
much larger optimisation budget. If they recover, the failure was ours, not theirs.
"""
import json, numpy as np
from caliper.activations import load_model, sample_corpus, collect
from caliper.estimator import fit, subspace_alignment

rec = json.load(open("results/e01.json"))["records"]
failing = [r for r in rec if not (r["raw_alignment"] > 0.95 and r["k2_gain"] < 0.01)]
neurons = np.array([r["neuron"] for r in failing])
print("re-fitting", list(neurons), flush=True)

m, tok = load_model("gpt2")
p = collect(m, tok, sample_corpus(n_docs=200, seed=0), layer=6,
            neurons=neurons, max_tokens=20000, seed=0)

out = []
for i, n in enumerate(p.neurons):
    w = p.weights[:, i] / np.linalg.norm(p.weights[:, i])
    before = [r for r in failing if r["neuron"] == int(n)][0]
    f1 = fit(p.stimulus, p.response[:, i], k=1, n_restarts=10, steps=3000, seed=1)
    f2 = fit(p.stimulus, p.response[:, i], k=2, n_restarts=10, steps=3000, seed=1)
    a = abs(subspace_alignment(f1.subspace, w[:, None]))
    row = {"neuron": int(n), "align_before": before["raw_alignment"],
           "align_after": a, "gain_before": before["k2_gain"],
           "gain_after": f2.test_r2 - f1.test_r2, "r2_after": f1.test_r2,
           "stability_after": f1.stability}
    out.append(row)
    print(f"  n{n:<5d} {before['raw_alignment']:.3f} -> {a:.3f}   "
          f"k2gain {before['k2_gain']:+.3f} -> {row['gain_after']:+.3f}  "
          f"R2={f1.test_r2:.3f} stab={f1.stability:.3f}", flush=True)

fixed = sum(1 for r in out if r["align_after"] > 0.95 and r["gain_after"] < 0.01)
print(f"\n  recovered with a larger budget: {fixed}/{len(out)}")
print("  VERDICT:", "optimisation budget, not neuron property" if fixed >= len(out) - 1
      else "a genuine failure mode remains")
json.dump(out, open("results/e01b_failure_mode.json", "w"), indent=2)
