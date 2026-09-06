"""E0.1h - does descending from k+1 recover the neurons the k=1 search loses?

E0.1g measured the diagnosis exactly: on the four surviving failures the objective at
the TRUE direction is 1.000, the k=1 search reaches 0.809, and the k=2 subspace already
contains the true direction at 0.997. The answer is found and then lost in the
parameterisation. fit_cascade searches inside the k+1 subspace instead.
"""
import json
import numpy as np
from caliper.activations import collect, load_model, sample_corpus
from caliper.estimator import fit, fit_cascade, subspace_alignment

FAILED = [2977, 230, 1543, 1989]
CONTROL = [824, 1561, 2796, 1945]
neurons = np.array(FAILED + CONTROL)

m, tok = load_model("gpt2")
p = collect(m, tok, sample_corpus(n_docs=200, seed=0), layer=6, neurons=neurons,
            max_tokens=20000, seed=0)

rows = []
print(f"{'neuron':>7} {'grp':>5} {'direct':>8} {'cascade':>9} {'delta':>8} {'R2':>7}")
for i, n in enumerate(p.neurons):
    wu = p.weights[:, i] / np.linalg.norm(p.weights[:, i])
    y = p.response[:, i]
    grp = "fail" if int(n) in FAILED else "ok"
    d = fit(p.stimulus, y, k=1, n_restarts=3, steps=2500, seed=0)
    c = fit_cascade(p.stimulus, y, k=1, n_restarts=3, steps=2500, seed=0)
    ad = abs(subspace_alignment(d.subspace, wu[:, None]))
    ac = abs(subspace_alignment(c.subspace, wu[:, None]))
    rows.append({"neuron": int(n), "group": grp, "direct": round(ad, 4),
                 "cascade": round(ac, 4), "delta": round(ac - ad, 4),
                 "r2_cascade": round(float(c.test_r2), 4)})
    print(f"{n:>7} {grp:>5} {ad:>8.3f} {ac:>9.3f} {ac-ad:>+8.3f} {c.test_r2:>7.3f}",
          flush=True)

json.dump(rows, open("results/e01h_cascade.json", "w"), indent=2)
f = [r for r in rows if r["group"] == "fail"]
o = [r for r in rows if r["group"] == "ok"]
print(f"\n  failing neurons: direct median {np.median([r['direct'] for r in f]):.3f}"
      f"  ->  cascade median {np.median([r['cascade'] for r in f]):.3f}")
print(f"  controls:        direct median {np.median([r['direct'] for r in o]):.3f}"
      f"  ->  cascade median {np.median([r['cascade'] for r in o]):.3f}")
print(f"  all above 0.95 under cascade: "
      f"{int(sum(r['cascade'] > 0.95 for r in rows))}/{len(rows)}")
