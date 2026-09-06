"""E0.1i - can held-out R^2 pick the right fit without knowing the answer?

E0.1h: the cascade rescues the neurons the direct search loses (n2977 0.452->0.999,
n230 0.866->0.999) but slightly damages some it does not (n1561 0.991->0.917). Neither
method dominates, so the estimator should run both and keep the better one.

The selection must not use ground truth - Phase A will not have any. The only legitimate
criterion is the objective itself: held-out R^2. This tests whether that criterion picks
the more accurate fit, at full precision, and whether it ever ties.
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
print(f"{'neuron':>7} {'direct':>17} {'cascade':>17} {'picked':>9} {'align':>7} {'ok':>4}")
print(f"{'':>7} {'align  R2':>17} {'align  R2':>17}")
for i, n in enumerate(p.neurons):
    wu = p.weights[:, i] / np.linalg.norm(p.weights[:, i])
    y = p.response[:, i]
    d = fit(p.stimulus, y, k=1, n_restarts=3, steps=2500, seed=0)
    c = fit_cascade(p.stimulus, y, k=1, n_restarts=3, steps=2500, seed=0)
    ad = abs(subspace_alignment(d.subspace, wu[:, None]))
    ac = abs(subspace_alignment(c.subspace, wu[:, None]))
    pick = "cascade" if c.test_r2 > d.test_r2 else "direct"
    chosen = ac if pick == "cascade" else ad
    rows.append({"neuron": int(n), "align_direct": round(ad, 4),
                 "r2_direct": float(d.test_r2), "align_cascade": round(ac, 4),
                 "r2_cascade": float(c.test_r2), "picked": pick,
                 "align_selected": round(chosen, 4),
                 "best_available": round(max(ad, ac), 4),
                 "r2_margin": float(abs(c.test_r2 - d.test_r2))})
    print(f"{n:>7} {ad:>7.3f} {d.test_r2:>9.6f} {ac:>7.3f} {c.test_r2:>9.6f} "
          f"{pick:>9} {chosen:>7.3f} {'ok' if chosen > 0.95 else 'FAIL':>4}", flush=True)

json.dump(rows, open("results/e01i_selection.json", "w"), indent=2)
sel = np.array([r["align_selected"] for r in rows])
best = np.array([r["best_available"] for r in rows])
margin = np.array([r["r2_margin"] for r in rows])
print(f"\n  selected median {np.median(sel):.4f}   min {sel.min():.4f}")
print(f"  oracle-best     {np.median(best):.4f}   min {best.min():.4f}")
print(f"  regret (oracle - selected): median {np.median(best - sel):.4f} "
      f"max {np.max(best - sel):.4f}")
print(f"  smallest R2 margin between methods: {margin.min():.2e}")
print(f"  above 0.95 after selection: {int((sel > 0.95).sum())}/{len(sel)}")
