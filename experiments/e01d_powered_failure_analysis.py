"""E0.1d - what actually predicts estimator failure, at adequate sample size.

Four candidate explanations were each tested on the 20 neurons of E0.1. Every one
returned the right direction with p ~ 0.13 and a roughly 2x separation between passing
and failing units: effective sample size, firing rate, lifetime kurtosis, silent mass.
That is the signature of an underpowered diagnostic, not of four weak effects.

This run measures every candidate on 100 neurons in one pass, then compares them in a
single multivariate model instead of a sequence of underpowered univariate tests.
Only the raw squared-error objective is used; the rank transform was shown to halve the
pass rate (0.75 -> 0.50) and is excluded.
"""
import argparse, json, time
import numpy as np
from scipy import stats
from caliper.activations import load_model, sample_corpus, collect, _blocks
from caliper.estimator import fit, subspace_alignment

ap = argparse.ArgumentParser()
ap.add_argument("--neurons", type=int, default=100)
ap.add_argument("--tokens", type=int, default=20000)
ap.add_argument("--layer", type=int, default=6)
ap.add_argument("--restarts", type=int, default=3)
ap.add_argument("--steps", type=int, default=1200)
ap.add_argument("--out", default="results/e01d.json")
a = ap.parse_args()

t0 = time.time()
m, tok = load_model("gpt2")
rng = np.random.default_rng(0)
neurons = rng.choice(3072, size=a.neurons, replace=False)
p = collect(m, tok, sample_corpus(n_docs=300, seed=0), layer=a.layer,
            neurons=neurons, max_tokens=a.tokens, seed=0)
bias = _blocks(m)[a.layer].mlp.c_fc.bias.detach().numpy()[neurons]
D = p.stimulus.shape[1]
print(f"collected {p.stimulus.shape} in {time.time()-t0:.0f}s", flush=True)

rows = []
for i, n in enumerate(p.neurons):
    w = p.weights[:, i]
    y = p.response[:, i]
    if y.std() < 1e-4:
        continue
    z = p.stimulus @ w + bias[i]
    w_unit = w / np.linalg.norm(w)

    f1 = fit(p.stimulus, y, k=1, n_restarts=a.restarts, steps=a.steps, seed=0)
    f2 = fit(p.stimulus, y, k=2, n_restarts=a.restarts, steps=a.steps, seed=0)
    row = {
        "neuron": int(n),
        "alignment": abs(subspace_alignment(f1.subspace, w_unit[:, None])),
        "r2_k1": f1.test_r2, "k2_gain": f2.test_r2 - f1.test_r2,
        "stability": f1.stability,
        # candidate predictors, all measured on the same data
        "n_eff": int((z > -0.75).sum()),
        "frac_active": float((z > 0).mean()),
        "kurtosis": float(stats.kurtosis(y, fisher=False)),
        "silent_mass": float((y < y.min() + 1e-3 * y.std()).mean()),
        "rho_a_z": float(stats.spearmanr(z, y).statistic),
        "resp_std": float(y.std()),
    }
    rows.append(row)
    if len(rows) % 10 == 0:
        print(f"  {len(rows)} neurons, {time.time()-t0:.0f}s", flush=True)

json.dump({"config": vars(a), "records": rows}, open(a.out, "w"), indent=2)

al = np.array([r["alignment"] for r in rows])
passing = (al > 0.95) & (np.array([r["k2_gain"] for r in rows]) < 0.01)
print(f"\n  n={len(rows)}  median alignment={np.median(al):.4f}  "
      f"fraction passing={passing.mean():.3f}")

print(f"\n  {'predictor':<14} {'spearman':>9} {'p':>9}   {'fail':>9} {'pass':>9}")
for key in ["n_eff", "frac_active", "kurtosis", "silent_mass", "rho_a_z", "resp_std"]:
    v = np.array([r[key] for r in rows])
    sr = stats.spearmanr(v, al)
    print(f"  {key:<14} {sr.statistic:>+9.3f} {sr.pvalue:>9.4f}   "
          f"{np.median(v[~passing]):>9.3g} {np.median(v[passing]):>9.3g}")

# One multivariate model instead of six univariate tests.
X = np.column_stack([stats.zscore(np.log10(np.array([r[k] for r in rows]) + 1e-9))
                     for k in ["n_eff", "kurtosis", "resp_std"]]
                    + [stats.zscore([r["rho_a_z"] for r in rows])])
X = np.column_stack([np.ones(len(X)), X])
beta, *_ = np.linalg.lstsq(X, al, rcond=None)
print(f"\n  multivariate (z-scored) coefficients on alignment:")
for name, b in zip(["intercept", "log n_eff", "log kurtosis", "log resp_std", "rho(a,z)"], beta):
    print(f"    {name:<14} {b:>+.4f}")
