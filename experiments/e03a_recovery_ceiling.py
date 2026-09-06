"""E0.3a - the empirical recovery ceiling as a function of effective sample size.

E0.1's pass criterion (95% of neurons above 0.95 alignment) assumed every neuron is
equally recoverable. It is not. These units sit mostly in GELU's flat, non-monotonic
region, so the number of *informative* positions varies more than tenfold across
neurons, and Paninski's minimax bound applies to every estimator, not just this one -
a direction cannot be recovered from a unit that barely fires.

So instead of trusting a theoretical constant, measure the ceiling directly: build
synthetic units on the REAL residual stream with a KNOWN direction, sweep how often they
fire, and record the best alignment achievable at each effective sample size. Real
neurons can then be scored against the ceiling at their own N_eff rather than against a
flat threshold. If they sit on it, the estimator is information-limited, and the
"failures" are not failures.
"""
import json
import numpy as np
from caliper.activations import collect, load_model, sample_corpus
from caliper.estimator import fit, subspace_alignment

FRAC_ACTIVE = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.80]
SEEDS = [0, 1, 2]

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

m, tok = load_model("gpt2")
p = collect(m, tok, sample_corpus(n_docs=300, seed=0), layer=6,
            neurons=np.array([0]), max_tokens=20000, seed=0)
X = p.stimulus
D = X.shape[1]
print(f"stimulus {X.shape}", flush=True)

rows = []
for frac in FRAC_ACTIVE:
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        w = rng.standard_normal(D)
        w /= np.linalg.norm(w)
        z = X @ w
        z = (z - z.mean()) / z.std()
        t = np.quantile(z, 1 - frac)
        a = gelu(z - t)                      # fires on `frac` of positions
        n_eff = int(((z - t) > -0.75).sum())

        f = fit(X, a, k=1, n_restarts=3, steps=1200, seed=seed)
        row = {"frac_active": frac, "seed": seed, "n_eff": n_eff,
               "n_eff_over_D": round(n_eff / D, 3),
               "alignment": abs(subspace_alignment(f.subspace, w[:, None])),
               "r2": f.test_r2}
        rows.append(row)
        print(f"  frac={frac:<6} seed={seed} N_eff={n_eff:>6} "
              f"(N_eff/D={row['n_eff_over_D']:>6.2f})  align={row['alignment']:.3f}  "
              f"R2={f.test_r2:.3f}", flush=True)

json.dump({"D": D, "records": rows},
          open("results/e03a_recovery_ceiling.json", "w"), indent=2)

print(f"\n  {'frac':>7} {'N_eff':>7} {'N_eff/D':>8} {'ceiling':>9}")
for frac in FRAC_ACTIVE:
    g = [r for r in rows if r["frac_active"] == frac]
    print(f"  {frac:>7} {int(np.median([r['n_eff'] for r in g])):>7} "
          f"{np.median([r['n_eff_over_D'] for r in g]):>8.2f} "
          f"{np.median([r['alignment'] for r in g]):>9.3f}")

# Score the real neurons of E0.1 against the ceiling at their own N_eff.
try:
    real = json.load(open("results/e01.json"))["records"]
    ne = np.array([r["n_eff_over_D"] for r in rows])
    ceil = np.array([r["alignment"] for r in rows])
    order = np.argsort(ne)
    print(f"\n  interpolated ceiling at selected N_eff/D:")
    for q in [2.5, 5, 10, 20]:
        print(f"    N_eff/D={q:>5}: ceiling ~= {np.interp(q, ne[order], ceil[order]):.3f}")
except FileNotFoundError:
    pass
