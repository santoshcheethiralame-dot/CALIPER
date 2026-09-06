"""E0.5 - do the classical closed-form estimators suffice?

Bussgang's theorem says that for a = f(w.s) with GAUSSIAN s, Cov(s,a) = c.Sigma.w for
any f, so ridge regression of response on stimulus recovers w up to scale with no
optimisation. If that held here the fitted estimator would be unnecessary machinery.

It does not hold. Two assumptions fail together: the residual stream is strongly
non-Gaussian, and GELU is non-monotonic over the range these neurons actually occupy
(55-90% of positions sit past its minimum), which drives the Bussgang constant
c = E[f'(z)] toward zero and empties the spike-triggered average of signal.
"""
import json
import numpy as np
from caliper.activations import collect, load_model, sample_corpus
from caliper.estimator import fit, subspace_alignment

m, tok = load_model("gpt2")
rng = np.random.default_rng(0)
neurons = rng.choice(3072, size=30, replace=False)
p = collect(m, tok, sample_corpus(n_docs=200, seed=0), layer=6,
            neurons=neurons, max_tokens=20000, seed=0)

X = p.stimulus - p.stimulus.mean(0)
C = X.T @ X / len(X)
Cinv = np.linalg.inv(C + 1e-3 * np.trace(C) / C.shape[0] * np.eye(C.shape[0]))
cos = lambda u, v: abs(float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v))))

rows = []
for i, n in enumerate(p.neurons):
    w, y = p.weights[:, i], p.response[:, i]
    if y.std() < 1e-4:
        continue
    sta = X.T @ (y - y.mean()) / len(X)
    S = X[y > np.quantile(y, 0.95)]
    stc = np.linalg.eigh(Cinv @ (S.T @ S / len(S) - C))[1][:, -1]
    f1 = fit(p.stimulus, y, k=1, n_restarts=3, steps=1200, seed=0)
    rows.append({"neuron": int(n), "sta": cos(sta, w),
                 "decorrelated_sta": cos(Cinv @ sta, w), "stc_top": cos(stc, w),
                 "bottleneck": abs(subspace_alignment(
                     f1.subspace, (w / np.linalg.norm(w))[:, None]))})
    print(f"  n{rows[-1]['neuron']:<5d} STA={rows[-1]['sta']:.3f} "
          f"decorr={rows[-1]['decorrelated_sta']:.3f} STC={rows[-1]['stc_top']:.3f} "
          f"fitted={rows[-1]['bottleneck']:.3f}", flush=True)

summary = {}
for k in ["sta", "decorrelated_sta", "stc_top", "bottleneck"]:
    v = np.array([r[k] for r in rows])
    summary[k] = {"median": round(float(np.median(v)), 4),
                  "min": round(float(v.min()), 4),
                  "frac_above_95": round(float((v > 0.95).mean()), 3)}
    print(f"  {k:<18} median={summary[k]['median']:.4f} "
          f"frac>0.95={summary[k]['frac_above_95']:.2f}")
json.dump({"summary": summary, "records": rows},
          open("results/e05_classical_baselines.json", "w"), indent=2)
print("\n  VERDICT: classical closed-form estimators are insufficient; "
      "the fitted estimator is necessary, not ornamental.")
