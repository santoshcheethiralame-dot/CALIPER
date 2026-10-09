"""F-0: why seed-only checks are blind to sampling error (docs/flagship-plan.md §5).

    python experiments/f0_simulation.py            # writes results/f0_simulation.json

A synthetic single-index unit, y = GELU(g * w.s / sd(w.s) + b), noiseless as in Paper 1, with
Gaussian stimulus of known covariance C, fitted by the same batched estimator Paper 1 uses.
Three regimes:
  well     C's spectrum spans 2 decades
  ill      6 decades, the order of the real residual streams (condition numbers 1e4-1e9)
  bias     ill, and every sample is read through the same fixed distortion A = I + 0.8 u v^T,
           so the estimand is consistently wrong: error that does not move when data are redrawn
Per unit: Euclidean alignment to w (truth); and four ground-truth-free signals:
  restart agreement   the estimator's own two-restart agreement (Paper 1's check)
  seed replicate      |cos| to a re-fit of the same data with another fit seed
  data replicate      |cos| to a fit of an independent sample, same settings (X-1's check)
  low-variance share  share of the fitted direction in C-hat's bottom-1%-variance directions
  held-out R2         the estimator's validation fit
Predictions (F-0): seed checks ~ uninformative where data replicates are informative; mean
data-replicate disagreement ~ 2x mean error when error is variance (well, ill) and ~ 0x when
it is bias; every signal blind in the bias regime.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from caliper.batched import fit_batch  # noqa: E402

D, N, UNITS, STEPS = 128, 8000, 120, 1600
torch.set_num_threads(4)


def auc(score, fail):
    from scipy.stats import rankdata
    score, fail = np.asarray(score, float), np.asarray(fail, bool)
    m, n = fail.sum(), (~fail).sum()
    if not m or not n:
        return None
    r = rankdata(score)
    return float((r[fail].sum() - m * (m + 1) / 2) / (m * n))


def sample(rng, evals, Q, A=None):
    s = (rng.standard_normal((N, D)) * np.sqrt(evals)) @ Q.T
    return s if A is None else s @ A.T, s


def responses(s_true, W, g, b):
    pre = s_true @ W
    pre = g * pre / pre.std(0) + b
    return torch.nn.functional.gelu(torch.as_tensor(pre)).numpy()


def fit(S, Y, seed):
    out = fit_batch(S.astype(np.float32), Y.astype(np.float32), k=1, n_restarts=2, steps=STEPS,
                    seed=seed, per_neuron_stop=True, unit_ids=np.arange(Y.shape[1]))
    V = np.stack([f.subspace[:, 0] / np.linalg.norm(f.subspace[:, 0]) for f in out], 1)
    return V, np.array([f.stability for f in out]), np.array([f.test_r2 for f in out])


def low_share(V, S):
    lam, vec = np.linalg.eigh(np.cov(S, rowvar=False))
    low = vec[:, np.cumsum(lam) / lam.sum() <= 0.01]
    return ((low.T @ V) ** 2).sum(0)


def regime(name, decades, bias, rng):
    evals = np.logspace(0, -decades, D)
    Q = np.linalg.qr(rng.standard_normal((D, D)))[0]
    W = rng.standard_normal((D, UNITS))
    W /= np.linalg.norm(W, axis=0)
    g = rng.uniform(1.0, 2.0, UNITS)
    b = rng.uniform(-2.5, 0.5, UNITS)
    A = None
    if bias:
        u, v = rng.standard_normal(D), rng.standard_normal(D)
        A = np.eye(D) + 0.8 * np.outer(u / np.linalg.norm(u), v / np.linalg.norm(v))
    Sa, Sa_true = sample(rng, evals, Q, A)
    Sb, Sb_true = sample(rng, evals, Q, A)
    Ya, Yb = responses(Sa_true, W, g, b), responses(Sb_true, W, g, b)
    Va, restart, r2 = fit(Sa, Ya, seed=0)
    Va2, _, _ = fit(Sa, Ya, seed=1)
    Vb, _, _ = fit(Sb, Yb, seed=0)
    align = np.abs((Va * W).sum(0))
    seed_rep = np.abs((Va * Va2).sum(0))
    data_rep = np.abs((Va * Vb).sum(0))
    share = low_share(Va, Sa)
    fail = align < 0.95
    cw = fail & (r2 > 0.99)
    signals = {"restart agreement": restart, "seed replicate": seed_rep,
               "data replicate": data_rep, "held-out R2": r2, "low-variance share": -share}
    rep = {"decades": decades, "bias": bias, "units": UNITS, "failures": int(fail.sum()),
           "converged_wrong": int(cw.sum()), "median_alignment": float(np.median(align)),
           # AUC at predicting failure; every signal oriented so that low means "worry".
           "auc_predicting_failure": {k: auc(-x, fail) for k, x in signals.items()},
           "auc_converged_wrong_vs_pass": {k: auc(-x[cw | ~fail], cw[cw | ~fail])
                                           for k, x in signals.items()} if cw.sum() else None,
           "mean_error_1_minus_cos": float(np.mean(1 - align)),
           "mean_data_disagreement_1_minus_cos": float(np.mean(1 - data_rep)),
           "mean_seed_disagreement_1_minus_cos": float(np.mean(1 - seed_rep)),
           "spearman_data_disagreement_vs_error": float(
               __import__("scipy.stats", fromlist=["spearmanr"]).spearmanr(1 - data_rep, 1 - align)[0]),
           "spearman_seed_disagreement_vs_error": float(
               __import__("scipy.stats", fromlist=["spearmanr"]).spearmanr(1 - seed_rep, 1 - align)[0])}
    rep["ratio_data_disagreement_to_error"] = (rep["mean_data_disagreement_1_minus_cos"]
                                               / rep["mean_error_1_minus_cos"])
    print(name, json.dumps(rep, indent=1), flush=True)
    return rep


def main():
    rng = np.random.default_rng(20261009)
    out = {"settings": {"d": D, "n": N, "units": UNITS, "steps": STEPS, "restarts": 2},
           "well": regime("well", 2, False, rng),
           "ill": regime("ill", 6, False, rng),
           "bias": regime("bias", 6, True, rng)}
    json.dump(out, open(ROOT / "results/f0_simulation.json", "w"), indent=1)


if __name__ == "__main__":
    main()
