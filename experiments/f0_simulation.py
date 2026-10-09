"""F-0: why seed-only checks are blind to sampling error (docs/flagship-plan.md §5).

    python experiments/f0_simulation.py --part a     # analytic, seconds
    python experiments/f0_simulation.py --part b     # Paper 1's estimator, about an hour on CPU

Part A, the theory in its cleanest form. A linear unit y = w.s + noise, fitted by least squares.
The fit is a deterministic function of the data, so restarts and seeds agree exactly (agreement
1) whatever the error. Two independent samples give E|v1 - v2|^2 = 2 tr(Sigma) while
E|v - w|^2 = tr(Sigma) + bias^2, with Sigma = sigma^2 (X'X)^-1, about sigma^2 C^-1 / n: the
error lives in C's low-eigenvalue directions, which is what the low-variance share measures.
Regimes: well (C spans 2 decades), ill (4), ill with a ridge penalty (regularisation shrinks the
poorly sampled directions, turning their variance into a consistent bias), and bias (well
conditioned, low noise, every sample read through the same strong fixed distortion, so the
estimand itself is wrong). Per-unit noise
varies, so units differ in difficulty.

Part B, the same three signals for Paper 1's estimator on noiseless GELU units: whether a
nonlinear fit with restarts behaves like Part A's prediction.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def auc(score, fail):
    score, fail = np.asarray(score, float), np.asarray(fail, bool)
    m, n = fail.sum(), (~fail).sum()
    if not m or not n:
        return None
    r = rankdata(score)
    return float((r[fail].sum() - m * (m + 1) / 2) / (m * n))


def covariance(rng, d, decades):
    evals = np.logspace(0, -decades, d)
    Q = np.linalg.qr(rng.standard_normal((d, d)))[0]
    return evals, Q


def draw(rng, n, evals, Q, A=None):
    s = (rng.standard_normal((n, len(evals))) * np.sqrt(evals)) @ Q.T
    return (s if A is None else s @ A.T), s


def low_share(V, S):
    lam, vec = np.linalg.eigh(np.cov(S, rowvar=False))
    low = vec[:, np.cumsum(lam) / lam.sum() <= 0.01]
    return ((low.T @ V) ** 2).sum(0), low


def unit(V):
    return V / np.linalg.norm(V, axis=0, keepdims=True)


def summarise(name, W, Va, Va_seed, Vb, r2, share, err_low, extra=None):
    align = np.abs((Va * W).sum(0))
    data_rep = np.abs((Va * Vb).sum(0))
    seed_rep = np.round(np.abs((Va * Va_seed).sum(0)), 9)   # exact ties read as ties, not as float noise
    fail = align < 0.95
    good_fit = r2 > np.quantile(r2, 0.5)
    hard = fail & good_fit                       # wrong although it fits well: the hard case
    signals = {"seed / restart agreement": seed_rep, "data replicate": data_rep,
               "held-out R2": r2, "low-variance share": -share}
    rep = {"units": len(align), "failures": int(fail.sum()),
           "fail_with_above_median_R2": int(hard.sum()),
           "median_alignment": float(np.median(align)),
           "seed_agreement_range": [float(seed_rep.min()), float(seed_rep.max())],
           "auc_predicting_failure": {k: auc(-x, fail) for k, x in signals.items()},
           "auc_hard_failures_vs_pass": ({k: auc(-x[hard | ~fail], hard[hard | ~fail])
                                         for k, x in signals.items()} if hard.sum() else None),
           "mean_error": float(np.mean(1 - align)),
           "mean_data_disagreement": float(np.mean(1 - data_rep)),
           "ratio_disagreement_to_error": float(np.mean(1 - data_rep) / np.mean(1 - align)),
           "spearman_disagreement_vs_error": float(spearmanr(1 - data_rep, 1 - align)[0]),
           "median_error_energy_in_low_variance": float(np.median(err_low)),
           **(extra or {})}
    print(name, json.dumps(rep, indent=1), flush=True)
    return rep


def part_a(seed=20261009, d=128, n=4000, units=500):
    out = {}
    regimes = (("well", 2, 0.0, 0.0, (2, 2000)), ("ill", 4, 0.0, 0.0, (2, 2000)),
               ("ill, ridge", 4, 0.0, 3e-3, (2, 2000)), ("bias", 2, 5.0, 0.0, (500, 5000)))
    for name, decades, strength, ridge, snr_range in regimes:
        bias = strength > 0
        rng = np.random.default_rng(seed)
        evals, Q = covariance(rng, d, decades)
        W = unit(rng.standard_normal((d, units)))
        A = None
        if bias:
            u, v = unit(rng.standard_normal((d, 1)))[:, 0], unit(rng.standard_normal((d, 1)))[:, 0]
            A = np.eye(d) + strength * np.outer(u, v)
        snr = np.exp(rng.uniform(*np.log(snr_range), units))     # per-unit difficulty

        def fit(Sobs, Strue):
            z = Strue @ W
            Y = z + rng.standard_normal(z.shape) * z.std(0) / np.sqrt(snr)
            k = n // 5
            X = Sobs[k:]
            C = X.T @ X / len(X)
            beta = np.linalg.solve(C + ridge * np.trace(C) / d * np.eye(d), X.T @ Y[k:] / len(X))
            pred = Sobs[:k] @ beta
            r2 = 1 - ((Y[:k] - pred) ** 2).sum(0) / ((Y[:k] - Y[:k].mean(0)) ** 2).sum(0)
            return unit(beta), r2

        Sa, Sa_t = draw(rng, n, evals, Q, A)
        Sb, Sb_t = draw(rng, n, evals, Q, A)
        Va, r2 = fit(Sa, Sa_t)
        Vb, _ = fit(Sb, Sb_t)
        share, low = low_share(Va, Sa)
        E = Va * np.sign((Va * W).sum(0)) - W
        err_low = ((low.T @ E) ** 2).sum(0) / (E ** 2).sum(0)
        out[name] = summarise(f"A/{name}", W, Va, Va.copy(), Vb, r2, share, err_low,
                              {"decades": decades, "distortion": strength, "ridge": ridge,
                               "snr_range": list(snr_range), "n": n, "d": d})
    return out


def part_b(seed=20261009, d=64, n=8000, units=24, steps=800):
    import torch
    from caliper.batched import fit_batch
    torch.set_num_threads(2)
    out = {}
    for name, decades in (("well", 2), ("ill", 6)):
        rng = np.random.default_rng(seed)
        evals, Q = covariance(rng, d, decades)
        W = unit(rng.standard_normal((d, units)))
        g, b = rng.uniform(1.0, 2.0, units), rng.uniform(-2.5, 0.5, units)

        def resp(S):
            z = S @ W
            return torch.nn.functional.gelu(torch.as_tensor(g * z / z.std(0) + b)).numpy()

        def fit(S, Y, s):
            f = fit_batch(S.astype(np.float32), Y.astype(np.float32), k=1, n_restarts=2,
                          steps=steps, seed=s, per_neuron_stop=True, unit_ids=np.arange(units))
            V = unit(np.stack([x.subspace[:, 0] for x in f], 1))
            return V, np.array([x.stability for x in f]), np.array([x.test_r2 for x in f])

        Sa, _ = draw(rng, n, evals, Q)
        Sb, _ = draw(rng, n, evals, Q)
        Va, restart, r2 = fit(Sa, resp(Sa), 0)
        Va2, _, _ = fit(Sa, resp(Sa), 1)
        Vb, _, _ = fit(Sb, resp(Sb), 0)
        share, low = low_share(Va, Sa)
        E = Va * np.sign((Va * W).sum(0)) - W
        err_low = ((low.T @ E) ** 2).sum(0) / np.maximum((E ** 2).sum(0), 1e-12)
        rep = summarise(f"B/{name}", W, Va, Va2, Vb, r2, share, err_low,
                        {"decades": decades, "n": n, "d": d, "steps": steps})
        rep["auc_predicting_failure"]["restart agreement (estimator's own)"] = auc(
            -restart, np.abs((Va * W).sum(0)) < 0.95)
        out[name] = rep
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("a", "b"), required=True)
    a = ap.parse_args()
    path = ROOT / "results/f0_simulation.json"
    rep = json.load(open(path)) if path.exists() else {}
    rep[f"part_{a.part}"] = part_a() if a.part == "a" else part_b()
    json.dump(rep, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
