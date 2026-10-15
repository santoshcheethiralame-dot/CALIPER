"""OXIDE: score recovered unit directions, and the checks used to trust them, against the exact reference.

    python -m caliper.oxide --model gpt2 --layer 6 --directions mine.npz [--checks mine.csv] [--out report.json]

An MLP unit's pre-activation is w . s + b, with s the output of the layer norm feeding the MLP and w the
unit's input weight column, so the direction the unit reads is w. Its component along n = 1/gamma (the
layer norm's gain, normalised) cannot be recovered from any data, because the layer norm gives every
token the same projection onto n. Each recovered direction is scored two ways:

  euclidean      |cos(v, w)|
  identifiable   |cos(P v, P w)| with P = I - n n'     (the label the data can determine)

A fit fails when its identifiable alignment is below --pass (0.95). Inputs:

  --directions   an .npz with `units` (unit ids, shape [k]) and `directions` (shape [k, d_model]),
                 in the coordinates of the layer norm's output
  --checks       optional .csv with a `unit` column and one column per check, higher meaning more
                 trusted (for example held-out R^2 or restart agreement)

For each check the report gives its AUC at telling failures from passes, with a stratified bootstrap
95% interval, and the difference from the first check with its interval. Nothing is fitted here.
"""
import argparse
import csv
import json

import numpy as np
from scipy.stats import rankdata


def reference(model_name, layer, units):
    """The units' weight columns (d_model x k) and the unit null direction n of the feeding layer norm."""
    import torch
    from caliper.activations import _blocks, _mlp_in_weight, _mlp_ln, load_model
    model, _ = load_model(model_name, dtype=torch.float32)
    block = _blocks(model)[layer]
    W = _mlp_in_weight(block)[:, np.asarray(units)].astype(np.float64)
    ln = _mlp_ln(block)
    if ln is None:
        n = None
    else:
        g = ln.weight.detach().double().numpy()
        n = (1.0 / g) / np.linalg.norm(1.0 / g)
    return W, n


def alignments(V, W, n=None):
    """Per-unit Euclidean and identifiable |cos| between rows of V (k x d) and columns of W (d x k)."""
    V = np.asarray(V, np.float64)
    W = np.asarray(W, np.float64).T
    unit = lambda X: X / np.linalg.norm(X, axis=1, keepdims=True)
    euc = np.abs(np.sum(unit(V) * unit(W), axis=1))
    if n is None:               # RMSNorm and other norms without a mean subtraction: nothing to remove
        return euc, euc.copy()
    P = lambda X: X - np.outer(X @ n, n)
    ident = np.abs(np.sum(unit(P(V)) * unit(P(W)), axis=1))
    return euc, ident


def auc(score, fail):
    """Mann-Whitney AUC, ties counted half; higher score = more suspect."""
    score, fail = np.asarray(score, float), np.asarray(fail, bool)
    m, k = fail.sum(), (~fail).sum()
    if m == 0 or k == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[fail].sum() - m * (m + 1) / 2) / (m * k))


def boot(scores, fail, n_boot=2000, seed=0):
    """Stratified bootstrap (failures and passes resampled separately) of each AUC and of each
    difference from the first check."""
    rng = np.random.default_rng(seed)
    fi, pi = np.flatnonzero(fail), np.flatnonzero(~fail)
    names = list(scores)
    draws = {k: [] for k in names}
    diffs = {k: [] for k in names[1:]}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(fi, len(fi)), rng.choice(pi, len(pi))])
        a = {k: auc(scores[k][idx], fail[idx]) for k in names}
        for k in names:
            draws[k].append(a[k])
        for k in names[1:]:
            diffs[k].append(a[k] - a[names[0]])
    ci = lambda x: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]
    return {k: ci(v) for k, v in draws.items()}, {k: ci(v) for k, v in diffs.items()}


def evaluate(units, euc, ident, checks=None, pass_bar=0.95):
    fail = ident < pass_bar
    rep = {"units": int(len(units)), "pass_bar": pass_bar,
           "failures_identifiable": int(fail.sum()), "failures_euclidean": int((euc < pass_bar).sum()),
           "median_identifiable": float(np.median(ident)),
           "per_unit": [{"unit": int(u), "euclidean": float(e), "identifiable": float(i)}
                        for u, e, i in zip(units, euc, ident)]}
    if checks and fail.any() and (~fail).any():
        scores = {k: -np.asarray(v, float) for k, v in checks.items()}   # higher value = more trusted
        ci, dci = boot(scores, fail)
        names = list(scores)
        rep["checks"] = {k: {"auc": auc(scores[k], fail), "ci95": ci[k]} for k in names}
        rep["difference_from_" + names[0]] = {
            k: {"diff": rep["checks"][k]["auc"] - rep["checks"][names[0]]["auc"], "ci95": dci[k]} for k in names[1:]}
    return rep


def read_checks(path, units):
    with open(path, newline="") as f:
        rows = {int(r["unit"]): r for r in csv.DictReader(f)}
    missing = [u for u in units if u not in rows]
    if missing:
        raise SystemExit(f"--checks has no row for units {missing[:5]}")
    names = [k for k in next(iter(rows.values())) if k != "unit"]
    return {k: np.array([float(rows[int(u)][k]) for u in units]) for k in names}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--directions", required=True)
    ap.add_argument("--checks")
    ap.add_argument("--pass", dest="pass_bar", type=float, default=0.95)
    ap.add_argument("--out")
    a = ap.parse_args()
    z = np.load(a.directions)
    units, V = np.asarray(z["units"]), np.asarray(z["directions"])
    W, n = reference(a.model, a.layer, units)
    euc, ident = alignments(V, W, n)
    rep = evaluate(units, euc, ident, read_checks(a.checks, units) if a.checks else None, a.pass_bar)
    rep.update(model=a.model, layer=a.layer)
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1)
    print(f"{rep['units']} units: {rep['failures_identifiable']} fail on the identifiable label, "
          f"{rep['failures_euclidean']} on the Euclidean label (bar {a.pass_bar})")
    for k, v in rep.get("checks", {}).items():
        print(f"  {k:<24} AUC {v['auc']:.3f}  [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")
    for k, v in next((rep[x] for x in rep if x.startswith("difference_from_")), {}).items():
        print(f"  {k} minus first check: {v['diff']:+.3f}  [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")


if __name__ == "__main__":
    main()
