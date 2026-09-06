"""E0.2 - the random-direction null: what does "found something" mean?

Sharpee, Rust & Bialek warned that with strongly correlated stimuli at D ~ 10^3, even a
projection onto a RANDOM vector can explain a large share of a unit's response. If that
holds here, then a high R^2 is not evidence of anything and every downstream claim needs
a threshold rather than a number.

This measures the null directly. For each random direction v we compute the best
achievable fit along it - not a trained nonlinearity but the exact optimum: bin the
projection into equal-count bins and take within-bin means, which is the best possible
function of one variable. So this is an upper bound on what a random direction can do,
which is the conservative thing to compare against.

Two nulls are reported per neuron:
  * R^2 null   - what a random direction explains
  * alignment null - how close a random direction lands to the true weight
and both are compared against the true direction and against the fitted estimator.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from caliper.activations import collect, load_model, sample_corpus
from caliper.estimator import fit, subspace_alignment

ROOT = Path(__file__).resolve().parents[1]


def best_1d_r2(z, y, n_bins=50):
    """Exact best R^2 of any function of z, via equal-count binning.

    No optimisation: sorting by z and averaging y within bins IS the least-squares
    optimal predictor for a piecewise-constant function of z at this resolution.
    """
    n = (len(z) // n_bins) * n_bins
    order = np.argsort(z, kind="stable")[:n]
    ys = y[order].reshape(n_bins, -1)
    pred = np.repeat(ys.mean(axis=1, keepdims=True), ys.shape[1], axis=1)
    ss_res = ((ys - pred) ** 2).sum()
    ss_tot = ((y[order] - y[order].mean()) ** 2).sum()
    return 1.0 - ss_res / ss_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neurons", type=int, default=12)
    ap.add_argument("--tokens", type=int, default=20_000)
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--n-random", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--bins", type=int, default=50)
    ap.add_argument("--fit-neurons", type=int, default=4,
                    help="how many neurons also get the full fitted estimator")
    ap.add_argument("--out", default="results/e02.json")
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(0)
    model, tokenizer = load_model("gpt2")
    neurons = rng.choice(3072, size=args.neurons, replace=False)
    p = collect(model, tokenizer, sample_corpus(n_docs=200, seed=0), layer=args.layer,
                neurons=neurons, max_tokens=args.tokens, seed=0)
    X = p.stimulus
    D = X.shape[1]
    print(f"stimulus {X.shape}  ({time.time()-t0:.0f}s)", flush=True)

    # Null over random directions, shared across neurons: one projection matrix,
    # reused for every unit, so the null is directly comparable across them.
    null_r2 = {int(n): [] for n in p.neurons}
    align_null = []
    W = p.weights / np.linalg.norm(p.weights, axis=0, keepdims=True)

    done = 0
    while done < args.n_random:
        b = min(args.batch, args.n_random - done)
        V = rng.standard_normal((D, b))
        V /= np.linalg.norm(V, axis=0, keepdims=True)
        Z = X @ V                                    # (n, b)
        align_null.append(np.abs(W.T @ V).ravel())
        for i, n in enumerate(p.neurons):
            y = p.response[:, i]
            for j in range(b):
                null_r2[int(n)].append(best_1d_r2(Z[:, j], y, args.bins))
        done += b
        print(f"  {done}/{args.n_random} random directions "
              f"({time.time()-t0:.0f}s)", flush=True)

    align_null = np.concatenate(align_null)

    rows = []
    for i, n in enumerate(p.neurons):
        y = p.response[:, i]
        z_true = X @ p.weights[:, i]
        nr = np.array(null_r2[int(n)])
        row = {
            "neuron": int(n),
            "true_direction_r2": round(float(best_1d_r2(z_true, y, args.bins)), 4),
            "null_r2_median": round(float(np.median(nr)), 4),
            "null_r2_p99": round(float(np.percentile(nr, 99)), 4),
            "null_r2_max": round(float(nr.max()), 4),
        }
        if i < args.fit_neurons:
            f = fit(X, y, k=1, n_restarts=3, steps=2500, seed=0)
            row["fitted_alignment"] = round(
                abs(subspace_alignment(f.subspace, W[:, i][:, None])), 4)
            row["fitted_r2"] = round(f.test_r2, 4)
        rows.append(row)
        print(f"  n{n:<5d} true={row['true_direction_r2']:.3f}  "
              f"null p99={row['null_r2_p99']:.3f} max={row['null_r2_max']:.3f}"
              + (f"  fitted={row['fitted_r2']:.3f}" if "fitted_r2" in row else ""),
              flush=True)

    summary = {
        "n_random_directions": args.n_random,
        "stimulus_dim": D,
        "alignment_null_median": round(float(np.median(align_null)), 5),
        "alignment_null_p99": round(float(np.percentile(align_null, 99)), 5),
        "alignment_null_max": round(float(align_null.max()), 5),
        "isotropic_expectation": round(float(np.sqrt(1 / D)), 5),
        "median_true_r2": round(float(np.median([r["true_direction_r2"] for r in rows])), 4),
        "median_null_p99": round(float(np.median([r["null_r2_p99"] for r in rows])), 4),
        "detection_threshold_r2": round(
            float(np.max([r["null_r2_p99"] for r in rows])), 4),
        "elapsed_s": round(time.time() - t0),
    }
    out = ROOT / args.out
    out.write_text(json.dumps({"config": vars(args), "summary": summary,
                               "records": rows}, indent=2))
    print("\n" + "=" * 66)
    for k, v in summary.items():
        print(f"  {k:.<40} {v}")
    print("=" * 66)
    print("  A fit must clear detection_threshold_r2 to be evidence of anything.")


if __name__ == "__main__":
    main()
