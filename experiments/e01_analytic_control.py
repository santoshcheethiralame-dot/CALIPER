"""E0.1 - analytic positive control.

An MLP neuron's pre-activation is exactly ``w . s`` where ``s`` is its own layer's
post-LayerNorm residual stream. With respect to that stimulus space its dependence is
exactly one-dimensional and the true direction is exactly ``w``. That is free, exact
ground truth in a real network, and the estimator must reproduce it.

Pass criteria (specification section 5, E0.1):
  * |v1 . w_hat| > 0.95
  * no significant held-out gain from k=2 over k=1
for at least 95% of tested neurons.

The run also measures how much of the true direction survives PCA truncation. That
bound is analytic and costs nothing, and the first pilot showed it is severe: MLP read
directions are close to isotropic in the residual stream's principal-component basis,
so truncation discards the signal it was meant to preserve. The primary configuration
therefore whitens without truncating.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from caliper.activations import collect, load_model, sample_corpus
from caliper.estimator import fit, subspace_alignment, whitening_frame

ROOT = Path(__file__).resolve().parents[1]


def pca_ceiling_curve(stimulus, weights, dims):
    """Fraction of each true direction retained by a top-D PCA subspace."""
    x = stimulus - stimulus.mean(0)
    _, s, vt = np.linalg.svd(x[: min(len(x), 20_000)], full_matrices=False)
    var = (s**2) / (s**2).sum()
    w = weights / np.linalg.norm(weights, axis=0, keepdims=True)
    rng = np.random.default_rng(0)
    rand = rng.standard_normal(w.shape)
    rand /= np.linalg.norm(rand, axis=0, keepdims=True)

    out = []
    for d in dims:
        p = vt[:d].T
        out.append({
            "d": int(d),
            "variance_kept": round(float(var[:d].sum()), 4),
            "median_weight_retained": round(
                float(np.median(np.linalg.norm(p @ (p.T @ w), axis=0))), 4),
            "median_random_retained": round(
                float(np.median(np.linalg.norm(p @ (p.T @ rand), axis=0))), 4),
            "isotropic_expectation": round(float(np.sqrt(d / w.shape[0])), 4),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--neurons", type=int, default=20)
    ap.add_argument("--tokens", type=int, default=20_000)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/e01.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    print(f"loading {args.model} ...", flush=True)
    model, tokenizer = load_model(args.model)
    d_mlp = model.transformer.h[args.layer].mlp.c_fc.weight.shape[1]
    neurons = rng.choice(d_mlp, size=args.neurons, replace=False)

    print(f"collecting {args.tokens} positions from layer {args.layer} ...", flush=True)
    probe = collect(model, tokenizer, sample_corpus(n_docs=200, seed=args.seed),
                    layer=args.layer, neurons=neurons,
                    max_tokens=args.tokens, seed=args.seed)
    print(f"  stimulus {probe.stimulus.shape}  ({time.time() - t0:.0f}s)", flush=True)

    alive = probe.response.std(0) > 1e-4
    print(f"  alive: {alive.sum()}/{len(alive)}", flush=True)

    ceiling = pca_ceiling_curve(probe.stimulus, probe.weights,
                                [16, 32, 64, 128, 256, 512, 768])
    print("  PCA truncation ceiling (why we whiten instead of truncating):")
    for row in ceiling:
        print(f"    D={row['d']:<4} var={row['variance_kept']:.3f} "
              f"weight kept={row['median_weight_retained']:.3f} "
              f"random={row['median_random_retained']:.3f}", flush=True)

    frame, mean = whitening_frame(probe.stimulus)
    centred = probe.stimulus - mean

    records = []
    for i, neuron in enumerate(probe.neurons):
        if not alive[i]:
            continue
        w_unit = probe.weights[:, i] / np.linalg.norm(probe.weights[:, i])
        y = probe.response[:, i]

        raw1 = fit(probe.stimulus, y, k=1, n_restarts=args.restarts,
                   steps=args.steps, seed=args.seed)
        raw2 = fit(probe.stimulus, y, k=2, n_restarts=args.restarts,
                   steps=args.steps, seed=args.seed)
        rec = {
            "neuron": int(neuron),
            "raw_alignment": abs(subspace_alignment(raw1.subspace, w_unit[:, None])),
            "raw_r2_k1": raw1.test_r2,
            "raw_r2_k2": raw2.test_r2,
            "k2_gain": raw2.test_r2 - raw1.test_r2,
            "raw_stability": raw1.stability,
        }
        records.append(rec)
        print(f"  n{neuron:<5d} raw={rec['raw_alignment']:.3f} "
              f"R2={raw1.test_r2:.3f} k2gain={rec['k2_gain']:+.3f} "
              f"stab={raw1.stability:.3f}", flush=True)

    summary = _summarise(records, time.time() - t0)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"config": vars(args), "summary": summary,
         "pca_ceiling_curve": ceiling, "records": records}, indent=2))

    print("\n" + "=" * 66)
    for k, v in summary.items():
        print(f"  {k:.<42} {v}")
    print("=" * 66)
    print(f"written to {out}")


def _summarise(records, elapsed):
    if not records:
        return {"verdict": "NO LIVE NEURONS"}
    align = np.array([r["raw_alignment"] for r in records])
    gain = np.array([r["k2_gain"] for r in records])
    passed = float(np.mean((align > 0.95) & (gain < 0.01)))
    return {
        "n_neurons": len(records),
        "median_raw_alignment": round(float(np.median(align)), 4),
        "min_raw_alignment": round(float(align.min()), 4),
        "median_k2_gain": round(float(np.median(gain)), 4),
        "median_raw_r2_k1": round(
            float(np.median([r["raw_r2_k1"] for r in records])), 4),
        "median_restart_stability": round(
            float(np.median([r["raw_stability"] for r in records])), 4),
        "fraction_passing": round(passed, 3),
        "verdict": "PASS" if passed >= 0.95 else "FAIL",
        "elapsed_s": round(elapsed),
    }


if __name__ == "__main__":
    main()
