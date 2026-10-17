"""Check a layer for the standardisation failure before fitting anything to it.

    python -m caliper.lint --model gpt2 [--layers 3,6] [--tokens 4000] [--out report.json]

Per layer, from the stimulus an MLP unit reads (the output of the layer norm feeding the MLP):
  near_constant   coordinates whose SD over the tokens is below --frac (1%) of the median SD. Fitting
                  in per-coordinate standardised coordinates magnifies any weight on them by
                  median / SD, so a fitted direction can collapse onto one, with every restart agreeing
                  (Paper 1; B-17, B-17c, F-3a, E-2b: three model families, two estimators).
  min_gain        the smallest |gain| of that layer norm; a near-zero gain is what makes a coordinate
                  nearly constant, and it can be read from the weights alone.
  top_share       the largest coordinate's share of the stimulus' mean squared norm (a massive
                  coordinate; it inflates raw cosines between directions read at that layer).
A layer is flagged when it has a near-constant coordinate. Remedies: drop the coordinate before
fitting, floor the scale, or score directions on the identifiable label (1/gamma removed).
"""
import argparse
import json

import numpy as np


def lint_layer(model, tok, texts, layer, tokens=4000, frac=0.01):
    from caliper.activations import _blocks, _mlp_ln, collect
    p = collect(model, tok, texts, layer=layer, neurons=np.arange(1), max_tokens=tokens, seed=0)
    S = np.asarray(p.stimulus, np.float64)
    sd = S.std(0)
    const = np.flatnonzero(sd < frac * np.median(sd))
    ln = _mlp_ln(_blocks(model)[layer])
    g = np.abs(ln.weight.detach().double().numpy()) if ln is not None else None
    ms = (S ** 2).mean(0)
    return {"layer": layer, "tokens": int(len(S)), "flagged": bool(len(const)),
            "near_constant": const.tolist(),
            "min_sd_over_median": float(sd.min() / np.median(sd)), "argmin_sd": int(sd.argmin()),
            "min_gain": float(g.min()) if g is not None else None,
            "argmin_gain": int(g.argmin()) if g is not None else None,
            "top_share": float(ms.max() / ms.sum()), "argmax_share": int(ms.argmax())}


def main():
    import torch
    from caliper.activations import _blocks, load_model, sample_corpus
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--tokens", type=int, default=4000)
    ap.add_argument("--frac", type=float, default=0.01)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    model, tok = load_model(a.model, dtype=torch.float32)
    n = len(_blocks(model))
    layers = range(n) if a.layers == "all" else [int(x) for x in a.layers.split(",")]
    texts = sample_corpus(n_docs=300, seed=0)
    rep = {"model": a.model, "frac": a.frac, "layers": []}
    for layer in layers:
        r = lint_layer(model, tok, texts, layer, a.tokens, a.frac)
        rep["layers"].append(r)
        flag = f"FLAG near-constant {r['near_constant']}" if r["flagged"] else "ok"
        gain = f"{r['min_gain']:.2e}" if r["min_gain"] is not None else "n/a"
        print(f"layer {layer:3d}  {flag:32s} min SD/median {r['min_sd_over_median']:.1e}  "
              f"min gain {gain}  top share {r['top_share']:.3f}", flush=True)
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
