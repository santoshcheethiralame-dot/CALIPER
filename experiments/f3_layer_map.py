"""F-3a layer map: near-constant MLP-input coordinates in OPT-125m, before any F-3a fit.

    python experiments/f3_layer_map.py      # writes results/f3_layer_map.json

The same map as B-17's (notebook Q3, 8 Oct): per layer, the coordinates of the stimulus the MLP
reads whose SD over e01_gate.py's 8,000 tokens is below 1% of the median SD, and the smallest
layer-norm gain.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from caliper.activations import _blocks, _mlp_ln, collect, load_model, sample_corpus  # noqa: E402

MODEL = "facebook/opt-125m"


def main():
    model, tok = load_model(MODEL, dtype=torch.float32)
    texts = sample_corpus(n_docs=300, seed=0)
    out = {"model": MODEL, "threshold": "SD < 1% of the layer's median SD", "layers": {}}
    for layer in range(len(_blocks(model))):
        p = collect(model, tok, texts, layer=layer, neurons=np.arange(4), max_tokens=8000, seed=0)
        sd = np.asarray(p.stimulus, float).std(0)
        g = np.abs(_mlp_ln(_blocks(model)[layer]).weight.detach().double().numpy())
        const = np.flatnonzero(sd < 1e-2 * np.median(sd))
        out["layers"][layer] = {"n_const": int(len(const)), "const_coords": const.tolist(),
                                "min_sd_over_median": float(sd.min() / np.median(sd)),
                                "argmin_sd": int(sd.argmin()), "min_gain": float(g.min()),
                                "argmin_gain": int(g.argmin()), "tokens": int(len(sd) and p.stimulus.shape[0])}
        print(layer, out["layers"][layer], flush=True)
    json.dump(out, open(ROOT / "results/f3_layer_map.json", "w"), indent=1)


if __name__ == "__main__":
    main()
