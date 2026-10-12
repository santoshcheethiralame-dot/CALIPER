"""The null-direction gain law (exploratory, 12 Oct 2026) and its forecast for a new layer.

    PYTHONPATH=. python experiments/gain_law.py --layer 0 --units results/g0_units.txt

For a unit with input weight column w at a layer whose layer norm has gain gamma (d coordinates):
  G_j = ||1/gamma|| * ||gamma (.) w_hat|| / sqrt(d)                          (weights only)
A fitted direction v carries a null-to-rest norm ratio r = sqrt(ns / (1 - ns)), ns = (u . v)^2,
u = (1/gamma)/||1/gamma||. The law: r = c * G_j / sqrt(d), with c an estimator constant whose
distribution is calibrated on the GPT-2 layer-6 arms with saved directions (B-14r, B-15a/b/c, D-1),
where G is about 1. A fit that is right on the identifiable label (identifiable cosine 0.99, the
calibration arms' median) then fails the filed Euclidean bar when 0.99 / sqrt(1 + (c G_j/sqrt d)^2)
< 0.95. The forecast for a set of units is the share of identifiable passes that fail the filed bar.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
HUB = Path.home() / ".cache/huggingface/hub"
CAL = ["results/b14r_primary_gpt2_dirs", "results/b15a_fitseed1", "results/b15b_corpusseed1",
       "results/b15c_seqsplit", "results/d1_gpt2_l6_fresh"]
BAR, IDENT = 0.95, 0.99


def weights(layer):
    path = next((HUB / "models--gpt2" / "snapshots").glob("*/model.safetensors"))
    with safe_open(str(path), "np") as f:
        g = f.get_tensor(f"h.{layer}.ln_2.weight").astype(np.float64)
        W = f.get_tensor(f"h.{layer}.mlp.c_fc.weight").astype(np.float64)    # (d, 4d): column j = w_j
    return g, W / np.linalg.norm(W, axis=0, keepdims=True)


def gains(layer, units):
    g, W = weights(layer)
    return np.linalg.norm(1 / g) * np.linalg.norm(g[:, None] * W[:, units], axis=0) / np.sqrt(len(g)), len(g)


def calibration():
    """c * sqrt(d) for every selected fit in the GPT-2 layer-6 calibration arms."""
    g, W = weights(6)
    u = (1 / g) / np.linalg.norm(1 / g)
    out = []
    for stem in CAL:
        for line in open(ROOT / f"{stem}.jsonl", encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            v = np.load(ROOT / f"{stem}_dirs/n{r['_key']}.npz")[r["picked"]][:, 0].astype(np.float64)
            v /= np.linalg.norm(v)
            ns = float((u @ v) ** 2)
            G = np.linalg.norm(1 / g) * np.linalg.norm(g * W[:, r["_key"]]) / np.sqrt(len(g))
            out.append(np.sqrt(ns / (1 - ns)) / G * np.sqrt(len(g)))
    return np.array(out)


def forecast(layer, units, n_pass=None, nsim=4000, seed=0):
    """Point forecast of the share of identifiable passes failing the filed bar, and the central 95%
    band of the share observed in n_pass passes (c resampled from calibration, units resampled)."""
    G, d = gains(layer, units)
    c = calibration()
    rng = np.random.default_rng(seed)
    fail = lambda cs, Gs: IDENT / np.sqrt(1 + (cs * Gs / np.sqrt(d)) ** 2) < BAR
    point = float(fail(rng.choice(c, 20000), rng.choice(G, 20000)).mean())
    out = {"layer": layer, "units": len(units), "G_median": float(np.median(G)), "c_sqrt_d_median": float(np.median(c)),
           "calibration_fits": len(c), "forecast_share": point}
    if n_pass:
        sims = [fail(rng.choice(c, n_pass), rng.choice(G, n_pass)).mean() for _ in range(nsim)]
        out["band95_at_n"] = {"n": n_pass, "band": [float(np.percentile(sims, 2.5)), float(np.percentile(sims, 97.5))]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--units", required=True, help="file with comma-separated unit ids")
    ap.add_argument("--n-pass", type=int, default=None)
    a = ap.parse_args()
    units = [int(x) for x in (ROOT / a.units).read_text().split(",")]
    print(json.dumps(forecast(a.layer, units, a.n_pass), indent=1))


if __name__ == "__main__":
    main()
