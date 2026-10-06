"""Per-unit diagnostics for fits with saved directions (B-17; works on any *_dirs run).

    python experiments/diagnose_units.py --rows results/b17_gptneo125m_l6.jsonl \
        --model EleutherAI/gpt-neo-125M --layer 6

For each unit it recomputes the stimulus exactly as e01_gate.py collected it, then reports:
  euclid_align  |cos(fitted, w)|, the quantity the bar is set on
  sigma_align   |corr(s . fitted, s . w)| over tokens: do the two directions produce the
                same projection on the stimulus actually seen?
  r2_exact      R2 of the recorded response against GELU(s . w + b) rebuilt from the captured
                stimulus and the weight column: the pipeline check, 1 to float precision if
                the hooks and the ground truth agree. (collect() defines the response as the
                erf GELU of the hooked pre-activation, so the same GELU is used here.)
  r2_true       R2 of a leave-one-out running-mean link on s . w: the attainable fit for a
                link-agnostic estimator at this token count
  r2_fitted     the same for the fitted direction
  pc1_share     share of stimulus variance on its first principal component
  cos_pc1       |cos(fitted, PC1)|

Reading (docs/preregistration-b17.md): r2_exact below 1 means the stimulus and response
do not share the ground truth, a pipeline fault. r2_fitted near r2_true with low
euclid_align but high sigma_align means the stimulus does not separate the two directions,
a property of the model's residual geometry. r2_fitted well below r2_true means the
optimiser did not get there.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from caliper.activations import _blocks, _mlp_in, collect, load_model, sample_corpus


def r2(y, yhat):
    return 1.0 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def link_r2(proj, y, half=12):
    """R2 of a leave-one-out running-mean link on a 1-D projection: each token is predicted
    by the mean response of its 2*half nearest neighbours in projection order. It follows
    any link shape, including GELU's dip and a sparse unit's tail, without a binning choice."""
    o = np.argsort(proj)
    ys = y[o]
    c = np.concatenate([[0.0], np.cumsum(ys)])
    n = len(ys)
    i = np.arange(n)
    lo, hi = np.clip(i - half, 0, n), np.clip(i + half + 1, 0, n)
    pred = np.empty(n)
    pred[o] = (c[hi] - c[lo] - ys) / (hi - lo - 1)
    return r2(y, pred)


def unit(v):
    return v / np.linalg.norm(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--tokens", type=int, default=8000)
    ap.add_argument("--route", choices=("selected", "direct", "cascade"), default="selected")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.rows, encoding="utf-8") if l.strip()]
    dirs = Path(a.rows.replace(".jsonl", "_dirs"))
    ids = np.array([r["_key"] for r in rows])
    model, tok = load_model(a.model)
    p = collect(model, tok, sample_corpus(n_docs=300, seed=0), layer=a.layer,
                neurons=ids, max_tokens=a.tokens, seed=0)
    block = _blocks(model)[a.layer]
    bias = _mlp_in(block).bias.detach().numpy()[ids]
    with torch.no_grad():
        exact = torch.nn.functional.gelu(torch.as_tensor(p.stimulus @ p.weights + bias)).numpy()
    S = p.stimulus - p.stimulus.mean(0)
    _, sv, vt = np.linalg.svd(S, full_matrices=False)
    pc1, pc1_share = vt[0], float(sv[0] ** 2 / (sv ** 2).sum())

    out = []
    for i, r in enumerate(rows):
        d = np.load(dirs / f"n{r['_key']}.npz")
        route = r["picked"] if a.route == "selected" else a.route
        f = unit(d[route][:, 0])
        w = unit(d["w"])
        y = p.response[:, i]
        out.append({
            "unit": int(r["_key"]), "route": route,
            "euclid_align": round(float(abs(f @ w)), 4),
            "sigma_align": round(float(abs(np.corrcoef(S @ f, S @ w)[0, 1])), 4),
            "r2_exact": round(float(r2(y, exact[:, i])), 6),
            "r2_true": round(float(link_r2(S @ w, y)), 4),
            "r2_fitted": round(float(link_r2(S @ f, y)), 4),
            "cos_pc1": round(float(abs(f @ pc1)), 4),
            "response_sd": round(float(y.std()), 5),
        })
    summary = {k: float(np.median([o[k] for o in out]))
               for k in ("euclid_align", "sigma_align", "r2_exact", "r2_true", "r2_fitted",
                         "cos_pc1")}
    rep = {"rows": a.rows, "model": a.model, "layer": a.layer, "pc1_share": pc1_share,
           "median": summary, "units": out}
    for o in out:
        print(o)
    print("pc1_share", round(pc1_share, 4), "median", {k: round(v, 4) for k, v in summary.items()})
    if a.out:
        json.dump(rep, open(a.out, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
