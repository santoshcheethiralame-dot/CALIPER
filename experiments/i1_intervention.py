"""I-1: what a collapsed direction does when used to intervene (docs/preregistration-i1-intervention.md).

    PYTHONPATH=. python experiments/i1_intervention.py      # writes results/i1_intervention.json (scores once)

For saved fits of MLP units, four unit-norm directions per unit: the weight column w (truth), the
fitted direction as returned, the same direction with the layer's near-constant coordinates zeroed,
and a random direction. Exact quantities, from the weights:
  efficacy     |w . u| / |w|: how much adding u to the MLP input moves the unit, per unit norm
               (for u = w it is 1)
  selectivity  (w . u)^2 / |W' u|^2: the share of the change in all of the layer's pre-activations
               that lands on the target unit
Measured by running the model with u added (at 0.25 x the median input norm) to the output of the
norm feeding the MLP, at every position of 2,000 tokens from unseen documents: the target unit's
pre-activation change (a check that the exact value holds) and the median next-token KL.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))
from caliper.activations import _blocks, _mlp_in, _mlp_in_weight, _mlp_ln, load_model, sample_corpus  # noqa: E402
from repair_table import directions, near_constant  # noqa: E402

NEO, OPT = "EleutherAI/gpt-neo-125M", "facebook/opt-125m"
RUNS = [  # (name, kind, stem, model, layer, format)
    ("GPT-Neo L6, least squares (B-17)", "flagged", "results/b17_gptneo125m_l6", NEO, 6, "e01"),
    ("GPT-2 L3, least squares (B-17c)", "flagged", "results/b17c_gpt2_l3", "gpt2", 3, "e01"),
    ("OPT-125m L5, least squares (F-3a)", "flagged", "results/f3a_opt125m_l5", OPT, 5, "e01"),
    ("GPT-Neo L6, MID (E-2b)", "flagged", "results/e2b_mid_gptneo_l6", NEO, 6, "mid"),
    ("GPT-2 L3, MID (E-2b)", "flagged", "results/e2b_mid_gpt2_l3", "gpt2", 3, "mid"),
    ("GPT-2 L6, least squares (E-1, control)", "control", "results/e1_gpt2_l6", "gpt2", 6, "mid"),
    ("GPT-2 L6, MID (E-2, control)", "control", "results/e2_mid_gpt2_l6", "gpt2", 6, "mid"),
]
N_UNITS, N_TOK, SEQ, DOSE = 20, 2000, 128, 0.25


def windows(tok, n):
    seen = set(sample_corpus(n_docs=300, seed=0))
    out, total = [], 0
    for t in sample_corpus(n_docs=300, seed=1):
        if t in seen:
            continue
        x = tok(t, return_tensors="pt", truncation=True, max_length=SEQ)["input_ids"]
        if x.shape[1] < 8:
            continue
        out.append(x)
        total += x.shape[1] - 1
        if total >= n:
            break
    return out


def run_model(model, block, X, add=None, unit=None):
    """Next-token log-probs and the unit's pre-activation, with `add` added to the MLP input."""
    ln, got = _mlp_ln(block), {}
    hs = []
    if add is not None:
        hs.append(ln.register_forward_hook(lambda _m, _i, o: o + add.to(o.dtype)))
    hs.append(_mlp_in(block).register_forward_hook(lambda _m, _i, o: got.__setitem__("z", o.detach())))
    lps, zs = [], []
    try:
        with torch.no_grad():
            for x in X:
                lps.append(model(input_ids=x).logits[0, :-1].float().log_softmax(-1))
                z = got["z"].reshape(x.shape[1], -1)
                zs.append(z[:-1, unit] if unit is not None else z[:-1, 0])
    finally:
        for h in hs:
            h.remove()
    return torch.cat(lps), torch.cat(zs)


def main():
    rep = {"dose": DOSE, "runs": {}}
    cache = {}
    for name, kind, stem, model_name, layer, fmt in RUNS:
        if model_name not in cache:
            cache = {model_name: load_model(model_name, dtype=torch.float32)}
        model, tok = cache[model_name]
        block = _blocks(model)[layer]
        ks, V = directions(stem, "mid" if fmt == "mid" else "e01")
        ks, V = ks[:N_UNITS], V[:N_UNITS]
        const = near_constant(stem, model_name, layer, ks, fmt)
        W = _mlp_in_weight(block).astype(np.float64)               # d x d_mlp
        X = windows(tok, N_TOK)
        ln = _mlp_ln(block)
        norms = []
        h = ln.register_forward_hook(lambda _m, _i, o: norms.append(o.detach().reshape(-1, o.shape[-1]).norm(dim=-1)))
        with torch.no_grad():
            model(input_ids=X[0])
        h.remove()
        alpha = DOSE * float(torch.cat(norms).median())
        rng = np.random.default_rng(0)
        rows = []
        for k, v in zip(ks, V):
            w = W[:, k]
            dirs = {"truth": w, "as fitted": v, "zeroed": np.where(np.isin(np.arange(len(v)), const), 0.0, v),
                    "random": rng.standard_normal(len(v))}
            r = {"unit": int(k)}
            base_lp, base_z = run_model(model, block, X, unit=int(k))
            for dname, u in dirs.items():
                u = u / np.linalg.norm(u)
                if dname != "truth" and float(u @ w) < 0:
                    u = -u                                          # a direction's sign is arbitrary
                eff = abs(float(u @ w)) / np.linalg.norm(w)
                sel = float(u @ w) ** 2 / float(np.sum((W.T @ u) ** 2))
                add = torch.as_tensor(alpha * u, dtype=torch.float32)
                lp, z = run_model(model, block, X, add=add, unit=int(k))
                kl = (base_lp.exp() * (base_lp - lp)).sum(-1)
                dz = float((z - base_z).median())
                r[dname] = {"efficacy": eff, "selectivity": sel, "median_kl": float(kl.median()),
                            "dz_measured": dz, "dz_exact": alpha * float(u @ w)}
            rows.append(r)
            print(name, k, {d: (round(r[d]["efficacy"], 3), round(r[d]["selectivity"], 4), round(r[d]["median_kl"], 4))
                            for d in dirs}, flush=True)
        summ = {d: {m: float(np.median([r[d][m] for r in rows])) for m in ("efficacy", "selectivity", "median_kl")}
                for d in ("truth", "as fitted", "zeroed", "random")}
        err = [abs(r[d]["dz_measured"] - r[d]["dz_exact"]) / max(abs(r[d]["dz_exact"]), 1e-9)
               for r in rows for d in ("truth", "as fitted", "zeroed")]
        rep["runs"][name] = {"kind": kind, "units": len(rows), "alpha": alpha, "near_constant": const.tolist(),
                             "median": summ, "median_rel_error_dz": float(np.median(err)), "per_unit": rows}
        print(name, json.dumps(summ), flush=True)
    R = rep["runs"]
    F = [r for r in R.values() if r["kind"] == "flagged"]
    C = [r for r in R.values() if r["kind"] == "control"]
    rep["P1 collapsed directions are unselective"] = {"holds": all(
        r["median"]["as fitted"]["selectivity"] < 0.1 * r["median"]["truth"]["selectivity"] for r in F)}
    rep["P2 zeroing restores selectivity where it restores alignment"] = {"holds": all(
        R[n]["median"]["zeroed"]["selectivity"] >= 0.5 * R[n]["median"]["truth"]["selectivity"]
        for n in ("GPT-Neo L6, least squares (B-17)", "GPT-Neo L6, MID (E-2b)", "GPT-2 L3, MID (E-2b)"))}
    rep["P3 controls are selective"] = {"holds": all(
        r["median"]["as fitted"]["selectivity"] >= 0.5 * r["median"]["truth"]["selectivity"] for r in C)}
    rep["P4 the measured change matches the exact one"] = {"holds": all(r["median_rel_error_dz"] <= 0.01 for r in R.values())}
    print(json.dumps({k: v for k, v in rep.items() if k.startswith("P")}, indent=1))
    json.dump(rep, open(ROOT / "results/i1_intervention.json", "w"), indent=1)


if __name__ == "__main__":
    main()
