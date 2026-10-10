"""Round-2 roadmap C1: a known-link baseline (inverse-GELU regression).

    PYTHONPATH=. python experiments/c1_known_link.py      # writes results/c1_known_link.json

The response is y = GELU(w.s + b) with GELU known. Where y > 0 the link is one-to-one, so z = w.s + b
is recovered exactly by inverting GELU, and ordinary least squares of z on [s, 1] estimates w in closed
form with no optimisation. This is the estimator a practitioner who knows the nonlinearity would use.

Each arm re-collects the stimulus exactly as its run did (same model, layer, corpus seed, split seed,
token budget and shuffle) and uses only the fit split (the run's first 20% of rows are held out).
Scored as filed (|cos| with w) and with the layer-norm null direction 1/gamma removed from both.
Part A repeats the classical-baselines units (E0.5: GPT-2 layer 6, 30 units, 20,000 tokens).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from caliper.activations import _blocks, _mlp_ln, collect, load_model, sample_corpus  # noqa: E402

PASS, WB = 0.95, 0.99
ARMS = [  # name, stem, model, layer, corpus seed, sequence split
    ("GPT-2 L6 re-fit, fit seed 1 (B-15a)", "results/b15a_fitseed1", "gpt2", 6, 0, False),
    ("GPT-2 L6 re-fit, corpus seed 1 (B-15b)", "results/b15b_corpusseed1", "gpt2", 6, 1, False),
    ("GPT-2 L6 re-fit, sequence split (B-15c)", "results/b15c_seqsplit", "gpt2", 6, 0, True),
    ("GPT-Neo-125m L10 (B-8b)", "results/b8b_gptneo125m_indep", "EleutherAI/gpt-neo-125M", 10, 0, False),
]


def inverse_gelu(y, iters=60):
    """z > 0 with GELU(z) = y, for y > 0, by bisection (GELU is increasing there)."""
    y = torch.as_tensor(y, dtype=torch.float64)
    lo, hi = torch.zeros_like(y), torch.full_like(y, 50.0)
    for _ in range(iters):
        mid = (lo + hi) / 2
        up = torch.nn.functional.gelu(mid) < y
        lo, hi = torch.where(up, mid, lo), torch.where(up, hi, mid)
    return ((lo + hi) / 2).numpy()


def known_link(S, y):
    on = y > 0
    z = inverse_gelu(y[on])
    A = np.hstack([S[on].astype(np.float64), np.ones((on.sum(), 1))])
    coef = np.linalg.lstsq(A, z, rcond=None)[0]
    return coef[:-1], int(on.sum())


def score(v, w, u):
    if not np.any(v):
        return 0.0, 0.0  # no token on the invertible branch: nothing estimated
    P = lambda x: x - u * (u @ x)
    raw = abs(v @ w) / (np.linalg.norm(v) * np.linalg.norm(w))
    pv, pw = P(v), P(w)
    return float(raw), float(abs(pv @ pw) / (np.linalg.norm(pv) * np.linalg.norm(pw)))


def enough_split(rows, est, cw, ident, unknowns):
    """Units with more invertible tokens than unknowns (d weights and a bias) against the rest."""
    ok = np.array([x["tokens_on_branch"] for x in rows]) > unknowns
    return {"unknowns": unknowns, "units_with_enough_tokens": int(ok.sum()),
            "known_link_pass_with_enough": int((ok & (ident >= PASS)).sum()),
            "known_link_min_with_enough": float(ident[ok].min()) if ok.any() else None,
            "estimator_pass_with_enough": int((ok & (est >= PASS)).sum()),
            "converged_wrong_with_enough": int((ok & cw).sum()),
            "converged_wrong_with_enough_recovered": int((ok & cw & (ident >= PASS)).sum()),
            "known_link_pass_too_few": int((~ok & (ident >= PASS)).sum())}


def null_direction(model, layer):
    g = _mlp_ln(_blocks(model)[layer]).weight.detach().double().numpy()
    return (1 / g) / np.linalg.norm(1 / g)


def arm(name, stem, model_name, layer, corpus_seed, seq_split):
    R = [json.loads(l) for l in open(ROOT / f"{stem}.jsonl", encoding="utf-8") if l.strip()]
    units = np.array([r["_key"] for r in R])
    model, tok = load_model(model_name)
    p = collect(model, tok, sample_corpus(n_docs=300, seed=corpus_seed), layer=layer, neurons=units,
                max_tokens=8000, seed=0, shuffle="sequence" if seq_split else "token")
    n_test = int(len(p.stimulus) * 0.2)
    S, Y = p.stimulus[n_test:], p.response[n_test:]
    u = null_direction(model, layer)
    rows = []
    for i, r in enumerate(R):
        w = p.weights[:, i].astype(np.float64)
        v, n_on = known_link(S, Y[:, i])
        raw, ident = score(v, w, u)
        rows.append({"unit": int(units[i]), "tokens_on_branch": n_on, "known_link_raw": raw,
                     "known_link_identifiable": ident, "estimator_raw": r["align_selected"],
                     "estimator_r2": r["r2_k1"]})
    raw = np.array([x["known_link_raw"] for x in rows])
    ident = np.array([x["known_link_identifiable"] for x in rows])
    est = np.array([x["estimator_raw"] for x in rows])
    cw = (est < PASS) & (np.array([x["estimator_r2"] for x in rows]) > WB)
    out = {"units": len(rows), "fit_rows": len(S),
           "median_tokens_on_branch": float(np.median([x["tokens_on_branch"] for x in rows])),
           "pass_known_link_raw": int((raw >= PASS).sum()),
           "pass_known_link_identifiable": int((ident >= PASS).sum()),
           "pass_estimator_raw": int((est >= PASS).sum()),
           "median_known_link_raw": float(np.median(raw)),
           "median_known_link_identifiable": float(np.median(ident)),
           "estimator_converged_wrong": int(cw.sum()),
           "converged_wrong_recovered_raw": int((cw & (raw >= PASS)).sum()),
           "converged_wrong_recovered_identifiable": int((cw & (ident >= PASS)).sum()),
           **enough_split(rows, est, cw, ident, S.shape[1] + 1),
           "rows": rows}
    print(name, {k: v for k, v in out.items() if k != "rows"}, flush=True)
    return out


def part_a():
    model, tok = load_model("gpt2")
    rng = np.random.default_rng(0)
    neurons = rng.choice(3072, size=30, replace=False)
    p = collect(model, tok, sample_corpus(n_docs=200, seed=0), layer=6, neurons=neurons,
                max_tokens=20000, seed=0)
    u = null_direction(model, 6)
    rows = []
    for i, n in enumerate(p.neurons):
        y = p.response[:, i]
        if y.std() < 1e-4:
            continue
        v, n_on = known_link(p.stimulus, y)
        raw, ident = score(v, p.weights[:, i].astype(np.float64), u)
        rows.append({"neuron": int(n), "tokens_on_branch": n_on, "raw": raw, "identifiable": ident})
    raw = np.array([r["raw"] for r in rows])
    ident = np.array([r["identifiable"] for r in rows])
    out = {"units": len(rows), "median_raw": float(np.median(raw)),
           "median_identifiable": float(np.median(ident)), "pass_raw": int((raw >= PASS).sum()),
           "pass_identifiable": int((ident >= PASS).sum()), "min_raw": float(raw.min()),
           "units_with_enough_tokens": int(sum(r["tokens_on_branch"] > p.stimulus.shape[1] + 1
                                               for r in rows)), "rows": rows}
    print("E0.5 units", {k: v for k, v in out.items() if k != "rows"}, flush=True)
    return out


def main():
    rep = {"E0.5 units (GPT-2 L6, 20k tokens)": part_a()}
    for spec in ARMS:
        rep[spec[0]] = arm(*spec)
    json.dump(rep, open(ROOT / "results/c1_known_link.json", "w"), indent=1)


if __name__ == "__main__":
    main()
