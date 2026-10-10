"""Flagship substrate pilot (discarded; sets token budgets only): pass rates per cell, as filed and
with the layer-norm null direction removed.

    python experiments/analyse_flagship_pilot.py [results/flagship_pilot1]

The unembedding reads the final norm's output, so a LayerNorm there has the same null direction
u = 1/gamma as the MLP's (docs/LAB_NOTEBOOK.md, round-2 review). RMSNorm (Qwen) subtracts no mean and
has none. OPT's gain is not loaded here: its fits are under-fitted, so the label does not decide them.
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NORM = {"f6_gpt2": ("gpt2", "transformer.ln_f"),
        "f6_pythia410m": ("EleutherAI/pythia-410m", "gpt_neox.final_layer_norm")}


def null_direction(model, path):
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32)
    for a in path.split("."):
        m = getattr(m, a)
    g = m.weight.detach().numpy().astype(np.float64)
    u = 1.0 / g
    return u / np.linalg.norm(u), float(np.abs(g).min()), float(np.median(np.abs(g)))


def main(d):
    rep = {}
    for f in sorted(glob.glob(os.path.join(d, "pilot*_*.jsonl"))):
        stem = os.path.basename(f)[:-6]
        R = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        al = np.array([r["align_selected"] for r in R])
        r2 = np.array([r["r2_k1"] for r in R])
        out = {"units": len(R), "complete": os.path.exists(os.path.join(d, stem + "_summary.json")),
               "pass": int((al >= 0.95).sum()), "converged_wrong": int(((al < 0.95) & (r2 > 0.99)).sum()),
               "median_alignment": float(np.median(al)), "median_r2": float(np.median(r2)),
               "median_restart_agreement": float(np.median([r["stability"] for r in R]))}
        key = next((k for k in NORM if stem.split("_", 1)[1].startswith(k + "_")), None)
        if key:
            u, gmin, gmed = null_direction(*NORM[key])
            P = lambda x: x - u * (u @ x)
            ident, share, agree = [], [], []
            for r in R:
                z = np.load(os.path.join(d, stem + "_dirs", f"n{r['_key']}.npz"))
                w = z["w"].astype(np.float64)
                v = z[r["picked"]][:, 0].astype(np.float64)
                v /= np.linalg.norm(v)
                pv, pw = P(v), P(w)
                ident.append(abs(pv @ pw) / (np.linalg.norm(pv) * np.linalg.norm(pw)))
                share.append(float((u @ v) ** 2))
                q = [P(x.astype(np.float64)) for x in z["direct_restarts"][:, :, 0]]
                agree.append(abs(q[0] @ q[1]) / (np.linalg.norm(q[0]) * np.linalg.norm(q[1])))
            ident = np.array(ident)
            out["null_removed"] = {"min_abs_gamma": gmin, "median_abs_gamma": gmed,
                                   "pass": int((ident >= 0.95).sum()),
                                   "median_alignment": float(np.median(ident)),
                                   "median_fit_share_on_null": float(np.median(share)),
                                   "median_restart_agreement": float(np.median(agree))}
        rep[stem] = out
        print(stem, json.dumps(out), flush=True)
    json.dump(rep, open(os.path.join(d, "pilot_analysis.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "results/flagship_pilot1"))
