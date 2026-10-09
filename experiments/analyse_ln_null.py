"""Round-2 roadmap A1: the layer-norm null direction (docs/LAB_NOTEBOOK.md, internal review round 2).

    python experiments/analyse_ln_null.py          # every arm with saved directions

The MLP reads s = gamma * z + beta with z zero-mean across coordinates, so s . (1/gamma) is the same
on every token and a direction's component along u = 1/gamma is invisible to the data. Each fit is
scored two ways:
  as filed      |cos(v, w)|, the pre-registered Euclidean label
  identifiable  |cos(P v, P w)|, P = I - u u', the target the data can determine
Per arm and label: failures, the class split (converged-wrong = fail with held-out R2 > 0.99), the
pre-registered comparison AUC(restart agreement) - AUC(held-out R2) with DeLong, a stratified bootstrap
CI and the paired permutation test; the fit's share on u as a ground-truth-free check; and the
converged-wrong comparison restricted to units with held-out R2 > 0.99, where the class definition
does not handicap held-out R2.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))
from analyse_review_round1 import auc, boot_diff, paired_perm  # noqa: E402
from b1_signal_calibration import delong  # noqa: E402

PASS, WB = 0.95, 0.99
ARMS = [  # name, stem, model, layer, pooled in the paper
    ("Pythia-160m L6 (B-2c)", "results/b2c_pythia160m_indep", "EleutherAI/pythia-160m", 6, True),
    ("GPT-Neo-125m L10 (B-8b)", "results/b8b_gptneo125m_indep", "EleutherAI/gpt-neo-125M", 10, True),
    ("GPT-2 L6 re-fit, fit seed 1 (B-15a)", "results/b15a_fitseed1", "gpt2", 6, False),
    ("GPT-2 L6 re-fit, corpus seed 1 (B-15b)", "results/b15b_corpusseed1", "gpt2", 6, False),
    ("GPT-2 L6 re-fit, sequence split (B-15c)", "results/b15c_seqsplit", "gpt2", 6, False),
    ("GPT-Neo-125m L10, new units (X-1a)", "results/x1a_gptneo_l10", "EleutherAI/gpt-neo-125M", 10, False),
    ("GPT-Neo-125m L10, new units, disjoint docs (X-1b)", "results/x1b_gptneo_l10", "EleutherAI/gpt-neo-125M", 10, False),
    ("Pythia-160m L6, float32 (B-2d)", "results/b2d_pythia160m_fp32", "EleutherAI/pythia-160m", 6, False),
    ("GPT-Neo-125m L6 (B-17)", "results/b17_gptneo125m_l6", "EleutherAI/gpt-neo-125M", 6, False),
    ("GPT-Neo-125m L6, coordinate dropped (B-17b)", "results/b17b_gptneo125m_l6_drop", "EleutherAI/gpt-neo-125M", 6, False),
    ("GPT-2 L3 (B-17c)", "results/b17c_gpt2_l3", "gpt2", 3, False),
    ("GPT-2 L3, coordinate dropped (B-17d)", "results/b17d_gpt2_l3_drop", "gpt2", 3, False),
    ("GPT-2 L6, noise SNR 1-19 (N-1a)", "results/n1a_snr_mixed", "gpt2", 6, False),
    ("GPT-2 L6, noise SNR 19 (N-1b)", "results/n1b_snr19", "gpt2", 6, False),
    ("GPT-2 L6, noise SNR 4 (N-1c)", "results/n1c_snr4", "gpt2", 6, False),
]
_GAMMA = {}


def null_direction(model, layer):
    key = (model, layer)
    if key not in _GAMMA:
        import torch
        from transformers import AutoModelForCausalLM
        from caliper.activations import _blocks, _mlp_ln
        m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32)
        g = _mlp_ln(_blocks(m)[layer]).weight.detach().numpy().astype(np.float64)
        u = 1.0 / g
        _GAMMA[key] = (u / np.linalg.norm(u), float(np.abs(g).min()))
    return _GAMMA[key]


def compare(st, r2, fail):
    if fail.sum() < 2 or (~fail).sum() < 2:
        return None
    a, b, d, se, z, p = delong(st, r2, fail)
    return {"failures": int(fail.sum()), "auc_restart": a, "auc_r2": b, "diff": d,
            "delong_p": p, "boot_ci95": boot_diff(st, r2, fail),
            "paired_perm_p": paired_perm(st, r2, fail, n_perm=2000)}


def arm(name, stem, model, layer):
    R = [json.loads(l) for l in open(ROOT / f"{stem}.jsonl", encoding="utf-8") if l.strip()]
    u, gmin = null_direction(model, layer)
    P = lambda x: x - u * (u @ x)
    raw, ident, share, r2, st, filed = [], [], [], [], [], []
    for r in R:
        z = np.load(ROOT / f"{stem}_dirs/n{r['_key']}.npz")
        w = z["w"].astype(np.float64)
        w /= np.linalg.norm(w)
        v = z[r["picked"]][:, 0].astype(np.float64)
        v /= np.linalg.norm(v)
        pv, pw = P(v), P(w)
        raw.append(abs(v @ w))
        ident.append(abs(pv @ pw) / (np.linalg.norm(pv) * np.linalg.norm(pw)))
        share.append(float((u @ v) ** 2))
        r2.append(r["r2_k1"])
        st.append(r["stability"])
        filed.append(r["align_selected"])
    raw, ident, share, r2, st, filed = map(np.array, (raw, ident, share, r2, st, filed))
    out = {"units": len(R), "min_abs_gamma": gmin,
           "max_abs_diff_raw_vs_filed_label": float(np.max(np.abs(raw - filed)))}
    for lab, al in (("as filed", raw), ("identifiable", ident)):
        fail = al < PASS
        cw, uf = fail & (r2 > WB), fail & (r2 <= WB)
        res = {"failures": int(fail.sum()), "converged_wrong": int(cw.sum()),
               "under_fitted": int(uf.sum()), "median_alignment": float(np.median(al)),
               "primary_comparison": compare(-st, -r2, fail)}
        keep = cw | ~fail
        if cw.sum() >= 2:
            res["converged_wrong_vs_pass"] = {
                "auc_restart": auc(-st[keep], cw[keep]), "auc_r2": auc(-r2[keep], cw[keep]),
                "auc_null_share": auc(share[keep], cw[keep])}
            hi = keep & (r2 > WB)
            if cw[hi].sum() >= 2 and (~cw[hi]).sum() >= 2:
                res["converged_wrong_vs_pass_r2_above_0.99"] = {
                    "passes_kept": int((~cw[hi]).sum()),
                    "auc_restart": auc(-st[hi], cw[hi]), "auc_r2": auc(-r2[hi], cw[hi]),
                    "auc_null_share": auc(share[hi], cw[hi])}
        res["auc_null_share_all_failures"] = auc(share, fail)
        out[lab] = res
    out["median_fit_share_on_null"] = float(np.median(share))
    out["verdicts_changed"] = int(((raw < PASS) != (ident < PASS)).sum())
    return out


def main():
    rep = {}
    for name, stem, model, layer, pooled in ARMS:
        if not (ROOT / f"{stem}_dirs").exists():
            rep[name] = "no saved directions"
            continue
        rep[name] = {"pooled_in_paper": pooled, **arm(name, stem, model, layer)}
        a, b = rep[name]["as filed"], rep[name]["identifiable"]
        pa, pb = a["primary_comparison"] or {}, b["primary_comparison"] or {}
        print(f"{name}: fail {a['failures']}->{b['failures']}, cw {a['converged_wrong']}->"
              f"{b['converged_wrong']}, diff {pa.get('diff', float('nan')):+.3f}->"
              f"{pb.get('diff', float('nan')):+.3f}", flush=True)
    rep["not computable"] = ("B-14 primary, B-7 depth arms and B-11c saved no fitted directions; "
                             "B-14 is re-run with directions saved (row-identical check first)")
    json.dump(rep, open(ROOT / "results/ln_null_analysis.json", "w"), indent=1)


if __name__ == "__main__":
    main()
