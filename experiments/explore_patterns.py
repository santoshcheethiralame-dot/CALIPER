"""Exploratory pattern pass across completed runs (8 Oct 2026). Not pre-registered.

    python experiments/explore_patterns.py [--only Q1 Q3 ...]

Q1  Identifiability index of the ground truth: share of ||w||^2 in the stimulus directions
    carrying the bottom 1% of variance. Does it predict converged-wrong before any fit?
Q2  Low-variance share of the FITTED direction, a ground-truth-free quantity. Does it flag
    converged-wrong fits? Also: Euclidean alignment after projecting both directions onto
    the well-sampled subspace (how many failures survive?).
Q3  Identifiability map: every layer of GPT-2 small and GPT-Neo-125m, condition number and
    the median w low-variance share over 300 random units.
Q4  Chronic vs flaky failures across the four fits of B-14's first 100 units.
Q5  Restart-agreement saturation: how often a failing fit has restart agreement >= 0.99.
Q6  Unit properties (firing fraction, response sd) by failure class.
Q7  T-SAE: failure and fit quality against latent firing count.
N-1 and X-1 are not read. Writes results/explore_patterns.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))
from analyse_review_round1 import auc, stimulus  # noqa: E402

PASS, WB = 0.95, 0.99
ARMS = {  # name: (rows stem, model, layer, fp16, corpus seed of the fit)
    "B-8b GPT-Neo L10": ("results/b8b_gptneo125m_indep", "EleutherAI/gpt-neo-125M", 10, False, 0),
    "B-15a GPT-2 L6": ("results/b15a_fitseed1", "gpt2", 6, False, 0),
    "B-15b GPT-2 L6": ("results/b15b_corpusseed1", "gpt2", 6, False, 1),
    "B-15c GPT-2 L6": ("results/b15c_seqsplit", "gpt2", 6, False, 0),
    "B-2c Pythia-160m L6": ("results/b2c_pythia160m_indep", "EleutherAI/pythia-160m", 6, True, 0),
    "B-17 GPT-Neo L6": ("results/b17_gptneo125m_l6", "EleutherAI/gpt-neo-125M", 6, False, 0),
}


def rows(stem):
    return {r["_key"]: r for r in map(json.loads, open(ROOT / f"{stem}.jsonl", encoding="utf-8"))}


def unit(v):
    return v / np.linalg.norm(v)


def low_subspace(S, frac=0.01):
    ev, evec = np.linalg.eigh(np.cov(S, rowvar=False))
    return ev, evec, evec[:, np.cumsum(ev) / ev.sum() <= frac]


def classes(R, ks):
    fail = np.array([R[k]["align_selected"] < PASS for k in ks])
    cw = fail & np.array([R[k]["r2_k1"] > WB for k in ks])
    return fail, cw, fail & ~cw


def q1_q2():
    out = {}
    for name, (stem, model, layer, fp16, cs) in ARMS.items():
        R = rows(stem)
        ks = sorted(R)
        S, _ = stimulus(model, layer, ks, fp16=fp16, corpus_seed=cs)
        ev, evec, low = low_subspace(S)
        hi = evec[:, ~(np.cumsum(ev) / ev.sum() <= 0.01)]
        w_share, v_share, proj_align = [], [], []
        for k in ks:
            z = np.load(ROOT / f"{stem}_dirs/n{k}.npz")
            w, v = unit(z["w"]), unit(z[R[k]["picked"]][:, 0])
            w_share.append(float(((low.T @ w) ** 2).sum()))
            v_share.append(float(((low.T @ v) ** 2).sum()))
            pw, pv = hi.T @ w, hi.T @ v
            proj_align.append(float(abs(pw @ pv) / (np.linalg.norm(pw) * np.linalg.norm(pv))))
        w_share, v_share, proj_align = map(np.array, (w_share, v_share, proj_align))
        fail, cw, uf = classes(R, ks)
        res = {"units": len(ks), "failures": int(fail.sum()), "converged_wrong": int(cw.sum()),
               "low_dims": int(low.shape[1]),
               "median_w_low_share": {"pass": float(np.median(w_share[~fail])),
                                      "converged-wrong": float(np.median(w_share[cw])) if cw.any() else None,
                                      "under-fitted": float(np.median(w_share[uf])) if uf.any() else None},
               "median_vhat_low_share": {"pass": float(np.median(v_share[~fail])),
                                         "converged-wrong": float(np.median(v_share[cw])) if cw.any() else None,
                                         "under-fitted": float(np.median(v_share[uf])) if uf.any() else None},
               "failures_passing_after_projection": int((fail & (proj_align >= PASS)).sum()),
               "converged_wrong_passing_after_projection": int((cw & (proj_align >= PASS)).sum())}
        for lab, pos in (("converged-wrong vs pass", cw), ("all failures", fail)):
            keep = pos | ~fail
            y = pos[keep]
            if y.sum() >= 2 and (~y).sum() >= 2:
                res[f"AUC w-share (ground truth), {lab}"] = auc(w_share[keep], y)
                res[f"AUC vhat-share (ground-truth-free), {lab}"] = auc(v_share[keep], y)
                res[f"AUC held-out R2, {lab}"] = auc(np.array([-R[k]["r2_k1"] for k in ks])[keep], y)
                res[f"AUC restart agreement, {lab}"] = auc(np.array([-R[k]["stability"] for k in ks])[keep], y)
        out[name] = res
    return out


def q3(n_units=300):
    from caliper.activations import collect, load_model, sample_corpus
    import torch
    out = {}
    for model_name in ("gpt2", "EleutherAI/gpt-neo-125M"):
        model, tok = load_model(model_name, dtype=torch.float32)
        ids = np.random.default_rng(0).choice(3072, n_units, replace=False)
        texts = sample_corpus(n_docs=300, seed=0)
        layers = {}
        for L in range(12):
            p = collect(model, tok, texts, layer=L, neurons=ids, max_tokens=8000, seed=0)
            S = np.asarray(p.stimulus, float)
            ev, evec, low = low_subspace(S)
            W = np.asarray(p.weights, float)
            W = W / np.linalg.norm(W, axis=0, keepdims=True)
            share = ((low.T @ W) ** 2).sum(0)
            pos = ev[ev > ev.max() * 1e-12]
            sd = S.std(0)
            layers[L] = {"near_constant_coords": int((sd < 1e-2 * np.median(sd)).sum()),
                         "min_sd_over_median": float(sd.min() / np.median(sd)),
                         "condition_number": float(ev.max() / pos.min()),
                         "low_dims": int(low.shape[1]),
                         "median_w_low_share": float(np.median(share)),
                         "frac_units_w_low_share_gt_0.5": float((share > 0.5).mean())}
            print(model_name, L, layers[L], flush=True)
        out[model_name] = layers
    return out


def q4():
    ref = rows("results/b14_primary_gpt2_indep")
    fits = [rows(s) for s in ("results/b15a_fitseed1", "results/b15b_corpusseed1", "results/b15c_seqsplit")]
    ks = sorted(set(ref) & set(fits[0]) & set(fits[1]) & set(fits[2]))
    nfail = np.array([sum(R[k]["align_selected"] < PASS for R in [ref] + fits) for k in ks])
    med_align = np.array([np.median([R[k]["align_selected"] for R in [ref] + fits]) for k in ks])
    dist = {int(c): int((nfail == c).sum()) for c in range(5)}
    cw_ever = np.array([any(R[k]["align_selected"] < PASS and R[k]["r2_k1"] > WB for R in [ref] + fits) for k in ks])
    return {"units": len(ks), "failing_in_k_of_4_fits": dist,
            "chronic_(4of4)_median_align": float(np.median(med_align[nfail == 4])) if (nfail == 4).any() else None,
            "flaky_(1-3of4)_median_align": float(np.median(med_align[(nfail > 0) & (nfail < 4)])),
            "units_ever_converged_wrong": int(cw_ever.sum()),
            "of_which_chronic": int((cw_ever & (nfail == 4)).sum())}


def q5():
    out = {}
    for name, stem in (("B-14 GPT-2 L6", "results/b14_primary_gpt2_indep"),
                       ("B-8b GPT-Neo L10", "results/b8b_gptneo125m_indep"),
                       ("B-2c Pythia-160m L6", "results/b2c_pythia160m_indep"),
                       ("B-17 GPT-Neo L6", "results/b17_gptneo125m_l6")):
        R = rows(stem)
        ks = sorted(R)
        fail, cw, uf = classes(R, ks)
        st = np.array([R[k]["stability"] for k in ks])
        out[name] = {"failures": int(fail.sum()),
                     "failing_with_restart_agreement_ge_0.99": int((fail & (st >= 0.99)).sum()),
                     "converged_wrong_with_ge_0.99": int((cw & (st >= 0.99)).sum()),
                     "under_fitted_with_ge_0.99": int((uf & (st >= 0.99)).sum()),
                     "passes_with_lt_0.99": int((~fail & (st < 0.99)).sum()),
                     "median_restart_agreement": {"pass": float(np.median(st[~fail])),
                                                  "failure": float(np.median(st[fail])) if fail.any() else None}}
    return out


def q6():
    out = {}
    for name, (stem, *_rest) in ARMS.items():
        diag = ROOT / f"{stem.replace('_indep', '').replace('_fitseed1', '').replace('_corpusseed1', '').replace('_seqsplit', '')}_diagnostics.json"
        cand = {"B-8b GPT-Neo L10": "results/b8b_diagnostics.json", "B-15a GPT-2 L6": "results/b15a_diagnostics.json",
                "B-15b GPT-2 L6": "results/b15b_diagnostics.json", "B-15c GPT-2 L6": "results/b15c_diagnostics.json",
                "B-2c Pythia-160m L6": "results/b2c_diagnostics.json", "B-17 GPT-Neo L6": "results/b17_diagnostics.json"}
        U = {u["unit"]: u for u in json.load(open(ROOT / cand[name]))["units"]}
        R = rows(stem)
        ks = sorted(set(R) & set(U))
        fail, cw, uf = classes(R, ks)
        res = {}
        for f in ("firing_frac", "response_sd"):
            x = np.array([U[k][f] for k in ks])
            res[f] = {c: (float(np.median(x[m])) if m.any() else None)
                      for c, m in (("pass", ~fail), ("converged-wrong", cw), ("under-fitted", uf))}
        out[name] = res
    return out


def q7():
    R = rows("results/tsae_main")
    ks = sorted(R)
    keys = list(R[ks[0]])
    out = {"fields": keys, "units": len(ks), "passes": int(sum(R[k]["align_selected"] >= PASS for k in ks))}
    al = np.array([R[k]["align_selected"] for k in ks])
    r2 = np.array([R[k]["r2_k1"] for k in ks])
    out["median_align"] = float(np.median(al))
    out["median_r2"] = float(np.median(r2))
    out["corr_align_r2"] = float(np.corrcoef(al, r2)[0, 1])
    out["align_quartiles"] = [float(q) for q in np.percentile(al, [25, 50, 75])]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    path = ROOT / "results/explore_patterns.json"
    out = json.load(open(path)) if path.exists() else {}
    fns = {"Q12": q1_q2, "Q3": q3, "Q4": q4, "Q5": q5, "Q6": q6, "Q7": q7}
    for name in (a.only or ["Q4", "Q5", "Q6", "Q7", "Q12", "Q3"]):
        print("==", name, flush=True)
        out[name] = fns[name]()
        json.dump(out, open(path, "w"), indent=1, default=float)
        print(json.dumps(out[name], indent=1, default=float)[:3000], flush=True)


if __name__ == "__main__":
    main()
