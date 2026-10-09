import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata, spearmanr
import torch
from transformers import AutoModelForCausalLM

ROOT = Path(r"C:/Users/carbo/projects/caliper")


def auc(score, pos):
    score, pos = np.asarray(score, float), np.asarray(pos, bool)
    m, n = pos.sum(), (~pos).sum()
    if not m or not n:
        return None
    r = rankdata(score)
    return float((r[pos].sum() - m * (m + 1) / 2) / (m * n))


def rows(stem):
    return {json.loads(l)["_key"]: json.loads(l) for l in open(ROOT / f"{stem}.jsonl")}


def ln_gamma(model, layer):
    m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32)
    blk = m.transformer.h[layer]
    return blk.ln_2.weight.detach().numpy().astype(np.float64)


out = {}
# V2: LN null direction removed from both fit and reference
for name, stem, model, layer in (("B-15a GPT-2 L6", "results/b15a_fitseed1", "gpt2", 6),
                                 ("B-15b GPT-2 L6", "results/b15b_corpusseed1", "gpt2", 6),
                                 ("B-15c GPT-2 L6", "results/b15c_seqsplit", "gpt2", 6),
                                 ("B-8b GPT-Neo L10", "results/b8b_gptneo125m_indep", "EleutherAI/gpt-neo-125M", 10),
                                 ("X-1a GPT-Neo L10", "results/x1a_gptneo_l10", "EleutherAI/gpt-neo-125M", 10),
                                 ("B-17 GPT-Neo L6", "results/b17_gptneo125m_l6", "EleutherAI/gpt-neo-125M", 6)):
    R = rows(stem)
    g = ln_gamma(model, layer)
    u = 1.0 / g
    u /= np.linalg.norm(u)
    P = lambda x: x - u * (u @ x)
    raw, nul, vshare, r2, rst, fitonly = [], [], [], [], [], []
    for k, r in R.items():
        z = np.load(ROOT / f"{stem}_dirs/n{k}.npz")
        w = z["w"].astype(np.float64); w /= np.linalg.norm(w)
        v = z[r["picked"]][:, 0].astype(np.float64); v /= np.linalg.norm(v)
        raw.append(abs(v @ w))
        pv, pw = P(v), P(w)
        nul.append(abs(pv @ pw) / np.linalg.norm(pv) / np.linalg.norm(pw))
        fitonly.append(abs(pv @ w) / np.linalg.norm(pv))
        vshare.append((u @ v) ** 2)
        r2.append(r["r2_k1"]); rst.append(r["stability"])
    raw, nul, vshare, r2, rst, fitonly = map(np.array, (raw, nul, vshare, r2, rst, fitonly))
    fail_raw, fail_nul = raw < 0.95, nul < 0.95
    cw_raw = fail_raw & (r2 > 0.99); cw_nul = fail_nul & (r2 > 0.99)
    out[name] = {"units": len(raw), "min_gamma_abs": float(np.abs(g).min()),
                 "fail_raw": int(fail_raw.sum()), "fail_null_removed_both": int(fail_nul.sum()),
                 "fail_null_removed_fit_only": int((fitonly < 0.95).sum()),
                 "cw_raw": int(cw_raw.sum()), "cw_null_removed": int(cw_nul.sum()),
                 "median_fit_share_on_null": float(np.median(vshare)),
                 "auc_null_share_cw_vs_pass_raw": auc(vshare[cw_raw | ~fail_raw], cw_raw[cw_raw | ~fail_raw]),
                 "dAUC_restart_minus_R2_raw": (auc(-rst, fail_raw) or 0) - (auc(-r2, fail_raw) or 0),
                 "dAUC_restart_minus_R2_null": (auc(-rst, fail_nul) or 0) - (auc(-r2, fail_nul) or 0)}
    print(name, json.dumps(out[name]), flush=True)

# V3: N-1 primary, SNR alone and within-SNR
R = [json.loads(l) for l in open(ROOT / "results/n1a_snr_mixed.jsonl")]
al = np.array([r["align_selected"] for r in R]); fail = al < 0.95
snr = np.array([r["snr"] for r in R]); r2 = np.array([r["r2_k1"] for r in R]); rst = np.array([r["stability"] for r in R])
ceil = snr / (1 + snr)
out["N-1a"] = {"auc_snr_alone": auc(-snr, fail), "auc_R2": auc(-r2, fail), "auc_restart": auc(-rst, fail),
               "spearman_R2_snr": float(spearmanr(r2, snr)[0]),
               "auc_R2_over_ceiling": auc(-(r2 / ceil), fail)}
print("N-1a", json.dumps(out["N-1a"]), flush=True)

# V4: X-1a / X-1b restart vs R2
for name, stem in (("X-1a", "results/x1a_gptneo_l10"), ("X-1b", "results/x1b_gptneo_l10")):
    Rr = list(rows(stem).values())
    al = np.array([r["align_selected"] for r in Rr]); fail = al < 0.95
    a_r = auc(-np.array([r["stability"] for r in Rr]), fail); a_q = auc(-np.array([r["r2_k1"] for r in Rr]), fail)
    out[name] = {"failures": int(fail.sum()), "auc_restart": a_r, "auc_R2": a_q, "dAUC": a_r - a_q}
    print(name, json.dumps(out[name]), flush=True)
json.dump(out, open(r"C:/Users/carbo/projects/caliper/results/review_round2_checks.json", "w"), indent=1)
