"""Paper 1, round-3 (strict blind) review, offline items B1-B6.

    PYTHONPATH=. python experiments/analyse_round3.py      # writes results/round3_fixes.json

B1  Route-matched comparison with each route's own held-out R2. The round-1 script scored the
    direct route's failure label against the selected fit's R2 (max of both routes). The direct
    route's own held-out R2 is the best over its restarts, stored as direct_r2_restarts.
B2  The cascade fallback. When the cascade's polish underperforms, fit_cascade returns a binned R2
    of its direction over all tokens as test_r2: in-sample. Such a unit is flagged when the cascade
    was selected and r2_k1 equals that all-token binned R2; its R2 is replaced by the binned R2 on
    the held-out rows (the first 20%), the route is re-selected, and the comparison re-scored.
B3  The 1/gamma table for the 1.4B arm, which did save directions (gain read from the hub).
B4  The token-budget sweep (B-10, coupled estimator) in the every-run format.
B5  Every filed secondary of the primary, as stored.
B6  Stability of the failure classes across fits of the same 100 GPT-2 layer-6 units.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))
from analyse_review_round1 import auc, boot_diff  # noqa: E402
from b1_signal_calibration import delong  # noqa: E402

PASS, WB = 0.95, 0.99
DIR_ARMS = [  # name, stem, model, layer, corpus seed, sequence split, fp16
    ("GPT-2 L6 primary replicate (B-14r)", "results/b14r_primary_gpt2_dirs", "gpt2", 6, 0, False, False),
    ("Pythia-160m L6 (B-2c)", "results/b2c_pythia160m_indep", "EleutherAI/pythia-160m", 6, 0, False, True),
    ("GPT-Neo-125m L10 (B-8b)", "results/b8b_gptneo125m_indep", "EleutherAI/gpt-neo-125M", 10, 0, False, False),
    ("Pythia-1.4B L12, 3200 steps (B-11c)", "data/b11/b11c_pythia-14b_s3200_indep", None, 12, 0, False, False),
    ("GPT-2 L6 re-fit, fit seed 1 (B-15a)", "results/b15a_fitseed1", "gpt2", 6, 0, False, False),
    ("GPT-2 L6 re-fit, corpus seed 1 (B-15b)", "results/b15b_corpusseed1", "gpt2", 6, 1, False, False),
    ("GPT-2 L6 re-fit, sequence split (B-15c)", "results/b15c_seqsplit", "gpt2", 6, 0, True, False),
    ("GPT-Neo-125m L10, new units (X-1a)", "results/x1a_gptneo_l10", "EleutherAI/gpt-neo-125M", 10, 0, False, False),
    ("GPT-Neo-125m L10, new units (X-1b)", "results/x1b_gptneo_l10", "EleutherAI/gpt-neo-125M", 10, 1, False, False),
]
B1_ONLY = [  # B1 reads saved directions only; these arms are not rebuilt for B2
    ("Pythia-160m L6, float32 (B-2d)", "results/b2d_pythia160m_fp32"),
    ("GPT-Neo-125m L6 (B-17)", "results/b17_gptneo125m_l6"),
    ("GPT-Neo-125m L6, coordinate dropped (B-17b)", "results/b17b_gptneo125m_l6_drop"),
    ("GPT-2 L3 (B-17c)", "results/b17c_gpt2_l3"),
    ("GPT-2 L3, coordinate dropped (B-17d)", "results/b17d_gpt2_l3_drop"),
    ("GPT-2 L6, SAE latents (T-SAE)", "results/tsae_main"),
    ("GPT-2 L6, noise SNR 1-19 (N-1a)", "results/n1a_snr_mixed"),
    ("GPT-2 L6, noise SNR 19 (N-1b)", "results/n1b_snr19"),
    ("GPT-2 L6, noise SNR 4 (N-1c)", "results/n1c_snr4"),
]


def rows(stem):
    p = ROOT / f"{stem}.jsonl"
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []


def compare(s1, s2, fail):
    if fail.sum() < 2 or (~fail).sum() < 2:
        return None
    a1, a2, d, se, z, p = delong(list(s1), list(s2), list(fail))
    return {"auc_restart": a1, "auc_r2": a2, "diff": d, "delong_p": p, "boot_ci95": boot_diff(s1, s2, fail)}


def b1():
    out = {}
    for name, stem, *_ in DIR_ARMS + B1_ONLY:
        R = rows(stem)
        if not R:
            continue
        own = np.array([float(np.max(np.load(ROOT / f"{stem}_dirs/n{r['_key']}.npz")["direct_r2_restarts"])) for r in R])
        lab = np.array([r["align_direct"] < PASS for r in R])
        st = np.array([r["stability"] for r in R])
        sel = np.array([r["r2_k1"] for r in R])
        out[name] = {"units": len(R), "direct_route_failures": int(lab.sum()),
                     "as_reported_selected_r2": compare(-st, -sel, lab),
                     "matched_own_r2": compare(-st, -own, lab)}
    return out


def stimulus(model, layer, ids, corpus_seed, seq_split, fp16):
    import torch
    from caliper.activations import collect, load_model, sample_corpus
    m, tok = load_model(model, dtype=torch.float16 if fp16 else torch.float32)
    p = collect(m, tok, sample_corpus(n_docs=300, seed=corpus_seed), layer=layer, neurons=np.asarray(ids),
                max_tokens=8000, seed=0, shuffle="sequence" if seq_split else "token")
    return np.asarray(p.stimulus, np.float32), np.asarray(p.response, np.float64)


def b2():
    from caliper.estimator import _binned_r2
    out = {}
    for name, stem, model, layer, cs, seq, fp16 in DIR_ARMS:
        R = rows(stem)
        if not R or model is None:
            continue
        X, Y = stimulus(model, layer, [r["_key"] for r in R], cs, seq, fp16)
        n_test = int(len(X) * 0.2)
        flagged, new_sel, new_r2 = [], [], []
        for i, r in enumerate(R):
            z = np.load(ROOT / f"{stem}_dirs/n{r['_key']}.npz")
            c = z["cascade"][:, 0].astype(np.float32)
            own = float(np.max(z["direct_r2_restarts"]))
            fb = r["picked"] == "cascade" and abs(_binned_r2(X @ c, Y[:, i]) - r["r2_k1"]) < 1e-5
            if fb:
                held = float(_binned_r2(X[:n_test] @ c, Y[:n_test, i], n_bins=40))
                use_c = held > own
                flagged.append({"unit": int(r["_key"]), "reported_r2": r["r2_k1"], "held_out_binned_r2": held,
                                "direct_r2": own, "reselected": "cascade" if use_c else "direct",
                                "align_before": r["align_selected"],
                                "align_after": r["align_cascade"] if use_c else r["align_direct"]})
                new_sel.append(r["align_cascade"] if use_c else r["align_direct"])
                new_r2.append(max(held, own) if use_c else own)
            else:
                new_sel.append(r["align_selected"])
                new_r2.append(r["r2_k1"])
        st = np.array([r["stability"] for r in R])
        f0 = np.array([r["align_selected"] < PASS for r in R])
        f1 = np.array(new_sel) < PASS
        out[name] = {"units": len(R), "fallback_units": len(flagged), "flagged": flagged,
                     "as_filed": compare(-st, -np.array([r["r2_k1"] for r in R]), f0),
                     "corrected": compare(-st, -np.array(new_r2), f1),
                     "verdicts_changed": int((f0 != f1).sum())}
        print(name, len(flagged), "fallbacks;", out[name]["as_filed"] and round(out[name]["as_filed"]["diff"], 3),
              "->", out[name]["corrected"] and round(out[name]["corrected"]["diff"], 3), flush=True)
    return out


def b3():
    import analyse_p2m as p2m
    src = p2m.Hub("EleutherAI/pythia-1.4b")
    t = p2m.tensors(src, lambda names: [k for k in names if k.endswith("layers.12.post_attention_layernorm.weight")])
    g = np.asarray(next(iter(t.values())), np.float64)
    u = (1 / g) / np.linalg.norm(1 / g)
    stem = "data/b11/b11c_pythia-14b_s3200_indep"
    R = rows(stem)
    P = lambda x: x - u * (u @ x)
    raw, ident, share = [], [], []
    for r in R:
        z = np.load(ROOT / f"{stem}_dirs/n{r['_key']}.npz")
        w = z["w"].astype(np.float64); w /= np.linalg.norm(w)
        v = z[r["picked"]][:, 0].astype(np.float64); v /= np.linalg.norm(v)
        pv, pw = P(v), P(w)
        raw.append(abs(v @ w)); ident.append(abs(pv @ pw) / np.linalg.norm(pv) / np.linalg.norm(pw))
        share.append(float((u @ v) ** 2))
    raw, ident = np.array(raw), np.array(ident)
    st, r2 = np.array([r["stability"] for r in R]), np.array([r["r2_k1"] for r in R])
    res = {"min_abs_gamma": float(np.abs(g).min()), "units": len(R)}
    for lab, al in (("as filed", raw), ("identifiable", ident)):
        fail = al < PASS
        res[lab] = {"failures": int(fail.sum()), "converged_wrong": int((fail & (r2 > WB)).sum()),
                    "comparison": compare(-st, -r2, fail)}
    res["verdicts_changed"] = int(((raw < PASS) != (ident < PASS)).sum())
    res["median_null_share"] = float(np.median(share))
    return res


def b4():
    R = rows("results/b10_required_n")
    out = {}
    for t in sorted({r["tokens"] for r in R}):
        g = [r for r in R if r["tokens"] == t]
        fail = np.array([r["align_selected"] < PASS for r in g])
        r2, st = np.array([r["r2_k1"] for r in g]), np.array([r["stability"] for r in g])
        out[str(t)] = {"units": len(g), "failures": int(fail.sum()),
                       "converged_wrong": int((fail & (r2 > WB)).sum()),
                       "comparison": compare(-st, -r2, fail)}
    return out


def b5():
    a = json.load(open(ROOT / "results/b14_analysis.json"))
    b = json.load(open(ROOT / "results/b15_analysis.json"))
    icc = {arm: {k: v for k, v in d.items() if "icc" in k.lower()} for arm, d in b["arms"].items()}
    return {"b14_signals": a.get("signals"), "b14_secondary_bh": a.get("secondary_bh"), "b15_iccs": icc}


def b6():
    fits = {"B-14": "results/b14_primary_gpt2_indep", "B-14r": "results/b14r_primary_gpt2_dirs",
            "B-15a": "results/b15a_fitseed1", "B-15b": "results/b15b_corpusseed1", "B-15c": "results/b15c_seqsplit"}
    tab = {k: {r["_key"]: r for r in rows(s)} for k, s in fits.items()}
    shared = sorted(set.intersection(*[set(v) for v in tab.values() if v]))
    cls = lambda r: "pass" if r["align_selected"] >= PASS else ("converged-wrong" if r["r2_k1"] > WB else "under-fitted")
    counts = {}
    for u in shared:
        c = [cls(tab[k][u]) for k in fits if tab[k]]
        counts[u] = {x: c.count(x) for x in set(c)}
    n_fits = len([k for k in fits if tab[k]])
    cw = [v.get("converged-wrong", 0) for v in counts.values()]
    fail = [n_fits - v.get("pass", 0) for v in counts.values()]
    return {"fits": n_fits, "shared_units": len(shared),
            "converged_wrong_in_k_fits": {str(k): int(sum(c == k for c in cw)) for k in range(n_fits + 1)},
            "failing_in_k_fits": {str(k): int(sum(f == k for f in fail)) for k in range(n_fits + 1)}}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", nargs="*", default=None,
                    help="run only these items (B1 ... B6) and keep the others from the existing json")
    only = ap.parse_args().items
    out = ROOT / "results/round3_fixes.json"
    rep = json.load(open(out)) if only and out.exists() else {}
    for key, fn in (("B1 route-matched with own R2", b1), ("B4 token sweep (B-10)", b4),
                    ("B5 filed secondaries", b5), ("B6 class stability", b6),
                    ("B3 1.4B arm on the identifiable label", b3), ("B2 cascade fallback", b2)):
        if only and key.split()[0] not in only:
            continue
        try:
            rep[key] = fn()
            print("done", key, flush=True)
        except Exception as e:  # report and keep going: each item is independent
            rep[key] = f"failed: {type(e).__name__}: {e}"
            print("FAILED", key, e, flush=True)
    json.dump(rep, open(out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
