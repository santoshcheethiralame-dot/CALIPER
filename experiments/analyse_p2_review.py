"""Paper 2, internal review of 10 Oct: the offline roadmap items A1-A5 and A7.

    PYTHONPATH=. python experiments/analyse_p2_review.py     # writes results/p2_review_analysis.json

A1  C8 on the log-odds scale. P(YES) is stored as sigmoid(YES - NO), so log-odds are recovered
    exactly except where P(YES) saturated at 0 or 1 (counted). Per cell: baseline and median change
    per framing, factual-NO minus introspective change paired by concept (Wilcoxon, bootstrap
    interval of the median), and the probability-scale ratio with a bootstrap interval (secondary).
A2  C5. Study 3: real against norm-matched random on the introspective prompt per strength, with the
    content-free share of the rise. Per framing, injected against uninjected P(YES) as an AUC (how
    well the framing's answer tells injection from none), at S-1M, in Study 3 and on Qwen-7B.
A3  C1 by concept type and recipe: every health check's AUC on concrete concepts only (the steering
    rule has associates only for them) and within each recipe; and the health checks at S-1M, the
    only result at the published operating point.
A4  C3: the P(YES)-shift inversion per recipe and on concrete concepts, with one-sided Mann-Whitney
    p-values (live vectors shift less) and Fisher's combination across the two Qwen sessions.
A5  C4: one set of vectors (S-1M) at one strength, KL by prompt.
A7  Live rate against own KL with intervals that resample vectors, not vector-dose pairs, and with
    the low-KL points the first version dropped.
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2, mannwhitneyu, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import analyse_p2_rule_sensitivity as rs  # noqa: E402
from analyse_p2m import auc  # noqa: E402
from kaggle_s3_positive_control import ASSOCIATES  # noqa: E402

RNG = np.random.default_rng(0)
NB = 2000
CHECKS = rs.CHECKS


def rows(p):
    return [json.loads(l) for l in open(ROOT / p, encoding="utf-8") if l.strip()]


def logodds(p):
    p = np.clip(np.asarray(p, float), 1e-300, 1 - 1e-16)
    return np.log(p) - np.log1p(-p)


def by_framing(path, framing):
    d = {}
    for r in rows(path):
        if r["framing"] == framing:
            d.setdefault(r["alpha"], {})[r["concept"]] = (r["p_yes"], r.get("kl", np.nan))
    return d


def boot_median(x):
    x = np.asarray(x)
    b = [np.median(RNG.choice(x, len(x))) for _ in range(NB)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


# ------------------------------------------------------------------------------------------- A1
def cell_a1(F, I, N, a_f, a_i):
    c = sorted(set(F[0.0]) & set(F[a_f]) & set(I[0.0]) & set(I[a_i]))
    pf0, pf = np.array([F[0.0][k][0] for k in c]), np.array([F[a_f][k][0] for k in c])
    pi0, pi = np.array([I[0.0][k][0] for k in c]), np.array([I[a_i][k][0] for k in c])
    df, di = logodds(pf) - logodds(pf0), logodds(pi) - logodds(pi0)
    diff = df - di
    out = {"concepts": len(c),
           "baseline_logodds_median": {"factual_no": float(np.median(logodds(pf0))),
                                       "introspective": float(np.median(logodds(pi0)))},
           "logodds_change_median": {"factual_no": float(np.median(df)), "introspective": float(np.median(di))},
           "factual_minus_introspective": {"median": float(np.median(diff)), "ci95": boot_median(diff),
                                           "factual_larger": int((diff > 0).sum()),
                                           "wilcoxon_p": float(wilcoxon(diff).pvalue)},
           "saturated_rows": int(sum(((v == 0) | (v == 1)).sum() for v in (pf0, pf, pi0, pi)))}
    if N is not None and a_i in N:
        cn = [k for k in c if k in N[0.0] and k in N[a_i]]
        dn = logodds([N[a_i][k][0] for k in cn]) - logodds([N[0.0][k][0] for k in cn])
        out["logodds_change_median"]["neutral"] = float(np.median(dn))
    ratio = lambda i: (pf[i] - pf0[i]).mean() / (pi[i] - pi0[i]).mean()
    b = [ratio(RNG.integers(0, len(c), len(c))) for _ in range(NB)]
    out["probability_ratio"] = {"value": float(ratio(np.arange(len(c)))),
                                "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}
    return out


def a1():
    rep = {}
    s1m = "results/s1_gemma27/s1_gemma27_s1m_forced_macar-release_fromtrial.jsonl"
    F = by_framing("results/p2f_gemma27/p2f_gemma27_forced_macar-release_factualno_fromtrial.jsonl", "factual_no")
    I, N = by_framing(s1m, "introspective"), by_framing(s1m, "neutral_matched")
    for a in (4.0, 8.0):
        rep[f"Gemma-3-27B released, strength {a:g}"] = cell_a1(F, I, N, a, a)
    for arm, tag in (("concept", ""), ("tail", ""), ("random", "_random")):
        F = by_framing(f"results/p2f_qwen7b/p2f_qwen7b_{arm}_forced{tag}_factualno.jsonl", "factual_no")
        src = f"results/s2_qwen7b/s2_qwen7b_{arm}_forced{tag}.jsonl"
        I, N = by_framing(src, "introspective"), by_framing(src, "neutral_matched")
        fa, ia = sorted(a for a in F if a), sorted(a for a in I if a)
        for frac, af, ai in zip(("0.25", "0.5", "1.0"), fa, ia):
            rep[f"Qwen2.5-7B {arm}, {frac} of the norm"] = cell_a1(F, I, N, af, ai)
    return rep


# ------------------------------------------------------------------------------------------- A2
def framing_auc(D, a):
    """AUC of P(YES) for injected (dose a) against uninjected (dose 0) trials of one framing."""
    x1, x0 = [v[0] for v in D[a].values()], [v[0] for v in D[0.0].values()]
    return auc(np.r_[x1, x0], np.r_[np.ones(len(x1), bool), np.zeros(len(x0), bool)])


def a2():
    rep = {"study3_real_vs_random_introspective": {}, "injected_vs_uninjected_auc": {}}
    R = by_framing("data/s3/s3_forced_real_sweep.jsonl", "introspective")
    Z = by_framing("data/s3/s3_forced_random.jsonl", "introspective")
    for a in (2.0, 4.0, 6.0):
        c = sorted(set(R[a]) & set(Z[a]) & set(R[0.0]))
        r, z, r0 = (np.array([D[al][k][0] for k in c]) for D, al in ((R, a), (Z, a), (R, 0.0)))
        rep["study3_real_vs_random_introspective"][f"alpha {a:g}"] = {
            "real_mean": float(r.mean()), "random_mean": float(z.mean()),
            "wilcoxon_p": float(wilcoxon(r - z).pvalue),
            "content_free_share": float((z - r0).mean() / (r - r0).mean())}
    s1m = "results/s1_gemma27/s1_gemma27_s1m_forced_macar-release_fromtrial.jsonl"
    for fr in ("introspective", "neutral_matched"):
        D = by_framing(s1m, fr)
        for a in (4.0, 8.0):
            rep["injected_vs_uninjected_auc"][f"S-1M {fr}, strength {a:g}"] = framing_auc(D, a)
    for fr in ("introspective", "neutral"):
        D = by_framing("data/s3/s3_forced_real_sweep.jsonl", fr)
        for a in (2.0, 4.0, 6.0):
            rep["injected_vs_uninjected_auc"][f"Study 3 real {fr}, alpha {a:g}"] = framing_auc(D, a)
    for arm, tag in (("concept", ""), ("tail", ""), ("random", "_random")):
        src = f"results/s2_qwen7b/s2_qwen7b_{arm}_forced{tag}.jsonl"
        for fr in ("introspective", "neutral_matched"):
            D = by_framing(src, fr)
            a = sorted(x for x in D if x)[1]  # 0.5 of the norm
            rep["injected_vs_uninjected_auc"][f"Qwen2.5-7B {arm} {fr}, 0.5 of the norm"] = framing_auc(D, a)
    return rep


# ------------------------------------------------------------------------------------------- A3
def check_aucs(lab, stats, keep):
    out = {}
    for chk in CHECKS:
        ks = [k for k in lab if keep(k) and stats.get(k, {}).get(chk) is not None]
        y = np.array([lab[k] for k in ks])
        x = np.array([stats[k][chk] for k in ks], float)
        if len(ks) and 2 <= y.sum() <= len(y) - 2:
            out[chk] = {"auc": auc(x, y), "live": int(y.sum()), "n": len(y)}
    return out


def a3():
    rep = {}
    sessions = [(f"S-2 {l}", lambda n=n, q=q, r=r: rs.session_s2(n, q, r)) for l, n, q, r in rs.S2] + \
               [(f"S-1 {l}", lambda n=n, r=r: rs.session_s1(n, r)) for l, n, r in rs.S1[:1]]
    for name, load in sessions:
        labels, stats = load()
        lab = labels["filed"]
        conc = lambda k: k[1] in ASSOCIATES
        rep[name] = {"live_concrete": [int(sum(lab[k] for k in lab if conc(k))), int(sum(conc(k) for k in lab))],
                     "live_abstract": [int(sum(lab[k] for k in lab if not conc(k))), int(sum(not conc(k) for k in lab))],
                     "concrete_only": check_aucs(lab, stats, conc),
                     "within_recipe": {arm: check_aucs(lab, stats, lambda k, a=arm: k[0] == a)
                                       for arm in sorted({k[0] for k in lab})},
                     # recipe and concept type both held fixed: what survives once neither can carry
                     # the signal; 20 vectors per cell, so read as direction, not as a test
                     "within_recipe_concrete_only": {arm: check_aucs(lab, stats, lambda k, a=arm: k[0] == a and conc(k))
                                                     for arm in sorted({k[0] for k in lab})}}
    cfg = json.load(open(ROOT / "results/s1_gemma27/s1_gemma27_s1m_steer_macar-release.config.json"))
    st = rows("results/s1_gemma27/s1_gemma27_s1m_steer_macar-release.jsonl")
    base = {r["concept"]: r["steered"] for r in st if r["alpha"] == 0}
    lab = {("released", r["concept"]): bool(r["steered"]) and not base[r["concept"]] for r in st if r["alpha"] == 4}
    stats = {("released", c): {"norm": h["norm"], "distinctness": -h["max_cos_other"], "stability": h["stability"],
                               "probe": h["probe"], "logit steering": h["steer_logit_delta"]}
             for c, h in cfg["health"].items()}
    rep["S-1M Gemma-3-27B released, strength 4"] = {"live": int(sum(lab.values())), "vectors": len(lab),
                                                     "auc": check_aucs(lab, stats, lambda k: True)}
    return rep


# ------------------------------------------------------------------------------------------- A4
def a4():
    rep, ps = {}, []
    for label, name, q, repo in rs.S2[:2]:
        P = json.load(open(ROOT / f"results/s2_{name}_analysis.json"))["per_vector"]
        live = np.array([p["live"] for p in P])
        x = np.array([p["P(YES) shift"] for p in P])
        arm = np.array([p["arm"] for p in P])
        conc = np.array([p["concept"] in ASSOCIATES for p in P])

        def one(m):
            if not (0 < live[m].sum() < m.sum()):
                return None
            p = mannwhitneyu(x[m][live[m]], x[m][~live[m]], alternative="less").pvalue
            return {"auc": auc(x[m], live[m]), "live": int(live[m].sum()), "n": int(m.sum()), "p_one_sided": float(p)}
        allv = one(np.ones(len(P), bool))
        ps.append(allv["p_one_sided"])
        rep[label] = {"all": allv, "concrete_only": one(conc),
                      "by_recipe": {a: one(arm == a) for a in sorted(set(arm))}}
    stat = -2 * np.sum(np.log(ps))
    rep["fisher_combined_p"] = float(chi2.sf(stat, 2 * len(ps)))
    return rep


# ------------------------------------------------------------------------------------------- A5
def a5():
    out = {}
    src = {"factual-NO": ("results/p2f_gemma27/p2f_gemma27_forced_macar-release_factualno_fromtrial.jsonl", "factual_no"),
           "introspective": ("results/s1_gemma27/s1_gemma27_s1m_forced_macar-release_fromtrial.jsonl", "introspective"),
           "neutral": ("results/s1_gemma27/s1_gemma27_s1m_forced_macar-release_fromtrial.jsonl", "neutral_matched"),
           "story (steer stage)": ("results/s1_gemma27/s1_gemma27_s1m_steer_macar-release.jsonl", None)}
    for name, (p, fr) in src.items():
        kl = [r["kl"] for r in rows(p) if r["alpha"] == 4.0 and (fr is None or r["framing"] == fr)]
        out[name] = {"median_kl": float(np.median(kl)), "n": len(kl)}
    return {"S-1M released vectors at strength 4": out}


# ------------------------------------------------------------------------------------------- A7
EDGES = np.array([0, 1e-5, 1e-3, 1e-2, 1e-1, 1, 10, 1e3])


def a7():
    rep = {}
    import analyse_p2_live_kl as lk
    for name, pat in lk.SESSIONS.items():
        pts = []
        for f in sorted(glob.glob(str(ROOT / pat))):
            if not any(t in os.path.basename(f) for t in lk.REAL):
                continue
            R = rows(os.path.relpath(f, ROOT))
            base = {r["concept"]: bool(r["steered"]) for r in R if r["alpha"] == 0}
            for r in R:
                if r["alpha"] and r["concept"] in base:
                    pts.append((os.path.basename(f) + "|" + r["concept"], float(r["kl"]),
                                bool(r["steered"]) and not base[r["concept"]]))
        vecs = sorted({p[0] for p in pts})
        by = {v: [p for p in pts if p[0] == v] for v in vecs}

        def rates(sample):
            P = [p for v in sample for p in by[v]]
            kl, lv = np.array([p[1] for p in P]), np.array([p[2] for p in P])
            return [(float(lv[(kl >= lo) & (kl < hi)].mean()) if ((kl >= lo) & (kl < hi)).sum() else np.nan)
                    for lo, hi in zip(EDGES[:-1], EDGES[1:])]
        point = rates(vecs)
        boots = np.array([rates(list(RNG.choice(vecs, len(vecs)))) for _ in range(500)])
        kl_all = np.array([p[1] for p in pts])
        bins = []
        for i, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
            m = (kl_all >= lo) & (kl_all < hi)
            if m.sum() >= 10:
                b = boots[:, i][~np.isnan(boots[:, i])]
                bins.append({"kl_lo": float(lo), "kl_hi": float(hi), "points": int(m.sum()),
                             "vectors": len({pts[j][0] for j in np.where(m)[0]}),
                             "median_kl": float(np.median(kl_all[m])), "live_rate": point[i],
                             "ci95_vector_bootstrap": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]})
        rep[name] = {"vectors": len(vecs), "points": len(pts), "bins": bins}
    return rep


def main():
    rep = {}
    for k, fn in (("A1 C8 on log-odds", a1), ("A2 C5 readout", a2), ("A3 C1 by concept type and recipe", a3),
                  ("A4 C3 inversion", a4), ("A5 C4 KL by prompt", a5), ("A7 live rate against own KL", a7)):
        rep[k] = fn()
        print("done", k, flush=True)
    json.dump(rep, open(ROOT / "results/p2_review_analysis.json", "w"), indent=1)


if __name__ == "__main__":
    main()
