"""Review round 1: the exploratory analyses filed in docs/review-round1-analysis-plan.md.

    python experiments/analyse_review_round1.py            # all sections
    python experiments/analyse_review_round1.py --only R3 R4

Writes results/review_round1.json. Every number here is exploratory: none changes a
pre-registered endpoint. Sections R2, R5 and R8 rebuild stimuli and load models; the rest read
archived rows only.

One deviation from the plan, made for runtime and recorded in the output: R5's bootstrap
reruns 10 CV repeats per resample (the point estimate uses the planned 50), with units grouped
so that a resampled duplicate never sits in both a training and a test fold.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, t as student_t
from scipy.special import betaln

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))
from b1_signal_calibration import PASS, delong  # noqa: E402
from analyse_b15 import icc_a1  # noqa: E402

WRONG_BASIN_R2 = 0.99
# name: (rows, estimator version, device, precision as loaded)
ARMS = {
    "GPT-2 L6 (B-14)": ("results/b14_primary_gpt2_indep.jsonl", "corrected", "CPU", "fp32"),
    "Pythia-160m L6 (B-2c)": ("results/b2c_pythia160m_indep.jsonl", "corrected", "CPU", "fp16"),
    "GPT-2 L2 (B-7)": ("results/b7_layer02_gpt2.jsonl", "coupled", "CPU", "fp32"),
    "GPT-2 L10 (B-7)": ("results/b7_layer10_gpt2.jsonl", "coupled", "CPU", "fp32"),
    "GPT-Neo-125m L10 (B-8b)": ("results/b8b_gptneo125m_indep.jsonl", "corrected", "CPU", "fp32"),
    "Pythia-1.4B L12, 3200 steps (B-11c)": ("data/b11/b11c_pythia-14b_s3200_indep.jsonl",
                                            "corrected", "Kaggle T4", "unrecorded"),
    "Pythia-70m (B-11)": ("data/b11/b11_pythia-70m.jsonl", "coupled", "CPU", "fp16"),
    "Pythia-410m (B-11)": ("data/b11/b11_pythia-410m.jsonl", "coupled", "CPU", "fp16"),
    "Pythia-410m, 3200 steps (B-11)": ("data/b11/b11_pythia-410m_s3200.jsonl", "coupled",
                                           "CPU", "fp16"),
    "GPT-2 L6 re-fit, fit seed 1 (B-15a)": ("results/b15a_fitseed1.jsonl", "corrected", "CPU", "fp32"),
    "GPT-2 L6 re-fit, corpus seed 1 (B-15b)": ("results/b15b_corpusseed1.jsonl", "corrected", "CPU",
                                               "fp32"),
    "GPT-2 L6 re-fit, sequence split (B-15c)": ("results/b15c_seqsplit.jsonl", "corrected", "CPU",
                                                "fp32"),
    "GPT-Neo-125m L6 (B-17)": ("results/b17_gptneo125m_l6.jsonl", "corrected", "CPU", "fp32"),
}
POOLED = ["GPT-2 L6 (B-14)", "Pythia-160m L6 (B-2c)", "GPT-2 L2 (B-7)", "GPT-2 L10 (B-7)",
          "GPT-Neo-125m L10 (B-8b)", "Pythia-1.4B L12, 3200 steps (B-11c)"]
CORRECTED = [a for a in POOLED if ARMS[a][1] == "corrected"]


def rows(path):
    return [json.loads(l) for l in open(ROOT / path, encoding="utf-8") if l.strip()]


def auc(score, fail):
    """Mann-Whitney AUC with midranks (ties count half); higher score = more suspect."""
    score, fail = np.asarray(score, float), np.asarray(fail, bool)
    m, n = fail.sum(), (~fail).sum()
    if m == 0 or n == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[fail].sum() - m * (m + 1) / 2) / (m * n))


def checks(R):
    return (np.array([-r["stability"] for r in R]), np.array([-r["r2_k1"] for r in R]))


def boot_diff(st, r2, fail, n_boot=2000, seed=0):
    """Stratified bootstrap CI of AUC(restart) - AUC(R2): failures and passes resampled apart."""
    rng = np.random.default_rng(seed)
    fail = np.asarray(fail, bool)
    pi, ni = np.where(fail)[0], np.where(~fail)[0]
    out = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pi, len(pi)), rng.choice(ni, len(ni))])
        out.append(auc(st[idx], fail[idx]) - auc(r2[idx], fail[idx]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def paired_perm(st, r2, fail, n_perm=10000, seed=0):
    """Paired permutation test of AUC equality: swap the two rank-standardised checks within
    each unit. Valid for correlated ROC curves on the same units, unlike permuting labels."""
    rng = np.random.default_rng(seed)
    a, b = rankdata(st) / len(st), rankdata(r2) / len(r2)
    obs = auc(a, fail) - auc(b, fail)
    hits = 0
    for _ in range(n_perm):
        m = rng.random(len(a)) < 0.5
        aa, bb = np.where(m, b, a), np.where(m, a, b)
        hits += abs(auc(aa, fail) - auc(bb, fail)) >= abs(obs) - 1e-12
    return (hits + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------------------- R1
def r1_route_matched():
    out = {}
    for name, (path, *_ ) in ARMS.items():
        R = rows(path)
        res = {}
        D = [r for r in R if r["picked"] == "direct"]
        for lab, sub, f in (("direct-picked, selected label", D,
                             [r["align_selected"] < PASS for r in D]),
                            ("all units, direct-route label", R,
                             [r["align_direct"] < PASS for r in R])):
            st, r2 = checks(sub)
            k = int(sum(f))
            res[lab] = {"n": len(sub), "failures": k,
                        "auc_restart": auc(st, f), "auc_r2": auc(r2, f),
                        "diff": auc(st, f) - auc(r2, f) if 0 < k < len(sub) else None}
        out[name] = res
    return out


# ---------------------------------------------------------------------------------------- R3
def r3_all_arms():
    out = {}
    for name, (path, version, device, prec) in ARMS.items():
        R = rows(path)
        fail = np.array([r["align_selected"] < PASS for r in R])
        st, r2 = checks(R)
        k = int(fail.sum())
        d = {"n": len(R), "failures": k, "estimator": version, "device": device,
             "precision": prec, "pooled": name in POOLED}
        if 0 < k < len(R):
            res = delong(list(st), list(r2), list(fail))
            d.update({"auc_restart": auc(st, fail), "auc_r2": auc(r2, fail),
                      "diff": auc(st, fail) - auc(r2, fail), "delong_p": res[5],
                      "delong_se": res[3], "boot_ci95": boot_diff(st, r2, fail)})
            if "route_agreement" in R[0]:
                ra = np.array([-r["route_agreement"] for r in R])
                d["auc_route_agreement"] = auc(ra, fail)
        out[name] = d
    return out


# ---------------------------------------------------------------------------------------- R4
def hksj(d, se):
    d, se = np.asarray(d), np.asarray(se)
    k = len(d)
    w = 1 / se ** 2
    mu_fe = (w * d).sum() / w.sum()
    q = (w * (d - mu_fe) ** 2).sum()
    tau2 = max(0.0, (q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    ws = 1 / (se ** 2 + tau2)
    mu = (ws * d).sum() / ws.sum()
    # Truncated at 1 (Roever et al. 2015): with Q < k - 1 the plain HKSJ interval would be
    # narrower than DerSimonian-Laird's, which is anti-conservative with few arms.
    var_hk = max(1.0, (ws * (d - mu) ** 2).sum() / (k - 1)) / ws.sum()
    se_hk = np.sqrt(var_hk)
    tq = student_t.ppf(0.975, k - 1)
    se_dl = np.sqrt(1 / ws.sum())
    pi_half = student_t.ppf(0.975, k - 2) * np.sqrt(tau2 + se_hk ** 2) if k > 2 else float("nan")
    return {"k": k, "pooled": float(mu), "tau2": float(tau2), "Q": float(q),
            "I2": float(max(0.0, (q - (k - 1)) / q)) if q > 0 else 0.0,
            "dl_ci95": [float(mu - 1.96 * se_dl), float(mu + 1.96 * se_dl)],
            "hksj_ci95": [float(mu - tq * se_hk), float(mu + tq * se_hk)],
            "prediction_interval95": [float(mu - pi_half), float(mu + pi_half)]}


def r4_pooling(r3):
    out = {}
    for label, arms in (("six filed arms", POOLED), ("corrected arms only", CORRECTED)):
        out[label] = {"arms": arms, **hksj([r3[a]["diff"] for a in arms],
                                           [r3[a]["delong_se"] for a in arms])}
    perm = {}
    for name in POOLED + [a for a in ARMS if "B-15" in a]:
        R = rows(ARMS[name][0])
        fail = np.array([r["align_selected"] < PASS for r in R])
        st, r2 = checks(R)
        perm[name] = paired_perm(st, r2, fail)
    out["paired_permutation_p"] = perm
    return out


# ----------------------------------------------------------------------------- shared stimuli
_STIM = {}


def stimulus(model_name, layer, ids, fp16=False, corpus_seed=0):
    """Rebuild a run's stimulus and responses exactly as e01_gate.py collected them."""
    import torch
    from caliper.activations import collect, load_model, sample_corpus
    key = (model_name, layer, tuple(ids), fp16, corpus_seed)
    if key not in _STIM:
        model, tok = load_model(model_name, dtype=torch.float16 if fp16 else torch.float32)
        p = collect(model, tok, sample_corpus(n_docs=300, seed=corpus_seed), layer=layer,
                    neurons=np.asarray(ids), max_tokens=8000, seed=0)
        _STIM[key] = (np.asarray(p.stimulus, float), np.asarray(p.response, float))
    return _STIM[key]


# ---------------------------------------------------------------------------------------- R2
def r2_function_space_agreement():
    path = "results/b15a_fitseed1.jsonl"
    R = rows(path)
    ids = [r["_key"] for r in R]
    S, _ = stimulus("gpt2", 6, ids)
    C = np.cov(S, rowvar=False)
    fresh = {u["unit"]: u for u in json.load(open(ROOT / "results/b15a_diagnostics_fresh.json"))["units"]}

    def med_cos(V, metric):
        cs = []
        for i in range(len(V)):
            for j in range(i + 1, len(V)):
                a, b = V[i], V[j]
                if metric == "C":
                    cs.append(abs(a @ C @ b) / np.sqrt((a @ C @ a) * (b @ C @ b)))
                else:
                    cs.append(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        return float(np.median(cs))

    feats = {"Euclidean, all 5 restarts": [], "Euclidean, random restarts 1-4": [],
             "covariance metric, all 5": [], "covariance metric, random 1-4": []}
    for r in R:
        V = np.load(ROOT / f"results/b15a_fitseed1_dirs/n{r['_key']}.npz")["direct_restarts"][:, :, 0]
        feats["Euclidean, all 5 restarts"].append(med_cos(V, "E"))
        feats["Euclidean, random restarts 1-4"].append(med_cos(V[1:], "E"))
        feats["covariance metric, all 5"].append(med_cos(V, "C"))
        feats["covariance metric, random 1-4"].append(med_cos(V[1:], "C"))
    labels = {"Euclidean < 0.95": [r["align_selected"] < PASS for r in R],
              "fresh sigma-align < 0.99": [fresh[r["_key"]]["sigma_align"] < 0.99 for r in R]}
    r2 = np.array([-r["r2_k1"] for r in R])
    out = {}
    for lab, f in labels.items():
        out[lab] = {"failures": int(sum(f)), "held-out R2": auc(r2, f),
                    "restart agreement as filed": auc(np.array([-r["stability"] for r in R]), f)}
        for k, v in feats.items():
            out[lab][k] = auc(-np.asarray(v), f)
    return out


# ---------------------------------------------------------------------------------------- R5
def cv_auc(X, y, groups, repeats, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    aucs = []
    for rep in range(repeats):
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + rep)
        pred = np.zeros(len(y))
        for tr, te in cv.split(X, y, groups):
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            m.fit(X[tr], y[tr])
            pred[te] = m.predict_proba(X[te])[:, 1]
        aucs.append(auc(pred, y))
    return float(np.mean(aucs))


def r5_incremental(n_boot=200):
    from scipy.stats import kurtosis
    specs = {"GPT-2 L6 (B-14)": ("gpt2", 6, False), "GPT-Neo-125m L10 (B-8b)": ("EleutherAI/gpt-neo-125M", 10, False),
             "Pythia-160m L6 (B-2c)": ("EleutherAI/pythia-160m", 6, True)}
    out = {"note": "point estimates: 50 x 5-fold; bootstrap: 10 x 5-fold per resample, units "
                   "grouped so duplicates stay in one fold"}
    for name, (model, layer, fp16) in specs.items():
        R = rows(ARMS[name][0])
        ids = [r["_key"] for r in R]
        _, Y = stimulus(model, layer, ids, fp16=fp16)
        base = np.column_stack([(Y > 0).mean(0), Y.mean(0), Y.std(0), kurtosis(Y, axis=0)])
        y = np.array([r["align_selected"] < PASS for r in R], int)
        r2 = np.array([[-r["r2_k1"]] for r in R]); st = np.array([[-r["stability"]] for r in R])
        sets = {"descriptors": base, "+ held-out R2": np.hstack([base, r2]),
                "+ restart agreement": np.hstack([base, st])}
        g = np.arange(len(y))
        point = {k: cv_auc(X, y, g, 50, 0) for k, X in sets.items()}
        rng = np.random.default_rng(0)
        inc = {"+ held-out R2": [], "+ restart agreement": []}
        for b in range(n_boot):
            idx = rng.choice(len(y), len(y))
            if y[idx].sum() < 10:
                continue
            a0 = cv_auc(sets["descriptors"][idx], y[idx], g[idx], 10, 1000 + b)
            for k in inc:
                inc[k].append(cv_auc(sets[k][idx], y[idx], g[idx], 10, 1000 + b) - a0)
        out[name] = {"cv_auc": point,
                     "increment": {k: {"point": point[k] - point["descriptors"],
                                       "boot_ci95": [float(np.percentile(v, 2.5)),
                                                     float(np.percentile(v, 97.5))]}
                                   for k, v in inc.items()}}
    return out


# ---------------------------------------------------------------------------------------- R6
def r6_pass_at_k():
    from scipy.optimize import minimize
    from scipy.stats import beta as beta_dist
    R = rows("results/b15a_fitseed1.jsonl")
    any_pass, sel_pass, any_rand = [], [], []
    for r in R:
        d = np.load(ROOT / f"results/b15a_fitseed1_dirs/n{r['_key']}.npz")
        V, r2, w = d["direct_restarts"][:, :, 0], d["direct_r2_restarts"], d["w"]
        al = np.abs(V @ w) / (np.linalg.norm(V, axis=1) * np.linalg.norm(w))
        any_pass.append([al[:k].max() >= PASS for k in range(1, 6)])
        sel_pass.append([al[np.argmax(r2[:k])] >= PASS for k in range(1, 6)])
        any_rand.append([al[1:1 + k].max() >= PASS for k in range(1, 5)])
    n = len(R)

    def ci(x):
        x = int(x)
        lo = beta_dist.ppf(0.025, x, n - x + 1) if x else 0.0
        hi = beta_dist.ppf(0.975, x + 1, n - x) if x < n else 1.0
        return [float(lo), float(hi)]

    def curve(M):
        M = np.asarray(M)
        return [{"k": k + 1, "rate": float(M[:, k].mean()), "ci95": ci(M[:, k].sum())}
                for k in range(M.shape[1])]

    obs = np.asarray(any_rand).sum(0)       # random restarts only: exchangeable draws
    ks = np.arange(1, len(obs) + 1)

    def nll(p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -(obs * np.log(p) + (n - obs) * np.log(1 - p)).sum()

    fits = {}
    res = minimize(lambda x: nll(1 - (1 - 1 / (1 + np.exp(-x[0]))) ** ks), [0.0])
    fits["one-class"] = {"q": float(1 / (1 + np.exp(-res.x[0]))), "nll": float(res.fun), "params": 1}
    res = minimize(lambda x: nll((1 / (1 + np.exp(-x[1]))) * (1 - (1 - 1 / (1 + np.exp(-x[0]))) ** ks)),
                   [0.0, 2.0])
    fits["two-class"] = {"q": float(1 / (1 + np.exp(-res.x[0]))),
                         "pi_unreachable": float(1 - 1 / (1 + np.exp(-res.x[1]))),
                         "nll": float(res.fun), "params": 2}
    res = minimize(lambda x: nll(1 - np.exp(betaln(np.exp(x[0]), np.exp(x[1]) + ks)
                                            - betaln(np.exp(x[0]), np.exp(x[1])))), [0.0, 0.0])
    fits["beta-geometric"] = {"a": float(np.exp(res.x[0])), "b": float(np.exp(res.x[1])),
                              "nll": float(res.fun), "params": 2}
    for f in fits.values():
        f["aic"] = 2 * f["params"] + 2 * f["nll"]
    return {"units": n, "any_restart_pass_at_k (with warm start)": curve(any_pass),
            "R2_selected_pass_at_k (with warm start)": curve(sel_pass),
            "any_random_restart_pass_at_k": curve(any_rand), "fits_on_random_restarts": fits,
            "note": "models describe 'any restart succeeds' over exchangeable random restarts; "
                    "nested k treated as independent binomials (descriptive AIC only)"}


# ---------------------------------------------------------------------------------------- R7
def r7_error_budget():
    ref = {r["_key"]: r for r in rows("results/b14_primary_gpt2_indep.jsonl")}
    out = {}
    for name in [a for a in ARMS if "B-15" in a]:
        R = {r["_key"]: r for r in rows(ARMS[name][0])}
        ks = sorted(set(R) & set(ref))
        a = np.array([ref[k]["align_selected"] < PASS for k in ks])
        b = np.array([R[k]["align_selected"] < PASS for k in ks])
        po = (a == b).mean()
        pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
        ang = lambda x: np.degrees(np.arccos(np.clip(x, 0, 1)))
        out[name] = {"shared": len(ks), "failures_reference": int(a.sum()),
                     "failures_replicate": int(b.sum()), "kappa": float((po - pe) / (1 - pe)),
                     "icc_alignment_angle": icc_a1(ang(np.array([ref[k]["align_selected"] for k in ks])),
                                                   ang(np.array([R[k]["align_selected"] for k in ks])))}
    return out


# ---------------------------------------------------------------------------------------- R8
def r8_error_spectrum():
    specs = {"GPT-2 L6 re-fit, fit seed 1 (B-15a)": ("gpt2", 6, False, "results/b15a_fitseed1"),
             "GPT-Neo-125m L10 (B-8b)": ("EleutherAI/gpt-neo-125M", 10, False, "results/b8b_gptneo125m_indep"),
             "Pythia-160m L6 (B-2c)": ("EleutherAI/pythia-160m", 6, True, "results/b2c_pythia160m_indep"),
             "GPT-Neo-125m L6 (B-17)": ("EleutherAI/gpt-neo-125M", 6, False, "results/b17_gptneo125m_l6")}
    out = {}
    for name, (model, layer, fp16, stem) in specs.items():
        R = rows(stem + ".jsonl")
        S, _ = stimulus(model, layer, [r["_key"] for r in R], fp16=fp16)
        ev, evec = np.linalg.eigh(np.cov(S, rowvar=False))      # ascending
        share = np.cumsum(ev) / ev.sum()
        low = evec[:, share <= 0.01]                             # bottom 1% of variance
        pos = ev[ev > ev.max() * 1e-12]
        res = {"dim": int(len(ev)), "condition_number": float(ev.max() / pos.min()),
               "effective_rank": float(np.exp(-(p := ev / ev.sum())[p > 0] @ np.log(p[p > 0]))),
               "low_variance_dims": int(low.shape[1]),
               "random_vector_expected_share": float(low.shape[1] / len(ev))}
        classes = {"converged-wrong": [], "under-fitted": []}
        for r in R:
            if r["align_selected"] >= PASS:
                continue
            d = np.load(ROOT / f"{stem}_dirs/n{r['_key']}.npz")
            f = d[r["picked"]][:, 0]
            f, w = f / np.linalg.norm(f), d["w"] / np.linalg.norm(d["w"])
            f = f * np.sign(f @ w) if f @ w != 0 else f
            e = f - w
            frac = float(((low.T @ e) ** 2).sum() / (e @ e))
            classes["converged-wrong" if r["r2_k1"] > WRONG_BASIN_R2 else "under-fitted"].append(frac)
        for c, v in classes.items():
            res[c] = {"n": len(v), "median_error_share_in_low_variance": float(np.median(v)) if v else None}
        out[name] = res
    return out


SECTIONS = {"R1": r1_route_matched, "R3": r3_all_arms, "R2": r2_function_space_agreement,
            "R5": r5_incremental, "R6": r6_pass_at_k, "R7": r7_error_budget, "R8": r8_error_spectrum}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    path = ROOT / "results/review_round1.json"
    out = json.load(open(path)) if path.exists() else {}
    todo = a.only or ["R1", "R3", "R4", "R2", "R6", "R7", "R8", "R5"]
    for s in todo:
        print("running", s, flush=True)
        if s == "R4":
            out["R4"] = r4_pooling(out.get("R3") or r3_all_arms())
        else:
            out[s] = SECTIONS[s]()
        json.dump(out, open(path, "w"), indent=1, default=float)
    print(json.dumps({k: out[k] for k in todo}, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
