"""B-14 analysis: the primary endpoint re-run, plus Addendum 1's secondaries.

Governed by docs/preregistration-b14-primary-rerun.md and
docs/preregistration-b14-addendum-1.md. Written and tested on B-1b's rows, which are
already unblinded, and on B-1b with its alignments scrambled across units. It is then run
on B-14 once. A change after that run is a deviation and is logged with the original
output kept.

    python experiments/analyse_b14.py --rows results/b1b_primary_gpt2.jsonl          # dev
    python experiments/analyse_b14.py --rows results/b1b_primary_gpt2.jsonl --scramble 0
    python experiments/analyse_b14.py --rows results/b14_primary_gpt2_indep.jsonl \
        --reference results/b1b_primary_gpt2.jsonl --out results/b14_analysis.json  # once

Two Addendum 1 items cannot be computed from what e01_gate.py archives, because it keeps
alignments but not the fitted directions: the identifiable-direction label (item 1) and
the functional correlation label (item 2). The script says so in its output rather than
skipping them silently. As the nearest computable substitute it reports, for each unit,
the cosine ceiling that the unidentifiable 1/gamma component imposes, and reruns the
primary test without units whose ceiling falls below the bar.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from b1_signal_calibration import PASS, benjamini_hochberg, delong, roc, wilson  # noqa: E402

# Orientation: higher = more suspect, the same convention as B-1b and B-10.
SIGNALS = {
    "held-out R2": lambda r: -r["r2_k1"],
    "restart agreement": lambda r: -r["stability"],
    "disagreement": lambda r: r["disagreement"],
    "restart R2 spread": lambda r: r["r2_spread"],
}
BARS = [0.90, 0.92, 0.95, 0.97, 0.98]
WRONG_BASIN_R2 = 0.99      # B-11 steps check's convergence line, fixed in the prereg
MIN_FAILURES = 36          # addendum 1 power figure for the primary arm


def load(path):
    rows = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            rows[int(str(r["_key"]).split("_")[-1])] = r
    return rows


def avg_precision(scores, labels):
    """Average precision with failures as positives; the PR-AUC reported beside ROC-AUC."""
    order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")
    y = np.asarray(labels, dtype=bool)[order]
    if y.sum() == 0:
        return float("nan")
    hits = np.cumsum(y)
    precision = hits / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


def spearman(a, b):
    from scipy.stats import spearmanr
    rho, p = spearmanr(a, b)
    return float(rho), float(p)


def decision(diff, p, n_fail):
    """The prereg's decision table, verbatim in logic."""
    if not np.isfinite(p):
        return "UNDEFINED - too few failures for DeLong"
    if p < 0.05 and diff > 0:
        return "REVERSAL - stop drafting and work out why"
    if p >= 0.05:
        verdict = "NOT SIGNIFICANT - B-1b's significance was partly a stopping artefact"
    elif diff <= -0.10:
        verdict = "HEADLINE STANDS on the fixed estimator"
    else:
        verdict = "RANKING STANDS, gap smaller than B-1b claimed"
    if n_fail < MIN_FAILURES:
        verdict += f" (UNDERPOWERED: {n_fail} failures < {MIN_FAILURES}; not extended)"
    return verdict


def paired_bootstrap(sa, sb, fail, n_boot=2000, seed=0):
    """Percentile CI on AUC(a) - AUC(b), resampling units within failure / pass strata."""
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(fail)
    neg = np.flatnonzero(~fail)
    out = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        lab = list(fail[idx])
        out.append(roc(list(sa[idx]), lab)[1] - roc(list(sb[idx]), lab)[1])
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def ln_null_ceiling(ids, layer):
    """Cosine ceiling from the 1/gamma component of each unit's weight column.

    The ln_2 output is gamma * z + beta with z zero-mean across coordinates, so s . (1/gamma)
    is constant on every token and w's component along 1/gamma cannot be recovered."""
    from caliper.activations import _blocks, _mlp_in_weight, load_model
    model, _ = load_model("gpt2")
    block = _blocks(model)[layer]
    u = 1.0 / block.ln_2.weight.detach().numpy()
    u /= np.linalg.norm(u)
    W = _mlp_in_weight(block)[:, ids]
    share = (u @ W) ** 2 / (W ** 2).sum(0)
    return np.sqrt(1.0 - share)


def nuisance_features(ids, layer, tokens):
    """Pre-fit unit descriptors, recomputed exactly as e01_gate.py collects its stimulus."""
    import torch
    from scipy.stats import kurtosis
    from caliper.activations import _blocks, _mlp_in, collect, load_model, sample_corpus
    model, tok = load_model("gpt2")
    p = collect(model, tok, sample_corpus(n_docs=300, seed=0), layer=layer,
                neurons=np.asarray(ids), max_tokens=tokens, seed=0)
    bias = _mlp_in(_blocks(model)[layer]).bias.detach().numpy()[ids]
    pre = p.stimulus @ p.weights + bias
    pre_t = torch.as_tensor(pre)
    normal = torch.distributions.Normal(0.0, 1.0)
    gelu_prime = (normal.cdf(pre_t) + pre_t * torch.exp(normal.log_prob(pre_t))).numpy()
    return {
        "active fraction": (pre > 0).mean(0),
        "z_mean": pre.mean(0),
        "response kurtosis": kurtosis(p.response, axis=0),
        "GELU first-order coefficient": gelu_prime.mean(0),
    }


def cv_auc(X, fail, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    prob = cross_val_predict(clf, X, fail, cv=cv, method="predict_proba")[:, 1]
    return roc(list(prob), list(fail))[1], prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--reference", default=None,
                    help="B-1b's rows, for the verdict-agreement McNemar (secondary 1)")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--tokens", type=int, default=8000)
    ap.add_argument("--summary", default=None, help="e01_gate summary json, for s/unit")
    ap.add_argument("--scramble", type=int, default=None,
                    help="dev mode: permute alignments across units with this seed")
    ap.add_argument("--no-model", action="store_true",
                    help="skip the parts that load GPT-2 (ceiling, nuisance baselines)")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = load(a.rows)
    ids = sorted(rows)
    R = [rows[i] for i in ids]
    align = np.array([r["align_selected"] for r in R])
    if a.scramble is not None:
        align = np.random.default_rng(a.scramble).permutation(align)
    fail = align < PASS
    n, n_fail = len(R), int(fail.sum())
    sig = {k: np.array([float(f(r)) for r in R]) for k, f in SIGNALS.items()}
    rep = {"rows": a.rows, "scrambled": a.scramble, "n": n, "n_failures": n_fail,
           "pass_rate": round(1 - n_fail / n, 4),
           "pass_wilson_95": [round(x, 4) for x in wilson(n - n_fail, n)]}

    # ---- primary
    d = delong(list(sig["restart agreement"]), list(sig["held-out R2"]), list(fail))
    rep["primary"] = {"auc_restart": d[0], "auc_r2": d[1], "diff_restart_minus_r2": d[2],
                      "se": d[3], "z": d[4], "p": d[5],
                      "verdict": decision(d[2], d[5], n_fail)}

    # ---- secondary 2: failure classification
    r2 = np.array([r["r2_k1"] for r in R])
    rep["failure_classes"] = {"wrong_basin_r2_gt_0.99": int((fail & (r2 > WRONG_BASIN_R2)).sum()),
                              "under_fitted": int((fail & (r2 <= WRONG_BASIN_R2)).sum())}

    # ---- secondary 1: agreement with the reference arm
    if a.reference:
        ref = load(a.reference)
        shared = [i for i in ids if i in ref]
        mine = {i: rows[i]["align_selected"] >= PASS for i in shared}
        theirs = {i: ref[i]["align_selected"] >= PASS for i in shared}
        b = sum(mine[i] and not theirs[i] for i in shared)
        c = sum(theirs[i] and not mine[i] for i in shared)
        from scipy.stats import binomtest
        rep["vs_reference"] = {
            "shared_units": len(shared), "agree": len(shared) - b - c,
            "pass_here_only": b, "pass_reference_only": c,
            "mcnemar_exact_p": float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0}
        rf = np.array([ref[i]["align_selected"] < PASS for i in shared])
        rr2 = np.array([ref[i]["r2_k1"] for i in shared])
        rep["vs_reference"]["reference_failure_classes"] = {
            "wrong_basin_r2_gt_0.99": int((rf & (rr2 > WRONG_BASIN_R2)).sum()),
            "under_fitted": int((rf & (rr2 <= WRONG_BASIN_R2)).sum())}

    # ---- secondary 3: paired bootstrap on the primary difference
    rep["primary"]["bootstrap_95_diff"] = paired_bootstrap(
        sig["restart agreement"], sig["held-out R2"], fail, n_boot=a.n_boot)

    # ---- secondary 4: the other two signals, BH across the secondary family
    sec_p = {}
    rep["signals"] = {}
    for k, s in sig.items():
        rep["signals"][k] = {"roc_auc": roc(list(s), list(fail))[1],
                             "pr_auc": avg_precision(s, fail),
                             "spearman_vs_alignment": spearman(-s, align)[0]}
    for k in ("disagreement", "restart R2 spread"):
        dd = delong(list(sig[k]), list(sig["held-out R2"]), list(fail))
        rep["signals"][k]["delong_minus_r2"] = {"diff": dd[2], "p": dd[5]}
        sec_p[f"{k} vs R2"] = dd[5]
    if "vs_reference" in rep:
        sec_p["mcnemar vs reference"] = rep["vs_reference"]["mcnemar_exact_p"]
    keys = list(sec_p)
    adj = benjamini_hochberg([sec_p[k] for k in keys])
    rep["secondary_bh"] = {k: {"p": sec_p[k], "q": q} for k, q in zip(keys, adj)}
    rep["pr_auc_prevalence_baseline"] = n_fail / n

    # ---- addendum 3: threshold sweep (descriptive)
    rep["threshold_sweep"] = {}
    for bar in BARS:
        fb = align < bar
        rep["threshold_sweep"][str(bar)] = {
            "failures": int(fb.sum()),
            **{k: (roc(list(s), list(fb))[1] if 0 < fb.sum() < n else None)
               for k, s in sig.items()}}

    # ---- addendum 7: calibration of a Platt-scaled R2 rule
    if 5 <= n_fail <= n - 5:
        auc_cv, prob = cv_auc(sig["held-out R2"][:, None], fail)
        q = np.quantile(prob, np.linspace(0, 1, 6))
        bins = np.clip(np.searchsorted(q, prob, side="right") - 1, 0, 4)
        rep["calibration_r2"] = {
            "brier": float(np.mean((prob - fail) ** 2)),
            "reliability_equal_mass_5": [
                {"mean_predicted": float(prob[bins == k].mean()),
                 "observed_rate": float(fail[bins == k].mean()), "n": int((bins == k).sum())}
                for k in range(5) if (bins == k).any()]}

    # ---- addendum 8: controls
    rng = np.random.default_rng(1)
    perm = []
    for _ in range(a.n_perm):
        pf = rng.permutation(fail)
        perm.append(delong(list(sig["restart agreement"]), list(sig["held-out R2"]),
                           list(pf))[2])
    perm = np.array(perm)
    rep["control_permuted_labels"] = {
        "n_perm": a.n_perm, "mean_diff": float(np.nanmean(perm)),
        "sd_diff": float(np.nanstd(perm)),
        "p_abs_ge_observed": float(np.mean(np.abs(perm) >= abs(d[2])))}
    rep["control_cheating_signal_auc"] = roc(list(-align), list(fail))[1]

    # ---- addendum 1/2 and 6: need the model
    rep["identifiable_direction_label"] = ("NOT COMPUTABLE: e01_gate.py archives alignments, "
                                           "not fitted directions (deviation from addendum 1 "
                                           "items 1-2)")
    if not a.no_model:
        ceil = ln_null_ceiling(ids, a.layer)
        limited = ceil < PASS
        rep["ln_null_ceiling"] = {"units_with_ceiling_below_bar": [int(i) for i in
                                                                   np.array(ids)[limited]],
                                  "median_ceiling": float(np.median(ceil))}
        if limited.any():
            keep = ~limited
            ds = delong(list(sig["restart agreement"][keep]), list(sig["held-out R2"][keep]),
                        list(fail[keep]))
            rep["ln_null_ceiling"]["primary_without_limited_units"] = {"diff": ds[2], "p": ds[5]}
        feats = nuisance_features(ids, a.layer, a.tokens)
        base = np.column_stack(list(feats.values()))
        rep["nuisance"] = {k: roc(list(-v if k != "response kurtosis" else v), list(fail))[1]
                           for k, v in feats.items()}
        rep["nuisance"]["orientation_note"] = ("AUC of each descriptor alone; active fraction, "
                                               "z_mean and GELU coefficient oriented low = suspect")
        if 5 <= n_fail <= n - 5:
            b0, _ = cv_auc(base, fail)
            b_r2, _ = cv_auc(np.column_stack([base, sig["held-out R2"]]), fail)
            b_st, _ = cv_auc(np.column_stack([base, sig["restart agreement"]]), fail)
            rep["incremental_cv_auc"] = {"baselines": b0, "baselines + R2": b_r2,
                                         "baselines + restart agreement": b_st}

    if a.summary:
        s = json.load(open(a.summary))
        rep["seconds_per_unit"] = s.get("seconds_per_neuron")

    p = rep["primary"]
    print(f"rows {a.rows}  n={n}  failures={n_fail}"
          + (f"  [SCRAMBLED seed {a.scramble}]" if a.scramble is not None else ""))
    print(f"PRIMARY  AUC restart {p['auc_restart']:.3f}  AUC R2 {p['auc_r2']:.3f}  "
          f"diff {p['diff_restart_minus_r2']:+.3f}  p={p['p']:.2g}  "
          f"bootstrap95 [{p['bootstrap_95_diff'][0]:+.3f}, {p['bootstrap_95_diff'][1]:+.3f}]")
    print(f"VERDICT  {p['verdict']}")
    print(f"failure classes {rep['failure_classes']}")
    if "vs_reference" in rep:
        print(f"vs reference {rep['vs_reference']}")
    print(f"permutation control: |diff| >= observed in "
          f"{rep['control_permuted_labels']['p_abs_ge_observed']:.3f} of {a.n_perm}")
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=2, default=float)
        print(f"wrote {a.out}")
    return rep


if __name__ == "__main__":
    main()
